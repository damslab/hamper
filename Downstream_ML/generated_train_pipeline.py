import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import (
    Compose,
    CenterCrop,
    InterpolationMode,
    PILToTensor,
    Resize,
)


_DATA_ROOT = Path("/home/xiaoma/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8")
_TRAIN_SPLITS = ("train.X1", "train.X2", "train.X3", "train.X4")
_VALIDATION_SPLIT = "val.X"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_EPOCHS = 30
_BATCH_SIZE = 64
_ACCUMULATION_STEPS = 2
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _ImageDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

        self.preprocess = Compose([
            Resize(
                256,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            CenterCrop((224, 224)),
            PILToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.preprocess(image)

        return image, target

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _scan_samples():
    split_paths = [_DATA_ROOT / name for name in (*_TRAIN_SPLITS, _VALIDATION_SPLIT)]
    class_names = sorted(
        {
            class_dir.name
            for split_path in split_paths
            for class_dir in split_path.iterdir()
            if class_dir.is_dir()
        }
    )
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    def scan(split_names):
        samples = []
        for split_name in split_names:
            split_path = _DATA_ROOT / split_name
            for class_name in class_names:
                class_path = split_path / class_name
                if class_path.is_dir():
                    image_paths = sorted(
                        path
                        for path in class_path.rglob("*")
                        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
                    )
                    samples.extend((str(path), class_to_idx[class_name]) for path in image_paths)
        return samples

    return scan(_TRAIN_SPLITS), scan((_VALIDATION_SPLIT,)), len(class_names)


def _loader(dataset, shuffle, generator, use_cuda, num_workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": _BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "drop_last": False,
        "worker_init_fn": _seed_worker,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _learning_rate(epoch):
    if epoch == 1:
        return 0.01
    if epoch == 2:
        return 0.055
    if epoch <= 15:
        return 0.1
    if epoch <= 23:
        return 0.01
    return 0.001


def _normalize(images, mean, std):
    return (images - mean) / std


def train_pipeline(aug_fn=None, seed=42):
    _seed_all(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    num_workers = min(8, os.cpu_count() or 0)

    train_samples, validation_samples, num_classes = _scan_samples()
    train_dataset = _ImageDataset(train_samples)
    validation_dataset = _ImageDataset(validation_samples)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = _loader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        use_cuda=use_cuda,
        num_workers=num_workers,
    )
    validation_loader = _loader(
        validation_dataset,
        shuffle=False,
        generator=None,
        use_cuda=use_cuda,
        num_workers=num_workers,
    )

    model = resnet50(weights=None, num_classes=num_classes).to(device)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        dampening=0,
        nesterov=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    mean = torch.tensor(_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    best_validation_accuracy = 0.0
    epoch_times = []

    for epoch in range(1, _EPOCHS + 1):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = _learning_rate(epoch)

        if use_cuda:
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0
        accumulated_samples = 0

        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device=device, non_blocking=True).float().div_(255.0)
            if use_cuda:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device=device, non_blocking=True)
            targets = F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)

            if aug_fn is not None:
                images, targets = aug_fn(images, targets)

            images = _normalize(images, mean, std)
            batch_samples = images.shape[0]
            autocast_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda
                else nullcontext()
            )
            with autocast_context:
                logits = model(images)
            per_sample_loss = -(
                targets * F.log_softmax(logits.float(), dim=1)
            ).sum(dim=1)
            loss_sum = per_sample_loss.sum()
            scaler.scale(loss_sum).backward()

            accumulated_batches += 1
            accumulated_samples += batch_samples
            final_batch = batch_index + 1 == len(train_loader)
            if accumulated_batches == _ACCUMULATION_STEPS or final_batch:
                scaler.unscale_(optimizer)
                inverse_sample_count = 1.0 / accumulated_samples
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(inverse_sample_count)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                accumulated_samples = 0

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_count = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images = images.to(device=device, non_blocking=True).float().div_(255.0)
                labels = labels.to(device=device, non_blocking=True)
                if use_cuda:
                    images = images.contiguous(memory_format=torch.channels_last)
                images = _normalize(images, mean, std)
                autocast_context = (
                    torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    if use_cuda
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(images)
                batch_loss = F.cross_entropy(logits.float(), labels, reduction="sum")
                validation_loss_sum += batch_loss.item()
                validation_correct += (logits.argmax(dim=1) == labels).sum().item()
                validation_count += labels.shape[0]

        validation_accuracy = validation_correct / validation_count
        best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)

        if use_cuda:
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)

        validation_loss = validation_loss_sum / validation_count
        print(
            f"epoch={epoch} lr={_learning_rate(epoch):.6g} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.6f} "
            f"epoch_time={epoch_times[-1]:.3f}s",
            flush=True,
        )

    result = {
        "aug_fn": "baseline" if aug_fn is None else getattr(aug_fn, "__name__", aug_fn.__class__.__name__),
        "target_metrics": {
            "best_validation_accuracy": float(best_validation_accuracy),
            "average_epoch_time": float(sum(epoch_times) / len(epoch_times)),
        },
    }
    return result


if __name__ == "__main__":
    print(json.dumps(train_pipeline(aug_fn=None, seed=42), sort_keys=True))
