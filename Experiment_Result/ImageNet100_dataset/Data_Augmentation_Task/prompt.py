train_spec = """
Please write a complete, runnable Python training script for image classification on an ImageNet100-style dataset using ResNet50.

Dataset details:
- The dataset root is: /home/xiaoma/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8
- Training images are stored in these folders:
  - train.X1
  - train.X2
  - train.X3
  - train.X4
- Validation images are stored in:
  - val.X
- Each split folder contains class subfolders named like ImageNet WNIDs, for example n01440764.
- The script should automatically scan all WNID class folders and build a consistent class_to_idx mapping.

Model:
- Use torchvision.models.resnet50.
- Train from scratch by default, with weights=None.
- Set num_classes according to the number of detected classes.

Preprocessing:
- The CPU Dataset should only read images, convert them to RGB, Resize the shorter side to 256, then CenterCrop to 224x224 deterministically if necessary, and return uint8 tensors with shape [3, 224, 224] in CHW format.
- After transferring each batch to GPU, convert images to float32, divide by 255, and normalize using ImageNet mean and std on GPU.

Training requirements:
- Use PyTorch.
- Support CUDA if available.
- Use mixed precision training with torch.cuda.amp when CUDA is available.
- Use SGD with momentum=0.9 and weight_decay=1e-4.
- Use a learning rate scheduler with warmup and milestone decay.
- Support gradient accumulation.
- Support channels_last memory format when CUDA is available.
- Save the best checkpoint according to validation top-1 accuracy.
- Only run exactly 20 epochs.


Reproducibility:
- Add a seed_all function.
- Seed Python random, torch, CUDA, and DataLoader workers.
- Use a torch.Generator for train DataLoader shuffling.

"""
user_message = """
ImageNet-100 is a subset of the ImageNet-1k dataset from the ImageNet Large Scale Visual Recognition Challenge 2012. It contains 100 randomly selected classes, as specified in the Labels.json file.

The training split contains 1300 images per class, and the validation split contains 50 images per class.

Design a data-augmentation pipeline for training a ResNet-50 classifier.

The two target metrics are:
1. validation accuracy
2. epoch time

The preprocessing pipeline already includes image resizing and center cropping. Focus only on augmentation methods that may improve validation accuracy while keeping epoch time reasonable.

Important implementation constraints:
1. The augmentation must be applied during training only.
2. The augmentation must be applied at batch level.
3. The augmentation must be placed after resize and center crop, but before normalization.
4. The augmentation receives a mini-batch of images after resize and center crop.
5. The augmentation must not change the original sample labels.
6. The augmentation must return the augmented image batch and the unchanged label batch.
7. Do not use dataset-level augmentation.
8. Do not use sample-level augmentation.
9. Do not use augmentation methods that require modifying labels or producing soft labels.
10. Validation and test-time data must not be augmented.
"""