import json
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

DATASET_PATH = "/home/xiaoma/Augmentation_Test/MustARD/data/sarcasm_data.json"
SPLIT_PATH = "/home/xiaoma/Augmentation_Test/MustARD/data/split_indices.p"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 512


def _initialize_random_states(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_samples() -> List[Tuple[Dict[str, Any], int]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    values = raw.values() if isinstance(raw, dict) else raw
    samples = []
    for item in values:
        inputs = {
            "utterance": str(item.get("utterance", "")),
            "speaker": str(item.get("speaker", "")),
            "context": [str(x) for x in item.get("context", [])],
            "context_speakers": [str(x) for x in item.get("context_speakers", [])],
        }
        samples.append((inputs, int(bool(item["sarcasm"]))))
    return samples


def _load_split_object() -> Any:
    with open(SPLIT_PATH, "rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _as_indices(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _extract_pair(item: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        train = item.get("train", item.get("train_idx", item.get("train_indices")))
        valid = item.get("test", item.get("test_idx", item.get("test_indices", item.get("val", item.get("validation")))))
        return _as_indices(train), _as_indices(valid)
    return _as_indices(item[0]), _as_indices(item[1])


def _official_folds(split_object: Any, sample_count: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_indices = np.arange(sample_count, dtype=np.int64)
    if isinstance(split_object, dict):
        if any(key in split_object for key in ("train", "train_idx", "train_indices")):
            trains = split_object.get("train", split_object.get("train_idx", split_object.get("train_indices")))
            tests = split_object.get("test", split_object.get("test_idx", split_object.get("test_indices", split_object.get("val", split_object.get("validation")))))
            if len(trains) == 5 and len(tests) == 5:
                return [(_as_indices(a), _as_indices(b)) for a, b in zip(trains, tests)]
            return [(_as_indices(trains), _as_indices(tests))]
        items = list(split_object.values())
    else:
        items = list(split_object)

    if len(items) == 2 and len(items[0]) == 5 and len(items[1]) == 5:
        first = items[0][0]
        second = items[1][0]
        if not isinstance(first, (int, np.integer)) and not isinstance(second, (int, np.integer)):
            return [(_as_indices(a), _as_indices(b)) for a, b in zip(items[0], items[1])]

    folds = []
    for item in items:
        if isinstance(item, dict) or (isinstance(item, tuple) and len(item) == 2):
            folds.append(_extract_pair(item))
        else:
            valid = _as_indices(item)
            train = np.setdiff1d(all_indices, valid, assume_unique=False)
            folds.append((train, valid))
    return folds


def _baseline_feature_fn(
    train_inputs: Sequence[Dict[str, Any]],
    train_labels: Sequence[int],
    eval_inputs: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    del train_labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def encode(records: Sequence[Dict[str, Any]]) -> np.ndarray:
        batches = []
        with torch.inference_mode():
            for start in range(0, len(records), BATCH_SIZE):
                batch = records[start : start + BATCH_SIZE]
                contexts = [" ".join(record["context"]) for record in batch]
                utterances = [record["utterance"] for record in batch]
                tokens = tokenizer(
                    contexts,
                    utterances,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(batches, axis=0)

    return encode(train_inputs), encode(eval_inputs)


def _classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("standardscaler", StandardScaler(copy=True, with_mean=True, with_std=True)),
            (
                "svc",
                SVC(
                    C=10,
                    kernel="rbf",
                    degree=3,
                    gamma="scale",
                    coef0=0.0,
                    shrinking=True,
                    probability=False,
                    tol=1e-3,
                    cache_size=200,
                    class_weight=None,
                    verbose=False,
                    max_iter=-1,
                    decision_function_shape="ovr",
                    break_ties=False,
                    random_state=42,
                ),
            ),
        ]
    )


def train_pipeline(aug_fn=None, feature_fn=None, seed=42):
    _initialize_random_states(seed)
    samples = _load_samples()
    folds = _official_folds(_load_split_object(), len(samples))
    extractor = _baseline_feature_fn if feature_fn is None else feature_fn
    fold_scores = []

    for train_indices, validation_indices in folds:
        train_data = [(dict(samples[i][0]), int(samples[i][1])) for i in train_indices]
        validation_inputs = [dict(samples[i][0]) for i in validation_indices]
        validation_labels = np.asarray([samples[i][1] for i in validation_indices], dtype=np.int64)

        if aug_fn is not None:
            train_data = aug_fn(train_data)

        train_inputs = [record for record, label in train_data]
        train_labels = np.asarray([label for record, label in train_data], dtype=np.int64)
        X_train, X_validation = extractor(train_inputs, train_labels, validation_inputs)
        X_train = np.asarray(X_train, dtype=np.float64)
        X_validation = np.asarray(X_validation, dtype=np.float64)

        classifier = _classifier()
        classifier.fit(X_train, train_labels)
        predictions = classifier.predict(X_validation)
        fold_scores.append(float(f1_score(validation_labels, predictions, average="weighted")))

    return {
        "target_metrics": {"f1_weighted": float(np.mean(fold_scores))},
        "aug_fn": "baseline_no_augmentation" if aug_fn is None else getattr(aug_fn, "__name__", aug_fn.__class__.__name__),
        "feature_fn": "frozen_bert_base_uncased_attention_mask_mean_pooling" if feature_fn is None else getattr(feature_fn, "__name__", feature_fn.__class__.__name__),
    }


if __name__ == "__main__":
    print(json.dumps(train_pipeline(), indent=2))
