import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union


def normalize_history_path(history_pipeline_path: Union[str, Path]) -> Path:
    if isinstance(history_pipeline_path, str):
        if not history_pipeline_path.strip():
            raise ValueError("history_pipeline_path must be a non-empty string")
        return Path(history_pipeline_path)

    if isinstance(history_pipeline_path, Path):
        return history_pipeline_path

    raise TypeError("history_pipeline_path must be a str or Path")


def filtered_history_result(
    history_pipeline_path: Union[str, Path],
    keys_to_keep = {
    "pipeline_id",
    "family",
    "arm_role",
    "noise_std",
    "target_metrics",
    "pipeline_description",
    "status",
    "round",
},
) -> List[Dict[str, Any]]:
    history_pipeline_path = normalize_history_path(history_pipeline_path)

    with open(history_pipeline_path, "r", encoding="utf-8") as f:
        history_result = [
            json.loads(line)
            for line in f
            if line.strip()
        ]
    filtered_result = [
        {k: v for k, v in item.items() if k in keys_to_keep}
        for item in history_result
        if item.get("status") == "success"
    ]

    return filtered_result


def select_pipeline_by_kround(
    pipeline_result: List[Dict[str, Any]],
    n: int,
    k: int = 2,
) -> List[Dict[str, Any]]:
    start_round = max(0, n - k)

    recent_pipeline_items = [
        item
        for item in pipeline_result
        if item.get("round") is not None
        and start_round <= int(item["round"]) <= n
    ]
    return recent_pipeline_items


def split_by_current_round(
    pipeline_result: List[Dict[str, Any]],
    n: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    current_round_items = []
    previous_round_items = []

    for item in pipeline_result:
        if item.get("round") is None:
            continue

        round_id = int(item["round"])

        if round_id == n:
            current_round_items.append(item)
        elif round_id < n:
            previous_round_items.append(item)

    return current_round_items, previous_round_items


