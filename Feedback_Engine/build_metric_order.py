import json
from typing import Any, List, Optional


MetricDirection = Optional[bool]
# True  = higher is better
# False = lower is better
# None  = no inherent ordering


def normalize_to_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be None")

    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} must not be empty")

    return text


def normalize_target_metrics(target_metrics: Any) -> List[str]:
    if isinstance(target_metrics, str):
        metrics = [normalize_to_str(target_metrics, "metric")]
    elif isinstance(target_metrics, list):
        metrics = [
            normalize_to_str(metric, "metric")
            for metric in target_metrics
        ]
    else:
        raise TypeError(
            "target_metrics must be a string or a list of strings"
        )

    if not metrics:
        raise ValueError("target_metrics must not be empty")

    return metrics


def build_higher_is_better_messages(
    task_info: Any,
    target_metrics: Any,
) -> list[dict]:
    task_info_text = normalize_to_str(task_info, "task_info")
    metrics = normalize_target_metrics(target_metrics)

    system_message = """
Determine the optimization direction of each ML metric.

Return true if higher is better, false if lower is better, and null if
the metric has no inherent ordering or is only a constraint/reference.

Use the metric name first and task_info only for ambiguous metrics.

Examples:
- true: accuracy, auc, f1, precision, recall, iou, dice, bleu, rouge,
  mAP, ndcg, r2
- false: loss, error, wer, cer, perplexity, mae, mse, rmse, latency,
  runtime, epoch_time, memory, energy
- usually null unless task_info says otherwise: dataset_size,
  sample_count, augmentation_ratio, parameter_count

Return only:
{"higher_is_better_list":[true,false,null]}

The list must match target_metrics in length and order. Each item must
be true, false, or null.
""".strip()

    user_message = json.dumps(
        {
            "task_info": task_info_text,
            "target_metrics": metrics,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def generate_higher_is_better_text(
    llm_client,
    task_info: Any,
    target_metrics: Any,
) -> str:
    messages = build_higher_is_better_messages(
        task_info=task_info,
        target_metrics=target_metrics,
    )
    return llm_client.generate(messages=messages)


def parse_higher_is_better_response(
    response_text: str,
    target_metrics: Any,
) -> List[MetricDirection]:
    metrics = normalize_target_metrics(target_metrics)

    data = json.loads(response_text)

    if not isinstance(data, dict):
        raise TypeError("LLM response must be a JSON object")

    higher_is_better_list = data.get("higher_is_better_list")

    if not isinstance(higher_is_better_list, list):
        raise TypeError("higher_is_better_list must be a list")

    if len(higher_is_better_list) != len(metrics):
        raise ValueError(
            "higher_is_better_list length must match target_metrics length"
        )

    for index, item in enumerate(higher_is_better_list):
        if item is not None and not isinstance(item, bool):
            raise TypeError(
                f"higher_is_better_list[{index}] must be true, false, or null"
            )

    return higher_is_better_list


def generate_higher_is_better_list(
    llm_client,
    task_info: Any,
    target_metrics: Any,
) -> List[MetricDirection]:
    response_text = generate_higher_is_better_text(
        llm_client=llm_client,
        task_info=task_info,
        target_metrics=target_metrics,
    )

    return parse_higher_is_better_response(
        response_text=response_text,
        target_metrics=target_metrics,
    )