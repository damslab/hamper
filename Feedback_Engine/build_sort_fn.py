import json
from typing import Callable, Dict, Any, List


def normalize_to_str(value: Any, field_name: str) -> str:
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value).strip()

    return text


def normalize_metric(metric: Any) -> str:
    if isinstance(metric, list):
        metric = metric[0]

    return normalize_to_str(metric, "metric")


def build_sort_order_messages(task_info: Any, metric: Any) -> list[dict]:
    task_info = normalize_to_str(task_info, "task_info")
    metric = normalize_metric(metric)

    system_message = """
You are an assistant that determines the correct sorting order for ranking machine learning pipelines.

Your task is to decide whether pipelines should be sorted in ascending order or descending order based on the given task information and metric name.

Decision rule:
- Return "desc" if a larger metric value means a better pipeline.
- Return "asc" if a smaller metric value means a better pipeline.
- Return "unknown" if the correct direction cannot be determined reliably.

General guidance:
- Metrics such as accuracy, acc, auc, f1, precision, recall, iou, dice, bleu, rouge, map, and ndcg are usually better when larger.
- Metrics such as loss, error, wer, cer, perplexity, mae, mse, rmse, latency, runtime, epoch_time, and memory are usually better when smaller.
- Use both the task information and the metric name to make the decision.
- If the metric meaning is ambiguous, infer from the task when possible.
- If you still cannot determine the direction confidently, return "unknown".

Output requirements:
- Output only one JSON object.
- Do not output any extra text.
- Do not use markdown code fences.

The JSON format must be exactly:
{
  "order": "asc" | "desc" | "unknown",
  "higher_is_better": true | false | null,
  "reason": "brief explanation"
}
""".strip()

    user_message = f"""
Task information:
{task_info}

Metric:
{metric}
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def generate_sort_order_text(llm_client, task_info: str, metric_name: str) -> str:
    messages = build_sort_order_messages(task_info, metric_name)
    return llm_client.generate(messages=messages)


def parse_sort_order_response(response_text: str) -> Dict[str, Any]:
    data = json.loads(response_text)
    return data


def build_sort_fn_from_order(
    order: str,
    metric_name: str,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    metric_name = normalize_metric(metric_name)
    reverse = order == "desc"

    def sort_fn(pipelines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            pipelines,
            key=lambda pipeline: float(pipeline["target_metrics"][metric_name]),
            reverse=reverse,
        )

    return sort_fn


def build_sort_fn_from_llm_response(
    llm_response_text: str,
    metric_name: str,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    parsed = parse_sort_order_response(llm_response_text)
    order = parsed["order"]

    if order == "unknown":
        raise ValueError(
            f"LLM could not determine sort order reliably for metric '{metric_name}'. "
            f"Reason: {parsed.get('reason')}"
        )

    return build_sort_fn_from_order(
        order=order,
        metric_name=metric_name,
    )


def generate_sort_fn(
    llm_client,
    task_info: str,
    metric_name: str,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    llm_response_text = generate_sort_order_text(
        llm_client=llm_client,
        task_info=task_info,
        metric_name=metric_name,
    )

    return build_sort_fn_from_llm_response(
        llm_response_text=llm_response_text,
        metric_name=metric_name,
    )