def metric_direction_info(higher_is_better):
    if higher_is_better is True:
        return {
            "higher_is_better": True,
            "optimization_direction": "maximize",
            "better_delta_sign": "positive",
        }

    if higher_is_better is False:
        return {
            "higher_is_better": False,
            "optimization_direction": "minimize",
            "better_delta_sign": "negative",
        }

    return {
        "higher_is_better": None,
        "optimization_direction": "unknown",
        "better_delta_sign": "unknown",
    }


def interpret_delta(delta, higher_is_better, tolerance=1e-12):
    if delta is None or higher_is_better is None:
        return "unknown"

    if abs(delta) <= tolerance:
        return "same"

    if higher_is_better:
        return "better" if delta > 0 else "worse"

    return "better" if delta < 0 else "worse"


def metric_gradient(metric_name, current_value, reference_value, higher_is_better):
    delta = current_value - reference_value

    info = metric_direction_info(higher_is_better)

    return {
        "metric_name": metric_name,
        "current": current_value,
        "reference": reference_value,
        "delta": delta,
        "relative_delta": delta / reference_value if reference_value != 0 else None,
        "higher_is_better": info["higher_is_better"],
        "optimization_direction": info["optimization_direction"],
        "better_delta_sign": info["better_delta_sign"],
        "change": interpret_delta(delta, higher_is_better),
    }


def build_metric_direction_map(target_metrics, higher_is_better_list):
    return {
        metric_name: higher_is_better
        for metric_name, higher_is_better in zip(
            target_metrics,
            higher_is_better_list,
        )
    }


def metric_gradients_for_all_metrics(
    current_metrics,
    baseline_metrics,
    target_metrics,
    higher_is_better_list,
    last_round_best_metrics=None,
    history_best_metrics=None,
    noise_std=None,
):
    metric_direction_map = build_metric_direction_map(
        target_metrics,
        higher_is_better_list,
    )

    gradients = {}

    for index, metric_name in enumerate(target_metrics):
        current_value = current_metrics[metric_name]
        higher_is_better = metric_direction_map[metric_name]

        vs_baseline = metric_gradient(
            metric_name,
            current_value,
            baseline_metrics[metric_name],
            higher_is_better,
        )

        if last_round_best_metrics is not None:
            vs_last = metric_gradient(
                metric_name,
                current_value,
                last_round_best_metrics[metric_name],
                higher_is_better,
            )
        else:
            vs_last = None

        if history_best_metrics is not None:
            vs_history = metric_gradient(
                metric_name,
                current_value,
                history_best_metrics[metric_name],
                higher_is_better,
            )
        else:
            vs_history = None

        direction_info = metric_direction_info(higher_is_better)

        gradients[metric_name] = {
            "higher_is_better": direction_info["higher_is_better"],
            "optimization_direction": direction_info["optimization_direction"],
            "better_delta_sign": direction_info["better_delta_sign"],
            "metric_value": current_value,


            "delta_vs_baseline": vs_baseline["delta"],
            "relative_delta_vs_baseline": vs_baseline["relative_delta"],
            "change_vs_baseline": vs_baseline["change"],

            "delta_vs_last_round_best": (
                vs_last["delta"] if vs_last else None
            ),
            "relative_delta_vs_last_round_best": (
                vs_last["relative_delta"] if vs_last else None
            ),
            "change_vs_last_round_best": (
                vs_last["change"] if vs_last else None
            ),

            "delta_vs_history_best": (
                vs_history["delta"] if vs_history else None
            ),
            "relative_delta_vs_history_best": (
                vs_history["relative_delta"] if vs_history else None
            ),
            "change_vs_history_best": (
                vs_history["change"] if vs_history else None
            ),
        }
        if index == 0 and noise_std is not None:
            gradients[metric_name]["noise_std"] = noise_std

    return gradients

def gradient_analysis(
    baseline,
    current_round_results,
    history_topk_so_far,
    last_round_best,
    target_metrics,
    higher_is_better_list,
):
    baseline_metrics = baseline["target_metrics"]

    last_round_best_metrics = None
    if last_round_best is not None:
        last_round_best_metrics = last_round_best["target_metrics"]

    history_best = history_topk_so_far[0] if history_topk_so_far else None
    history_best_metrics = None
    if history_best is not None:
        history_best_metrics = history_best["target_metrics"]

    current_round_gradient_summary = []

    for pipeline_result in current_round_results:
        current_metrics = pipeline_result["target_metrics"]

        pipeline_gradient_summary = {
            "round": pipeline_result["round"],
            "pipeline_id": pipeline_result["pipeline_id"],
            "target_metrics": current_metrics,
            "metric_gradients": metric_gradients_for_all_metrics(
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                target_metrics=target_metrics,
                higher_is_better_list=higher_is_better_list,
                last_round_best_metrics=last_round_best_metrics,
                history_best_metrics=history_best_metrics,
                noise_std=pipeline_result.get("noise_std"),
),
        }

        current_round_gradient_summary.append(pipeline_gradient_summary)

    return {
        "target_metrics": target_metrics,
        "higher_is_better_list": higher_is_better_list,
        "baseline": {
            "pipeline_id": baseline.get("pipeline_id"),
            "target_metrics": baseline_metrics,
        },
        "last_round_best": (
            {
                "round": last_round_best["round"],
                "pipeline_id": last_round_best["pipeline_id"],
                "target_metrics": last_round_best_metrics,
            }
            if last_round_best is not None
            else None
        ),
        "history_best": (
            {
                "round": history_best["round"],
                "pipeline_id": history_best["pipeline_id"],
                "target_metrics": history_best_metrics,
            }
            if history_best is not None
            else None
        ),
        "current_round_gradient_summary": current_round_gradient_summary,
    }

def build_round_conclusion_pipeline_summaries(
    gradient_ana,
    pipeline_items,
):
    items = gradient_ana["current_round_gradient_summary"]

    pipeline_map = {
        item["pipeline_id"]: item
        for item in pipeline_items
    }

    pipeline_summaries = []

    for item in items:
        pipeline_id = item["pipeline_id"]
        pipeline_item = pipeline_map.get(pipeline_id, {})

        pipeline_summary = {
        "round": item["round"],
        "pipeline_id": pipeline_id,
        "family": pipeline_item.get("family"),
        "arm_role": pipeline_item.get("arm_role"),
        "pipeline_description": pipeline_item.get("pipeline_description"),
        "target_metrics": item["target_metrics"],
        "metric_gradients": item["metric_gradients"],
        }

        pipeline_summaries.append(pipeline_summary)

    return pipeline_summaries