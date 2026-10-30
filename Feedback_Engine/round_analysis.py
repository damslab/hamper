import json
from typing import Any, Dict, List
from pathlib import Path
ROUND_NOISE_ENABLED = """
Noise analysis:
- After multi-seed evaluation, the system updates the PRIMARY metric to
  its mean. noise_std is its across-seed sample standard deviation,
  not the standard error or across-fold dispersion.
- noise_std = 0 means the pipeline's PRIMARY-metric result is unaffected
  by random-seed variation across the tested seeds under the fixed
  evaluation setup.
- Missing noise_std means unavailable, not zero.
- Compare PRIMARY means using both pipelines' available noise information.
  Differences comparable to noise are inconclusive; exceeding noise alone
  does not prove significance.
- Never apply PRIMARY noise_std to SECONDARY metrics.
- Do not propose seed-only replications; the system handles them.
""".strip()

ROUND_NOISE_DISABLED = """
Noise analysis:
- Random-seed analysis is disabled. Ignore random-seed effects and any
  seed-noise information in the inputs.
- Focus on identifying effective augmentation methods using the reported
  metrics, optimization directions, controlled comparisons, and trade-offs.
- Do not attribute metric differences to random seeds, discuss seed
  uncertainty, request noise estimates, or recommend seed-only replications.
- Do not mark an outcome inconclusive merely because seed-noise information
  or multi-seed confirmation is absent.
- Continue checking implementation and design confounds. Do not claim
  statistical significance or seed robustness.
""".strip()
def load_round_records(
    jsonl_path: str | Path,
    round_index: int,
) -> list[dict]:
    """
    从 jsonl 文件中读取指定 round_index 的所有记录。
    不过滤 type，不只返回 round_summary。
    """

    jsonl_path = Path(jsonl_path)

    if not jsonl_path.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    records = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {e}"
                ) from e

            if record.get("round_index") == round_index:
                records.append(record)

    return records
def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
def build_round_analysis_messages(
    spec: Dict[str, Any],
    round_hypothesis: Any,
    round_gradient: Any,
    noise_analysis_enabled: bool = False,
) -> List[Dict[str, str]]:
    dataset_info = spec.get("dataset_info", {})
    task_info = spec.get("task_info", {})
    user_goal=spec.get("user_goal", {})
    system_prompt = """
You analyze one completed round of an iterative data-augmentation search.
Your output guides the model that proposes the next round.

Inputs:
- dataset/task information
- the round hypothesis, written before pipeline generation
- measured gradients containing structured pipeline operations and metrics

Use the gradient as the sole source of measured results. Dataset/task
information may explain results but must not override them.

Metric rules:
- The first target_metrics item is PRIMARY; others are SECONDARY.
- Interpret every metric using its provided direction metadata. Never assume
  a positive delta is beneficial.
- Judge pipelines mainly by the PRIMARY metric while reporting meaningful
  SECONDARY movements and trade-offs.
- State whether each cited change is vs baseline, previous-round best, or
  history best.

Hypothesis judgment:
1. Check design fidelity by comparing each designed quantity against its
   realized value. Fidelity is read from measured quantities, never from
   description or rationale. Only after no deviation is found may op_name,
   scope and parameters be used to confirm the intended operator was applied.
   If realized values are unavailable, fidelity is "unknown", not "faithful".
2. Judge the measured outcome.
If the intended design was not implemented faithfully, use "inconclusive",
not "not_supported".

Analysis rules:
- Do not merely restate deltas. Each finding must add at least one of:
  a task-grounded mechanism hypothesis, controlled comparison, trade-off,
  or confounder.
- Mark mechanisms as hypotheses, not facts.
- Attribute evidence to op_name plus relevant parameters/scope.
- Report controlled comparisons when pipelines differ in one variable.
- Surface shared settings that may confound conclusions.
- Do not call a primary-metric gain a clean positive if an important
  secondary metric meaningfully worsens.
- Leave arrays empty when evidence is insufficient.

Leads:
- Describe evidence-backed experimental openings, not complete pipeline designs.
- Name the relevant op, scope, or parameter direction.
- Include any observed metric trade-off.

Do not recompute metrics or design the next round.
Return JSON only and follow the supplied schema exactly.
Keep the response compact: at most 4 key findings, 3 items per signal list,
3 uncertainties, and 3 leads.
""".strip()

    noise_policy = (ROUND_NOISE_ENABLED if noise_analysis_enabled else ROUND_NOISE_DISABLED)
    system_prompt += "\n\n" + noise_policy
    user_prompt = f"""
Analyze the completed augmentation round.

## Dataset_info:
{compact_json(dataset_info)}

## Task_info:
{compact_json(task_info)}

##User Goal:
{compact_json(user_goal)}
## Round Hypothesis and goal
{compact_json(round_hypothesis)}

## Round Gradient

{compact_json(round_gradient)}

## Required Output JSON Schema

Return exactly:
{{
  "round_analysis": {{
    "summary": "...",
    "primary_metric": "...",
    "secondary_metrics": [],
    "hypothesis_result": {{
      "status": "supported|partially_supported|not_supported|inconclusive",
      "reason": "..."
    }},
    "key_findings": [],
    "positive_signals": [],
    "negative_signals": [],
    "tradeoffs": [],
    "uncertainties": [],
    "leads_from_this_round": []
  }}
}}
""".strip()


    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
def generate_round_analysis(
    llm_client,
    spec,
    round_hypothesis,
    round_gradient,
    noise_analysis_enabled: bool = False,
) -> str:
    messages = build_round_analysis_messages(
        spec=spec,
        round_hypothesis=round_hypothesis,
        round_gradient=round_gradient,
        noise_analysis_enabled=noise_analysis_enabled,
    )
    return llm_client.generate(messages=messages)