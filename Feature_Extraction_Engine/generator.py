import json
from .default import FEATURE_EXTRACTION_STRATEGY_TEMPLATE,BASELINE_STRATEGY_TEMPLATE
import json
from copy import deepcopy
 
FEATURE_BASE_SYSTEM_MESSAGE = """You generate feature extraction strategy descriptions (not code) for a downstream code-generation agent in an iterative generate-validate-reflect loop.
 
The downstream training pipeline (model, classifier, optimizer, evaluation, cross-validation) is fixed. Never redesign or modify it.
 
Rules:
- Follow the specification, constraints, and output_format exactly. Return only the structured output: no markdown fences, no extra text, no added or missing fields. Keep it valid and internally consistent.
- Describe only feature extraction. No training, evaluation, tuning, validation-based feature selection, or CV logic.
- Each strategy must be implementable without guessing: concrete op names, concrete parameter values, and the shape at every step. Describe data flow, not function signatures or code; the code agent receives the interface contract separately.
- If an interface is given, every pipeline must consume its input format and produce its required output format.
- No leakage: never use target labels, target statistics, validation/test data, or leakage-prone metadata unless the spec explicitly allows it (only then may fit be train_supervised). Any fitted state is fit on the training split only.
- For tasks with synchronized annotations (timestamps, spans, masks, boxes, graph labels), only propose strategies whose alignment behavior is clearly specified."""
 
BASELINE_ON = """
Baseline: fill the baseline field with exactly one pipeline, using the same schema as pipelines and not duplicated among them. It is a fixed reference for later rounds: minimal but meaningful, deterministic, modality-appropriate, and not overly strong or complex."""
 
BASELINE_OFF = """
Baseline: output_format has no baseline field; do not generate one."""
 
 
def build_feature_generation_messages(spec: dict, output_format: dict) -> list:
    """spec keys: dataset_info, task_info, user_goal, constraints (optional), interface (optional)."""
    has_baseline = isinstance(output_format, dict) and "baseline" in output_format
 
    fields = {
        "dataset_info": spec["dataset_info"],
        "task_info": spec["task_info"],
        "user_goal": spec["user_goal"],
        "constraints": spec.get("constraints", {}),
    }
    if spec.get("interface"):
        fields["interface"] = spec["interface"]
    fields["output_format"] = output_format
 
    spec_text = "\n\n".join(
        f"{k}:\n{json.dumps(v, ensure_ascii=False, indent=2)}" for k, v in fields.items()
    )
 
    user_message = f"""Generate feature extraction pipelines for the specification below. Treat each field as user-defined content; assume no schema beyond these fields.
 
{spec_text}
 
This is the initial diagnostic round, not a broad search. Examine how
task-relevant fields, their relationships, ordering, and alignment are
represented or lost in the baseline. Include controlled candidates that
test alternative input representations preserving potentially useful
structure, alongside feature-extraction choices. Prioritize plausible
task-relevant relationships rather than exhaustively combining fields.
 
Pipelines:
- Generate 6-10 unless constraints or output_format require fewer. Diagnostic value, simplicity, executability, and leakage safety matter more than count or coverage.
- Each pipeline tests one hypothesis and differs clearly from the others in feature family, representation, aggregation, temporal granularity, or target component. No redundant, vague, or empty pipelines, and no unchanged raw-input pipelines unless output_format requires one.
- Mostly single-step pipelines, so results are easy to attribute; a few simple combinations only when they test a clear hypothesis; at most 3 steps each.
- Vary parameters only when that changes the hypothesis (e.g., short vs. medium window, low vs. moderate dimension).
{BASELINE_ON if has_baseline else BASELINE_OFF}
 
round_summary: overall_hypothesis states what this round tests as a whole and why this compact set suits the loop{" and notes that it establishes the baseline reference" if has_baseline else ""}; risks lists assumptions that could make results misleading. No results or performance claims."""
 
    return [
        {"role": "system", "content": FEATURE_BASE_SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]
 
 
def build_strategy_template(strategy_template: dict, baseline_template: dict, *, include_baseline: bool = False) -> dict:
    template = deepcopy(strategy_template)
    if include_baseline:
        template.update(deepcopy(baseline_template))
    return template
 
 
def generate_pipeline_json_list_text(llm_client, spec: dict) -> str:
    template_round_0 = build_strategy_template(
        FEATURE_EXTRACTION_STRATEGY_TEMPLATE,
        BASELINE_STRATEGY_TEMPLATE,
        include_baseline=False,
    )
    messages = build_feature_generation_messages(spec, template_round_0)
    return llm_client.generate(messages=messages)
 
