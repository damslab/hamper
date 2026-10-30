import json

from.default import PIPELINE_OUTPUT_FORMAT_TEMPLATE
BASE_SYSTEM_MESSAGE = """You generate data-augmentation strategies for an iterative optimization loop.

Rules:

- Follow the specification, constraints, allowed_ops, and output_format exactly.
- The downstream training and evaluation pipeline is fixed and must not be modified or redesigned.
- Augmentation is the only optimization variable; candidate quality is determined only by the specified target metrics.
- Generate augmentation strategies only; do not execute or evaluate experiments.
- Preserve labels unless explicitly allowed otherwise.
- Do not invent evaluation results.
- Avoid invalid or behaviorally redundant candidates.
- Do not assume unavailable external files or resources.
- The first and last output characters must be `{` and `}`.
- Do not include Markdown, headings, or extra text.
- Return exactly one top-level object of the form
  {"generation_round_summary": {...}, "pipelines": [...]};
  these two fields must be siblings.
- Keep descriptions, operation descriptions, parameters, and notes concise.
  Include only details needed to implement or distinguish each candidate.
"""
def compact_json(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
def _allowed_ops_text(allowed_ops=None) -> str:
    if allowed_ops is None:
        return (
            "Allowed operators are unrestricted. "
            "You may use any operators that satisfy the specification, "
            "constraints, and output format."
        )

    if len(allowed_ops) == 0:
        return (
            "Allowed operators are unrestricted. "
            "You may use any operators that satisfy the specification, "
            "constraints, and output format."
        )

    return (
        "You may only use operators from this allowed set:\n"
        + {compact_json(allowed_ops)}
    )

def build_generation_messages(spec: dict, default_output_format=PIPELINE_OUTPUT_FORMAT_TEMPLATE):

    dataset_info = spec["dataset_info"]
    task_info = spec["task_info"]
    user_goal = spec["user_goal"]
    constraints = spec.get("constraints", {})
    output_format = default_output_format
    allowed_ops = spec.get("allowed_ops", None)

    user_message = f"""Generate candidate augmentations according to the following specification.

You only need to follow these six top-level fields from the specification:
- dataset_info
- task_info
- user_goal
- constraints
- output_format
- allowed_ops


Do not assume any schema beyond these six top-level fields.
Treat the content inside each field as user-defined specification content and follow it directly.

dataset_info:
{compact_json(dataset_info)}

task_info:
{compact_json(task_info)}

user_goal:
{compact_json(user_goal)}

constraints:
{compact_json(constraints)}

output_format:
{compact_json(output_format)}

allowed_ops guidance:
{_allowed_ops_text(allowed_ops)}

Generation objective:

Generate 6–10 diagnostic candidates for the initial round.

- Prefer simple, interpretable, executable, label-preserving strategies.
- Each candidate should test one main augmentation hypothesis.
- Prefer one-step pipelines; use at most two operations when a combination tests a clear hypothesis.
- Cover meaningfully different perturbation types, strengths, or target components.
- Prefer conservative perturbations whose validation effects can be clearly attributed.
- Do not include an unchanged baseline.
- Include label/annotation synchronization when required by the task.
- Each pipeline must be fully self-contained and must not reference
  definitions, controls, fields, or resources outside that pipeline.
"""
    return [
        {"role": "system", "content": BASE_SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]

def generate_pipeline_json_list_text(llm_client, spec: dict) -> str:
    messages = build_generation_messages(spec)
    return llm_client.generate(messages=messages)