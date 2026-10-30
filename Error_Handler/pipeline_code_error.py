import json


FIX_PIPELINE_SYSTEM_MESSAGE = """You repair Python augmentation pipeline code.

Preserve the pipeline specification and make the smallest change needed
to fix the reported traceback. Do not redesign, simplify, or remove
specified operations. Do not modify downstream training or evaluation.
Do not hide errors with broad exception handling, add fallback strategies,
or use unbounded loops.
- Execute all augmentation components on GPU whenever they support GPU execution and CUDA is available. Explicitly place every generated model and all associated tensors on the current CUDA device; do not leave GPU-capable augmentation components on CPU. Use CPU only for operations that do not support GPU execution or when CUDA is unavailable. Do not hard-code a specific CUDA device index.
- Do not use any unbounded loops or retry-until-success logic; all loops and generation attempts must have explicit finite bounds.
- Use only externally seeded global RNG state for all randomness; do not set or hard-code seeds, create independent RNG/generator objects, or add/pass seed, rng, random_state, or generator arguments.
- Do not include any explicit input/output validation, assertions, or deliberately raised exceptions (raise) inside the generated code; interface compliance is validated externally.
- Include only required imports, directly required helpers, `aug_code`, and `AUG_OUTPUT_STOCHASTIC`.
Return only the requested valid JSON object without markdown fences or
additional text.

<STOCHASTICITY_FLAG>
Define a module-level boolean `AUG_OUTPUT_STOCHASTIC`.

True if and only if the code you actually wrote can produce different augmented
output when run twice on identical inputs under different global RNG state.
Judge your code, not the specification's intent.

Easy-to-miss True cases: sampled decoding (do_sample=True, temperature > 0,
top_k/top_p); dropout or other stochastic layers left active at inference;
random selection of donors, contexts, spans, or positions buried inside
otherwise rule-based logic.

False only if output is content-identical across runs: deterministic rules,
fixed templates, greedy decoding, deterministically ordered selection.

If uncertain, set True.
</STOCHASTICITY_FLAG>
"""


def _code_format_text(code_format=None) -> str:
    if code_format is None:
        return "No additional format constraints."

    if isinstance(code_format, str):
        return code_format

    return json.dumps(code_format, ensure_ascii=False, separators=(",", ":"))


def _render_text_field(value) -> str:
    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_fix_messages(pipeline_info: dict, code_format=None):
    code = pipeline_info["code"]
    traceback_text = pipeline_info["traceback"]
    pipeline_specification = pipeline_info["pipeline_description"]

    user_message = f"""Repair this failed augmentation pipeline.

Pipeline specification:
{_render_text_field(pipeline_specification)}

Traceback:
{traceback_text}

Current code:
{code}

Additional code-format constraints:
{_code_format_text(code_format)}

The repaired code must remain complete and executable and must define
`aug_code`. Preserve all specified operations, parameters, ordering,
scope, filtering, and metadata behavior.

Return exactly:
{{"fixed_code":"","fix_summary":""}}
"""

    return [
        {
            "role": "system",
            "content": FIX_PIPELINE_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def generate_fixed_pipeline_text(
    llm_client,
    pipeline_info: dict,
    code_format=None,
) -> str:
    messages = build_fix_messages(
        pipeline_info,
        code_format=code_format,
    )
    return llm_client.generate(messages=messages)