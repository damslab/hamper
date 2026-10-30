
import ast
import json
from typing import Any


def build_coding_system_message(
    aug_code_format: str,
    aug_interface_type: str,
    aug_output_effect: str,
) -> str:

    return f"""
You are a Python ML engineer implementing one data-augmentation pipeline.

Translate the provided pipeline specification into executable augmentation code exactly as specified.

Rules:

- Implement the augmentation strategy only.
- Do not redesign, optimize, simplify, expand, reorder, or reinterpret the strategy.
- Follow all specified operations, parameters, scope, ordering, filtering, and metadata-preservation behavior.
- The downstream training and evaluation pipeline is fixed and must not be modified or reproduced.
- Do not include training, validation, evaluation, feature extraction, classifier logic, cross-validation, dataset loading, benchmarking, or metric computation.
- Do not measure or report target metrics or augmentation processing time; these are handled externally by the fixed evaluator.
- AUG_CODE_FORMAT, AUG_INTERFACE_TYPE, and AUG_OUTPUT_EFFECT are mandatory interface constraints.
- Include only required imports, directly required helpers, `aug_fn`, and `AUG_OUTPUT_STOCHASTIC`.
- Use third-party dependencies only when directly required by the specified strategy.
- Use only externally seeded global RNG state for all randomness; do not set or hard-code seeds, create independent RNG/generator objects, or add/pass seed, rng, random_state, or generator arguments.
- Execute all augmentation components on GPU whenever they support GPU execution and CUDA is available. Explicitly place every generated model and all associated tensors on the current CUDA device; do not leave GPU-capable augmentation components on CPU. Use CPU only for operations that do not support GPU execution or when CUDA is unavailable. Do not hard-code a specific CUDA device index.
- Do not assume unspecified external files or resources.
- Do not add augmentation operations, fallback strategies, filters, or heuristics that are not specified.
- Keep the implementation complete and executable.
- The returned code must define `aug_fn`.
- Do not use any unbounded loops or retry-until-success logic; all loops and generation attempts must have explicit finite bounds.
- Do not include any explicit input/output validation, assertions, or deliberately raised exceptions (raise) inside the generated code; interface compliance is validated externally.

Return only executable Python source code.
Do not return JSON.
Do not use markdown fences.
Do not include explanations or any text outside the Python code.
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
<AUG_CODE_FORMAT>
{aug_code_format}
</AUG_CODE_FORMAT>

<AUG_INTERFACE_TYPE>
{aug_interface_type}
</AUG_INTERFACE_TYPE>

<AUG_OUTPUT_EFFECT>
{aug_output_effect}
</AUG_OUTPUT_EFFECT>
""".strip()


def build_pipeline_message(
    pipeline_description: Any,
) -> str:

    if isinstance(pipeline_description, str):
        text = pipeline_description.strip()

    else:
        text = json.dumps(
            pipeline_description,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return f"""Implement this pipeline specification exactly:

<PIPELINE_DESCRIPTION>
{text}
</PIPELINE_DESCRIPTION>"""


def generate_aug_code(
    llm_client,
    spec,
    pipeline_description,
) -> str:

    system_message = build_coding_system_message(
        aug_code_format=spec["aug_code_format"],
        aug_interface_type=spec["aug_interface_type"],
        aug_output_effect=spec["aug_output_effect"],
    )

    user_message = build_pipeline_message(
        pipeline_description
    )

    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    response = llm_client.generate(
        messages=messages
    )

    if not isinstance(response, str):
        raise TypeError(
            f"Expected string, got "
            f"{type(response).__name__}"
        )

    code = response.strip()

    if not code:
        raise ValueError(
            "Coding Engine returned empty code."
        )

    # Validate Python syntax without executing it
    try:
        tree = ast.parse(code)

    except SyntaxError as e:
        raise ValueError(
            f"Coding Engine returned invalid Python: {e}\n"
            f"Code:\n{code}"
        ) from e

    # Verify aug_code exists
    has_aug_code = any(
        isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and node.name == "aug_fn"
        for node in ast.walk(tree)
    )

    if not has_aug_code:
        raise ValueError(
            "Generated code does not define aug_fn."
        )

    return code