import json
from textwrap import dedent
from typing import Any
import json
import tiktoken

def _to_text(value: Any) -> str:
    """Convert arbitrary input into readable text for prompt injection."""
    if value is None:
        return "None"
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_ml_codegen_prompt(
    spec,
    dataset_path,
    train_spec=None,
) -> str:

    data_info_text = _to_text(spec["dataset_info"])
    task_info_text = _to_text(spec["task_info"])
    constraint=spec["constraints"]
    dataset_path_text = str(dataset_path).strip()

    constraint_text = (
        _to_text(constraint)
        if constraint is not None
        else "No additional constraints."
    )

    prompt = f"""
You are an expert machine learning engineer and code generator.

Generate a complete, runnable Python ML training pipeline from the provided
dataset, task, path, constraints, and training specification.

Return STRICT JSON ONLY.
No markdown fences, commentary, or text outside the JSON.

The JSON must contain exactly these six top-level fields:

{{
  "code": "<complete Python source code>",
  "aug_code_format": "<compact implementation-ready augmentation API contract>",
  "aug_interface_type": "<sample_transform|batch_transform|dataset_callable>",
  "aug_output_effect": "<rewrite_existing_samples|generate_new_samples|mixed|unknown>",
  "target_metrics": ["<metric_name>", "..."]
}}

<DATA_INFO>
{data_info_text}
</DATA_INFO>

<TASK_INFO>
{task_info_text}
</TASK_INFO>

<DATASET_PATH>
{dataset_path_text}
</DATASET_PATH>

<CONSTRAINT>
{constraint_text}
</CONSTRAINT>

<TRAINING_SPECIFICATION>
{train_spec}
</TRAINING_SPECIFICATION>


GENERAL REQUIREMENTS

- Follow DATA_INFO, TASK_INFO, CONSTRAINT, and TRAINING_SPECIFICATION exactly.
- Generate complete executable Python, not pseudocode.
- Embed required dataset/task/path/config information directly in the code.
- Use suitable common ML libraries for the task and modality.
- Use the train_pipeline `seed` argument for all random seeding and support CPU fallback.
- Do not require manual modification before execution.
- Avoid TODOs, placeholders, or unfinished branches.
- Treat the dataset directory as read-only.
- Do not introduce modality-specific augmentation methods unless explicitly
  required by the provided task or constraints.
- Do not add explicit validation checks or deliberately raise exceptions in the
  generated training code. Let genuine runtime errors propagate naturally.

TRAINING ENTRY INTERFACE

The generated code must expose exactly one public training entry function:

    def train_pipeline(aug_fn=None, seed=42):
        ...

`seed` must be the optional second argument with default value 42.
At the beginning of each call, use `seed` to initialize all global random
states used by Python, NumPy, PyTorch, CUDA, data splitting, data loading,
model initialization, training, and augmentation.
Do not pass `seed` to aug_fn; augmentation uses the externally initialized
global random states.
No additional argument may be required.

When aug_fn is None, run the baseline training pipeline without augmentation.

When aug_fn is provided, apply it through the single augmentation interface
selected for this training script.

train_pipeline must return a JSON-serializable object:

{{
  "aug_fn": "<baseline or augmentation description>",
  "target_metrics": {{
    "<metric_name>": <numeric_value>
  }}
}}

The generated pipeline must include all task-appropriate components required
for real execution, such as dataset loading, preprocessing, model, loss,
optimizer/training procedure, validation/evaluation, and metric computation.


TARGET METRICS

- If TASK_INFO explicitly specifies metrics, use exactly those metrics.
- Otherwise infer a reasonable task-appropriate metric set.
- Metric names must use lowercase snake_case.
- Only choose metrics that the generated pipeline can actually compute.
- The top-level "target_metrics" field must be a list.
- Its items must exactly match the keys of
  train_pipeline(... )["target_metrics"].
- Every declared metric must actually be computed by the generated code.


SINGLE-STAGE AUGMENTATION INTERFACE

Select exactly one augmentation stage for the generated training script:

1. dataset
   Use when augmentation needs whole-dataset access, changes dataset size,
   rewrites/rebalances records, or creates/removes training examples.

   Canonical interface:

       def aug_fn(train_data):
           return augmented_train_data

2. sample
   Use when each example can be transformed independently before batching.

   Canonical interfaces:

       def aug_fn(x):
           return x_aug

   or:

       def aug_fn(x, y):
           return x_aug, y_aug

   If only x is transformed, preserve the original target.

3. batch
   Use when augmentation depends on multiple examples, batch-level
   interactions/statistics, or naturally transforms batched targets.

   Canonical interface:

       def aug_fn(batch_x, batch_y):
           return batch_x_aug, batch_y_aug


STAGE SELECTION

- Select the stage from TASK_INFO, CONSTRAINT, and TRAINING_SPECIFICATION.
- An explicitly requested stage takes priority.
- Otherwise select the stage from augmentation semantics, not implementation
  convenience.
- Do not classify an independent per-sample transformation as batch-stage only
  because vectorized execution is possible.
- The selected stage is fixed for the generated training script.
- The canonical aug_fn form is always a plain Python callable.
- Do not accept dictionary forms containing dataset/sample/batch callables.
- Apply aug_fn only to training data and only at the selected stage.



TARGET-AWARE AUGMENTATION

Do not assume augmentation is only x -> x.

The selected interface must support target transformation whenever appropriate.

The training and loss logic must safely support the original target format and
any augmented target format permitted by the selected interface.

Examples of possible target structures include:
- hard class labels
- soft/probabilistic class targets
- regression targets
- tensors or arrays
- lists or dictionaries
- structured targets for sequence, detection, segmentation, ranking, or other
  task-specific problems


AUGMENTATION PLACEMENT

dataset stage:
- Apply once to the training data before dataset/DataLoader construction.
- Never modify source dataset files in place.

sample stage:
- Apply independently to training examples before batching.

batch stage:
- Apply after creation of a training mini-batch and before model forward/loss
  computation.

Augmentation must never be applied to validation or test data unless explicitly
required by the provided specification.


AUG_CODE_FORMAT

"aug_code_format" is the API contract passed to a separate augmentation
code-generation model. That model should be able to generate compatible
augmentation code without reading the full training script.

Therefore aug_code_format must describe only the selected stage and include:

- selected stage
- exact callable signature
- where it is applied in train_pipeline
- expected input types, shapes, structures, devices, and dtypes when known
- expected output types, shapes, and structures when known
- target format and whether targets may change
- relevant hard/soft target handling
- a minimal compatible callable skeleton

It must explicitly state that the other augmentation stages are not accepted.

Do not include:
- baseline behavior
- training metrics
- unrelated training implementation details
- specific augmentation algorithms unless required by the provided task

Keep aug_code_format compact, concrete, implementation-ready, and fully
consistent with the generated code.


AUG_INTERFACE_TYPE

Set according to the selected augmentation stage:

dataset -> "dataset_callable"
sample  -> "sample_transform"
batch   -> "batch_transform"

Return exactly one value.


AUG_OUTPUT_EFFECT

Describe the primary effect allowed by the selected augmentation interface:

- "rewrite_existing_samples":
  primarily transforms existing examples one-for-one

- "generate_new_samples":
  primarily creates additional examples

- "mixed":
  the interface may validly support both behaviors

- "unknown":
  the effect cannot be determined safely from the task specification

The value must be consistent with code and aug_code_format.




EXECUTION VALIDATION

Before returning the final JSON:

1. Write the generated code to a Python file in the current working directory.
2. Verify that it imports successfully.
3. Verify that the dataset path is readable.
4. Run a lightweight smoke test of train_pipeline(aug_fn=None, seed=42).
5. Verify at least a minimal executable training/evaluation step.
6. Verify that the returned object contains "aug_fn" and "target_metrics"
   with the required metric keys.
7. If validation fails, fix the generated code and rerun the test.
8. Do not return the final result until validation succeeds.

Keep validation inexpensive:
- prefer a small subset
- use one epoch or a few optimization steps when full training is expensive
- use minimal workers/batch size when appropriate
- validate executability and interface correctness rather than model quality

All temporary files, caches, checkpoints, logs, and outputs created during
validation must remain inside the current working directory.


FINAL CONSISTENCY CHECK

Before returning, ensure:

- exactly six required top-level JSON fields are present
- code and imports agree
- code and aug_code_format agree
- aug_interface_type matches the selected stage
- aug_output_effect matches the implemented interface
- exactly one augmentation stage is supported
- augmentation is applied at the correct location
- target handling is compatible with the training loss
- top-level target_metrics exactly match returned metric keys
- the generated code passed the smoke test
- train_pipeline has the exact signature train_pipeline(aug_fn=None, seed=42)

Return the strict JSON object only.
"""

    return dedent(prompt).strip()


def count_tokens(text: str, model="gpt-5.6-luna") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    return len(enc.encode(text))


import json
from typing import Any


def run_downstream_training_prompt(
    spec,
    dataset_path,
    train_spec,
    client: Any,
) -> Any:

    prompt=build_ml_codegen_prompt(spec,dataset_path,train_spec)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    messages = [
        {
            "role": "user",
            "content": prompt.strip(),
        }
    ]


    
    response = client.generate(messages)
    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise TypeError(
            "LLM response must be a JSON object."
        )

    return result
