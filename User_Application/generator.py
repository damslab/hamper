import json
import tiktoken

SYSTEM_MESSAGE = """
You convert a user's free-form request about a dataset, ML task,
data augmentation (DA), and feature extraction (FE) into structured
JSON for the corresponding optimization engines.

Your role is information extraction and organization, not strategy generation.
Preserve the meaning and level of specificity of the user's request.

Return valid JSON only.
Do not wrap JSON in markdown fences.
Do not output any extra text.

The output must be one JSON object containing exactly these top-level fields:

- "dataset_info": an object
- "task_info": an object
- "user_goal": an object
- "constraints": an object

If no information is available for any field, use an empty object `{}`.
Never return null, an array, or a string for any of these four fields.

Extraction rules:

- Use only information stated or directly entailed by the user's message.
- Do not invent pipelines, operations, filters, thresholds,
  evaluation procedures, quality controls, or implementation requirements.
- Do not convert examples, preferences, warnings, dataset properties, or
  plausible implementation choices into hard constraints.
- Do not infer prohibitions from omitted information.
- Preserve unspecified choices as unspecified rather than filling defaults.
- Keep constraints minimal. Every constraint must be traceable to a specific
  statement in the user's message.
- Place desired outcomes and preferences in user_goal.
- Place only mandatory restrictions in constraints.
- Reorganize information without narrowing or expanding the requested
  pipelines' search space.
- Preserve ambiguity and leave unspecified operations, scopes, sources, and
  strategy families unrestricted.

Before returning, verify that every generated value is supported by the user's
message and remove assumption-based additions.
"""


def build_user_messages(user_message: str, system_message=SYSTEM_MESSAGE):
    user_prompt = f"""Convert the following user request into structured JSON.

User request:
{user_message}

Return JSON only.
"""
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

def count_tokens(text: str, model="gpt-5.6-luna") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    return len(enc.encode(text))


def generate_user_json_text(llm_client, user_message: str, system_message=SYSTEM_MESSAGE):
    messages = build_user_messages(user_message,system_message)

    
    input_tokens = sum(
        count_tokens(m["content"])
        for m in messages
    )

    response = llm_client.generate(messages=messages)


    output_tokens = count_tokens(response)

    usage = {
        "prompt_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    return response, usage