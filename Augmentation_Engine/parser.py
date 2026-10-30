import json


def parse_pipeline_json(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM output is not valid JSON: {e}\n"
            f"Raw text length: {len(raw_text)}\n"
            f"Raw text:\n{raw_text!r}"
        ) from e