import json
from typing import Any, Dict


import json


def parse_pipeline_json(raw_text: str):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}") from e