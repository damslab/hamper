import json
from .generator import generate_user_json_text,SYSTEM_MESSAGE
from .validator import validate_user_output

def build_engine_input(llm_client, user_message: str) -> dict:
    """
    Convert one free-form user message into the json.
    
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("user_message must be a non-empty string")

    raw_text, usage = generate_user_json_text(
        llm_client=llm_client,
        user_message=user_message,
    )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"user_app_layer returned invalid JSON: {e}") from e

    validate_user_output(parsed)

    return {
        "dataset_info": parsed["dataset_info"],
        "task_info": parsed["task_info"],
        "user_goal": parsed["user_goal"],
        "constraints": parsed.get("constraints", {}),
        "allowed_ops": None,
    }

