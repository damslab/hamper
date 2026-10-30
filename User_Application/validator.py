def _require_non_empty_object(obj, key: str):
    if key not in obj:
        raise ValueError(f"Missing required field: {key}")
    if not isinstance(obj[key], dict):
        raise ValueError(f"{key} must be an object")
    if len(obj[key]) == 0:
        raise ValueError(f"{key} must not be empty")


def validate_user_output(obj: dict):
    if not isinstance(obj, dict):
        raise ValueError("User output must be a JSON object")

    _require_non_empty_object(obj, "dataset_info")
    _require_non_empty_object(obj, "task_info")
    _require_non_empty_object(obj, "user_goal")

    if "constraints" in obj:
        if not isinstance(obj["constraints"], dict):
            raise ValueError("constraints must be an object")

    return True