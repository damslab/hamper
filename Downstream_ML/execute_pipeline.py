import json
def train_with_pipeline(aug_code):
    try:
        from .generated_train_pipeline import train_pipeline
    except ImportError:
        raise ImportError("Missing downstream training pipeline")
    result=train_pipeline(aug_code)
    return result
def append_result_to_history(path,result):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def train_with_da_fe_pipeline(aug_fn,fe_fn):
    try:
        from .generated_train_pipeline import train_pipeline
    except ImportError:
        raise ImportError("Missing downstream training pipeline")
    result=train_pipeline(aug_fn,fe_fn)
    return result