from Llm_Client.claude_code_client import ClaudeCodeChatClient
from Llm_Client.codex_client import CodexClient
from Downstream_ML.engine import MLEngine
from Augmentation_Engine.engine import AugEngine
from Coding_Engine.engine import PipelineCodeEngine
from Feedback_Engine.engine import FeedbackEngine
from Feature_Extraction_Engine.engine import FeatureEngine
import traceback
import json
from pathlib import Path
json_path = "/home/xiaoma/Agent_hamper/merged.json"

with open(json_path, "r", encoding="utf-8") as f:
    spec = json.load(f)

output_path = Path("pipeline_results.jsonl")
output_hypothesis_path = Path("round_hypothesis.jsonl")
output_roundanalysis_path = Path("round_analysis.jsonl")
workdir=Path("/home/xiaoma/Agent_hamper/Downstream_ML").resolve()   
downstream_ml_engine=MLEngine(workdir,None,spec,None,output_path,None)

aug_client = ClaudeCodeChatClient(
    workdir="/home/xiaoma/Empty",
    model="claude-opus-5-5",
    timeout=600,
    safe_mode=True,
    no_session_persistence=True,
    disable_tools=True,
    max_turns=1,
    system_prompt_mode="replace",
)

work_dir="/home/xiaoma/Empty"


aug_engine = AugEngine(aug_client,spec)

coding_client=CodexClient(workdir="/home/xiaoma/Empty",model="gpt-5.6-luna")
code_engine = PipelineCodeEngine(llm_client=coding_client,spec=spec,work_dir=work_dir)

round_start= aug_engine.generate_initialiation_round()
hypothesis=round_start['generation_round_summary']
pipelines = json.loads(round_start["pipelines"])

record = {
    "round_index": 0,
    "type": "round_generation_summary",
    "round_summary": hypothesis
}

with output_hypothesis_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
#Test genertate pipeline
pipe_id=1
for i in range(0,1):

    round_id=0
    pipeline = pipelines[i]
    description = pipeline.get("description", "")
    current_pipe_id = pipe_id
    pipe_id += 1
    result_record = {
        "pipeline_id": current_pipe_id,
        "pipeline_description": pipeline,
        "description": description,
        "code": None,
        "status": "init",
        "stochastic": None,
        "round":round_id,
        "error_stage": None,
        "error_message": None,
    }
    
    try:
        result= code_engine.generation_augmentation_code(pipeline)
        result_record["code"] = result
    except Exception as e:
        result_record["status"] = "failed"
        result_record["error_stage"] = "code_generation"
        result_record["error_message"] = repr(e)
        downstream_ml_engine.append_result(result_record)
        continue

    try:
        namespace = {}
        exec(result, namespace)
    except Exception as e:
        result_record["status"] = "failed"
        result_record["error_stage"] = "code_exec"
        result_record["error_message"] = repr(e)
        downstream_ml_engine.append_result(result_record)
        continue

    try:
        aug_code = namespace["aug_fn"]
        declared = namespace.get("AUG_OUTPUT_STOCHASTIC", None)
        use_random = True if declared is None else bool(declared)
        result_record["stochastic"] = use_random
        if not callable(aug_code):
            raise TypeError("Generated aug_code exists but is not callable.")
    except Exception as e:
        result_record["status"] = "failed"
        result_record["error_stage"] = "aug_code_validation"
        result_record["error_message"] = repr(e)
        downstream_ml_engine.append_result(result_record)
        continue

    try:
        train_result = downstream_ml_engine.train(aug_code)
        train_result["pipeline_id"] = current_pipe_id
        train_result["pipeline_description"] = pipeline
        train_result["code"] = result
        train_result["status"] = "success"
        train_result["round"]=round_id
        train_result["stochastic"] = use_random
        downstream_ml_engine.append_result(train_result)
    except Exception as e:
        result_record["status"] = "failed"
        result_record["error_stage"] = "train_pipeline"
        result_record["error_message"] = repr(e)
        result_record["traceback"]=traceback.format_exc()
        downstream_ml_engine.append_result(result_record)
    
        continue
