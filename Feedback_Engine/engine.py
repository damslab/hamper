import json
from pathlib import Path
from typing import Any, Callable, Dict, Union
from .build_sort_fn import generate_sort_fn
from .build_metric_order import generate_higher_is_better_list
from .round_gradient import gradient_analysis, build_round_conclusion_pipeline_summaries
from .result_grouping import filtered_history_result, select_pipeline_by_kround
from Augmentation_Engine.parser import parse_pipeline_json
from .round_analysis import generate_round_analysis, load_round_records

class FeedbackEngine:
    def __init__(
        self,
        local_llm_client,
        global_llm_client,
        spec: Dict[str, Any],
        history_pipeline_path: Union[str, Path],
        reset_history_on_init: bool = False,    
    ):
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict")

        if "task_info" not in spec:
            raise ValueError("spec must contain key 'task_info'")

        task_info = str(spec["task_info"]).strip()
        if not task_info:
            raise ValueError("spec['task_info'] must not be empty")



        if isinstance(history_pipeline_path, str):
            if not history_pipeline_path.strip():
                raise ValueError("history_pipeline_path must be a non-empty string")
            history_pipeline_path = Path(history_pipeline_path)
        elif not isinstance(history_pipeline_path, Path):
            raise TypeError("history_pipeline_path must be a str or Path")

        if not isinstance(reset_history_on_init, bool):
            raise TypeError("reset_history_on_init must be a bool")

        self.llm_client = local_llm_client
        self.spec = spec
        self.gloabl_client=global_llm_client
        self.history_pipeline_path = history_pipeline_path
        self.reset_history_on_init = reset_history_on_init
        self.primary_metric = self.spec["target_metrics"][0]
        self.sort_fn=generate_sort_fn(self.llm_client,self.spec['task_info'],self.primary_metric)
        self.metric_order=generate_higher_is_better_list(self.llm_client,self.spec['task_info'],self.spec["target_metrics"])
    def topk_pipelines(self, results,k):
        return self.sort_fn(results)[:k]
    
    def baseline_result(self):
        history_result=filtered_history_result(self.history_pipeline_path)
        baseline_items = [item for item in history_result if item.get("pipeline_id") == 0]
        return baseline_items[0]
    
    def round_gradient(self,n,k=3):
        history_result=filtered_history_result(self.history_pipeline_path)
        topk_pipelines=[]
        baseline=self.baseline_result()
        current_round_results=select_pipeline_by_kround(history_result,n,0)
        last_round_best=None
        if n>=1:
            topk_pipelines=self.topk_pipelines(select_pipeline_by_kround(history_result,n-1,n-1),k)
            last_round_results=select_pipeline_by_kround(history_result,n-1,0)
            last_round_top1 = self.topk_pipelines(last_round_results, 1)
            last_round_best = last_round_top1[0] if last_round_top1 else None
        gradient_ana=gradient_analysis(baseline,current_round_results,topk_pipelines,last_round_best,self.spec["target_metrics"],self.metric_order)
        return gradient_ana
    
    def round_conclusion(self,round_hypothesis,n,k=3):
        round_gradient=self.round_gradient(n,k)
        history_result=filtered_history_result(self.history_pipeline_path)
        current_round_results=select_pipeline_by_kround(history_result,n,0)
        round_gradient=build_round_conclusion_pipeline_summaries(round_gradient,current_round_results)
        round_conclusion=generate_round_analysis(self.llm_client,self.spec,round_hypothesis,round_gradient)
        return parse_pipeline_json(round_conclusion)
    
    def save_round_conclusion(self,file_path,round_hypothesis,n,k=3):
        round_conclusion=self.round_conclusion(round_hypothesis,n,k)
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "round": n,
            "hypothesis": round_hypothesis,
            "conclusion": round_conclusion
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        return round_conclusion

    def load_round_hypothesis(self,path,n):
        return load_round_records(path,n) 
        

    


