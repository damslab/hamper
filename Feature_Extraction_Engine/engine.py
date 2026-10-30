from .generator import build_feature_generation_messages, generate_pipeline_json_list_text
from .parser import parse_pipeline_json
from .generate_with_history_result import generate_pipelines_with_history_result
class FeatureEngine:
    def __init__(self, llm_client,spec):
        self.llm_client = llm_client
        self.spec=spec

    def generate_feature_extraction_strategy(self):
        pipelines=generate_pipeline_json_list_text(self.llm_client,self.spec)
        return parse_pipeline_json(pipelines)
    
    def generate_feature_extraction_strategy_with_history_result(self,history_result):
        pipelines= generate_pipelines_with_history_result(self.llm_client,self.spec,history_result)
        return parse_pipeline_json(pipelines)