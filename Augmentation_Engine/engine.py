from .parser import parse_pipeline_json
from .generator import generate_pipeline_json_list_text

from .generator_with_round_analysis import generate_pipeline_with_history_analysis
import json
from pathlib import Path
class AugEngine:
    def __init__(self, llm_client,spec):
        self.llm_client = llm_client
        self.spec=spec

    def generate_initialiation_round(self):
        raw_text = generate_pipeline_json_list_text(
            llm_client=self.llm_client,
            spec=self.spec,
        )

        objs = parse_pipeline_json(raw_text)
        return objs

    def load_round_conclusions(self, file_path, n=None):
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        results = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if n is None or record.get("round_index") <= n:
                    results.append(record)

        return results
    
    def generate_pipelines_with_history_anaysis(self,file_path,n):
        history_pipelines=None
        round_conclusions=self.load_round_conclusions(file_path,n)
        raw_text = generate_pipeline_with_history_analysis(self.llm_client,self.spec,history_pipelines,round_conclusions)
        objs = parse_pipeline_json(raw_text)
        return objs
