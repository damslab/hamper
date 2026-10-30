from .augmentation_generator import run_downstream_training_prompt
from .execute_pipeline import train_with_pipeline, append_result_to_history

class MLEngine:
    def __init__(self, work_dir,llm_client,spec,dataset_path,output_path,train_spec):
        self.llm_client = llm_client
        self.spec=spec
        self.work_dir=work_dir
        self.path=dataset_path
        self.output_path=output_path
        self.train_spec=train_spec
    def generate_augmentation_train_script(self):
        return run_downstream_training_prompt(self.spec,self.path,self.train_spec,self.llm_client)
    def augmentation_initialization(self):
        output_path = self.work_dir / "generated_train_pipeline.py"
        result=self.generate_augmentation_train_script()
        code = result.get("code")
        self.pipeline_format=result.get("aug_code_format")
        self.metric_names=result.get("target_metrics")
        output_path.write_text(code, encoding="utf-8")
        return result.get("aug_code_format"), result.get("target_metrics")
    def adding_setup(self,text):
        self.setup=text
    def train(self,aug_code):
        return train_with_pipeline(aug_code)
    def append_result(self,result):
        append_result_to_history(self.output_path,result)
    def create_result_file(self,path):
        if path==None:

            with open(self.output_path, "w", encoding="utf-8") as f:
                pass
        else:
            with open(path, "w", encoding="utf-8") as f:
                pass

    
