

from .augmentation_code_generation import generate_aug_code

class PipelineCodeEngine:
    def __init__(self, llm_client,spec,work_dir):
        self.llm_client = llm_client
        self.spec=spec
        self.workdir=work_dir
    
    def generation_augmentation_code(self,pipeline_description,):
        code_text = generate_aug_code(
            llm_client=self.llm_client,
            spec=self.spec,
            pipeline_description=pipeline_description,
        )

        return code_text