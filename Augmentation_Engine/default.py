PIPELINE_OUTPUT_FORMAT_TEMPLATE = {
    "generation_round_summary": {
        "round_hypothesis": "string",
        "exploration_objective": "string",
        "strategy_selection_rationale": "string",
    },
    "pipelines": [
        {
            "pipeline_id": "string",
            "family": "string",
            "arm_role": "canonical|restricted_variant|refinement|recombination|repair|control|replication",
            "description": "string",
            "steps": [
                {
                    "step_id": "string",
                    "op_name": "string",
                    "operation_description": "string",
                    "parameters": {},
                    "scope": "string|null",
                    "label_effect": "preserve|transform|unknown",
                    "notes": "string|null",
                }
            ],
        }
    ],
}