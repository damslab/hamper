FEATURE_EXTRACTION_STRATEGY_TEMPLATE = {
    "round_summary": {
        "overall_hypothesis": "What this round tests as a whole and why these pipelines were chosen",
        "risks": ["An assumption or risk that could make a result misleading"],
    },
    "pipelines": [
        {
            "pipeline_id": "Unique id, e.g. p1",
            "hypothesis": "One sentence: which signal this pipeline tests",
            "description": "Brief description of the complete pipeline",
            "steps": [
                {
                    "step_id": "string",
                    "op_name": "Concrete operation name, e.g. mfcc, tfidf, pca",
                    "input": "Raw input or a previous step_id, with its shape",
                    "operation_description": "The transformation, precise enough to implement without guessing",
                    "fit": "none | train_unsupervised | train_supervised",
                    "parameters": {"param_name": "concrete value, not a range"},
                    "output": "What this step produces and its shape",
                }
            ],
            "output": "Final representation: dtype and exact dimension or dimension rule",
        }
    ],
}


BASELINE_STRATEGY_TEMPLATE = {
    "baseline": {
        "candidate_id": "string",

        # A precise and directly implementable feature definition.
        # Include the input field, transformation method, important
        # parameters, aggregation method, and temporal constraints when needed.
        "feature_name": "string",

        "modality": (
            "audio | text | image | tabular | time_series | "
            "graph | multimodal | other"
        ),

        "safety_constraints": {
            "must_not_use_labels": True,
            "must_fit_on_train_only": True,
            "must_preserve_sample_order": True,
            "must_preserve_sample_count": True,
            "must_not_use_future_information": True
        },

        "validation_checks": [
            "string"
        ]
    }
}