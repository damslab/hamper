import copy
import json
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .pipeline_code_error import generate_fixed_pipeline_text


class ErrorHandlerEngine:
    def __init__(
        self,
        llm_client,
        history_pipeline_path: Union[str, Path],
        validate_fn: Callable[
            [Callable[..., Any]],
            Dict[str, Any],
        ],
        spec: Dict[str, Any],
        max_repair_rounds: int = 3,
    ):
        self.llm_client = llm_client
        self.history_pipeline_path = Path(
            history_pipeline_path
        )
        self.validate_fn = validate_fn
        self.code_format = spec.get("aug_code_format")
        self.max_repair_rounds = max_repair_rounds

    def _load_history_pipeline(
        self,
    ) -> List[Dict[str, Any]]:
        if not self.history_pipeline_path.exists():
            raise FileNotFoundError(
                "History pipeline file not found: "
                f"{self.history_pipeline_path}"
            )

        with self.history_pipeline_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        return data

    def _save_history_pipeline(
        self,
        history_data: List[Dict[str, Any]],
    ) -> None:
        with self.history_pipeline_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in history_data:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _parse_fix_response(
        self,
        response_text: str,
    ) -> str:
        response_text = (
            response_text or ""
        ).strip()

        if not response_text:
            raise ValueError("Empty LLM response")

        parsed = json.loads(response_text)

        if not isinstance(parsed, dict):
            raise ValueError(
                "LLM response is not a JSON object"
            )

        fixed_code = parsed.get("fixed_code")

        if not isinstance(fixed_code, str):
            raise ValueError(
                "'fixed_code' must be a string"
            )

        if not fixed_code.strip():
            raise ValueError(
                "Missing or empty 'fixed_code'"
            )

        return fixed_code

    def _repair_one_pipeline_code(
        self,
        pipeline_info: Dict[str, Any],
    ) -> str:
        response_text = generate_fixed_pipeline_text(
            llm_client=self.llm_client,
            pipeline_info=pipeline_info,
            code_format=self.code_format,
        )

        return self._parse_fix_response(
            response_text
        )

    def _extract_error_message(
        self,
        validation_result: Any,
    ) -> str:
        if not isinstance(validation_result, dict):
            return str(validation_result)

        error_message = validation_result.get(
            "error_message"
        )

        if error_message:
            return str(error_message)

        error_stage = validation_result.get(
            "error_stage"
        )

        if error_stage:
            return (
                "Validation failed at stage: "
                f"{error_stage}"
            )

        return json.dumps(
            validation_result,
            ensure_ascii=False,
        )

    def _copy_pipeline_metadata(
        self,
        result: Dict[str, Any],
        pipeline_info: Dict[str, Any],
    ) -> None:
        metadata_fields = (
            "pipeline_id",
            "pipeline_description",
            "description",
            "round",
        )

        for field in metadata_fields:
            if (
                result.get(field) is None
                and pipeline_info.get(field) is not None
            ):
                result[field] = pipeline_info[field]

    def _set_failed_result(
        self,
        pipeline: Dict[str, Any],
        fixed_code: str,
        error_stage: str,
        error_message: str,
        repair_round: int,
        traceback_text: str = "",
    ) -> Dict[str, Any]:
        result = copy.deepcopy(pipeline)

        result["code"] = fixed_code
        result["status"] = "failed"
        result["error_stage"] = error_stage
        result["error_message"] = error_message
        result["traceback"] = traceback_text
        result["repair_status"] = (
            "validation_failed"
        )
        result["repair_round"] = repair_round

        return result

    def repair_single_pipeline_with_retry(
        self,
        pipeline_info: Dict[str, Any],
        max_rounds: Optional[int] = None,
    ) -> Dict[str, Any]:
        if max_rounds is None:
            max_rounds = self.max_repair_rounds

        if max_rounds < 1:
            raise ValueError(
                "max_rounds must be at least 1"
            )

        current_pipeline = copy.deepcopy(
            pipeline_info
        )
        last_result = copy.deepcopy(
            pipeline_info
        )

        for round_idx in range(max_rounds):
            repair_round = round_idx + 1

            # ----------------------------------------
            # 1. Ask the LLM to repair the latest code
            # ----------------------------------------
            try:
                fixed_code = (
                    self._repair_one_pipeline_code(
                        current_pipeline
                    )
                )

                print(
                    f"Repair round: {repair_round}"
                )
                print(fixed_code)

            except Exception as error:
                last_result = copy.deepcopy(
                    current_pipeline
                )

                last_result["repair_status"] = (
                    "failed"
                )
                last_result[
                    "repair_error_message"
                ] = repr(error)
                last_result[
                    "repair_traceback"
                ] = traceback.format_exc()
                last_result["repair_round"] = (
                    repair_round
                )

                continue

            # ----------------------------------------
            # 2. Execute the new code
            # ----------------------------------------
            namespace: Dict[str, Any] = {}

            try:
                exec(fixed_code, namespace)

            except Exception as error:
                current_pipeline = (
                    self._set_failed_result(
                        pipeline=current_pipeline,
                        fixed_code=fixed_code,
                        error_stage="code_exec",
                        error_message=repr(error),
                        traceback_text=(
                            traceback.format_exc()
                        ),
                        repair_round=repair_round,
                    )
                )

                last_result = copy.deepcopy(
                    current_pipeline
                )

                # The next repair round receives the
                # newly generated code and new error.
                continue

            # ----------------------------------------
            # 3. Obtain aug_code from the namespace
            # ----------------------------------------
            aug_code = namespace.get("aug_fn")
            declared = namespace.get("AUG_OUTPUT_STOCHASTIC", None)
            use_random = True if declared is None else bool(declared)
            if not callable(aug_code):
                current_pipeline = (
                    self._set_failed_result(
                        pipeline=current_pipeline,
                        fixed_code=fixed_code,
                        error_stage=(
                            "aug_code_validation"
                        ),
                        error_message=(
                            "Repaired code does not "
                            "define a callable aug_code."
                        ),
                        traceback_text="",
                        repair_round=repair_round,
                    )
                )

                last_result = copy.deepcopy(
                    current_pipeline
                )

                # No exception is raised. The latest
                # code and error go to the next LLM
                # repair round.
                continue

            # ----------------------------------------
            # 4. Run downstream training
            # ----------------------------------------
            try:
                validation_result = (
                    self.validate_fn(aug_code)
                )

            except Exception as error:
                current_pipeline = (
                    self._set_failed_result(
                        pipeline=current_pipeline,
                        fixed_code=fixed_code,
                        error_stage=(
                            "train_pipeline"
                        ),
                        error_message=repr(error),
                        traceback_text=(
                            traceback.format_exc()
                        ),
                        repair_round=repair_round,
                    )
                )

                last_result = copy.deepcopy(
                    current_pipeline
                )

                # The next round receives the current
                # fixed code and the training error.
                continue

            # ----------------------------------------
            # 5. Validate downstream return type
            # ----------------------------------------
            if not isinstance(
                validation_result,
                dict,
            ):
                current_pipeline = (
                    self._set_failed_result(
                        pipeline=current_pipeline,
                        fixed_code=fixed_code,
                        error_stage=(
                            "validation_result"
                        ),
                        error_message=(
                            "validate_fn returned a "
                            "non-dict result: "
                            f"{validation_result!r}"
                        ),
                        traceback_text="",
                        repair_round=repair_round,
                    )
                )

                last_result = copy.deepcopy(
                    current_pipeline
                )

                continue

            validation_result = copy.deepcopy(
                validation_result
            )

            self._copy_pipeline_metadata(
                validation_result,
                pipeline_info,
            )

            validation_result["code"] = fixed_code
            validation_result["repair_round"] = (
                repair_round
            )

            # ----------------------------------------
            # 6. Successful downstream result
            # ----------------------------------------
            if (
                validation_result.get("status")
                != "failed"
            ):
                validation_result["status"] = (
                    "success"
                )
                validation_result[
                    "repair_status"
                ] = "success"

                validation_result[
                    "stochastic"
                ] = use_random

                validation_result.pop(
                    "error_stage",
                    None,
                )
                validation_result.pop(
                    "error_message",
                    None,
                )
                validation_result.pop(
                    "traceback",
                    None,
                )
                validation_result.pop(
                    "repair_error_message",
                    None,
                )
                validation_result.pop(
                    "repair_traceback",
                    None,
                )

                return validation_result

            # ----------------------------------------
            # 7. Structured downstream failure
            # ----------------------------------------
            validation_result["status"] = "failed"

            if not validation_result.get(
                "error_stage"
            ):
                validation_result["error_stage"] = (
                    "train_pipeline"
                )

            if not validation_result.get(
                "error_message"
            ):
                validation_result[
                    "error_message"
                ] = self._extract_error_message(
                    validation_result
                )

            if "traceback" not in validation_result:
                validation_result["traceback"] = ""

            validation_result[
                "repair_status"
            ] = "validation_failed"

            current_pipeline = copy.deepcopy(
                validation_result
            )
            last_result = copy.deepcopy(
                validation_result
            )

        return last_result

    def repair_all_failed_pipelines(
        self,
    ) -> None:
        history_data = (
            self._load_history_pipeline()
        )

        for idx, pipeline_info in enumerate(
            history_data
        ):
            if not isinstance(
                pipeline_info,
                dict,
            ):
                continue

            # Only repair downstream training failures.
            # Existing successful pipelines remain
            # unchanged.
            if (
                pipeline_info.get("status")
                != "failed"

            ):
                continue

            repaired_result = (
                self.repair_single_pipeline_with_retry(
                    pipeline_info
                )
            )

            if not isinstance(
                repaired_result,
                dict,
            ):
                continue

            # Replace the failed record at the same
            # position and retain all other records.
            history_data[idx] = repaired_result

            # Preserve JSONL format: one pipeline per
            # line.
            self._save_history_pipeline(
                history_data
            )