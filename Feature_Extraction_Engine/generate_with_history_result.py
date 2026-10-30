import json
from typing import Any, Dict, List
from .default import FEATURE_EXTRACTION_STRATEGY_TEMPLATE

FEATURE_ITERATION_SYSTEM_MESSAGE = """
Generate next-round feature-extraction (FE) strategy descriptions for an
iterative generate-validate-reflect loop. Output is for a downstream
code-generation agent, not executable code or evaluation feedback.

Rules:
- Follow the specification, constraints, allowed_ops when provided, and
  output schema. Do not assume undocumented fields. Return valid JSON only,
  without fences, explanations, extra fields, or missing required fields.
- Change FE only. Keep augmentation, the downstream model and its
  hyperparameters, its training procedure, and the evaluation protocol
  fixed. FE components themselves may be trained, fine-tuned, or
  supervised on training data when the specification permits.
- Fit learned components, statistics, vocabularies, and transforms only on
  training data; apply the same fitted mapping to evaluation inputs.
  Supervised fitting is allowed only when explicitly permitted. Never use
  evaluation data for fitting or leak targets or prohibited metadata.
- Respect the required modality, interface, output representation, and
  sample/annotation alignment. Feature dimensions may vary across
  candidates unless explicitly fixed, but training and evaluation outputs
  must have consistent dimensions and feature meanings within a candidate.
- Make strategies concrete enough to implement without pipeline redesign.
  Do not regenerate the fixed baseline or add a baseline field unless
  explicitly requested by the output schema.
- Feature scaling, normalization, and relative block weighting are
  part of FE and may be varied, since the downstream model's
  hyperparameters are fixed.
""".strip()


def build_iterative_feature_generation_messages(
    spec: Dict[str, Any],
    relevant_results: Any,
    default_output_format=FEATURE_EXTRACTION_STRATEGY_TEMPLATE,
) -> List[Dict[str, str]]:

    payload = {
        key: spec.get(key, {})
        for key in (
            "dataset_info",
            "task_info",
            "user_goal",
            "constraints",
            "target_metrics",
        )
    }

    if spec.get("allowed_ops") is not None:
        payload["allowed_ops"] = spec["allowed_ops"]

    payload["history_result"] = relevant_results
    payload["output_format"] = default_output_format

    user_message = f"""
Candidate design:
- Generate 6–8 candidates unless constraints or the schema specify
  otherwise. Use historical results to identify promising, weak, and
  underexplored feature-extraction directions. Prioritize exploration
  of diverse hypotheses rather than refinement of the best-performing
  pipeline. Consider both standalone extraction approaches and
  extensions or combinations of existing representations.
  Do not assume that retaining historically successful components
  is necessary for a promising candidate.

- Candidates may refine, extend, simplify, replace, or combine any
  previously explored strategy when justified by evidence, but are
  not required to build upon existing pipelines. Allocate candidates
  adaptively based on task relevance, historical evidence,
  unexplored opportunities, feasibility, and expected information
  gain, without predetermined preferences or quotas.

- First reason about what information the prediction task needs
  (semantic, contextual, structural, relational, statistical, or
  domain-specific signals, as fits the task and modality) and use it
  to motivate hypotheses, without favoring task-specific features
  over general-purpose representations.

- Consider the full range of feasible feature-extraction approaches,
  including substantially different sources, forms, and methods of
  representing predictive information. Do not limit exploration to
  architectures, algorithms, representations, or feature families
  appearing in historical results.
  Familiarity, implementation convenience, or frequency of
  appearance in history should not be treated as evidence of
  predictive value.

- Exploration axes:
  (a) task-aware signals not captured by existing features;
  (b) feature extractor: algorithms, architectures, pretrained
      models, learning objectives, feature families, or other
      methods of deriving predictive representations;
  (c) input representation: construction, selection, granularity,
      resolution, organization, and available information;
  (d) aggregation and interaction modeling: pooling, layer
      selection, attention, spatial, temporal, spectral,
      hierarchical, or cross-modal relationships as applicable;
  (e) transformation: normalization, scaling, dimensionality
      reduction, projection, or supervised transforms when permitted;
  (f) combination, fusion, or alignment of complementary
      features, representations, views, or modalities.

- These axes are exploratory guides, not an exhaustive taxonomy
  or mandatory categories. Candidates may introduce methods
  or hypotheses that do not fit neatly within them.

- Within any axis, a change may replace, add (including higher
  dimensionality via a block, view, layer, modality, or larger
  extractor), remove, simplify, or re-parameterize components.
  These directions are equally legitimate. Judge added dimensions
  by the new information they bring, not by size; avoid blocks
  that merely repeat existing information.

- Spread the round across hypotheses and directions where
  justified, but do not force diversity for its own sake.
  Concentrating on a promising direction or exploring a
  substantially different approach are both legitimate when
  supported by evidence or expected information gain.
  Exploration candidates are valued for the information they
  provide, not only expected performance. Weakly used or unused
  inputs, signals, relationships, and methods are natural
  hypothesis sources.

- Each candidate tests one main hypothesis and differs meaningfully
  from the others (information, family, representation, aggregation,
  granularity, component, parameters, or diagnostic purpose), not
  merely in name. Refinements make one controlled main change;
  standalone candidates may introduce entirely different extraction
  approaches without being constrained to minimal modifications
  of existing pipelines. When an existing pipeline has several
  components, consider ablating its weakest or least-justified one.

- Do not repeat FE strategies already evaluated under the current
  augmentation except as a justified control, ablation, replication,
  or implementation repair. Distinguish execution failures from
  poor performance. Revisit weak families only with a materially
  different hypothesis; one exploratory attempt is weak evidence
  against a family. A family weak standalone may return as a
  complementary block only if the candidate explains what it adds.

- Keep augmentation and downstream behavior fixed, but account
  for them.

- Within the output schema, specify concrete operations,
  parameters, inputs, ordering, aggregation, fit/transform scope,
  output representation and dimensions or their derivation,
  and alignment/leakage handling as needed, compatible with
  the fixed downstream interface.
<SPECIFICATION>
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
</SPECIFICATION>

Return only JSON matching output_format.
""".strip()

    return [
        {
            "role": "system",
            "content": FEATURE_ITERATION_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

def generate_pipelines_with_history_result(
    llm_client,
    spec: Dict[str, Any],
    hsitory_results: Any,
) -> str:

    messages = build_iterative_feature_generation_messages(
        spec=spec,
        relevant_results=hsitory_results,
        default_output_format=FEATURE_EXTRACTION_STRATEGY_TEMPLATE,
    )

    return llm_client.generate(messages=messages)