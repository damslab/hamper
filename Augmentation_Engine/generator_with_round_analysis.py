import json
from typing import Any, Dict, List, Optional

from.default import PIPELINE_OUTPUT_FORMAT_TEMPLATE

NOISE_POLICY_ENABLED = """
# Noise handling
- The system performs multi-seed evaluation and updates the PRIMARY metric
  to its mean; noise_std is its across-seed sample standard deviation.
- noise_std = 0 means the pipeline's PRIMARY-metric result is unaffected
  by random-seed variation across the tested seeds under the fixed
  evaluation setup.
  Missing noise_std means unavailable, not zero.
- Compare PRIMARY means using both pipelines' available noise information.
  Differences comparable to noise are inconclusive; exceeding noise alone
  does not prove significance.
- Never apply PRIMARY noise to SECONDARY metrics or propose seed-only
  replications; the system handles these evaluations automatically.
- A stable pipeline result does not establish a conclusion about its family.
"""

NOISE_POLICY_DISABLED = """
# Noise handling
- Random-seed analysis is disabled. Ignore random-seed effects and any
  seed-noise information in current or historical feedback.
- Focus on discovering effective augmentation methods using the reported
  metrics, optimization directions, constraints, and behavioral evidence.
- Do not discuss seed uncertainty, request noise estimates, propose seed-only
  replications, or require multi-seed confirmation for search decisions.
- Rules elsewhere requiring seed-noise estimates or seed-based repetition
  do not apply in this mode, including family status changes.
- Continue checking implementation and design confounds. Ignoring seed
  effects does not establish statistical significance or seed robustness.
"""

BASE_SYSTEM_MESSAGE = """You generate candidate data augmentation strategies for an iterative optimization loop.
 
Rules:
- Follow the specification, constraints, allowed_ops, and output_format exactly.
- The specification is the only source of hard restrictions. Do not introduce
  additional restrictions of your own, and do not treat conventions implied by
  past candidates as rules.
- If allowed_ops is empty or absent, no operator whitelist applies; scanning the
  full strategy space is then your own responsibility.
- If a restriction in the specification is ambiguous, apply the least restrictive
  reasonable reading. 
- Preserve targets: every example the pipeline produces or modifies must carry a
  target that is still correct for it. Where a transformation of the input has a
  corresponding transformation of the target (geometric ops on masks or boxes,
  resampling on sequence targets, unit changes on regression targets), applying
  both is target-preserving. This rule constrains what may be produced; it does
  not restrict which components of an example, or how many source examples, an
  operation may draw on. Whether a particular recombination preserves the target
  is an empirical question to be tested, not a reason to omit the candidate.
- Return 4-8 valid candidates as the required JSON
  only. Distinctness is judged at the level of mechanism and information source,
  not wording or parameter values. Do not pad the set with variants to reach the
  count; 4 strong candidates are better than 8 with
  filler.
- The first and last output characters must be `{` and `}`.
- Do not include Markdown, headings, or extra text.
- Return exactly one top-level object of the form
  {"generation_round_summary": {...}, "pipelines": [...]};
  these two fields must be siblings.
- Keep descriptions, operation descriptions, parameters, and notes concise.
  Include only details needed to implement or distinguish each candidate.
"""


GENERATION_POLICY = """
Generation policy:
 
# Definitions
- An "example" is one unit of the dataset. Its "components" are whatever parts the
  specification exposes: fields, channels, views, segments, sub-structures, or the
  whole input. A "family" is defined by its mechanism and its information source,
  not by the component it acts on. Two operations acting on the same component
  belong to different families if one uses only that example's own content while
  the other imports content from elsewhere, or if they draw on different
  information sources (raw input, targets, auxiliary attributes, model state,
  external models or data).
- "Repeated" means at least two independent observations, i.e. different seeds or
  different evaluation partitions. Two parameter variants within one round are not
  repetition.
 

# Mode selection
- Choose the generation mode from the historical feedback: exploitation,
  exploration, ablation, repair, recombination, simplification, diversification,
  or risk-taking. No mode is preferred by default.
- When feedback is weak, ambiguous, or confounded, prioritize diverse plausible
  exploration over refinement of any apparent leader.
- When feedback is strong and consistent, refinement is appropriate, but the set
  must still carry meaningful behavioral variation.
- Treat history as evidence, not as a rule. You may introduce a new strategy
  family that history does not support, provided it is plausible for the task and
  compliant with the specification. Independently decide how much weight the
  history deserves; a result produced under conditions that no longer hold, or
  under an unresolved confound, deserves little.
 
# Evidence handling
- Evaluate evidence jointly across the primary metric and every secondary quantity
  the specification tracks, plus constraint compliance; do not overfocus on one
  result or one metric.
- Read the dispersion across evaluation partitions before interpreting any
  difference. When a reported difference is small relative to that dispersion,
  treat the direction as weakly informative at best and say so, rather than
  building the round's hypothesis on it.
- Treat effects confounded by implementation, yield, filtering, composition,
  coverage, scope, strength, scale, or randomness as unresolved. Prefer candidates
  that isolate the confound or challenge the current hypothesis over candidates
  that assume it away.
- Distinguish designed from realized quantities. When a realized quantity departs
  from its design, the result is confounded by yield and no family status may
  change on that evidence.
- Interpret negative history cautiously. A weak or negative result is a reason to
  avoid, repair, isolate the failing component, or revisit the idea under a
  meaningfully different design -- not a reason to discard the family. When
  revisiting, seek a better-controlled design that isolates the useful component,
  not merely a safer or narrower one; narrowing without a control adds a second
  unattributed restriction rather than resolving the first.
 
# First test of a family
- For an untested or unresolved family, first propose a canonical version subject
  only to mandatory constraints. Safeguards, filters, narrow scopes, similarity or
  eligibility conditions on what may be combined, and any other restriction that
  shrinks the operating range are separately attributable variants or repairs,
  never part of the canonical version.
- A canonical version must not be evaluated only at the minimum admissible
  strength, coverage, or scale. Test it at two settings spanning the admissible
  range, or at a mid-to-high setting. A result obtained only at the minimum
  setting is not evidence about the family.
- Every candidate must declare its `family` and its `arm_role`
  (canonical | restricted_variant | refinement | recombination | repair | control |
  replication), so family status can be tracked across rounds.
 
# Status changes
- Promote a family to refinement only after controlled support that is not
  explained by a confound listed above.
- Deprioritize a family when its canonical version was tested faithfully and
  compliantly and showed no advantage. Deprioritization lowers its budget share; it
  does not remove it from the search space, and a later hypothesis may restore it.
  For each deprioritized family, consider whether a controlled design could isolate
  a useful component before letting its share fall to zero.
- Retire a family only in the confirmation stage, and only if all of the following
  hold: its canonical version was tested; a noise estimate was available under
  matched conditions; the runs were faithful, compliant, and minimally confounded;
  and it failed on repeated observations. If any condition fails, its status is
  unresolved.
- Status records must carry the evidence conditions that produced them. Whenever a
  variance estimate becomes available, or a systematic defect in an earlier round is
  identified, re-audit every deprioritization and retirement that relied on that
  round and return the failing entries to unresolved.
- Never infer prohibition, rejection, or retirement from earlier omission, a
  restricted variant, or a historical claim about user preference. Only the original
  specification defines hard restrictions.
- Do not adopt implicit invariants the specification does not state. If every past
  candidate has stayed inside one scope, granularity, intervention level, or unit of
  operation, that is an uncovered region to probe, not a rule to honour.
 
# Budget allocation
- Allocate at least one third of the set to previously untested mechanisms or
  information sources while in the exploration stage, and at least one candidate in
  any stage, unless no eligible untested or unresolved family remains.
- Allocate the remainder in this order: unresolved confounds and untested canonical
  versions of promising families first, then refinements of families with controlled
  support.
- At most two candidates may be parameter, template, strength, coverage, or scope
  variants of a single family, and the set must never consist only of such variants.
  Each refinement must state the behavioral hypothesis it tests, not merely the
  parameter it changes.
- Maintain diversity across mechanism, information source, sample scope
  (within-example vs cross-example), granularity, intervention level (input,
  representation, or training procedure -- only those levels the specification
  admits), target dependence, and risk. Judge coverage of these axes by current
  family status: an axis touched only by a candidate that is now deprioritized,
  retired, or unresolved counts as uncovered.
- Information sources to consider, where the specification exposes them: the
  example itself; other examples sharing its target; semantically or structurally
  similar examples; class or cluster prototypes; auxiliary attributes and metadata;
  model state such as difficulty or uncertainty; external models or data; and the
  recorded failure cases of earlier rounds. This list is a coverage prompt, not a
  whitelist.
- Vary risk deliberately: include at least one conservative refinement of a
  supported motif and at least one higher-novelty candidate. A higher-novelty
  candidate may carry safeguards, but the unguarded canonical version must be scheduled
  either in the same round or explicitly in the next.
- If refinements of the leading family have produced no further movement, shift the
  budget toward mechanistically distinct directions. Revisit the family only when a
  new hypothesis addresses the observed bottleneck.
- State in `generation_round_summary` which prior-round hypothesis this round
  follows up on and whether it was supported, refuted, or left unresolved.
 
# Compliance
- Any quantity fitted on data -- donor or neighbor pools, normalization statistics,
  vocabularies, class priors, difficulty or uncertainty estimates, learned
  generators, or selection thresholds -- must be fitted using only the training
  portion of the evaluation partition it will be used in. State the partition scope
  in the candidate.
- An intervention that changes the target distribution (class balance, target
  density, per-group coverage) is confounded with that change by construction.
  Either pair it with a distribution-matched control in the same round, or state the
  induced shift explicitly so later attribution can account for it.
- Generated examples must differ from their sources unless the specification permits
  identical copies. State how distinctness is ensured; a pipeline that can silently
  collapse to duplicates is not interpretable.
- Avoid exact and behaviorally redundant historical candidates, except replication
  arms in the confirmation stage. Before returning, verify plausibility, compliance,
  distinctness, plausibility for the dataset and goal, absence of invented results,
  completeness of output format.
"""

def build_static_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_info": spec.get("dataset_info", {}),
        "task_info": spec.get("task_info", {}),
        "user_goal": spec.get("user_goal", {}),
        "constraints": spec.get("constraints", {}),
        "allowed_ops": spec.get("allowed_ops"),
        "output_format": PIPELINE_OUTPUT_FORMAT_TEMPLATE,
    }


def normalize_history(value: Optional[Any]) -> Any:
    """Convert JSON strings to objects when possible."""
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_generation_messages(
    spec: Dict[str, Any],
    history_pipelines: Optional[Any] = None,
    history_round_conclusion: Optional[Any] = None,
    noise_analysis_enabled: bool = False,
) -> List[Dict[str, str]]:
    generation_input = build_static_spec(spec)

    generation_input["historical_pipeline_feedback"] = normalize_history(
        history_pipelines
    )
    generation_input["historical_round_conclusion"] = normalize_history(
        history_round_conclusion
    )

    noise_policy = (
        NOISE_POLICY_ENABLED
        if noise_analysis_enabled
        else NOISE_POLICY_DISABLED
    )

    system_message = (
        BASE_SYSTEM_MESSAGE
        + "\n\n"
        + GENERATION_POLICY
        + "\n\n"
        + noise_policy
    )

    user_message = (
        "Generate the next round of candidate augmentation pipeline "
        "strategies from this input:\n"
        + compact_json(generation_input)
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

def generate_pipeline_with_history_analysis(
    llm_client,
    spec: Dict[str, Any],
    history_pipelines: Optional[Any] = None,
    history_round_conclusion: Optional[Any] = None,
    noise_analysis_enabled: bool = False,
) -> str:
    messages = build_generation_messages(
        spec=spec,
        history_pipelines=history_pipelines,
        history_round_conclusion=history_round_conclusion,
        noise_analysis_enabled=noise_analysis_enabled,
    )

    return llm_client.generate(messages=messages)