"""Rubric-grounded grading prompt, ported from grade_intermediate_steps_azure.py
/ reverse_grading_resume.py / failed_instance_resume.py (previously
copy-pasted identically into all three; single source of truth now). Same
content as Listing 1 in the Part II paper (sec:grading), restructured for
prompt caching (see build_stable_prefix docstring below).

Two grading policies are supported (`partial_credit_policy`):

- "step_calibrated" (default): the original policy behind every persisted
  result so far -- frames each image as an intermediate correction step and
  explicitly permits early steps to get partial credit just for moving in
  the right direction. Kept byte-identical to preserve reproducibility.

- "standalone_submission": a second, independent experiment (does not
  replace or alter the first -- see analysis.acceptance.
  partial_credit_on_unsatisfied_rules for what motivated it). Each image is
  graded as if it were a complete, standalone submission to the exercise:
  no mention of correction steps, no expected-progress hint, no reference
  image sent alongside it (see grading/pipeline.py and azure_client.py,
  where `send_reference_image` gates that). Partial credit must be
  justified strictly by closeness to each rubric rule's own
  quantity/relation/expected threshold. The goal of this policy is not to
  study the grader's leniency (that's the first experiment) but to judge
  the *generation method*: does each independently-graded step look like an
  increasingly-correct standalone answer on its own merits, with whatever
  step-to-step score variance that reveals, uncontaminated by "it's early,
  go easy on it" framing?
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ..domain import StepInfo

_STEP_CALIBRATED_RECEIVES = """\
1. A generated intermediate graph image.
2. The correct reference answer image.
3. The exercise instruction.
4. The exact JSON rubric."""

_STEP_CALIBRATED_POLICY = """\
Your task:
Grade the generated intermediate image according to EACH rubric rule.
You must assign partial credit for every rule.

General grading policy (applies to every intermediate step of this exercise):
- This image is an intermediate correction step, not necessarily the final answer.
- Apply the rubric, but calibrate harshness to the intermediate step:
  - early steps may receive partial credit if they move in the correct direction
  - step 05 should be close to the reference and should be graded more strictly
- Do NOT grade the reference answer.
- Penalize any extra non-graph artifacts, hooks, side fragments, unrelated lines, wrong direction, wrong function family, wrong slope sign, wrong axes, or missing labels if labels are expected.
- Reward realistic partial progress toward the reference."""

_STANDALONE_SUBMISSION_RECEIVES = """\
1. A submitted graph image, to grade exactly as a complete, standalone answer.
2. The exercise instruction.
3. The exact JSON rubric."""

_STANDALONE_SUBMISSION_POLICY = """\
Your task:
Grade the submitted image as if it were a student's complete, final answer to this
exercise. You are not told, and must not assume, whether this image was produced as
part of some other process, a draft, or a revision -- judge it purely on its own
merits against the instruction and rubric below, exactly as you would any standalone
submission.

Partial-credit policy (applies uniformly to every rule):
- Each rule specifies a quantity, a relation, and an expected threshold (for example "dfdx gt 0.1").
- Award partial credit ONLY based on how close the image's observed value for that quantity is to
  satisfying its own expected threshold -- never based on assumed effort, assumed intent, or assumed
  progress toward some other image you have not been shown.
- A rule whose observed value is on the correct side of the threshold but has not yet reached it may
  receive partial credit, scaled by how close the observed value is to that threshold.
- A rule whose observed value has the wrong sign, or is far from its threshold, must receive
  near-zero credit.
- Penalize any extra non-graph artifacts, hooks, side fragments, unrelated lines, wrong direction, wrong function family, wrong slope sign, wrong axes, or missing labels if labels are expected."""

_POLICIES: Dict[str, Dict[str, Any]] = {
    "step_calibrated": {
        "receives": _STEP_CALIBRATED_RECEIVES,
        "task_and_policy": _STEP_CALIBRATED_POLICY,
        "include_progress_fields": True,
        "send_reference_image": True,
    },
    "standalone_submission": {
        "receives": _STANDALONE_SUBMISSION_RECEIVES,
        "task_and_policy": _STANDALONE_SUBMISSION_POLICY,
        "include_progress_fields": False,
        "send_reference_image": False,
    },
}


def _policy_config(partial_credit_policy: str) -> Dict[str, Any]:
    if partial_credit_policy not in _POLICIES:
        raise ValueError(f"Unknown partial_credit_policy: {partial_credit_policy!r}")
    return _POLICIES[partial_credit_policy]


def sends_reference_image(partial_credit_policy: str) -> bool:
    """Whether grade_one()/AzureVisionGrader.grade() should include the
    reference (correct-answer) image in the request for this policy."""
    return _policy_config(partial_credit_policy)["send_reference_image"]


def includes_progress_fields(partial_credit_policy: str) -> bool:
    """Whether the per-step suffix/output schema should reveal step meaning
    and expected-progress for this policy (see build_grading_prompt_from_prefix)."""
    return _policy_config(partial_credit_policy)["include_progress_fields"]


def build_rule_schema_description(rubric: Dict[str, Any], include_progress_fields: bool = True) -> str:
    max_score = rubric["max_score"]
    rules = rubric["rules"]

    rule_objects = [
        {
            "id": rule.get("id"),
            "points": rule.get("points"),
            "awarded": 0,
            "quantity": rule.get("quantity"),
            "relation": rule.get("relation"),
            "expected": rule.get("expected"),
            "observed": "describe what you see in the generated image",
            "satisfied": False,
            "partial_credit_reason": "explain awarded points",
        }
        for rule in rules
    ]

    schema = {
        "max_score": max_score,
        "total_awarded": 0,
        "score_ratio": 0.0,
        "score_100": 0,
        "step_file": "filename.png",
    }
    if include_progress_fields:
        schema["step_expected_progress"] = 0
    schema["rules"] = rule_objects
    schema["feedback"] = "short global feedback"
    schema["is_graph_only"] = True
    if include_progress_fields:
        schema["is_progress_plausible"] = True
    return json.dumps(schema, indent=2, ensure_ascii=False)


def build_stable_prefix(
    exercise: int, instruction_text: str, rubric: Dict[str, Any], partial_credit_policy: str = "step_calibrated",
) -> str:
    """Everything about a grading prompt that is identical across all five
    steps, all instances, and all rounds of one exercise -- i.e. everything
    except which specific image is being graded right now.

    Putting the invariant content (task description, rubric JSON,
    instruction text, output-schema spec) in one contiguous block, with the
    per-image specifics appended afterward (see build_grading_prompt),
    maximizes the length of the prompt prefix that is byte-identical across
    requests. Azure OpenAI / OpenAI apply automatic prompt caching to the
    longest matching prefix between consecutive requests (no special request
    parameter needed for Chat Completions) -- so this reordering is what
    actually lets that caching engage across roughly 1,500 gradings per
    exercise, on top of the explicit client-side cache in pipeline.py that
    avoids rebuilding this string per image in the first place.

    See the module docstring for what `partial_credit_policy` selects.
    """
    policy = _policy_config(partial_credit_policy)
    rubric_json = json.dumps(rubric, indent=2, ensure_ascii=False)
    required_schema = build_rule_schema_description(rubric, policy["include_progress_fields"])

    return f"""
You are a strict grading assistant for kinematics graph images.

You will receive:
{policy["receives"]}

{policy["task_and_policy"]}

Exercise number: {exercise}

Exercise instruction:
{instruction_text}

Rubric JSON:
{rubric_json}

Rule interpretation guidance:
- "points" is the maximum for that rule.
- "awarded" must be between 0 and that rule's "points".
- "total_awarded" must equal the sum of all rule awarded values.
- "score_ratio" must equal total_awarded / max_score.
- "score_100" must equal round(score_ratio * 100).
- Preserve all rubric rule ids.
- Preserve each rule's points, quantity, relation, and expected values in your output.
- For "observed", describe the visual evidence from the generated image.
- For "partial_credit_reason", explain why the rule got that many points.

Return ONLY valid JSON, no markdown, no explanations outside JSON.

Required JSON shape:
{required_schema}
""".strip()


def build_grading_prompt_from_prefix(
    stable_prefix: str, step_filename: str, step_info: StepInfo, include_progress_fields: bool = True,
) -> str:
    """Append the per-image specifics to an already-built (and ideally
    cached) stable prefix. Prefer this over build_grading_prompt() in any
    loop that grades multiple steps of the same exercise.

    `include_progress_fields` must match the policy the stable_prefix was
    built with -- False (standalone_submission) omits the step meaning and
    expected-progress hint, since revealing "this is step 2 of a 5-step
    correction, ~40% of the way there" is itself a leniency signal that
    policy is specifically testing the absence of.
    """
    suffix_lines = ["--- This specific image ---", f"Step file: {step_filename}"]
    if include_progress_fields:
        expected_progress = step_info.expected_progress if step_info.expected_progress is not None else 50
        step_name = step_info.step_name.replace("_", " ")
        suffix_lines += [f"Step meaning: {step_name}", f"Expected approximate progress toward final answer: {expected_progress}%"]
    suffix_lines += ["", "Grade the image described above now, following the general grading policy stated earlier in this prompt."]

    return f"{stable_prefix}\n\n{chr(10).join(suffix_lines)}"


def build_grading_prompt(
    exercise: int,
    step_filename: str,
    step_info: StepInfo,
    instruction_text: str,
    rubric: Dict[str, Any],
    partial_credit_policy: str = "step_calibrated",
) -> str:
    """Convenience one-shot version (rebuilds the stable prefix every call).
    Grading pipelines that call this repeatedly for the same exercise should
    use build_stable_prefix() once (cached) + build_grading_prompt_from_prefix()
    per image instead -- see grading/pipeline.py."""
    policy = _policy_config(partial_credit_policy)
    stable_prefix = build_stable_prefix(exercise, instruction_text, rubric, partial_credit_policy)
    return build_grading_prompt_from_prefix(stable_prefix, step_filename, step_info, policy["include_progress_fields"])
