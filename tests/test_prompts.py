import pytest

from kinematics_grading.domain import parse_step_info
from kinematics_grading.grading.prompts import (
    build_grading_prompt,
    build_grading_prompt_from_prefix,
    build_rule_schema_description,
    build_stable_prefix,
    sends_reference_image,
)


def test_prompt_contains_step_specific_calibration(rubric_ex2):
    step_info = parse_step_info("02_more_curve_adjusted.png")
    prompt = build_grading_prompt(
        exercise=2,
        step_filename="02_more_curve_adjusted.png",
        step_info=step_info,
        instruction_text="Some instruction text.",
        rubric=rubric_ex2,
    )
    assert "02_more_curve_adjusted.png" in prompt
    assert "40%" in prompt  # expected progress for step 02
    assert "step 05 should be close to the reference" in prompt
    assert "Exercise number: 2" in prompt


def test_prompt_embeds_full_rubric_json(rubric_ex2):
    step_info = parse_step_info("05_labels_fixed.png")
    prompt = build_grading_prompt(
        exercise=2,
        step_filename="05_labels_fixed.png",
        step_info=step_info,
        instruction_text="",
        rubric=rubric_ex2,
    )
    for rule_id in ("forward", "parking", "reverse"):
        assert rule_id in prompt


def test_required_schema_preserves_every_rule_id(rubric_ex2):
    schema_json = build_rule_schema_description(rubric_ex2)
    for rule_id in ("forward", "parking", "reverse"):
        assert f'"id": "{rule_id}"' in schema_json


def test_stable_prefix_is_byte_identical_across_steps(rubric_ex2):
    """The whole point of build_stable_prefix: grading five different steps
    of the same exercise must reuse the exact same prefix string, so that
    (a) the client-side cache in grading/pipeline.py is a real cache hit and
    (b) the provider's prompt-prefix cache has something stable to match."""
    prefix_a = build_stable_prefix(2, "Some instruction.", rubric_ex2)
    prefix_b = build_stable_prefix(2, "Some instruction.", rubric_ex2)
    assert prefix_a == prefix_b


def test_stable_prefix_excludes_step_specific_fields(rubric_ex2):
    prefix = build_stable_prefix(2, "Some instruction.", rubric_ex2)
    assert "Step file:" not in prefix
    assert "Expected approximate progress" not in prefix


def test_full_prompt_from_prefix_matches_one_shot_builder(rubric_ex2):
    """build_grading_prompt (one-shot) and build_stable_prefix +
    build_grading_prompt_from_prefix (cached path) must produce byte-identical
    prompts for the same inputs -- callers should get the same request either way."""
    step_info = parse_step_info("03_main_trend_corrected.png")

    one_shot = build_grading_prompt(
        exercise=2, step_filename="03_main_trend_corrected.png", step_info=step_info,
        instruction_text="Some instruction.", rubric=rubric_ex2,
    )

    prefix = build_stable_prefix(2, "Some instruction.", rubric_ex2)
    cached_path = build_grading_prompt_from_prefix(prefix, "03_main_trend_corrected.png", step_info)

    assert one_shot == cached_path


def test_different_steps_share_prefix_but_differ_in_suffix(rubric_ex2):
    prefix = build_stable_prefix(2, "Some instruction.", rubric_ex2)
    step_02 = parse_step_info("02_more_curve_adjusted.png")
    step_04 = parse_step_info("04_closer_to_target.png")

    prompt_02 = build_grading_prompt_from_prefix(prefix, "02_more_curve_adjusted.png", step_02)
    prompt_04 = build_grading_prompt_from_prefix(prefix, "04_closer_to_target.png", step_04)

    assert prompt_02.startswith(prefix)
    assert prompt_04.startswith(prefix)
    assert prompt_02 != prompt_04
    assert "40%" in prompt_02 and "80%" in prompt_04


def test_unknown_partial_credit_policy_raises(rubric_ex2):
    with pytest.raises(ValueError):
        build_stable_prefix(2, "Some instruction.", rubric_ex2, partial_credit_policy="not_a_real_policy")


def test_sends_reference_image_differs_by_policy():
    assert sends_reference_image("step_calibrated") is True
    assert sends_reference_image("standalone_submission") is False


def test_standalone_submission_drops_step_and_progress_language(rubric_ex2):
    step_info = parse_step_info("02_more_curve_adjusted.png")
    prompt = build_grading_prompt(
        exercise=2, step_filename="02_more_curve_adjusted.png", step_info=step_info,
        instruction_text="Some instruction.", rubric=rubric_ex2, partial_credit_policy="standalone_submission",
    )
    # No hint that this is part of a progression...
    assert "40%" not in prompt
    assert "Expected approximate progress" not in prompt
    assert "Step meaning" not in prompt
    assert "intermediate" not in prompt.lower()
    assert "correction step" not in prompt.lower()
    # ...and no mention of a reference/correct-answer image to compare against.
    assert "reference answer" not in prompt.lower()
    assert "correct reference" not in prompt.lower()
    # The rubric and instruction are still fully present.
    for rule_id in ("forward", "parking", "reverse"):
        assert rule_id in prompt
    assert "Some instruction." in prompt


def test_standalone_submission_still_ties_partial_credit_to_rubric_thresholds(rubric_ex2):
    step_info = parse_step_info("01_slight_correction.png")
    prompt = build_grading_prompt(
        exercise=2, step_filename="01_slight_correction.png", step_info=step_info,
        instruction_text="Some instruction.", rubric=rubric_ex2, partial_credit_policy="standalone_submission",
    )
    assert "expected threshold" in prompt
    assert "near-zero credit" in prompt


def test_standalone_submission_omits_progress_fields_from_schema(rubric_ex2):
    schema_with_progress = build_rule_schema_description(rubric_ex2, include_progress_fields=True)
    schema_without_progress = build_rule_schema_description(rubric_ex2, include_progress_fields=False)
    assert "step_expected_progress" in schema_with_progress
    assert "is_progress_plausible" in schema_with_progress
    assert "step_expected_progress" not in schema_without_progress
    assert "is_progress_plausible" not in schema_without_progress


def test_standalone_submission_prefix_is_byte_identical_across_steps(rubric_ex2):
    prefix_a = build_stable_prefix(2, "Some instruction.", rubric_ex2, partial_credit_policy="standalone_submission")
    prefix_b = build_stable_prefix(2, "Some instruction.", rubric_ex2, partial_credit_policy="standalone_submission")
    assert prefix_a == prefix_b


def test_step_calibrated_and_standalone_submission_prefixes_differ(rubric_ex2):
    step_calibrated = build_stable_prefix(2, "Some instruction.", rubric_ex2, partial_credit_policy="step_calibrated")
    standalone = build_stable_prefix(2, "Some instruction.", rubric_ex2, partial_credit_policy="standalone_submission")
    assert step_calibrated != standalone
