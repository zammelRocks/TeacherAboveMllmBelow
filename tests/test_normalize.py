import pytest

from kinematics_grading.grading.normalize import extract_json_from_text, normalize_grade_json


def test_extract_json_handles_plain_json():
    assert extract_json_from_text('{"a": 1}') == {"a": 1}


def test_extract_json_handles_markdown_fence():
    text = '```json\n{"a": 1}\n```'
    assert extract_json_from_text(text) == {"a": 1}


def test_extract_json_handles_surrounding_prose():
    text = 'Here is the grade:\n{"a": 1}\nThanks.'
    assert extract_json_from_text(text) == {"a": 1}


def test_extract_json_raises_on_no_json():
    with pytest.raises(ValueError):
        extract_json_from_text("no json here at all")


def test_normalize_reproduces_case_study_from_paper(rubric_ex2):
    """forward: 3.0/3.0, parking: 2.0/4.0, reverse: 3.0/3.0 -> score_ratio 0.80
    (Part II paper, sec:res-case-study)."""
    raw = {
        "rules": [
            {"id": "forward", "awarded": 3.0, "observed": "..."},
            {"id": "parking", "awarded": 2.0, "observed": "..."},
            {"id": "reverse", "awarded": 3.0, "observed": "..."},
        ],
        "feedback": "Good progress.",
    }
    grade = normalize_grade_json(raw, rubric_ex2, step_file="02_more_curve_adjusted.png", expected_progress=40)
    assert grade["total_awarded"] == pytest.approx(8.0)
    assert grade["score_ratio"] == pytest.approx(0.80)
    assert grade["score_100"] == 80


def test_normalize_reproduces_exact_threshold_arithmetic(rubric_ex2):
    """forward: 3/3, parking: 0/4, reverse: 3/3 -> exactly 0.60, the
    acceptance threshold -- even with zero credit on the one unmet rule."""
    raw = {
        "rules": [
            {"id": "forward", "awarded": 3.0},
            {"id": "parking", "awarded": 0.0},
            {"id": "reverse", "awarded": 3.0},
        ],
    }
    grade = normalize_grade_json(raw, rubric_ex2, step_file="02_more_curve_adjusted.png", expected_progress=40)
    assert grade["score_ratio"] == pytest.approx(0.60)


def test_normalize_clips_out_of_range_awarded(rubric_ex2):
    raw = {"rules": [{"id": "forward", "awarded": 999}, {"id": "parking", "awarded": -5}, {"id": "reverse", "awarded": 3}]}
    grade = normalize_grade_json(raw, rubric_ex2, step_file="x.png", expected_progress=50)
    by_id = {r["id"]: r["awarded"] for r in grade["rules"]}
    assert by_id["forward"] == 3.0  # clipped to rule's max points
    assert by_id["parking"] == 0.0  # clipped to 0


def test_normalize_handles_missing_rule_in_response(rubric_ex2):
    """If the model's response omits a rule entirely, it should be treated
    as zero credit rather than raising."""
    raw = {"rules": [{"id": "forward", "awarded": 3.0}]}
    grade = normalize_grade_json(raw, rubric_ex2, step_file="x.png", expected_progress=50)
    by_id = {r["id"]: r["awarded"] for r in grade["rules"]}
    assert by_id["parking"] == 0.0
    assert by_id["reverse"] == 0.0
    assert len(grade["rules"]) == 3  # every rubric rule id is preserved regardless
