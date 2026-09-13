"""Regression test for the axis-label bug documented in the Part II paper
(sec:confound): infer_labels() previously mapped Exercise 3 -> "v(t)" and
Exercise 4 -> "a(t)", contradicting each exercise's own instruction text and
rubric quantity. GPT-5.1 caught this reliably in the original corpus
(quoted directly from grading_results/intermediate_step_rule_grades.jsonl
in the paper); this test exists so a future refactor can't silently
reintroduce it without a test failing first.
"""

import json

import pytest

from kinematics_grading.config import DATA_DIR
from kinematics_grading.generation.rendering import infer_labels


@pytest.mark.parametrize(
    "exercise,expected_quantity_symbol",
    [
        (1, "x(t)"),
        (2, "x(t)"),
        (3, "x(t)"),  # position graph; dfdx/d2fdx2 rubric checks its derivatives
        (4, "v(t)"),  # velocity graph, checked via the raw value f directly
    ],
)
def test_infer_labels_matches_rubric_quantity(exercise, expected_quantity_symbol):
    _, ylabel = infer_labels(exercise)
    assert ylabel.startswith(expected_quantity_symbol), (
        f"exercise {exercise} labeled {ylabel!r}, expected it to start with {expected_quantity_symbol!r}"
    )


@pytest.mark.parametrize("exercise", [1, 2, 3, 4])
def test_infer_labels_consistent_with_rubric_json(exercise):
    """Cross-check against the actual rubric files rather than a hardcoded
    expectation, so this fails loudly if the rubrics themselves ever change
    without the label mapping being revisited."""
    rubric_path = DATA_DIR / f"exercice_{exercise}" / f"rubric_{exercise}.json"
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    quantities = {rule["quantity"] for rule in rubric["rules"]}
    _, ylabel = infer_labels(exercise)

    if quantities <= {"f"}:
        # raw-value rubric: the plotted quantity IS what the rubric checks
        pass  # exercise 4 -- v(t) plotted and checked directly; nothing further to assert generically
    elif quantities & {"dfdx", "d2fdx2"}:
        # derivative rubric: the plotted quantity is one differentiation
        # level below what the rubric checks (x(t) plotted, v/a derived)
        assert not ylabel.startswith(("v(t)", "a(t)")), (
            f"exercise {exercise} has a derivative-based rubric {quantities} "
            f"but is labeled {ylabel!r}, which names a derived quantity as if it were plotted directly"
        )
