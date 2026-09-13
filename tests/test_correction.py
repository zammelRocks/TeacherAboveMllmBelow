import numpy as np
import pytest

from kinematics_grading.generation.correction import (
    CORRECTION_STRATEGIES,
    correction_mask,
    make_intermediate_curve,
)
from kinematics_grading.generation.optuna_tuning import score_curve


DEFAULT_PARAMS = dict(
    strategy="left_to_right",
    window_width=0.2,
    window_center=0.5,
    easing=1.0,
    global_blend=0.3,
    scale_error=0.0,
    vertical_shift=0.0,
    early_smooth_k=5,
    late_smooth_k=1,
)


@pytest.mark.parametrize("strategy", CORRECTION_STRATEGIES)
def test_correction_mask_bounded_0_1(strategy):
    x = np.linspace(0, 1, 100)
    mask = correction_mask(x, strategy, progress=0.5, width=0.2, center=0.5)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_correction_mask_rejects_unknown_strategy():
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        correction_mask(x, "not_a_strategy", 0.5, 0.2, 0.5)


def test_intermediate_curve_moves_toward_target():
    x = np.linspace(0, 1, 240)
    source_y = np.full_like(x, 0.2)
    target_y = np.full_like(x, 0.8)
    valid = np.ones_like(x, dtype=bool)

    y1, _ = make_intermediate_curve(source_y, target_y, x, valid, DEFAULT_PARAMS, step_idx=1)
    y5, _ = make_intermediate_curve(source_y, target_y, x, valid, DEFAULT_PARAMS, step_idx=5)

    dist_1 = np.mean(np.abs(y1 - target_y))
    dist_5 = np.mean(np.abs(y5 - target_y))
    assert dist_5 < dist_1, "step 5 should be closer to the target than step 1"


def test_optuna_objective_penalizes_non_monotonic_error_by_construction():
    """The generator's own curve-fit loss (used only for Optuna tuning, never
    as a grading signal) should decrease across steps for a well-behaved
    blend -- this is the monotonicity guarantee described in the Part II
    paper's Methodology (sec:optuna)."""
    x = np.linspace(0, 1, 240)
    source_y = np.full_like(x, 0.2)
    target_y = np.full_like(x, 0.8)
    valid = np.ones_like(x, dtype=bool)

    losses = []
    for step in range(1, 6):
        y, v = make_intermediate_curve(source_y, target_y, x, valid, DEFAULT_PARAMS, step)
        losses.append(score_curve(y, target_y, v))

    assert losses == sorted(losses, reverse=True), f"expected monotonically decreasing loss, got {losses}"
