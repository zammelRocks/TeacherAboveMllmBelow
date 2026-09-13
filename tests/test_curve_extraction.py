import numpy as np

from kinematics_grading.generation.curve_extraction import (
    detect_plot_box,
    extract_curve,
    largest_contiguous_run,
)


def test_detect_plot_box_returns_reasonable_bounds(synthetic_increasing_image):
    left, top, right, bottom = detect_plot_box(synthetic_increasing_image)
    assert 0 <= left < right <= 512
    assert 0 <= top < bottom <= 512


def test_extract_curve_recovers_increasing_trend(synthetic_increasing_image):
    x, y, valid = extract_curve(synthetic_increasing_image)
    assert valid.any(), "extraction should find at least some of the drawn line"
    ys = y[valid]
    # a synthetic line drawn from bottom-left to top-right should decode as increasing
    assert ys[-1] > ys[0]


def test_extract_curve_recovers_flat_trend(synthetic_flat_image):
    x, y, valid = extract_curve(synthetic_flat_image)
    assert valid.any()
    ys = y[valid]
    assert np.std(ys) < 0.05, "a horizontal line should decode as nearly constant"


def test_largest_contiguous_run_picks_longest_segment():
    valid = np.array([True, True, False, True, True, True, False, True])
    run = largest_contiguous_run(valid)
    # indices 3,4,5 form the longest contiguous True run
    assert run.tolist() == [False, False, False, True, True, True, False, False]


def test_largest_contiguous_run_handles_all_false():
    valid = np.zeros(10, dtype=bool)
    run = largest_contiguous_run(valid)
    assert not run.any()
