import pandas as pd
import pytest

from kinematics_grading.analysis.classification import classify_confusing_robust, confusing_robust_summary


def test_confusing_case_close_but_failing():
    df = pd.DataFrame([{"exercise": 3, "dist_to_ref": 0.05, "score_ratio": 0.40}])
    out = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    assert out.iloc[0]["is_confusing"]
    assert not out.iloc[0]["is_robust_to_noise"]


def test_robust_to_noise_case_far_but_passing():
    df = pd.DataFrame([{"exercise": 1, "dist_to_ref": 0.20, "score_ratio": 0.75}])
    out = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    assert out.iloc[0]["is_robust_to_noise"]
    assert not out.iloc[0]["is_confusing"]


def test_neither_flag_for_close_and_passing():
    df = pd.DataFrame([{"exercise": 1, "dist_to_ref": 0.05, "score_ratio": 0.90}])
    out = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    assert not out.iloc[0]["is_confusing"]
    assert not out.iloc[0]["is_robust_to_noise"]


def test_neither_flag_for_far_and_failing():
    df = pd.DataFrame([{"exercise": 1, "dist_to_ref": 0.20, "score_ratio": 0.30}])
    out = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    assert not out.iloc[0]["is_confusing"]
    assert not out.iloc[0]["is_robust_to_noise"]


def test_threshold_boundaries_are_exclusive_inclusive_as_documented():
    """dist exactly at threshold counts as 'far' (>=), score exactly at
    threshold counts as 'passing' (>=) -- matches sec:res-confusing."""
    df = pd.DataFrame([{"exercise": 1, "dist_to_ref": 0.10, "score_ratio": 0.60}])
    out = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    assert out.iloc[0]["is_robust_to_noise"]


def test_summary_percentages_per_exercise():
    df = pd.DataFrame(
        [
            {"exercise": 1, "dist_to_ref": 0.20, "score_ratio": 0.75},  # robust
            {"exercise": 1, "dist_to_ref": 0.20, "score_ratio": 0.30},  # neither
            {"exercise": 3, "dist_to_ref": 0.05, "score_ratio": 0.40},  # confusing
        ]
    )
    classified = classify_confusing_robust(df, embedding_distance_threshold=0.10, score_threshold=0.60)
    summary = confusing_robust_summary(classified)

    ex1 = summary[summary["exercise"] == 1].iloc[0]
    ex3 = summary[summary["exercise"] == 3].iloc[0]
    assert ex1["pct_robust_to_noise"] == pytest.approx(50.0)
    assert ex3["pct_confusing"] == pytest.approx(100.0)
