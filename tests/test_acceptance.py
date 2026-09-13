import pandas as pd
import pytest

from kinematics_grading.analysis.acceptance import (
    acceptance_summary_by_exercise,
    first_accepted_step,
    first_accepted_step_distribution,
    monotonicity_summary_by_exercise,
    partial_credit_on_unsatisfied_rules,
    per_step_pass_rate,
    trajectory_score_monotonicity,
    trajectory_summary,
)


def _trajectory_rows(scores, dists=None, exercise=2, round_=1, source=1, instance="instance_001"):
    dists = dists or [0.15] * len(scores)
    return [
        {
            "exercise": exercise, "round": round_, "source_exercise": source, "instance": instance,
            "step_order": i + 1, "score_ratio": s, "dist_to_ref": d,
        }
        for i, (s, d) in enumerate(zip(scores, dists))
    ]


def test_first_accepted_step_returns_first_crossing():
    df = pd.DataFrame(_trajectory_rows([0.2, 0.5, 0.65, 0.9, 1.0]))
    row = first_accepted_step(df, score_threshold=0.60)
    assert row is not None
    assert row["step_order"] == 3


def test_first_accepted_step_none_when_never_accepted():
    df = pd.DataFrame(_trajectory_rows([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert first_accepted_step(df, score_threshold=0.60) is None


def test_trajectory_summary_matches_first_accepted_step():
    rows = _trajectory_rows([0.2, 0.5, 0.65, 0.9, 1.0])
    df = pd.DataFrame(rows)
    summary = trajectory_summary(df, score_threshold=0.60)
    assert len(summary) == 1
    r = summary.iloc[0]
    assert r["has_acceptance"]
    assert r["first_accepted_step"] == 3
    assert r["first_accepted_dist_to_ref"] == pytest.approx(0.15)


def test_acceptance_summary_by_exercise_percentages():
    accepted = _trajectory_rows([0.7, 0.8, 0.9, 1.0, 1.0], instance="instance_001")
    never = _trajectory_rows([0.1, 0.2, 0.3, 0.4, 0.5], instance="instance_002")
    df = pd.DataFrame(accepted + never)

    traj_df = trajectory_summary(df, score_threshold=0.60)
    summary = acceptance_summary_by_exercise(traj_df)

    assert len(summary) == 1
    assert summary.iloc[0]["pct_accepted"] == pytest.approx(50.0)


def test_per_step_pass_rate_reflects_only_that_steps_own_score():
    rows = (
        _trajectory_rows([0.7, 0.7, 0.7, 0.7, 0.7], instance="instance_A")
        + _trajectory_rows([0.1, 0.1, 0.1, 0.7, 0.7], instance="instance_B")
        + _trajectory_rows([0.1, 0.1, 0.1, 0.1, 0.1], instance="instance_C")
    )
    df = pd.DataFrame(rows)
    rates = per_step_pass_rate(df, score_threshold=0.60)

    by_step = dict(zip(rates["step_order"], rates["pct_pass"]))
    assert by_step[1] == pytest.approx(100.0 / 3)
    assert by_step[3] == pytest.approx(100.0 / 3)
    assert by_step[4] == pytest.approx(200.0 / 3)  # A and B both pass once B catches up at step 4
    assert by_step[5] == pytest.approx(200.0 / 3)
    assert set(rates["exercise"]) == {2}


def test_first_accepted_step_distribution_sums_to_100_and_finds_the_mode():
    rows = (
        _trajectory_rows([0.7, 0.8, 0.9, 1.0, 1.0], instance="instance_A")  # first accepted at step 1
        + _trajectory_rows([0.1, 0.7, 0.8, 0.9, 1.0], instance="instance_B")  # first accepted at step 2
        + _trajectory_rows([0.1, 0.7, 0.8, 0.9, 1.0], instance="instance_C")  # first accepted at step 2
        + _trajectory_rows([0.1, 0.1, 0.1, 0.1, 0.1], instance="instance_D")  # never accepted
    )
    df = pd.DataFrame(rows)
    traj_df = trajectory_summary(df, score_threshold=0.60)
    dist = first_accepted_step_distribution(traj_df)

    assert dist["pct_of_trajectories"].sum() == pytest.approx(100.0)  # 5 step-rows + 1 "never" row per exercise

    by_step = {
        row.step: row.pct_of_trajectories for row in dist.itertuples() if pd.notna(row.step)
    }
    assert by_step[1] == pytest.approx(25.0)
    assert by_step[2] == pytest.approx(50.0)  # the mode -- "highest acceptance rate" step for this exercise
    assert by_step[3] == pytest.approx(0.0)

    never_row = dist[dist["step"].isna()].iloc[0]
    assert never_row["pct_of_trajectories"] == pytest.approx(25.0)

    last_step_row = dist[dist["step"] == 5].iloc[0]
    assert last_step_row["cumulative_pct_accepted_by_step"] == pytest.approx(75.0)


def _graded_row(exercise, score_ratio, rules):
    return {"exercise": exercise, "score_ratio": score_ratio, "rules": rules}


def test_partial_credit_on_unsatisfied_rules_matches_real_example_arithmetic():
    # The exact Exercise-2, step-1 record pulled from the real GPT-6-astra corpus:
    # forward satisfied (2.5/3), parking UNSATISFIED (1.0/4), reverse satisfied (2.5/3) -> 6.0/10 = 0.60.
    rules = [
        {"points": 3.0, "awarded": 2.5, "satisfied": True},
        {"points": 4.0, "awarded": 1.0, "satisfied": False},
        {"points": 3.0, "awarded": 2.5, "satisfied": True},
    ]
    df = pd.DataFrame([_graded_row(2, 0.60, rules)])
    out = partial_credit_on_unsatisfied_rules(df, score_threshold=0.60)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_passing"] == 1
    assert row["pct_with_unsatisfied_rule"] == pytest.approx(100.0)
    assert row["avg_pct_credit_from_unsatisfied"] == pytest.approx(100.0 * 1.0 / 6.0)


def test_partial_credit_on_unsatisfied_rules_zero_when_all_rules_satisfied():
    rules = [{"points": 4.0, "awarded": 4.0, "satisfied": True}]
    df = pd.DataFrame([_graded_row(1, 1.0, rules)])
    out = partial_credit_on_unsatisfied_rules(df, score_threshold=0.60)

    row = out.iloc[0]
    assert row["pct_with_unsatisfied_rule"] == pytest.approx(0.0)
    assert row["avg_pct_credit_from_unsatisfied"] == pytest.approx(0.0)


def test_partial_credit_on_unsatisfied_rules_excludes_non_passing_records():
    failing = _graded_row(1, 0.30, [{"points": 4.0, "awarded": 1.2, "satisfied": False}])
    passing = _graded_row(1, 0.90, [{"points": 4.0, "awarded": 3.6, "satisfied": True}])
    df = pd.DataFrame([failing, passing])
    out = partial_credit_on_unsatisfied_rules(df, score_threshold=0.60)

    assert len(out) == 1
    assert out.iloc[0]["n_passing"] == 1  # only the passing record counted


def test_partial_credit_on_unsatisfied_rules_empty_when_nothing_passes():
    df = pd.DataFrame([_graded_row(1, 0.10, [{"points": 4.0, "awarded": 0.4, "satisfied": False}])])
    out = partial_credit_on_unsatisfied_rules(df, score_threshold=0.60)
    assert out.empty


def test_partial_credit_on_unsatisfied_rules_grouped_per_exercise():
    ex1 = _graded_row(1, 1.0, [{"points": 4.0, "awarded": 4.0, "satisfied": True}])
    ex2 = _graded_row(
        2, 0.60,
        [
            {"points": 3.0, "awarded": 2.5, "satisfied": True},
            {"points": 4.0, "awarded": 1.0, "satisfied": False},
            {"points": 3.0, "awarded": 2.5, "satisfied": True},
        ],
    )
    df = pd.DataFrame([ex1, ex2])
    out = partial_credit_on_unsatisfied_rules(df, score_threshold=0.60)

    assert set(out["exercise"]) == {1, 2}
    by_ex = out.set_index("exercise")
    assert by_ex.loc[1, "avg_pct_credit_from_unsatisfied"] == pytest.approx(0.0)
    assert by_ex.loc[2, "avg_pct_credit_from_unsatisfied"] == pytest.approx(100.0 / 6.0)


def test_trajectory_score_monotonicity_flags_a_regression():
    df = pd.DataFrame(_trajectory_rows([0.70, 0.40, 0.90, 0.96, 1.00]))
    out = trajectory_score_monotonicity(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert not row["is_monotonic_nondecreasing"]
    assert row["n_regressions"] == 1
    assert row["max_drop"] == pytest.approx(0.30)


def test_trajectory_score_monotonicity_true_for_nondecreasing_scores():
    df = pd.DataFrame(_trajectory_rows([0.20, 0.50, 0.65, 0.90, 1.00]))
    out = trajectory_score_monotonicity(df)

    row = out.iloc[0]
    assert row["is_monotonic_nondecreasing"]
    assert row["n_regressions"] == 0
    assert row["max_drop"] == pytest.approx(0.0)


def test_trajectory_score_monotonicity_true_for_flat_or_tied_scores():
    df = pd.DataFrame(_trajectory_rows([0.50, 0.50, 0.50, 0.50, 0.50]))
    out = trajectory_score_monotonicity(df)

    row = out.iloc[0]
    assert row["is_monotonic_nondecreasing"]  # non-decreasing allows ties
    assert row["n_regressions"] == 0


def test_trajectory_score_monotonicity_counts_multiple_regressions():
    df = pd.DataFrame(_trajectory_rows([0.80, 0.50, 0.70, 0.30, 0.90]))
    out = trajectory_score_monotonicity(df)

    row = out.iloc[0]
    assert row["n_regressions"] == 2  # 0.80->0.50 and 0.70->0.30
    assert row["max_drop"] == pytest.approx(0.40)  # the larger of the two drops


def test_monotonicity_summary_by_exercise_aggregates_correctly():
    regressing = _trajectory_rows([0.70, 0.40, 0.90, 0.96, 1.00], instance="instance_A")
    clean = _trajectory_rows([0.20, 0.50, 0.65, 0.90, 1.00], instance="instance_B")
    df = pd.DataFrame(regressing + clean)

    mono_df = trajectory_score_monotonicity(df)
    summary = monotonicity_summary_by_exercise(mono_df)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["n_trajectories"] == 2
    assert row["pct_with_any_regression"] == pytest.approx(50.0)
    assert row["mean_n_regressions"] == pytest.approx(0.5)
    assert row["largest_drop"] == pytest.approx(0.30)
