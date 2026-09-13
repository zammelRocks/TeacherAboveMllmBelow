"""Acceptance-point analysis: when does a trajectory first cross the passing
threshold, and how far is it from the reference at that point.

Ported from reverse_corpus_analysis_notebook.ipynb's first_high_step /
trajectory_summary logic, as pure pandas functions with no notebook/plotting
side effects, so they're directly unit-testable.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def first_accepted_step(group: pd.DataFrame, score_threshold: float, step_col: str = "step_order", score_col: str = "score_ratio") -> Optional[pd.Series]:
    """Return the row of the first step in `group` with score >= threshold, or None."""
    g = group.sort_values(step_col)
    accepted = g[g[score_col] >= score_threshold]
    return None if accepted.empty else accepted.iloc[0]


def trajectory_summary(
    df: pd.DataFrame,
    score_threshold: float,
    group_cols=("exercise", "round", "source_exercise", "instance"),
    step_col: str = "step_order",
    score_col: str = "score_ratio",
    dist_col: str = "dist_to_ref",
) -> pd.DataFrame:
    """One row per trajectory: whether/when it was accepted, and the
    embedding distance to reference at that point (or at the final step, if
    never accepted)."""
    rows = []
    for keys, group in df.groupby(list(group_cols)):
        g = group.sort_values(step_col)
        first = first_accepted_step(g, score_threshold, step_col, score_col)

        row = dict(zip(group_cols, keys))
        row["max_score_ratio"] = float(g[score_col].max())
        row["mean_score_ratio"] = float(g[score_col].mean())
        row["has_acceptance"] = first is not None
        row["n_steps"] = len(g)

        if first is not None:
            row["first_accepted_step"] = int(first[step_col])
            row["first_accepted_score"] = float(first[score_col])
            row["first_accepted_dist_to_ref"] = float(first[dist_col]) if dist_col in first else None
        else:
            row["first_accepted_step"] = None
            row["first_accepted_score"] = None
            row["first_accepted_dist_to_ref"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def acceptance_summary_by_exercise(traj_df: pd.DataFrame) -> pd.DataFrame:
    """Headline per-exercise stats matching Table 2 (tab:acceptance) in the
    Part II paper: % ever accepted, median first-accepted step, mean dist."""
    rows = []
    for ex, g in traj_df.groupby("exercise"):
        accepted = g[g["has_acceptance"]]
        rows.append(
            {
                "exercise": ex,
                "n_trajectories": len(g),
                "pct_accepted": 100.0 * len(accepted) / len(g) if len(g) else 0.0,
                "median_first_accepted_step": accepted["first_accepted_step"].median() if not accepted.empty else None,
                "mean_dist_at_acceptance": accepted["first_accepted_dist_to_ref"].mean() if not accepted.empty else None,
            }
        )
    return pd.DataFrame(rows)


def per_step_pass_rate(
    df: pd.DataFrame, score_threshold: float, step_col: str = "step_order", score_col: str = "score_ratio"
) -> pd.DataFrame:
    """For each (exercise, step), the % of *that step's* graded images which
    independently score >= threshold -- unlike first_accepted_step, this is
    not "first time crossing the line", it's "does this specific step pass
    on its own". Since correction is monotonic by construction, this rises
    with step_col within an exercise, but the shape of that rise (a slow
    climb vs. a sudden jump at one step) is itself the signal for "where does
    the grader start letting things through"."""
    out = (
        df.assign(_passed=df[score_col] >= score_threshold)
        .groupby(["exercise", step_col])["_passed"]
        .agg(n="count", pct_pass="mean")
        .reset_index()
    )
    out["pct_pass"] = out["pct_pass"] * 100.0
    return out.sort_values(["exercise", step_col]).reset_index(drop=True)


def first_accepted_step_distribution(traj_df: pd.DataFrame) -> pd.DataFrame:
    """For each exercise, what % of trajectories are first accepted at each
    step (or never) -- rows sum to 100% per exercise. The step with the
    highest `pct_of_trajectories` is "the step with the highest acceptance
    rate" for that exercise. `cumulative_pct_accepted_by_step` answers "when
    does the model start failing (accepting insufficient corrections in
    bulk)": the running total of how many trajectories have been accepted by
    step k or earlier.
    """
    rows = []
    for ex, g in traj_df.groupby("exercise"):
        n = len(g)
        cumulative = 0.0
        for step in range(1, 6):
            count = int((g["first_accepted_step"] == step).sum())
            pct = 100.0 * count / n if n else 0.0
            cumulative += pct
            rows.append(
                {
                    "exercise": ex, "step": step, "n_first_accepted_here": count,
                    "pct_of_trajectories": pct, "cumulative_pct_accepted_by_step": cumulative,
                }
            )
        never = int((~g["has_acceptance"]).sum())
        rows.append(
            {
                "exercise": ex, "step": None, "n_first_accepted_here": never,
                "pct_of_trajectories": 100.0 * never / n if n else 0.0,
                "cumulative_pct_accepted_by_step": cumulative,
            }
        )
    return pd.DataFrame(rows)


def partial_credit_on_unsatisfied_rules(
    df: pd.DataFrame, score_threshold: float, score_col: str = "score_ratio", rules_col: str = "rules"
) -> pd.DataFrame:
    """For every *passing* graded image (score >= threshold), how much of its
    awarded score came from rules the grader itself marked `satisfied: false`.

    This is the mechanism behind early/premature acceptance, distinct from
    first_accepted_step_distribution (which only says *when* a trajectory
    first passes, not *why* a given pass was possible): rubric criteria are
    additive, and the grader routinely awards 25-90% partial credit for
    "plausible direction, not yet met" even on a rule it flags as
    unsatisfied. Once enough of those partial credits stack up across a
    rubric's rules, the weighted sum crosses the threshold despite the
    grader's own per-rule judgment saying the correction isn't done.

    Pass a pre-filtered `df` (e.g. steps 1-2 only) to restrict the analysis
    to a specific subset -- this function itself makes no assumption about
    which steps are included.
    """
    passing = df[df[score_col] >= score_threshold].copy()
    if passing.empty:
        return pd.DataFrame(
            columns=["exercise", "n_passing", "pct_with_unsatisfied_rule", "avg_pct_credit_from_unsatisfied"]
        )

    def _frac_credit_from_unsatisfied(rules) -> float:
        total = sum(r["awarded"] for r in rules)
        unsatisfied = sum(r["awarded"] for r in rules if not r["satisfied"])
        return unsatisfied / total if total > 0 else 0.0

    passing["_frac_unsatisfied"] = passing[rules_col].apply(_frac_credit_from_unsatisfied)
    passing["_any_unsatisfied"] = passing[rules_col].apply(lambda rs: any(not r["satisfied"] for r in rs))

    rows = []
    for ex, g in passing.groupby("exercise"):
        rows.append(
            {
                "exercise": ex,
                "n_passing": len(g),
                "pct_with_unsatisfied_rule": 100.0 * g["_any_unsatisfied"].mean(),
                "avg_pct_credit_from_unsatisfied": 100.0 * g["_frac_unsatisfied"].mean(),
            }
        )
    return pd.DataFrame(rows)


def trajectory_score_monotonicity(
    df: pd.DataFrame,
    group_cols=("exercise", "round", "source_exercise", "instance"),
    step_col: str = "step_order",
    score_col: str = "score_ratio",
) -> pd.DataFrame:
    """Per trajectory: does score_ratio rise monotonically across its graded
    steps, or does it dip anywhere?

    Motivated by the "standalone_submission" grading policy (see
    grading/prompts.py): grading each step as an independent submission, with
    no "it's early, go easy" framing, should surface genuine step-to-step
    quality variance in the *generation method* itself -- under
    "step_calibrated", explicit leniency instructions largely manufacture a
    monotonic climb regardless of whether the underlying images actually
    improve monotonically. Running this on both policies' graded corpora and
    comparing is the point of the second experiment.

    Returns one row per trajectory: n_steps, is_monotonic_nondecreasing,
    n_regressions (how many steps scored lower than the previous step), and
    max_drop (the single largest such decrease, 0.0 if none).
    """
    rows = []
    for keys, group in df.groupby(list(group_cols)):
        g = group.sort_values(step_col)
        scores = g[score_col].to_numpy(dtype=float)
        diffs = scores[1:] - scores[:-1]

        row = dict(zip(group_cols, keys))
        row["n_steps"] = len(scores)
        row["is_monotonic_nondecreasing"] = bool((diffs >= 0).all()) if len(diffs) else True
        row["n_regressions"] = int((diffs < 0).sum())
        row["max_drop"] = float(-diffs.min()) if len(diffs) and diffs.min() < 0 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def monotonicity_summary_by_exercise(mono_df: pd.DataFrame) -> pd.DataFrame:
    """Per-exercise rollup of trajectory_score_monotonicity: the headline
    numbers for "how much step-to-step score variance does this grading
    policy reveal"."""
    rows = []
    for ex, g in mono_df.groupby("exercise"):
        rows.append(
            {
                "exercise": ex,
                "n_trajectories": len(g),
                "pct_with_any_regression": 100.0 * (~g["is_monotonic_nondecreasing"]).mean(),
                "mean_n_regressions": float(g["n_regressions"].mean()),
                "mean_max_drop": float(g["max_drop"].mean()),
                "largest_drop": float(g["max_drop"].max()),
            }
        )
    return pd.DataFrame(rows)
