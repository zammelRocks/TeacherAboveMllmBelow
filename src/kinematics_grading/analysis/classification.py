"""Confusing vs. robust-to-noise classification (Part II paper, sec:res-confusing).

Ported from reverse_xai_pipeline.ipynb. The original notebook documented a
SIM_THRESHOLD of 0.25 in a comment but the executed cell hardcoded 0.10 --
the two never matched. This module has exactly one threshold, sourced from
config/default.yaml (analysis.embedding_distance_threshold), so that
divergence cannot recur silently.
"""

from __future__ import annotations

import pandas as pd


def classify_confusing_robust(
    df: pd.DataFrame,
    embedding_distance_threshold: float,
    score_threshold: float,
    dist_col: str = "dist_to_ref",
    score_col: str = "score_ratio",
) -> pd.DataFrame:
    """Adds boolean `is_confusing` and `is_robust_to_noise` columns.

    confusing        := visually close to the reference, yet scored below threshold
                         (dist < embedding_distance_threshold AND score < score_threshold)
    robust_to_noise   := visually far from the reference, yet scored as passing
                         (dist >= embedding_distance_threshold AND score >= score_threshold)
    """
    out = df.copy()
    out["is_confusing"] = (out[dist_col] < embedding_distance_threshold) & (out[score_col] < score_threshold)
    out["is_robust_to_noise"] = (out[dist_col] >= embedding_distance_threshold) & (out[score_col] >= score_threshold)
    return out


def confusing_robust_summary(df: pd.DataFrame, group_col: str = "exercise") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_col):
        n = len(g)
        rows.append(
            {
                group_col: key,
                "n_total": n,
                "n_confusing": int(g["is_confusing"].sum()),
                "pct_confusing": 100.0 * g["is_confusing"].mean() if n else 0.0,
                "n_robust_to_noise": int(g["is_robust_to_noise"].sum()),
                "pct_robust_to_noise": 100.0 * g["is_robust_to_noise"].mean() if n else 0.0,
            }
        )
    return pd.DataFrame(rows)
