"""Combined per-trajectory XAI panels: one figure per generated trajectory,
showing the 5 graded correction steps (01-05) across 4 rows (original /
IG+SmoothGrad / Occlusion / Guided Grad-CAM), with the first step that
crosses the acceptance threshold highlighted.

Ported from reverse_xai_trajectory_panels.ipynb's build_panel_items /
select_representative_trajectories / plot_combined_xai_panel, split into pure,
unit-testable selection logic (this module) and a thin rendering function,
rather than one notebook cell that mixes both.

Two deliberate deviations from the notebook, both requested after reviewing
real output:
- 00 (incorrect submission) and 06 (reference answer) are dropped. Neither
  is ever graded, so neither carries rubric evidence; they added 2 of every
  7 columns' worth of XAI compute for no analytical value.
- LIME and plain Grad-CAM are replaced by Occlusion and Guided Grad-CAM (see
  xai/attributions.py) -- on this task (a thin curve on a mostly-white
  background) LIME's felzenszwalb superpixels frequently put top-weighted
  segments on background/border/axis content, and Grad-CAM's 7x7 layer4
  feature map is too coarse to localize a 1-2px-wide curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..analysis.acceptance import first_accepted_step
from ..domain import STEP_META


@dataclass(frozen=True)
class PanelItem:
    step_id: str  # "01".."05"
    label: str
    path: Path
    score_ratio: float
    score_text: str
    is_first_accepted: bool


def step_label(step_id: str) -> str:
    name, _ = STEP_META[step_id]
    return f"{step_id} {name.replace('_', ' ')}"


def build_panel_items(
    group: pd.DataFrame, score_threshold: float
) -> Tuple[List[PanelItem], Optional[int]]:
    """Reconstruct the 01->05 graded sequence for one trajectory.

    `group` must hold every graded row (steps 01-05, one per step) for
    exactly one (exercise, round, source_exercise, instance) trajectory, with
    `step_order`, `score_ratio`, and `generated_image` columns.

    Returns (items, first_accepted_step_order) -- the latter is None if no
    step in the trajectory ever reached score_threshold.
    """
    if group.empty:
        return [], None

    g = group.sort_values("step_order").reset_index(drop=True)
    first_row = first_accepted_step(g, score_threshold)
    accepted_step = int(first_row["step_order"]) if first_row is not None else None

    items: List[PanelItem] = []
    for _, row in g.iterrows():
        step_id = f"{int(row['step_order']):02d}"
        score_ratio = float(row["score_ratio"])
        score_text = f"{score_ratio:.2f}"
        total_awarded, max_score = row.get("total_awarded"), row.get("max_score")
        if pd.notna(total_awarded) and pd.notna(max_score):
            score_text += f" ({total_awarded:g}/{max_score:g})"
        items.append(
            PanelItem(
                step_id, step_label(step_id), Path(row["generated_image"]), score_ratio, score_text,
                accepted_step is not None and int(row["step_order"]) == accepted_step,
            )
        )

    return items, accepted_step


def select_representative_trajectories(traj_df: pd.DataFrame, n_per_pair: int) -> pd.DataFrame:
    """Top `n_per_pair` accepted trajectories per (exercise, source_exercise)
    pair: earliest acceptance first, and among ties, the largest embedding
    distance to the reference at that step (the more visually-surprising
    acceptances -- see analysis/classification.py's robust-to-noise idea).

    Trajectories that were never accepted are excluded: there is no
    first-accepted step to highlight or to select "the more surprising one"
    around.
    """
    accepted = traj_df[traj_df["has_acceptance"]].copy()
    if accepted.empty:
        return accepted
    accepted = accepted.sort_values(
        ["exercise", "source_exercise", "first_accepted_step", "first_accepted_dist_to_ref"],
        ascending=[True, True, True, False],
    )
    return (
        accepted.groupby(["exercise", "source_exercise"], group_keys=False)
        .head(n_per_pair)
        .reset_index(drop=True)
    )
