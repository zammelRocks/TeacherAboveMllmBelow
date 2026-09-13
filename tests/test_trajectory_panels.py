"""Unit tests for the pure selection/reconstruction logic behind the
combined per-trajectory XAI panels (5 graded steps, 4 methods), kept
separate from the matplotlib rendering in cli.py so they run fast and
without torch/captum.
"""

import pandas as pd
import pytest

from kinematics_grading.xai.trajectory_panels import (
    build_panel_items,
    select_representative_trajectories,
    step_label,
)


def _graded_group(scores: dict) -> pd.DataFrame:
    rows = []
    for step_order, score in scores.items():
        rows.append(
            {
                "step_order": step_order, "score_ratio": score,
                "generated_image": f"/fake/{step_order:02d}_step.png",
                "total_awarded": round(score * 4, 1), "max_score": 4.0,
            }
        )
    return pd.DataFrame(rows)


def test_build_panel_items_covers_all_five_graded_steps():
    group = _graded_group({1: 0.25, 2: 0.75, 3: 1.0, 4: 1.0, 5: 1.0})
    items, first_accepted = build_panel_items(group, score_threshold=0.60)

    assert [it.step_id for it in items] == ["01", "02", "03", "04", "05"]
    assert first_accepted == 2
    assert step_label("01") == "01 slight correction"
    assert step_label("05") == "05 labels fixed"


def test_build_panel_items_highlights_only_the_first_accepted_step():
    group = _graded_group({1: 0.25, 2: 0.75, 3: 1.0, 4: 1.0, 5: 1.0})
    items, _ = build_panel_items(group, score_threshold=0.60)

    highlighted = [it.step_id for it in items if it.is_first_accepted]
    assert highlighted == ["02"]


def test_build_panel_items_no_step_reaches_threshold():
    group = _graded_group({1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50})
    items, first_accepted = build_panel_items(group, score_threshold=0.60)

    assert first_accepted is None
    assert not any(it.is_first_accepted for it in items)
    assert len(items) == 5


def test_build_panel_items_empty_group_returns_nothing():
    items, first_accepted = build_panel_items(pd.DataFrame(), score_threshold=0.60)
    assert items == []
    assert first_accepted is None


def test_build_panel_items_partial_group_only_includes_present_steps():
    group = _graded_group({1: 0.9})
    items, _ = build_panel_items(group, score_threshold=0.60)
    assert [it.step_id for it in items] == ["01"]


def _traj_row(exercise, source_exercise, first_step, dist, has_acceptance=True):
    return {
        "exercise": exercise, "round": 1, "source_exercise": source_exercise, "instance": f"instance_{first_step}_{dist}",
        "has_acceptance": has_acceptance,
        "first_accepted_step": first_step, "first_accepted_dist_to_ref": dist,
    }


def test_select_representative_trajectories_excludes_unaccepted():
    df = pd.DataFrame(
        [
            _traj_row(1, 2, 2, 0.3),
            _traj_row(1, 2, None, None, has_acceptance=False),
        ]
    )
    selected = select_representative_trajectories(df, n_per_pair=2)
    assert len(selected) == 1
    assert selected.iloc[0]["has_acceptance"]


def test_select_representative_trajectories_prioritizes_earliest_then_farthest():
    df = pd.DataFrame(
        [
            _traj_row(1, 2, 4, 0.5),  # later acceptance
            _traj_row(1, 2, 2, 0.1),  # earliest acceptance, close to reference
            _traj_row(1, 2, 2, 0.9),  # earliest acceptance, far from reference -- should win over the above
        ]
    )
    selected = select_representative_trajectories(df, n_per_pair=2)
    assert len(selected) == 2
    assert selected.iloc[0]["first_accepted_dist_to_ref"] == 0.9
    assert selected.iloc[1]["first_accepted_dist_to_ref"] == 0.1


def test_select_representative_trajectories_caps_per_pair():
    df = pd.DataFrame([_traj_row(1, 2, s, 0.5) for s in range(1, 6)])
    selected = select_representative_trajectories(df, n_per_pair=2)
    assert len(selected) == 2


def test_select_representative_trajectories_groups_are_independent_per_pair():
    df = pd.DataFrame(
        [
            _traj_row(1, 2, 1, 0.5), _traj_row(1, 2, 2, 0.5),
            _traj_row(1, 3, 1, 0.5), _traj_row(1, 3, 2, 0.5),
        ]
    )
    selected = select_representative_trajectories(df, n_per_pair=1)
    assert len(selected) == 2
    assert set(zip(selected["exercise"], selected["source_exercise"])) == {(1, 2), (1, 3)}
