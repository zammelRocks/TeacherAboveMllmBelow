"""CLI-level test for `trajectory-panels`, the combined 00->06 x
(original/IG+SmoothGrad/LIME/Grad-CAM) panel view ported from
reverse_xai_trajectory_panels.ipynb. Calls cli.main() directly, following
test_cli_train_and_xai.py's rationale: CLI-wiring bugs (missing args, import
errors) don't show up in unit tests of the underlying functions.

Reuses tiny_graded_jsonl from test_cli_train_and_xai.py -- it already
generates a tiny real corpus for exercise 2 sourced from exercises 1, 3, 4
(3 (exercise, source_exercise) pairs), which is exactly what's needed to
exercise per-pair trajectory selection here.
"""

import json

import pytest
import yaml
from PIL import Image

from kinematics_grading.cli import main as cli_main
from kinematics_grading.config import DATA_DIR, DEFAULT_CONFIG_PATH, GenerationConfig
from kinematics_grading.domain import TrajectoryKey, parse_step_info
from kinematics_grading.generation.pipeline import generate_for_target
from kinematics_grading.grading.pipeline import discover_gradable_images

from test_cli_train_and_xai import fast_surrogate_config, tiny_graded_jsonl  # noqa: F401


@pytest.fixture(scope="module")
def fast_trajectory_panels_config(tmp_path_factory):
    """fast_surrogate_config's short training schedule, plus
    trajectories_per_pair=1 to keep this test's LIME/IG/Grad-CAM workload
    (run per selected trajectory, per step) small."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["surrogate"].update({"unfreeze_after_epoch": 2, "total_epochs": 4, "batch_size": 8, "val_fraction": 0.3})
    raw["xai"]["trajectories_per_pair"] = 1
    path = tmp_path_factory.mktemp("config") / "fast_trajectory_panels.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_trajectory_panels_cli_end_to_end(tiny_graded_jsonl, fast_trajectory_panels_config, tmp_path):
    surrogate_out = tmp_path / "surrogate_out"
    assert (
        cli_main(
            [
                "train-surrogate", "--grading-results", str(tiny_graded_jsonl), "--out", str(surrogate_out),
                "--config", str(fast_trajectory_panels_config),
            ]
        )
        == 0
    )

    panels_out = tmp_path / "trajectory_panels"
    exit_code = cli_main(
        [
            "trajectory-panels", "--grading-results", str(tiny_graded_jsonl),
            "--model", str(surrogate_out / "proxy_model.pt"),
            "--out", str(panels_out),
            "--config", str(fast_trajectory_panels_config),
        ]
    )
    assert exit_code == 0

    pngs = list(panels_out.glob("ex*.png"))
    # exercise 2 sourced from 1, 3, 4 => 3 pairs x trajectories_per_pair=1,
    # capped by however many of the 4 instances per pair were accepted
    # (all of them, given the fixture's monotonically increasing scores).
    assert len(pngs) == 3

    with Image.open(pngs[0]) as img:
        width, height = img.size
    assert width > 0 and height > 0


def test_trajectory_panels_cli_resumes_skipping_existing_output(
    tiny_graded_jsonl, fast_trajectory_panels_config, tmp_path
):
    """trajectory-panels can run for hours (LIME/Occlusion/IG at full-corpus
    scale) and gets interrupted in practice -- it must not redo trajectories
    it already rendered, the same way generate/grade already skip completed
    work."""
    surrogate_out = tmp_path / "surrogate_out_resume"
    assert (
        cli_main(
            [
                "train-surrogate", "--grading-results", str(tiny_graded_jsonl), "--out", str(surrogate_out),
                "--config", str(fast_trajectory_panels_config),
            ]
        )
        == 0
    )

    panels_out = tmp_path / "trajectory_panels_resume"
    args = [
        "trajectory-panels", "--grading-results", str(tiny_graded_jsonl),
        "--model", str(surrogate_out / "proxy_model.pt"),
        "--out", str(panels_out), "--config", str(fast_trajectory_panels_config),
    ]
    assert cli_main(args) == 0
    pngs = sorted(panels_out.glob("ex*.png"))
    assert len(pngs) == 3
    mtimes_before = {p: p.stat().st_mtime_ns for p in pngs}

    # Delete one output, as if it never finished saving before an interrupt.
    pngs[0].unlink()

    assert cli_main(args) == 0
    pngs_after = sorted(panels_out.glob("ex*.png"))
    assert pngs_after == pngs  # same 3 filenames, none added or removed
    # The deleted one was regenerated -- same path, fresh (newer) mtime.
    assert pngs[0].stat().st_mtime_ns > mtimes_before[pngs[0]]
    # ...and the two untouched ones were skipped, not silently re-rendered.
    for p in pngs[1:]:
        assert p.stat().st_mtime_ns == mtimes_before[p]


@pytest.fixture(scope="module")
def tiny_mixed_acceptance_jsonl(tmp_path_factory):
    """Exactly one accepted and one never-accepted trajectory, to exercise
    --all-trajectories: the curated selection (select_representative_trajectories)
    always excludes never-accepted trajectories by design, so this is the only
    way to prove --all-trajectories actually renders one anyway."""
    tmp = tmp_path_factory.mktemp("cli_all_trajectories")
    generated_dir = tmp / "generated"
    optuna_dir = tmp / "optuna_logs"

    tiny_cfg = GenerationConfig(
        exercises=[2], source_map={2: [1]}, n_rounds=1, instances_per_round=2, optuna_trials=5,
        image_size=512, n_curve_points=240,
    )
    generate_for_target(2, tiny_cfg, generated_dir, optuna_dir)

    images = discover_gradable_images(generated_dir, [2])
    by_instance = {}
    for img in images:
        by_instance.setdefault(TrajectoryKey.from_path(img).instance, []).append(img)
    instances = sorted(by_instance)
    assert len(instances) == 2, f"expected exactly 2 instances, got {instances}"
    accepted_instance, unaccepted_instance = instances

    grading_dir = tmp / "grading_results"
    grading_dir.mkdir()
    rows = []
    for instance, imgs in by_instance.items():
        scores_by_step = (
            {"01": 0.3, "02": 0.5, "03": 0.7, "04": 0.85, "05": 0.95} if instance == accepted_instance
            else {"01": 0.1, "02": 0.2, "03": 0.3, "04": 0.4, "05": 0.5}
        )
        for img in imgs:
            key = TrajectoryKey.from_path(img)
            score = scores_by_step[parse_step_info(img.name).step_id]
            rows.append(
                {
                    "exercise": key.exercise, "round": key.round, "source_exercise": key.source_exercise,
                    "instance": key.instance, "step_file": img.name, "generated_image": str(img.resolve()),
                    "correct_image": str((DATA_DIR / "exercice_2" / "correct_2.png").resolve()),
                    "max_score": 10.0, "total_awarded": round(score * 10, 1), "score_ratio": score,
                    "score_100": int(round(score * 100)), "rules": [],
                    "feedback": "SYNTHETIC TEST DATA, not a real grade", "is_graph_only": True,
                    "is_progress_plausible": True,
                }
            )
    with (grading_dir / "intermediate_step_rule_grades.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return grading_dir


def test_trajectory_panels_all_trajectories_includes_never_accepted(
    tiny_mixed_acceptance_jsonl, fast_trajectory_panels_config, tmp_path
):
    surrogate_out = tmp_path / "surrogate_out_mixed"
    assert (
        cli_main(
            [
                "train-surrogate", "--grading-results", str(tiny_mixed_acceptance_jsonl), "--out", str(surrogate_out),
                "--config", str(fast_trajectory_panels_config),
            ]
        )
        == 0
    )

    curated_out = tmp_path / "panels_curated"
    assert (
        cli_main(
            [
                "trajectory-panels", "--grading-results", str(tiny_mixed_acceptance_jsonl),
                "--model", str(surrogate_out / "proxy_model.pt"), "--out", str(curated_out),
                "--config", str(fast_trajectory_panels_config),
            ]
        )
        == 0
    )
    assert len(list(curated_out.glob("ex*.png"))) == 1  # curated selection skips the never-accepted trajectory

    all_out = tmp_path / "panels_all"
    assert (
        cli_main(
            [
                "trajectory-panels", "--grading-results", str(tiny_mixed_acceptance_jsonl),
                "--model", str(surrogate_out / "proxy_model.pt"), "--out", str(all_out),
                "--config", str(fast_trajectory_panels_config), "--all-trajectories",
            ]
        )
        == 0
    )
    assert len(list(all_out.glob("ex*.png"))) == 2  # --all-trajectories renders both


def test_trajectory_panels_cli_missing_grading_results_fails_cleanly(tmp_path):
    exit_code = cli_main(
        [
            "trajectory-panels", "--grading-results", str(tmp_path / "does_not_exist"),
            "--model", str(tmp_path / "no_such_model.pt"),
            "--out", str(tmp_path / "panels_out"),
        ]
    )
    assert exit_code == 1


def test_trajectory_panels_cli_missing_model_fails_cleanly(tiny_graded_jsonl, tmp_path):
    exit_code = cli_main(
        [
            "trajectory-panels", "--grading-results", str(tiny_graded_jsonl),
            "--model", str(tmp_path / "no_such_model.pt"),
            "--out", str(tmp_path / "panels_out"),
        ]
    )
    assert exit_code == 1
