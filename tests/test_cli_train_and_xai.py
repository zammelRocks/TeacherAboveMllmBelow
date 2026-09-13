"""CLI-level tests for `train-surrogate` and `xai`.

These call cli.main() directly rather than only testing train_proxy() /
saliency_map() underneath it, because the earlier real grading run in this
project failed on a missing --env-file argument that only surfaced by
actually invoking the CLI -- the same class of bug these commands are
equally exposed to (argument wiring, import paths, file existence checks).
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from kinematics_grading.cli import main as cli_main
from kinematics_grading.config import DATA_DIR, DEFAULT_CONFIG_PATH
from kinematics_grading.domain import TrajectoryKey, parse_step_info
from kinematics_grading.generation.pipeline import generate_for_target
from kinematics_grading.grading.pipeline import discover_gradable_images


@pytest.fixture(scope="module")
def fast_surrogate_config(tmp_path_factory):
    """A copy of config/default.yaml with a drastically shortened surrogate
    training schedule, so these CLI tests don't run the real 200-epoch
    production schedule."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["surrogate"].update(
        {"unfreeze_after_epoch": 2, "total_epochs": 4, "batch_size": 8, "val_fraction": 0.3}
    )
    path = tmp_path_factory.mktemp("config") / "fast.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def tiny_graded_jsonl(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cli_train_xai")
    generated_dir = tmp / "generated"
    optuna_dir = tmp / "optuna_logs"

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    gen = raw["generation"]
    from kinematics_grading.config import GenerationConfig

    tiny_cfg = GenerationConfig(
        exercises=[2],
        source_map={int(k): list(v) for k, v in gen["source_map"].items()},
        n_rounds=1, instances_per_round=4, optuna_trials=5,
        image_size=gen["image_size"], n_curve_points=gen["n_curve_points"],
    )
    generate_for_target(2, tiny_cfg, generated_dir, optuna_dir)

    images = discover_gradable_images(generated_dir, [2])
    assert images, "tiny corpus generation produced no gradable images"

    rng = np.random.default_rng(1)
    grading_dir = tmp / "grading_results"
    grading_dir.mkdir()
    rows = []
    for img in images:
        key = TrajectoryKey.from_path(img)
        step = parse_step_info(img.name)
        base = {"01": 0.3, "02": 0.5, "03": 0.7, "04": 0.85, "05": 0.95}[step.step_id]
        score = float(np.clip(base + rng.uniform(-0.05, 0.05), 0.0, 1.0))
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


def test_train_surrogate_cli_end_to_end(tiny_graded_jsonl, fast_surrogate_config, tmp_path):
    out_dir = tmp_path / "surrogate_out"
    exit_code = cli_main(
        [
            "train-surrogate", "--grading-results", str(tiny_graded_jsonl), "--out", str(out_dir),
            "--config", str(fast_surrogate_config),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "proxy_model.pt").exists()
    assert (out_dir / "embeddings_cache.npz").exists()

    log = pd.read_csv(out_dir / "proxy_training_log.csv")
    assert len(log) == 4  # total_epochs in fast_surrogate_config
    assert {"pearson_r", "spearman_r", "mae", "rmse", "accuracy_within_0_15"} <= set(log.columns)


def test_train_surrogate_cli_missing_grading_results_fails_cleanly(tmp_path):
    exit_code = cli_main(
        ["train-surrogate", "--grading-results", str(tmp_path / "does_not_exist"), "--out", str(tmp_path / "out")]
    )
    assert exit_code == 1


def test_xai_cli_end_to_end(tiny_graded_jsonl, fast_surrogate_config, tmp_path):
    surrogate_out = tmp_path / "surrogate_out"
    assert (
        cli_main(
            [
                "train-surrogate", "--grading-results", str(tiny_graded_jsonl), "--out", str(surrogate_out),
                "--config", str(fast_surrogate_config),
            ]
        )
        == 0
    )

    xai_out = tmp_path / "xai_out"
    exit_code = cli_main(
        [
            "xai", "--grading-results", str(tiny_graded_jsonl),
            "--model", str(surrogate_out / "proxy_model.pt"),
            "--out", str(xai_out), "--n-per-exercise", "2",
            "--config", str(fast_surrogate_config),
        ]
    )
    assert exit_code == 0
    pngs = list(xai_out.glob("*.png"))
    assert len(pngs) == 2  # n_per_exercise=2, one exercise (2) present in the fixture


def test_xai_cli_missing_model_fails_cleanly(tiny_graded_jsonl, tmp_path):
    exit_code = cli_main(
        [
            "xai", "--grading-results", str(tiny_graded_jsonl),
            "--model", str(tmp_path / "no_such_model.pt"),
            "--out", str(tmp_path / "xai_out"),
        ]
    )
    assert exit_code == 1
