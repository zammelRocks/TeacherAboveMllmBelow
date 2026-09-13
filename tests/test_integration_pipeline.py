"""Hermetic integration test for the code paths that, until this test
existed, had never actually been executed by anything: the `analyze` CLI
subcommand, surrogate training, and XAI attribution. Both live smoke-test
attempts (scripts/run_smoke_test.py) returned zero successful gradings
(external Azure deployment issue) and bailed out before reaching any of
these steps.

This test generates a tiny real corpus (no Azure call needed) and grades it
with fabricated scores, since the point is to prove the downstream
analysis/surrogate/XAI code runs correctly end-to-end, not to produce a
real result -- the scores are clearly synthetic and never treated as such.
Slower than the rest of the suite (real Optuna tuning + a few epochs of
real backprop through ResNet-50); kept as one file so it's easy to skip
with `-k "not integration"` if that matters later.
"""

import numpy as np
import pandas as pd
import pytest

from kinematics_grading.config import DATA_DIR, load_settings
from kinematics_grading.domain import TrajectoryKey
from kinematics_grading.generation.pipeline import generate_for_target
from kinematics_grading.grading.pipeline import discover_gradable_images
from kinematics_grading.analysis.acceptance import acceptance_summary_by_exercise, trajectory_summary
from kinematics_grading.analysis.classification import classify_confusing_robust, confusing_robust_summary
from kinematics_grading.analysis.embeddings import EmbeddingCache
from kinematics_grading.domain import parse_step_info
from kinematics_grading.surrogate.train import train_proxy
from kinematics_grading.xai.attributions import gradcam_map, saliency_map


@pytest.fixture(scope="module")
def tiny_graded_corpus(tmp_path_factory):
    settings = load_settings()
    tmp = tmp_path_factory.mktemp("integration_corpus")
    generated_dir = tmp / "generated"
    optuna_dir = tmp / "optuna_logs"

    tiny_cfg = settings.generation.__class__(
        exercises=[2],
        source_map=settings.generation.source_map,
        n_rounds=1,
        instances_per_round=4,
        optuna_trials=5,
        image_size=settings.generation.image_size,
        n_curve_points=settings.generation.n_curve_points,
    )
    generate_for_target(2, tiny_cfg, generated_dir, optuna_dir)

    images = discover_gradable_images(generated_dir, [2])
    assert images, "tiny corpus generation produced no gradable images"

    rng = np.random.default_rng(0)
    rows = []
    for img in images:
        # Each source directory restarts instance numbering at instance_001,
        # so source_exercise must come from the actual path (TrajectoryKey),
        # not be assumed constant -- otherwise instances from different
        # sources collide into the same (exercise, round, source, instance)
        # trajectory-group key.
        key = TrajectoryKey.from_path(img)
        step = parse_step_info(img.name)
        base = {"01": 0.3, "02": 0.5, "03": 0.7, "04": 0.85, "05": 0.95}[step.step_id]
        score = float(np.clip(base + rng.uniform(-0.05, 0.05), 0.0, 1.0))
        rows.append(
            {
                "exercise": key.exercise, "round": key.round, "source_exercise": key.source_exercise,
                "instance": key.instance,
                "step_file": img.name, "generated_image": str(img.resolve()),
                "correct_image": str((DATA_DIR / "exercice_2" / "correct_2.png").resolve()),
                "max_score": 10.0, "total_awarded": round(score * 10, 1), "score_ratio": score,
                "score_100": int(round(score * 100)), "rules": [],
                "feedback": "SYNTHETIC TEST DATA, not a real grade", "is_graph_only": True,
                "is_progress_plausible": True,
            }
        )
    df = pd.DataFrame(rows)
    df["step_order"] = df["step_file"].apply(lambda s: parse_step_info(s).order)
    return df, settings


def test_analysis_pipeline_runs_end_to_end(tiny_graded_corpus, tmp_path):
    df, settings = tiny_graded_corpus
    cache = EmbeddingCache(tmp_path / "embeddings_cache.npz")
    ref_emb = cache.get(DATA_DIR / "exercice_2" / "correct_2.png")

    df = df.copy()
    df["dist_to_ref"] = [
        EmbeddingCache.cosine_distance(cache.get(p), ref_emb) for p in df["generated_image"]
    ]

    traj_df = trajectory_summary(df, settings.analysis.score_threshold)
    assert len(traj_df) == 4  # 4 instances generated

    acc_summary = acceptance_summary_by_exercise(traj_df)
    assert set(acc_summary.columns) >= {"exercise", "n_trajectories", "pct_accepted"}

    classified = classify_confusing_robust(
        df, settings.analysis.embedding_distance_threshold, settings.analysis.score_threshold
    )
    cr_summary = confusing_robust_summary(classified)
    assert set(cr_summary.columns) >= {"n_confusing", "n_robust_to_noise"}


def test_surrogate_and_xai_run_end_to_end(tiny_graded_corpus, tmp_path):
    df, settings = tiny_graded_corpus
    cache = EmbeddingCache(tmp_path / "embeddings_cache2.npz")
    embeddings = np.stack([cache.get(p) for p in df["generated_image"]])
    scores = df["score_ratio"].to_numpy()

    tiny_surrogate_cfg = settings.surrogate.__class__(
        unfreeze_layer="layer4", unfreeze_after_epoch=2, total_epochs=4,
        batch_size=8, val_fraction=0.3, huber_delta=0.10,
    )
    model, history = train_proxy(embeddings, scores, tiny_surrogate_cfg)
    assert len(history) == 4
    assert all(np.isfinite(h.mae) for h in history)

    sample_img = df.iloc[0]["generated_image"]
    _, sal = saliency_map(model, sample_img, settings.xai)
    _, cam = gradcam_map(model, sample_img, settings.xai)

    for arr in (sal, cam):
        assert arr.shape == (224, 224)
        assert arr.min() >= 0.0 and arr.max() <= 1.0
