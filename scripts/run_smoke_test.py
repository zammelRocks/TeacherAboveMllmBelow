"""End-to-end smoke test: generation -> real Azure grading -> analysis ->
surrogate -> XAI, all through the new package, at a small scale.

This is NOT a fidelity benchmark (far too few samples for that) -- it exists
to prove the pipeline actually works end-to-end against your configured
grading endpoint, on a handful of real images, before committing to a full
run. Run with:

    python scripts/run_smoke_test.py                    # uses .env in the project root
    python scripts/run_smoke_test.py --env-file path/to/.env
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from kinematics_grading.config import DATA_DIR, load_azure_config, load_settings
from kinematics_grading.generation.pipeline import generate_for_target
from kinematics_grading.grading.pipeline import discover_gradable_images, run_grading
from kinematics_grading.domain import parse_step_info
from kinematics_grading.analysis.embeddings import EmbeddingCache
from kinematics_grading.analysis.acceptance import trajectory_summary, acceptance_summary_by_exercise
from kinematics_grading.analysis.classification import classify_confusing_robust, confusing_robust_summary
from kinematics_grading.surrogate.train import train_proxy
from kinematics_grading.xai.attributions import saliency_map, gradcam_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smoke_test")

SMOKE_DIR = Path(__file__).resolve().parents[1] / "smoke_test_output"
GENERATED_DIR = SMOKE_DIR / "generated"
OPTUNA_DIR = SMOKE_DIR / "optuna_logs"
GRADING_DIR = SMOKE_DIR / "grading_results"
ANALYSIS_DIR = SMOKE_DIR / "analysis"

DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

TARGET_EXERCISE = 2
TINY_INSTANCES_PER_ROUND = 6  # small, real cost; not a fidelity run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args = parser.parse_args()
    env_file = args.env_file

    settings = load_settings()
    tiny_gen_cfg = replace(
        settings.generation,
        n_rounds=1,
        instances_per_round=TINY_INSTANCES_PER_ROUND,
        optuna_trials=8,  # smaller search budget, still exercises the real Optuna path
    )

    log.info("STEP 1/5: generation (exercise %s, %d total instances)", TARGET_EXERCISE, TINY_INSTANCES_PER_ROUND)
    generate_for_target(TARGET_EXERCISE, tiny_gen_cfg, GENERATED_DIR, OPTUNA_DIR)

    images = discover_gradable_images(GENERATED_DIR, [TARGET_EXERCISE])
    log.info("Generated %d gradable images", len(images))
    assert images, "generation produced no gradable images"

    log.info("STEP 2/5: grading via the real Azure endpoint (%s)", env_file)
    azure_cfg = load_azure_config(env_file)
    counters = asyncio.run(run_grading(images, azure_cfg, GRADING_DIR, DATA_DIR))
    log.info("Grading result: %s", counters)

    jsonl_path = GRADING_DIR / "intermediate_step_rule_grades.jsonl"
    if not jsonl_path.exists() or counters["succeeded"] == 0:
        log.error(
            "No successful gradings (%s) -- stopping before analysis/surrogate/XAI steps. "
            "Check the Azure deployment/credentials in %s and re-run.",
            counters, env_file,
        )
        return
    df = pd.read_json(jsonl_path, lines=True)
    df["step_order"] = df["step_file"].apply(lambda s: parse_step_info(s).order)

    step5 = df[df["step_file"].str.startswith("05")]
    log.info("Step-5 rows graded: %d (checking corrected axis label took effect)", len(step5))
    if not step5.empty:
        log.info("Step-5 mean score_ratio (post-fix): %.3f", step5["score_ratio"].mean())
        for _, row in step5.iterrows():
            log.info("  step-5 feedback: %s", str(row.get("feedback", ""))[:160])

    log.info("STEP 3/5: analysis (acceptance point + confusing/robust-to-noise)")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(ANALYSIS_DIR / "embeddings_cache.npz")
    ref_path = DATA_DIR / f"exercice_{TARGET_EXERCISE}" / f"correct_{TARGET_EXERCISE}.png"
    ref_emb = cache.get(ref_path)
    df["dist_to_ref"] = [
        EmbeddingCache.cosine_distance(cache.get(Path(p)), ref_emb) for p in df["generated_image"]
    ]
    cache.save()

    traj_df = trajectory_summary(df, settings.analysis.score_threshold)
    acc_summary = acceptance_summary_by_exercise(traj_df)
    log.info("Acceptance summary:\n%s", acc_summary.to_string(index=False))

    classified = classify_confusing_robust(df, settings.analysis.embedding_distance_threshold, settings.analysis.score_threshold)
    cr_summary = confusing_robust_summary(classified)
    log.info("Confusing/robust-to-noise summary:\n%s", cr_summary.to_string(index=False))

    log.info("STEP 4/5: surrogate training smoke test (functional check, NOT a fidelity result at this sample size)")
    embeddings = np.stack([cache.get(Path(p)) for p in df["generated_image"]])
    scores = df["score_ratio"].to_numpy()
    tiny_surrogate_cfg = replace(settings.surrogate, total_epochs=5, unfreeze_after_epoch=3, batch_size=8, val_fraction=0.3)
    try:
        model, history = train_proxy(embeddings, scores, tiny_surrogate_cfg)
        log.info("Surrogate trained %d epochs without error; final val MAE=%.3f", len(history), history[-1].mae)
    except Exception as e:
        log.error("Surrogate training smoke test FAILED: %s", e)
        raise

    log.info("STEP 5/5: XAI attribution smoke test (saliency + Grad-CAM on one image)")
    sample_img = Path(df.iloc[0]["generated_image"])
    try:
        _, sal = saliency_map(model, sample_img, settings.xai)
        _, cam = gradcam_map(model, sample_img, settings.xai)
        log.info("Saliency map shape=%s range=[%.3f, %.3f]", sal.shape, sal.min(), sal.max())
        log.info("Grad-CAM map shape=%s range=[%.3f, %.3f]", cam.shape, cam.min(), cam.max())
    except Exception as e:
        log.error("XAI smoke test FAILED: %s", e)
        raise

    log.info("SMOKE TEST COMPLETE. Outputs under: %s", SMOKE_DIR)


if __name__ == "__main__":
    main()
