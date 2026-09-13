"""Optuna-guided monotonicity tuning for the correction schedule.

Ported from generate.py's score_curve / make_objective / choose_instance_params.
Guarantees the intended difficulty ordering across the five steps is
monotonic (see Part II paper, sec:optuna) independent of which grader is
later used to score the corpus.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import optuna
from PIL import Image

from .correction import CORRECTION_STRATEGIES, make_intermediate_curve
from .curve_extraction import extract_curve, largest_contiguous_run


def score_curve(y: np.ndarray, target_y: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return 1e6
    diff = np.abs(y[valid] - target_y[valid])
    slope_diff = np.abs(np.gradient(y)[valid] - np.gradient(target_y)[valid])
    return float(np.mean(diff) + 0.35 * np.mean(slope_diff))


def make_objective(source_img: Image.Image, target_img: Image.Image) -> Callable[[optuna.Trial], float]:
    x, source_y, source_valid = extract_curve(source_img, smooth_k=5)
    _, target_y, target_valid = extract_curve(target_img, smooth_k=5)
    common_valid = largest_contiguous_run(source_valid & target_valid)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "strategy": trial.suggest_categorical("strategy", list(CORRECTION_STRATEGIES)),
            "window_width": trial.suggest_float("window_width", 0.08, 0.42),
            "window_center": trial.suggest_float("window_center", 0.20, 0.80),
            "easing": trial.suggest_float("easing", 0.65, 1.60),
            "global_blend": trial.suggest_float("global_blend", 0.15, 0.70),
            "scale_error": trial.suggest_float("scale_error", -0.12, 0.12),
            "vertical_shift": trial.suggest_float("vertical_shift", -0.05, 0.05),
            "early_smooth_k": trial.suggest_int("early_smooth_k", 3, 17, step=2),
            "late_smooth_k": trial.suggest_int("late_smooth_k", 1, 7, step=2),
        }

        losses, prev_loss, penalty = [], None, 0.0
        for step_idx in range(1, 6):
            y, valid = make_intermediate_curve(source_y, target_y, x, common_valid, params, step_idx)
            loss = score_curve(y, target_y, valid)
            losses.append(loss)
            if prev_loss is not None and loss > prev_loss:
                penalty += 0.1 * (loss - prev_loss)
            prev_loss = loss

        return losses[-1] + 0.25 * np.mean(losses) + penalty

    return objective


def choose_instance_params(study: optuna.Study, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    trials = sorted((t for t in study.trials if t.value is not None), key=lambda t: t.value)
    top = trials[: min(8, len(trials))]
    base = dict(top[int(rng.integers(0, len(top)))].params) if top else dict(study.best_params)

    if rng.random() < 0.55:
        base["strategy"] = rng.choice(CORRECTION_STRATEGIES)

    base["window_width"] = float(np.clip(base["window_width"] + rng.normal(0, 0.08), 0.06, 0.50))
    base["window_center"] = float(np.clip(base["window_center"] + rng.normal(0, 0.18), 0.08, 0.92))
    base["easing"] = float(np.clip(base["easing"] + rng.normal(0, 0.22), 0.45, 1.90))
    base["global_blend"] = float(np.clip(base["global_blend"] + rng.normal(0, 0.15), 0.05, 0.85))
    base["scale_error"] = float(np.clip(base["scale_error"] + rng.normal(0, 0.08), -0.22, 0.22))
    base["vertical_shift"] = float(np.clip(base["vertical_shift"] + rng.normal(0, 0.035), -0.10, 0.10))
    base["line_width"] = int(rng.integers(3, 6))
    return base
