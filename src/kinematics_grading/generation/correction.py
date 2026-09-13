"""Correction strategies: blend a wrong curve toward the correct one.

Ported 1:1 from generate.py's correction_mask / make_intermediate_curve.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .curve_extraction import largest_contiguous_run, prune_endpoints

CORRECTION_STRATEGIES = ("left_to_right", "right_to_left", "middle_out", "local_then_global")

TOTAL_STEPS = 5


def smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) <= k:
        return y.copy()
    return np.convolve(y, np.ones(int(k)) / int(k), mode="same")


def correction_mask(x: np.ndarray, strategy: str, progress: float, width: float, center: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    width = float(np.clip(width, 0.05, 0.8))

    if strategy == "left_to_right":
        boundary = progress
        return 1.0 / (1.0 + np.exp((x - boundary) / 0.035))
    if strategy == "right_to_left":
        boundary = 1.0 - progress
        return 1.0 / (1.0 + np.exp((boundary - x) / 0.035))
    if strategy == "middle_out":
        dist = np.abs(x - center)
        radius = progress * 0.65
        return 1.0 / (1.0 + np.exp((dist - radius) / 0.035))
    if strategy == "local_then_global":
        local = np.exp(-0.5 * ((x - center) / width) ** 2)
        global_part = progress * np.ones_like(x)
        return np.clip(0.75 * local + 0.35 * global_part, 0.0, 1.0)

    raise ValueError(f"Unknown correction strategy: {strategy!r}")


def make_intermediate_curve(
    source_y: np.ndarray,
    target_y: np.ndarray,
    x: np.ndarray,
    common_valid: np.ndarray,
    params: dict,
    step_idx: int,
    total_steps: int = TOTAL_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    raw_progress = step_idx / total_steps
    progress = raw_progress ** float(params["easing"])

    local_mask = correction_mask(x, params["strategy"], progress, params["window_width"], params["window_center"])

    global_blend = params["global_blend"] * progress
    alpha = np.clip(global_blend + (1.0 - global_blend) * local_mask * progress, 0.0, 1.0)
    y = (1.0 - alpha) * source_y + alpha * target_y

    valid = largest_contiguous_run(common_valid.copy())
    valid = prune_endpoints(y, valid)

    decay = 1.0 - progress
    y_mean = float(np.mean(y[valid])) if np.any(valid) else float(np.mean(y))
    y = y_mean + (y - y_mean) * (1.0 + decay * params["scale_error"])
    y = y + decay * params["vertical_shift"]

    k = int(round((1.0 - progress) * params["early_smooth_k"] + progress * params["late_smooth_k"]))
    y = np.clip(smooth(y, max(1, k)), 0.02, 0.98)
    return y, valid
