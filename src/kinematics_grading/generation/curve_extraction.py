"""Extract the plotted curve from a rendered kinematics graph image.

Ported 1:1 from the original diffusion_module/diffuser/generate.py (same
masking thresholds, same pruning heuristics) so generated corpora remain
comparable; only reorganized into a testable module with type hints.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

N_CURVE_POINTS_DEFAULT = 240


def gray_np(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def detect_plot_box(img: Image.Image) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) pixel bounds of the plot axes."""
    g = gray_np(img)
    edges = cv2.Canny(g, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 50:
        return 72, 55, 470, 430

    h, w = g.shape
    valid = (ys > int(0.05 * h)) & (ys < int(0.95 * h)) & (xs > int(0.03 * w)) & (xs < int(0.97 * w))
    xs, ys = xs[valid], ys[valid]
    if len(xs) < 50:
        return 72, 55, 470, 430

    left = int(np.percentile(xs, 3))
    right = int(np.percentile(xs, 97))
    top = int(np.percentile(ys, 3))
    bottom = int(np.percentile(ys, 97))

    left = max(40, min(left, 120))
    right = min(490, max(right, 390))
    top = max(35, min(top, 90))
    bottom = min(470, max(bottom, 360))
    return left, top, right, bottom


def curve_mask(img: Image.Image) -> np.ndarray:
    """Binary mask (0/255) of pixels belonging to the plotted (blue/green) curve."""
    arr = np.array(img)
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    left, top, right, bottom = detect_plot_box(img)

    blue = (b > 90) & (b > r + 25) & (b > g + 8)
    green = (g > 60) & (g > r + 12) & (g > b + 2)
    mask = (blue | green).astype(np.uint8)

    clean = np.zeros_like(mask, dtype=np.uint8)
    margin = 8
    clean[top + margin:bottom - margin, left + margin:right - margin] = mask[
        top + margin:bottom - margin, left + margin:right - margin
    ]
    clean = (clean * 255).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    filtered = np.zeros_like(clean)

    for comp_id in range(1, num):
        x, y, w, h, area = stats[comp_id]
        if area < 20:
            continue
        aspect = w / max(1, h)

        if aspect < 0.18 and h > 40:
            continue
        if x <= left + margin + 2 or x + w >= right - margin - 2:
            if h > 20:
                continue

        filtered[labels == comp_id] = 255

    return cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def largest_contiguous_run(valid: np.ndarray) -> np.ndarray:
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return valid.copy()

    breaks = np.where(np.diff(idx) > 1)[0]
    runs, start = [], 0
    for b in breaks:
        runs.append(idx[start:b + 1])
        start = b + 1
    runs.append(idx[start:])
    best = max(runs, key=len)

    out = np.zeros_like(valid, dtype=bool)
    out[best] = True
    return out


def prune_endpoints(y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Remove suspicious hook-like endpoint zones (rendering artifacts)."""
    out = valid.copy()
    idx = np.where(out)[0]
    if len(idx) < 12:
        return out

    trim = max(3, int(0.04 * len(idx)))
    out[idx[:trim]] = False
    out[idx[-trim:]] = False

    idx = np.where(out)[0]
    if len(idx) < 12:
        return out

    dy = np.gradient(y)
    local = np.abs(dy[idx])
    med = np.median(local) + 1e-6
    threshold = max(0.18, 3.0 * med)

    left_i = 0
    while left_i < min(10, len(idx) - 1) and abs(dy[idx[left_i]]) > threshold:
        out[idx[left_i]] = False
        left_i += 1

    right_i = len(idx) - 1
    steps = 0
    while right_i >= max(0, len(idx) - 10) and steps < 10 and abs(dy[idx[right_i]]) > threshold:
        out[idx[right_i]] = False
        right_i -= 1
        steps += 1

    return largest_contiguous_run(out)


def extract_curve(
    img: Image.Image,
    smooth_k: int = 5,
    n_curve_points: int = N_CURVE_POINTS_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_grid, y_grid, valid_mask) for the normalized [0,1] curve."""
    left, top, right, bottom = detect_plot_box(img)
    mask = curve_mask(img)

    xs, ys = [], []
    for x in range(left, right + 1):
        yy = np.where(mask[:, x] > 0)[0]
        yy = yy[(yy >= top) & (yy <= bottom)]
        if len(yy) > 0:
            xs.append(x)
            ys.append(float(np.median(yy)))

    grid = np.linspace(0.0, 1.0, n_curve_points)

    if len(xs) < 5:
        y_grid = 0.5 + 0.2 * grid
        valid = (grid >= 0.15) & (grid <= 0.85)
        return grid, y_grid, valid

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    if smooth_k > 1 and len(ys) > smooth_k:
        ys = np.convolve(ys, np.ones(int(smooth_k)) / int(smooth_k), mode="same")

    x_norm = (xs - left) / max(1, right - left)
    y_norm = 1.0 - (ys - top) / max(1, bottom - top)

    x_min, x_max = float(np.min(x_norm)), float(np.max(x_norm))
    valid = (grid >= x_min) & (grid <= x_max)

    y_grid = np.full_like(grid, np.nan, dtype=np.float32)
    y_grid[valid] = np.interp(grid[valid], x_norm, y_norm)
    y_grid = np.clip(np.nan_to_num(y_grid, nan=0.0), 0.02, 0.98)

    valid = largest_contiguous_run(valid)
    valid = prune_endpoints(y_grid, valid)

    return grid, y_grid, valid
