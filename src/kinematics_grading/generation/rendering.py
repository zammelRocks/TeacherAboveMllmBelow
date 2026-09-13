"""Render intermediate curves in the same visual style as the reference image.

Ported from generate.py's draw_curve_like_reference / infer_labels / make_panel,
with one deliberate change: infer_labels() now maps each exercise to the
quantity it actually plots, per its own instruction text and rubric quantity
(see Part II paper, sec:confound). The original mapping swapped Exercise 3
("x(t)", a position graph whose slope/curvature the rubric checks via dfdx/
d2fdx2) with "v(t)", and Exercise 4 ("v(t)", checked via the raw value f)
with "a(t)" -- a labeling bug that GPT-5.1 reliably caught and penalized in
the original corpus, but that should not be reproduced going forward.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .curve_extraction import detect_plot_box

# Exercise -> (x-axis label, y-axis label). Fixed mapping: see module docstring.
AXIS_LABELS = {
    1: ("t [sec]", "x(t) [m]"),
    2: ("t [sec]", "x(t) [m]"),
    3: ("t [sec]", "x(t) [m]"),
    4: ("t [sec]", "v(t) [m/s]"),
}

# Every correct_N.png reference image (all 4 exercises, verified directly)
# uses this exact fixed [-10, 10] scale on both axes, with tick marks at the
# same 9 positions as the gridlines already drawn below. The original
# generate.py never rendered these numeric tick values on generated steps at
# all -- only the gridlines and (at step 5) the unit labels -- so a
# generated step 05, even correctly labeled, still looked visibly less
# "finished" than the reference it's supposed to be approaching.
AXIS_TICK_VALUES = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]


def infer_labels(target_ex: int) -> Tuple[str, str]:
    return AXIS_LABELS.get(target_ex, ("t", "f(t)"))


def get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return None


def _draw_text_anchored(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font, anchor: str) -> None:
    """draw.text(..., anchor=...) requires real font metrics; Pillow's
    fallback bitmap font (used when get_font() can't find arial.ttf, e.g. on
    a non-Windows machine) doesn't support arbitrary anchors and raises. Degrade
    to an unanchored draw at the given point rather than losing the tick
    labels entirely on such a machine."""
    try:
        draw.text(xy, text, fill=(0, 0, 0), font=font, anchor=anchor)
    except (ValueError, TypeError):
        draw.text(xy, text, fill=(0, 0, 0), font=font)


def draw_curve_like_reference(
    target_img: Image.Image,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    target_ex: int,
    title: str,
    show_labels: bool,
    line_width: int,
    image_size: int = 512,
) -> Image.Image:
    w = h = image_size
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = detect_plot_box(target_img)

    for gx in np.linspace(left, right, 9):
        draw.line((gx, top, gx, bottom), fill=(225, 225, 225), width=1)
    for gy in np.linspace(top, bottom, 9):
        draw.line((left, gy, right, gy), fill=(225, 225, 225), width=1)

    draw.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)
    draw.line((left, bottom, left, top), fill=(20, 20, 20), width=2)

    px = left + x * (right - left)
    py = bottom - y * (bottom - top)

    idx = np.where(valid)[0]
    if len(idx) >= 2:
        splits = np.where(np.diff(idx) > 1)[0]
        segments, start = [], 0
        for s in splits:
            segments.append(idx[start:s + 1])
            start = s + 1
        segments.append(idx[start:])
        for seg in segments:
            if len(seg) >= 2:
                pts = list(zip(px[seg].astype(float), py[seg].astype(float)))
                draw.line(pts, fill=(31, 119, 180), width=int(line_width))

    font = get_font(18)
    small = get_font(15)
    tick_font = get_font(12)
    draw.text((w // 2 - 120, 16), title, fill=(0, 0, 0), font=font)

    if show_labels:
        xlabel, ylabel = infer_labels(target_ex)
        draw.text((w // 2 - 45, h - 38), xlabel, fill=(0, 0, 0), font=small)
        draw.text((22, h // 2 - 25), ylabel, fill=(0, 0, 0), font=small)

        # Numeric tick values, matching the fixed [-10, 10] scale used by
        # every real correct_N.png reference (verified directly; see
        # AXIS_TICK_VALUES). Only drawn alongside the unit labels (i.e. at
        # step 5), matching the existing steps-1-4-look-unlabeled design.
        for gx, value in zip(np.linspace(left, right, 9), AXIS_TICK_VALUES):
            _draw_text_anchored(draw, (gx, bottom + 6), f"{value:.1f}", tick_font, anchor="mt")
        for gy, value in zip(np.linspace(top, bottom, 9), reversed(AXIS_TICK_VALUES)):
            _draw_text_anchored(draw, (left - 6, gy), f"{value:.1f}", tick_font, anchor="rm")
    return img


def make_panel(step_images: List[Tuple[str, Image.Image]]) -> Image.Image:
    thumbs = []
    for name, img in step_images:
        thumb = img.resize((260, 260), Image.LANCZOS)
        canvas = Image.new("RGB", (260, 300), "white")
        canvas.paste(thumb, (0, 30))
        d = ImageDraw.Draw(canvas)
        d.text((8, 8), name.replace(".png", "").replace("_", " ")[:32], fill=(0, 0, 0), font=get_font(13))
        thumbs.append(canvas)

    panel = Image.new("RGB", (260 * len(thumbs), 300), "white")
    for i, t in enumerate(thumbs):
        panel.paste(t, (i * 260, 0))
    return panel
