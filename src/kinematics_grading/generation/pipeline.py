"""Orchestrates the full generation pipeline: Optuna tuning -> per-instance
trajectory rendering -> disk output. Ported from generate.py's
generate_round_for_source / generate_for_target / build_step_images, made
resumable (skips instances already fully written) and config-driven instead
of relying on module-level globals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import optuna
from PIL import Image

from ..config import GenerationConfig, DATA_DIR
from .correction import make_intermediate_curve
from .curve_extraction import extract_curve, largest_contiguous_run
from .optuna_tuning import choose_instance_params, make_objective
from .rendering import draw_curve_like_reference, make_panel

logger = logging.getLogger(__name__)

STEP_NAMES = [
    "01_slight_correction.png",
    "02_more_curve_adjusted.png",
    "03_main_trend_corrected.png",
    "04_closer_to_target.png",
    "05_labels_fixed.png",
]
STEP_TITLES = [
    "Step 1 - slight correction",
    "Step 2 - more adjusted",
    "Step 3 - trend corrected",
    "Step 4 - closer to target",
    "Step 5 - labels fixed",
]


def image_path(ex_id: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"exercice_{ex_id}" / f"correct_{ex_id}.png"


def load_rgb(path: Path, image_size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_step_images(
    source_img: Image.Image,
    target_img: Image.Image,
    target_ex: int,
    params: dict,
) -> List[Tuple[str, Image.Image]]:
    x, source_y, source_valid = extract_curve(source_img, smooth_k=5)
    _, target_y, target_valid = extract_curve(target_img, smooth_k=5)
    common_valid = largest_contiguous_run(source_valid & target_valid)

    steps = [("00_incorrect_submission.png", source_img)]
    for step_idx, (fname, title) in enumerate(zip(STEP_NAMES, STEP_TITLES), start=1):
        y, valid = make_intermediate_curve(source_y, target_y, x, common_valid, params, step_idx)
        img = draw_curve_like_reference(target_img, x, y, valid, target_ex, title, step_idx >= 5, params["line_width"])
        steps.append((fname, img))

    steps.append(("06_reference_answer.png", target_img))
    return steps


def _instance_complete(instance_dir: Path) -> bool:
    return (instance_dir / "06_reference_answer.png").exists()


def generate_round_for_source(
    target_ex: int,
    round_idx: int,
    src_ex: int,
    n_instances: int,
    cfg: GenerationConfig,
    out_dir: Path,
    optuna_dir: Path,
    data_dir: Path = DATA_DIR,
) -> None:
    target_img = load_rgb(image_path(target_ex, data_dir), cfg.image_size)
    source_img = load_rgb(image_path(src_ex, data_dir), cfg.image_size)

    study_name = f"pruned_ex{target_ex}_r{round_idx}_src{src_ex}"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(optuna_dir / (study_name + '.db')).as_posix()}"

    logger.info("Optuna tuning: %s", study_name)
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(make_objective(source_img, target_img), n_trials=cfg.optuna_trials)

    base_out = out_dir / f"exercice_{target_ex}" / f"round_{round_idx}" / f"from_correct_{src_ex}"
    base_out.mkdir(parents=True, exist_ok=True)

    for idx in range(1, n_instances + 1):
        instance_dir = base_out / f"instance_{idx:03d}"
        if _instance_complete(instance_dir):
            continue

        seed = target_ex * 100000 + round_idx * 10000 + src_ex * 1000 + idx
        params = choose_instance_params(study, seed)
        steps = build_step_images(source_img, target_img, target_ex, params)

        for fname, img in steps:
            save(img, instance_dir / fname)
        save(make_panel(steps), instance_dir / "panel.png")

        if idx % 10 == 0 or idx == n_instances:
            logger.info("  saved %03d/%03d", idx, n_instances)


def generate_for_target(
    target_ex: int,
    cfg: GenerationConfig,
    out_dir: Path,
    optuna_dir: Path,
    data_dir: Path = DATA_DIR,
) -> None:
    if not image_path(target_ex, data_dir).exists():
        logger.warning("Missing target image for exercise %s", target_ex)
        return

    source_ids = [s for s in cfg.source_map[target_ex] if image_path(s, data_dir).exists()]
    if not source_ids:
        logger.warning("No valid sources for exercise %s", target_ex)
        return

    for round_idx in range(1, cfg.n_rounds + 1):
        base = cfg.instances_per_round // len(source_ids)
        rem = cfg.instances_per_round % len(source_ids)
        allocation = {src: base + (1 if i < rem else 0) for i, src in enumerate(source_ids)}

        for src_ex in source_ids:
            generate_round_for_source(
                target_ex, round_idx, src_ex, allocation[src_ex], cfg, out_dir, optuna_dir, data_dir
            )
