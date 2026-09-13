"""Discover gradable images and grade them concurrently, with resume support.

Replaces three separate scripts (grade_intermediate_steps_azure.py,
reverse_grading_resume.py, failed_instance_resume.py) with one pipeline that
takes a `mode` argument instead of being three copies of the same logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from ..config import AzureConfig, DATA_DIR
from ..domain import TrajectoryKey, is_gradable_step, parse_step_info
from .azure_client import AzureVisionGrader
from .prompts import build_grading_prompt_from_prefix, build_stable_prefix, includes_progress_fields, sends_reference_image
from .normalize import normalize_grade_json
from .store import GradingStore

logger = logging.getLogger(__name__)


def correct_image_path(exercise: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"exercice_{exercise}" / f"correct_{exercise}.png"


def instruction_path(exercise: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"exercice_{exercise}" / f"instruction_{exercise}.txt"


def rubric_path(exercise: int, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"exercice_{exercise}" / f"rubric_{exercise}.json"


@lru_cache(maxsize=None)
def load_rubric(exercise: int, data_dir: Path = DATA_DIR) -> dict:
    """Cached: identical for every one of an exercise's ~1,500 graded images
    within a run, so there's no reason to re-read + re-parse the file each time."""
    path = rubric_path(exercise, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing rubric file: {path}")
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if "max_score" not in rubric or "rules" not in rubric:
        raise ValueError(f"Rubric must contain max_score and rules: {path}")
    return rubric


@lru_cache(maxsize=None)
def _cached_instruction_text(exercise: int, data_dir: Path = DATA_DIR) -> str:
    return instruction_path(exercise, data_dir).read_text(encoding="utf-8", errors="ignore")


@lru_cache(maxsize=None)
def cached_stable_prompt_prefix(exercise: int, data_dir: Path = DATA_DIR, partial_credit_policy: str = "step_calibrated") -> str:
    """The part of the grading prompt shared by every step/instance/round of
    one exercise (see prompts.build_stable_prefix): built once per exercise
    (and per policy) per process, not once per image. This is the client-side
    half of the caching story; build_stable_prefix's docstring covers the
    server-side (Azure/OpenAI automatic prompt-prefix caching) half, which
    this prefix reuse is what makes possible in the first place -- if we
    rebuilt a freshly-formatted-but-byte-different string per call, the
    provider's prefix match would never hit even though the content is the
    same. `partial_credit_policy` is part of the lru_cache key so grading
    under both policies in the same process (e.g. a comparison run) doesn't
    collide."""
    rubric = load_rubric(exercise, data_dir)
    instruction_text = _cached_instruction_text(exercise, data_dir)
    return build_stable_prefix(exercise, instruction_text, rubric, partial_credit_policy)


def discover_gradable_images(generated_dir: Path, exercises: List[int]) -> List[Path]:
    images: List[Path] = []
    for ex in exercises:
        ex_dir = generated_dir / f"exercice_{ex}"
        if not ex_dir.exists():
            logger.warning("generated exercise folder does not exist: %s", ex_dir)
            continue
        images.extend(p for p in ex_dir.rglob("*.png") if is_gradable_step(p.name))
    images.sort(key=lambda p: (str(TrajectoryKey.from_path(p)), p.name))
    return images


async def grade_one(
    grader: AzureVisionGrader,
    store: GradingStore,
    img_path: Path,
    data_dir: Path,
    partial_credit_policy: str = "step_calibrated",
) -> Optional[dict]:
    key = TrajectoryKey.from_path(img_path)
    if key is None:
        logger.warning("could not parse trajectory metadata from path: %s", img_path)
        return None

    step_info = parse_step_info(img_path.name)
    if step_info is None or not step_info.is_graded:
        return None

    correct_img = correct_image_path(key.exercise, data_dir)
    use_reference_image = sends_reference_image(partial_credit_policy)
    if use_reference_image and not correct_img.exists():
        raise FileNotFoundError(f"Missing correct image: {correct_img}")

    rubric = load_rubric(key.exercise, data_dir)
    stable_prefix = cached_stable_prompt_prefix(key.exercise, data_dir, partial_credit_policy)
    prompt = build_grading_prompt_from_prefix(
        stable_prefix, img_path.name, step_info, includes_progress_fields(partial_credit_policy)
    )

    raw_grade = await grader.grade(img_path, correct_img if use_reference_image else None, prompt)
    grade = normalize_grade_json(
        raw=raw_grade,
        rubric=rubric,
        step_file=img_path.name,
        expected_progress=step_info.expected_progress or 50,
    )

    row = {
        "exercise": key.exercise,
        "round": key.round,
        "source_exercise": key.source_exercise,
        "instance": key.instance,
        "step_file": img_path.name,
        "generated_image": str(img_path.resolve()),
        "correct_image": str(correct_img.resolve()),
        **grade,
    }
    await store.append_success(row)
    return row


async def run_grading(
    images: List[Path],
    azure_cfg: AzureConfig,
    out_dir: Path,
    data_dir: Path = DATA_DIR,
    csv_every: int = 20,
    partial_credit_policy: str = "step_calibrated",
) -> Dict[str, int]:
    """Grade `images` concurrently (bounded by azure_cfg.max_concurrency),
    skipping any already present in the store (resume-by-default)."""
    store = GradingStore(out_dir)
    existing = store.load_existing()

    todo = [p for p in images if str(p.resolve()) not in existing]
    logger.info("Resuming: %d/%d images already graded, %d remaining", len(existing), len(images), len(todo))

    counters = {"succeeded": 0, "failed": 0, "skipped": len(images) - len(todo)}
    if not todo:
        store.write_csv()
        return counters

    async def _grade_with_path(grader: AzureVisionGrader, path: Path):
        # asyncio.as_completed() wraps each awaitable in a new object with a
        # different identity than what you pass in, so a dict keyed on the
        # original futures can't be used to recover which path finished --
        # closing over `path` here and returning it alongside the result
        # sidesteps that identity mismatch entirely.
        try:
            row = await grade_one(grader, store, path, data_dir, partial_credit_policy)
            return path, row, None
        except Exception as e:
            return path, None, e

    async with AzureVisionGrader(azure_cfg) as grader:
        pending = [asyncio.ensure_future(_grade_with_path(grader, p)) for p in todo]
        done_count = 0
        for coro in asyncio.as_completed(pending):
            img_path, row, error = await coro

            if error is None:
                if row is not None:
                    counters["succeeded"] += 1
            else:
                counters["failed"] += 1
                key = TrajectoryKey.from_path(img_path)
                await store.append_failure(
                    {
                        "exercise": key.exercise if key else None,
                        "round": key.round if key else None,
                        "source_exercise": key.source_exercise if key else None,
                        "instance": key.instance if key else None,
                        "generated_image": str(img_path.resolve()),
                        "error": str(error),
                    }
                )
                logger.error("[FAILED] %s: %s", img_path, error)

            done_count += 1
            if done_count % csv_every == 0:
                store.write_csv()

    store.write_csv()
    return counters


async def run_retry_failed(
    azure_cfg: AzureConfig, out_dir: Path, data_dir: Path = DATA_DIR, partial_credit_policy: str = "step_calibrated",
) -> Dict[str, int]:
    store = GradingStore(out_dir)
    failed = store.load_failed()
    images = [Path(row["generated_image"]) for row in failed.values() if row.get("generated_image")]
    logger.info("Retrying %d previously failed images", len(images))
    return await run_grading(images, azure_cfg, out_dir, data_dir, partial_credit_policy=partial_credit_policy)


def _outstanding_failures(store: GradingStore) -> List[Path]:
    """failed.jsonl is append-only: an image that failed in round 1 and
    then succeeded in round 2 is still listed there. What actually still
    needs retrying is failed-and-not-already-in-the-success-store."""
    existing = store.load_existing()
    failed = store.load_failed()
    outstanding = []
    for row in failed.values():
        img = row.get("generated_image")
        if img and str(Path(img).resolve()) not in existing:
            outstanding.append(Path(img))
    return outstanding


async def run_retry_until_clean(
    azure_cfg: AzureConfig,
    out_dir: Path,
    data_dir: Path = DATA_DIR,
    max_rounds: int = 20,
    pause_seconds: float = 20.0,
    partial_credit_policy: str = "step_calibrated",
) -> Dict[str, int]:
    """Repeatedly retry outstanding failures (e.g. a flaky/transient network
    or DNS outage that recurs every so often) instead of needing a fresh
    manual `--retry-failed` invocation every single time it happens.
    Stops when nothing is left, when a round makes no further progress
    (treated as a non-transient failure, not worth looping on forever), or
    after max_rounds."""
    store = GradingStore(out_dir)
    total_succeeded = 0
    prev_remaining = None
    remaining_count = 0

    for round_idx in range(1, max_rounds + 1):
        outstanding = _outstanding_failures(store)
        if not outstanding:
            logger.info("Retry round %d: no outstanding failures remain.", round_idx)
            remaining_count = 0
            break

        logger.info("Retry round %d/%d: %d outstanding failed images", round_idx, max_rounds, len(outstanding))
        counters = await run_grading(outstanding, azure_cfg, out_dir, data_dir, partial_credit_policy=partial_credit_policy)
        total_succeeded += counters["succeeded"]

        remaining = _outstanding_failures(store)
        remaining_count = len(remaining)
        logger.info(
            "Retry round %d done: %d succeeded this round, %d still outstanding",
            round_idx, counters["succeeded"], remaining_count,
        )

        if not remaining:
            break
        if prev_remaining is not None and remaining_count >= prev_remaining:
            logger.warning(
                "No progress in retry round %d (%d -> %d outstanding); stopping rather than looping "
                "forever. Remaining failures are likely non-transient -- check %s.",
                round_idx, prev_remaining, remaining_count, store.failed_path,
            )
            break
        prev_remaining = remaining_count

        if round_idx < max_rounds:
            await asyncio.sleep(pause_seconds)

    return {"succeeded_total": total_succeeded, "still_outstanding": remaining_count}
