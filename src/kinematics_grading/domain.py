"""Shared value types used across generation, grading, and analysis.

Centralizing these avoids the original codebase's pattern of every script
re-deriving path metadata (exercise/round/source/instance/step) with its own
slightly different regex and dict shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

STEP_FILE_PATTERN = re.compile(r"^(0[0-6])_.*\.png$", re.IGNORECASE)

STEP_META = {
    "00": ("incorrect_submission", None),
    "01": ("slight_correction", 20),
    "02": ("more_curve_adjustment", 40),
    "03": ("main_trend_corrected", 60),
    "04": ("closer_to_target", 80),
    "05": ("labels_fixed", 95),
    "06": ("reference_answer", None),
}


@dataclass(frozen=True)
class TrajectoryKey:
    """Identifies one generated correction trajectory (00 -> 06)."""

    exercise: int
    round: int
    source_exercise: int
    instance: str

    @staticmethod
    def from_path(path: Path) -> Optional["TrajectoryKey"]:
        exercise = round_ = source_exercise = instance = None
        for part in path.parts:
            if part.startswith("exercice_"):
                exercise = int(part.rsplit("_", 1)[-1])
            elif part.startswith("round_"):
                round_ = int(part.rsplit("_", 1)[-1])
            elif part.startswith("from_correct_"):
                source_exercise = int(part.rsplit("_", 1)[-1])
            elif part.startswith("instance_"):
                instance = part

        if None in (exercise, round_, source_exercise, instance):
            return None
        return TrajectoryKey(exercise, round_, source_exercise, instance)

    def instance_dir(self, base_dir: Path) -> Path:
        return (
            base_dir
            / f"exercice_{self.exercise}"
            / f"round_{self.round}"
            / f"from_correct_{self.source_exercise}"
            / self.instance
        )


@dataclass(frozen=True)
class StepInfo:
    step_id: str  # "00".."06"
    step_name: str
    expected_progress: Optional[int]  # None for 00/06 (never graded)

    @property
    def order(self) -> int:
        return int(self.step_id)

    @property
    def is_graded(self) -> bool:
        return self.step_id not in ("00", "06")


def parse_step_info(filename: str) -> Optional[StepInfo]:
    match = STEP_FILE_PATTERN.match(filename)
    if not match:
        return None
    step_id = match.group(1)
    name, progress = STEP_META[step_id]
    return StepInfo(step_id=step_id, step_name=name, expected_progress=progress)


def is_gradable_step(filename: str) -> bool:
    info = parse_step_info(filename)
    return info is not None and info.is_graded
