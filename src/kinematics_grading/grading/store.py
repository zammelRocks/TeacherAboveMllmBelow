"""Append-only JSONL result store with resume support.

Unifies the persistence logic that was previously copy-pasted across
grade_intermediate_steps_azure.py, reverse_grading_resume.py, and
failed_instance_resume.py. Adds an asyncio.Lock so concurrent grading tasks
(azure_client.py now runs several in parallel) don't interleave writes to
the same JSONL file.
"""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


class GradingStore:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.jsonl_path = out_dir / "intermediate_step_rule_grades.jsonl"
        self.csv_path = out_dir / "intermediate_step_rule_grades.csv"
        self.failed_path = out_dir / "intermediate_step_rule_grades_failed.jsonl"
        self._lock = asyncio.Lock()
        out_dir.mkdir(parents=True, exist_ok=True)

    def load_existing(self) -> Dict[str, Dict[str, Any]]:
        """Keyed by resolved generated_image path -> already-graded row."""
        existing: Dict[str, Dict[str, Any]] = {}
        if not self.jsonl_path.exists():
            return existing
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = row.get("generated_image")
                if img:
                    existing[str(Path(img).resolve())] = row
        return existing

    def load_failed(self) -> Dict[str, Dict[str, Any]]:
        failed: Dict[str, Dict[str, Any]] = {}
        if not self.failed_path.exists():
            return failed
        with self.failed_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                img = row.get("generated_image")
                if img:
                    failed[str(Path(img).resolve())] = row
        return failed

    async def append_success(self, row: Dict[str, Any]) -> None:
        async with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    async def append_failure(self, row: Dict[str, Any]) -> None:
        async with self._lock:
            with self.failed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

    def write_csv(self) -> Optional[Path]:
        if not self.jsonl_path.exists():
            return None

        rows = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not rows:
            return None

        fields = [
            "exercise", "round", "source_exercise", "instance", "step_file",
            "generated_image", "correct_image", "max_score", "total_awarded",
            "score_ratio", "score_100", "step_expected_progress", "feedback",
            "is_graph_only", "is_progress_plausible", "rules_json",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["rules_json"] = json.dumps(row.get("rules", []), ensure_ascii=False)
                writer.writerow(csv_row)
        return self.csv_path
