"""Parse and normalize the grader's JSON response.

Ported from the (identical, x3) normalize_grade_json / extract_json_from_text
in the original grading scripts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def normalize_grade_json(
    raw: Dict[str, Any],
    rubric: Dict[str, Any],
    step_file: str,
    expected_progress: int,
) -> Dict[str, Any]:
    max_score = float(rubric.get("max_score", 0))
    rubric_rules = rubric.get("rules", [])

    raw_by_id = {
        str(r["id"]): r for r in raw.get("rules", []) if isinstance(r, dict) and "id" in r
    }

    normalized_rules = []
    total = 0.0

    for rr in rubric_rules:
        rid = str(rr.get("id"))
        max_points = float(rr.get("points", 0))
        model_rule = raw_by_id.get(rid, {})

        try:
            awarded = float(model_rule.get("awarded", 0))
        except Exception:
            awarded = 0.0
        awarded = max(0.0, min(max_points, awarded))
        total += awarded

        normalized_rules.append(
            {
                "id": rr.get("id"),
                "points": max_points,
                "awarded": awarded,
                "quantity": rr.get("quantity"),
                "relation": rr.get("relation"),
                "expected": rr.get("expected"),
                "observed": model_rule.get("observed", ""),
                "satisfied": bool(model_rule.get("satisfied", awarded >= max_points)),
                "partial_credit_reason": model_rule.get("partial_credit_reason", ""),
            }
        )

    total = max(0.0, min(max_score, total)) if max_score > 0 else total
    score_ratio = float(total / max_score) if max_score > 0 else 0.0
    score_100 = int(round(score_ratio * 100))

    return {
        "max_score": max_score,
        "total_awarded": total,
        "score_ratio": score_ratio,
        "score_100": score_100,
        "step_file": step_file,
        "step_expected_progress": expected_progress,
        "rules": normalized_rules,
        "feedback": raw.get("feedback", ""),
        "is_graph_only": raw.get("is_graph_only"),
        "is_progress_plausible": raw.get("is_progress_plausible"),
    }
