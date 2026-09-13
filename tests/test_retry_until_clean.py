"""Tests for the retry-until-clean logic added in response to a real,
recurring transient network/DNS outage during a live grading run.

Two things worth pinning down with tests rather than trusting by eye:
1. failed.jsonl is append-only, so an image that failed once and later
   succeeded must not be double-counted as "still outstanding".
2. The retry loop must actually stop -- both when everything clears, and
   when a persistent (non-transient) failure would otherwise retry forever.
"""

import asyncio
import json

import httpx
import pytest

from kinematics_grading.config import AzureConfig
from kinematics_grading.grading.pipeline import _outstanding_failures, run_retry_until_clean
from kinematics_grading.grading.store import GradingStore


def make_cfg(**overrides) -> AzureConfig:
    defaults = dict(
        api_key="k", endpoint="https://example.openai.azure.com/", deployment="dep",
        max_retries=1, request_timeout=5, connect_timeout=5, max_concurrency=4,
    )
    defaults.update(overrides)
    return AzureConfig(**defaults)


def test_outstanding_failures_excludes_already_succeeded(tmp_path):
    store = GradingStore(tmp_path)
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    asyncio.run(store.append_failure({"generated_image": str(img.resolve()), "error": "boom"}))
    assert len(_outstanding_failures(store)) == 1

    # Same image later succeeds (e.g. a subsequent retry round) -- the
    # append-only failed.jsonl still has the old record, but it should no
    # longer count as outstanding.
    asyncio.run(store.append_success({"generated_image": str(img.resolve()), "score_ratio": 0.5}))
    assert len(_outstanding_failures(store)) == 0


@pytest.mark.asyncio
async def test_retry_until_clean_stops_once_everything_succeeds(tmp_path, monkeypatch):
    store = GradingStore(tmp_path)
    imgs = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        imgs.append(p)
        await store.append_failure({"generated_image": str(p.resolve()), "error": "prior network blip"})

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        body = {"choices": [{"message": {"content": json.dumps({"feedback": "ok", "rules": []})}}]}
        return httpx.Response(200, json=body)

    from kinematics_grading.grading import pipeline as pipeline_mod

    async def fake_grade_one(grader, store_, img_path, data_dir, partial_credit_policy="step_calibrated"):
        # bypass rubric/instruction lookups; only exercise the retry-loop bookkeeping
        await store_.append_success({"generated_image": str(img_path.resolve()), "score_ratio": 0.9})
        return {"generated_image": str(img_path.resolve()), "score_ratio": 0.9}

    monkeypatch.setattr(pipeline_mod, "grade_one", fake_grade_one)

    result = await run_retry_until_clean(make_cfg(), tmp_path, max_rounds=5, pause_seconds=0)

    assert result["still_outstanding"] == 0
    assert result["succeeded_total"] == 3
    assert _outstanding_failures(store) == []


@pytest.mark.asyncio
async def test_retry_until_clean_stops_on_no_progress(tmp_path, monkeypatch):
    store = GradingStore(tmp_path)
    p = tmp_path / "always_fails.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    await store.append_failure({"generated_image": str(p.resolve()), "error": "persistent error"})

    from kinematics_grading.grading import pipeline as pipeline_mod

    round_calls = {"n": 0}

    async def always_failing_grade_one(grader, store_, img_path, data_dir, partial_credit_policy="step_calibrated"):
        round_calls["n"] += 1
        raise RuntimeError("still broken")

    monkeypatch.setattr(pipeline_mod, "grade_one", always_failing_grade_one)
    result = await run_retry_until_clean(make_cfg(), tmp_path, max_rounds=20, pause_seconds=0)

    # Should give up after detecting no progress (round 2 has the same
    # outstanding count as round 1), not loop all the way to max_rounds=20.
    assert result["still_outstanding"] == 1
    assert round_calls["n"] <= 3
