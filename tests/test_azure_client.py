"""Offline tests for AzureVisionGrader's retry policy, using httpx.MockTransport
so no network calls (and no API cost) are involved. Written directly in
response to a live run hitting a real `404 DeploymentNotFound` and retrying
it 5x with exponential backoff before failing -- these tests pin down the
fix (non-retryable 4xx fails fast) so it can't silently regress.
"""

import json

import httpx
import pytest

from kinematics_grading.config import AzureConfig
from kinematics_grading.grading.azure_client import AzureVisionGrader, NonRetryableGradingError


def make_cfg(**overrides) -> AzureConfig:
    defaults = dict(
        api_key="k", endpoint="https://example.openai.azure.com/", deployment="dep",
        max_retries=5, request_timeout=5, connect_timeout=5, max_concurrency=4,
    )
    defaults.update(overrides)
    return AzureConfig(**defaults)


def _ok_response() -> httpx.Response:
    body = {"choices": [{"message": {"content": json.dumps({"feedback": "ok"})}}]}
    return httpx.Response(200, json=body)


@pytest.mark.asyncio
async def test_non_retryable_4xx_fails_after_exactly_one_attempt(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(404, json={"error": {"code": "DeploymentNotFound", "message": "nope"}})

    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    grader = AzureVisionGrader(make_cfg())
    grader._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    ref = tmp_path / "b.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(NonRetryableGradingError):
        await grader.grade(img, ref, "prompt")

    assert call_count["n"] == 1, "a permanent config error should not be retried"
    await grader.aclose()


@pytest.mark.asyncio
async def test_rate_limit_retries_then_succeeds(monkeypatch, tmp_path):
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return _ok_response()

    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    grader = AzureVisionGrader(make_cfg())
    grader._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    ref = tmp_path / "b.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")

    result = await grader.grade(img, ref, "prompt")

    assert call_count["n"] == 3
    assert result == {"feedback": "ok"}
    await grader.aclose()


@pytest.mark.asyncio
async def test_correct_img_path_none_omits_reference_image_from_payload(monkeypatch, tmp_path):
    """The standalone_submission grading policy grades each image without
    showing a reference answer at all -- confirm that actually reaches the
    wire, not just the Python call signature."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _ok_response()

    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    grader = AzureVisionGrader(make_cfg())
    grader._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    await grader.grade(img, None, "prompt")
    await grader.aclose()

    content = captured["body"]["messages"][1]["content"]
    image_blocks = [c for c in content if c["type"] == "image_url"]
    assert len(image_blocks) == 1, "only the submitted image should be sent, no reference image"


async def _no_sleep(*args, **kwargs):
    return None
