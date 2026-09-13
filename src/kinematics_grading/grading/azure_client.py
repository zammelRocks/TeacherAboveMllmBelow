"""Async Azure OpenAI vision-grading client.

The original codebase issued one synchronous `requests.post` per image, so
grading the full 6,000-image corpus meant 6,000 sequential round trips. This
client bounds concurrency with a semaphore (default 8, configurable) instead,
which is the main scalability change from the original grading scripts --
same request shape, same retry/backoff policy, just no longer serialized.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from ..config import AzureConfig
from .normalize import extract_json_from_text

logger = logging.getLogger(__name__)

_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# Identical on every single grading call, by design: this is the other half
# of the prompt-caching story alongside prompts.build_stable_prefix (see its
# docstring). Keeping it as one named constant, rather than an inline
# literal re-typed at each call site, is what guarantees it stays
# byte-identical -- required for the provider's prefix cache to match it.
SYSTEM_PROMPT = (
    "You are a strict grading assistant for kinematics graph images. "
    "You grade according to explicit rubric checkpoints and partial credit. "
    "You always return valid JSON only."
)


class NonRetryableGradingError(RuntimeError):
    """A permanent (config/auth) failure -- retrying will not help."""


def encode_image_to_data_url(img_path: Path) -> str:
    mime = _MIME_BY_SUFFIX.get(img_path.suffix.lower(), "image/png")
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


class AzureVisionGrader:
    """Bounded-concurrency async client for the grading endpoint."""

    def __init__(self, cfg: AzureConfig):
        self.cfg = cfg
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.request_timeout, connect=cfg.connect_timeout))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AzureVisionGrader":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def grade(self, generated_img_path: Path, correct_img_path: Optional[Path], prompt: str) -> Dict[str, Any]:
        """Send one grading request; returns the parsed JSON grade dict.

        `correct_img_path=None` omits the reference image from the request
        entirely -- used by the "standalone_submission" grading policy (see
        grading/prompts.py), which grades each image as a complete answer on
        its own, not by comparison against a shown reference.

        Raises after cfg.max_retries exhausted attempts, mirroring the
        original scripts' behavior (caller is expected to catch and log to
        a failed-queue for later retry).
        """
        content: list[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image_to_data_url(generated_img_path), "detail": "high"}},
        ]
        if correct_img_path is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": encode_image_to_data_url(correct_img_path), "detail": "high"}}
            )

        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"api-key": self.cfg.api_key, "Content-Type": "application/json"}

        last_error: Exception | None = None
        async with self._semaphore:
            for attempt in range(1, self.cfg.max_retries + 1):
                try:
                    response = await self._client.post(self.cfg.chat_url, headers=headers, json=payload)

                    if response.status_code == 429:
                        wait = min(60, 2 ** attempt)
                        logger.warning("[RATE LIMIT] waiting %ss (attempt %s)", wait, attempt)
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code >= 500:
                        wait = min(60, 2 ** attempt)
                        logger.warning("[SERVER ERROR %s] waiting %ss (attempt %s)", response.status_code, wait, attempt)
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code != 200:
                        # Any other 4xx (auth failure, bad request, or --
                        # observed live during this rewrite's own smoke
                        # test -- "DeploymentNotFound") is a configuration
                        # problem, not a transient one. Retrying it 5x with
                        # backoff, across every concurrent request, wastes
                        # minutes confirming the same permanent failure;
                        # fail immediately instead so the caller (and the
                        # user) find out right away.
                        raise NonRetryableGradingError(
                            f"Azure OpenAI API error {response.status_code} (not retrying): {response.text}"
                        )

                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    return extract_json_from_text(content)

                except NonRetryableGradingError:
                    raise

                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_error = e
                    logger.warning("[NETWORK ERROR] attempt %s/%s: %s", attempt, self.cfg.max_retries, e)
                    if attempt < self.cfg.max_retries:
                        await asyncio.sleep(min(60, 2 ** attempt))

                except Exception as e:
                    last_error = e
                    logger.warning("[ERROR] attempt %s/%s: %s", attempt, self.cfg.max_retries, e)
                    if attempt < self.cfg.max_retries:
                        await asyncio.sleep(min(60, 2 ** attempt))

        raise RuntimeError(f"grading failed after {self.cfg.max_retries} attempts: {last_error}")
