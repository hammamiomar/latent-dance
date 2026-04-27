"""Async OpenRouter API client for VLM annotation.

Handles rate limiting, retries, and multi-format response parsing
for the VLM ensemble (Stages 3 + 6). Models are swapped by
changing a single parameter — same endpoint, same format.

Usage:
    async with OpenRouterClient() as client:
        resp = await client.annotate(
            model=PAID_MODELS["qwen3_vl_235b"],
            system_prompt="...",
            images_b64=[img1, img2, ...],
        )
        print(resp.label, resp.category, resp.confidence)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_api_key() -> str:
    """Load OpenRouter API key from environment or .env file.

    Checks os.environ first, then reads .env in project root.
    Raises RuntimeError with actionable message if not found.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    # Try loading from .env file at project root
    from hambajuba2ba.labeling.config import PROJECT_ROOT

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.replace("export", "").strip()
            value = value.strip().strip("\"'")
            if name == "OPENROUTER_API_KEY" and value:
                return value

    raise RuntimeError(
        "OPENROUTER_API_KEY not found. Set it via:\n"
        "  set OPENROUTER_API_KEY in your environment\n"
        "  or add it to .env in the project root."
    )

# ─── Model registry ──────────────────────────────────────────────

PAID_MODELS: dict[str, str] = {
    "qwen3_vl_235b": "qwen/qwen3-vl-235b-a22b-instruct",
    "glm_4_6v": "z-ai/glm-4.6v",
    "kimi_k2_5": "moonshotai/kimi-k2.5",
}

FREE_MODELS: dict[str, str] = {
    "qwen3_vl_30b": "qwen/qwen3-vl-30b-a3b-thinking",
}

ALL_MODELS: dict[str, str] = {**PAID_MODELS, **FREE_MODELS}

# Models that support reasoning token control
_REASONING_MODELS = {"moonshotai/kimi-k2.5"}


# ─── Response data ────────────────────────────────────────────────


@dataclass
class VLMResponse:
    """Parsed response from a VLM annotation call."""

    label: str
    category: str
    confidence: str  # "high", "medium", "low"
    raw: str         # Original response text


# ─── Client ───────────────────────────────────────────────────────


class OpenRouterClient:
    """Async OpenRouter API client with rate limiting and retries.

    Args:
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        concurrency: Max concurrent requests (default 50).
        timeout: Per-request timeout in seconds (default 120).
        max_retries: Retry attempts on transient failures (default 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        concurrency: int = 50,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or _load_api_key()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenRouterClient:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def annotate(
        self,
        model: str,
        system_prompt: str,
        images_b64: list[str],
        user_text: str = "Label this feature.",
        max_tokens: int = 150,
    ) -> VLMResponse:
        """Send images to a VLM and parse the feature label response.

        Args:
            model: OpenRouter model ID (e.g. "qwen/qwen3-vl-235b-a22b-instruct")
            system_prompt: Block-specific annotation prompt.
            images_b64: Base64-encoded JPEG images (OFF first, then ON ascending).
            user_text: User message text appended after images.
            max_tokens: Max output tokens.

        Returns:
            Parsed VLMResponse with label, category, confidence.
        """
        content: list[dict] = [
            *(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                }
                for img in images_b64
            ),
            {"type": "text", "text": user_text},
        ]

        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
        }

        # Disable reasoning tokens for thinking-capable models
        if model in _REASONING_MODELS:
            body["reasoning"] = {"enabled": False}

        raw = await self._request(body)
        return parse_vlm_response(raw)

    async def _request(self, body: dict) -> str:
        """Make an API request with retries and exponential backoff."""
        assert self._client is not None, "Use as async context manager"

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client.post(BASE_URL, json=body)

                    if resp.status_code == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", 2 ** attempt)
                        )
                        logger.warning(
                            "Rate limited, retrying in %.1fs (attempt %d/%d)",
                            retry_after, attempt + 1, self.max_retries,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

                except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait = 2 ** attempt
                    logger.warning(
                        "%s, retrying in %ds (attempt %d/%d)",
                        type(e).__name__, wait, attempt + 1, self.max_retries,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError("Exhausted retries")


# ─── Response parsing ─────────────────────────────────────────────


def parse_vlm_response(raw: str) -> VLMResponse:
    """Parse VLM output into structured label data.

    Tries multiple formats in order:
    1. JSON: {"label": "...", "category": "...", "confidence": "..."}
    2. Markdown list: - label: ... / - category: ... / - confidence: ...
    3. Key-value: Label: ... / Category: ... / Confidence: ...
    4. Fallback: first non-empty line as label, category=unknown, confidence=low
    """
    raw = raw.strip()

    # Try JSON
    parsed = _try_json(raw)
    if parsed:
        return parsed

    # Try markdown list or key-value pairs
    parsed = _try_key_value(raw)
    if parsed:
        return parsed

    # Fallback: first non-empty line
    first_line = next((line.strip() for line in raw.split("\n") if line.strip()), raw)
    return VLMResponse(
        label=first_line[:100],
        category="unknown",
        confidence="low",
        raw=raw,
    )


def _try_json(raw: str) -> VLMResponse | None:
    """Try to parse as JSON (with optional markdown code fences)."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "label" in data:
            return VLMResponse(
                label=data["label"],
                category=data.get("category", "unknown"),
                confidence=data.get("confidence", "medium"),
                raw=raw,
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _try_key_value(raw: str) -> VLMResponse | None:
    """Try to parse as markdown list or key-value pairs."""
    label = _extract_field(raw, "label")
    if not label:
        return None
    return VLMResponse(
        label=label,
        category=_extract_field(raw, "category") or "unknown",
        confidence=_extract_field(raw, "confidence") or "medium",
        raw=raw,
    )


def _extract_field(text: str, field: str) -> str | None:
    """Extract a field value from various text formats."""
    patterns = [
        rf"[-*]\s*{field}\s*:\s*(.+)",       # - label: value
        rf"{field}\s*:\s*(.+)",               # label: value
        rf"\*\*{field}\*\*\s*:\s*(.+)",       # **label**: value
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'")
    return None


# ─── Image encoding helper ────────────────────────────────────────


def encode_image_b64(path: Path) -> str:
    """Read an image file and return base64-encoded string."""
    return base64.b64encode(path.read_bytes()).decode("ascii")
