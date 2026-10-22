"""DeepSeek vision captioning for figures extracted from documents.

Uses the ``openai`` Python client pointed at DeepSeek's OpenAI-compatible API
to turn an image into a short, factual caption. Transient errors are retried
with exponential backoff; a bad request (or any non-rate-limit 4xx) or a final
failure returns ``None`` instead of raising, so a single bad image never aborts
a whole document load.
"""

from __future__ import annotations

import base64
import logging
from functools import lru_cache

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from verirag.config import settings

logger = logging.getLogger(__name__)

# Only transient failures are worth retrying. A BadRequestError / 4xx (other
# than 429) means the request itself is wrong and will never succeed.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

_CAPTION_PROMPT = (
    "You are describing a figure extracted from a document for a retrieval "
    "system. In 2-4 concise, factual sentences, describe what this figure or "
    "chart shows: the type of visual, any axis titles, labels, or legend text "
    "you can read, and the single key takeaway. Do not speculate beyond what "
    "is visible. If it is not a meaningful figure, say so briefly."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Lazily build (and cache) the DeepSeek-pointed OpenAI client."""
    return OpenAI(
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
    )


def _detect_mime(image_bytes: bytes) -> str:
    """Best-effort mime sniffing for the data URL; defaults to PNG."""
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
)
def _caption_with_retry(data_url: str) -> str | None:
    """Single captioning call; retried by tenacity on transient errors only."""
    response = _client().chat.completions.create(
        model=settings.deepseek_vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content


def caption_image(image_bytes: bytes) -> str | None:
    """Caption an image with the DeepSeek vision model.

    Returns the caption text, or ``None`` if captioning ultimately failed
    (transient errors exhausted, a bad-request/4xx, or an empty response).
    Never raises.
    """
    mime = _detect_mime(image_bytes)
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    try:
        caption = _caption_with_retry(data_url)
    except Exception as exc:  # noqa: BLE001 - deliberately swallow; caller skips image
        logger.warning(
            "Vision captioning failed (%s): %s", type(exc).__name__, exc
        )
        return None

    if not caption:
        logger.warning("Vision captioning returned an empty response.")
        return None
    return caption.strip() or None
