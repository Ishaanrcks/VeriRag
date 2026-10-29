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
from dataclasses import dataclass
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
    "system. State what kind of visual it is, its title, any axis titles, "
    "labels and legend text you can read, and the single key takeaway. "
    "If the figure plots or labels numbers (a chart, graph, dot plot or any "
    "table-like graphic), you MUST state the actual numeric values you can "
    "read, each tied to its category, series or group (for example 'U.S., "
    "ages 30-49: 72%'), rather than only describing the trend. List them "
    "compactly. Do not speculate beyond what is visible and do not invent "
    "values you cannot read. If it is not a meaningful figure, say so briefly."
)

# Hidden reasoning tokens count against max_tokens, so a hard chart can spend
# nearly the whole budget thinking (measured: 1,686 of 2,048 on one Pew chart).
# A response that stops on the length limit is retried once with more headroom.
# A value-listing caption on a dense chart exceeded 2,048 on 3 of 3 attempts, so
# the first call gets 4,096 (a cap costs nothing unless used) and the retry 8,192.
_CAPTION_MAX_TOKENS = 4096
_CAPTION_RETRY_MAX_TOKENS = 8192


@dataclass(frozen=True)
class Caption:
    """A generated caption and whether the model's output hit its length limit."""

    text: str
    truncated: bool = False


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
def _caption_with_retry(
    data_url: str, max_tokens: int = _CAPTION_MAX_TOKENS
) -> tuple[str | None, str | None]:
    """Single captioning call; retried by tenacity on transient errors only.

    Returns ``(content, finish_reason)``; ``finish_reason == "length"`` means
    the output (reasoning included) hit ``max_tokens`` and may be cut short.
    """
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
        # The vision model is a reasoning model: its hidden reasoning tokens
        # count against this limit. At 300, two of three test figures returned
        # EMPTY content (all 300 tokens spent reasoning, finish_reason=length)
        # and the third was cut mid-sentence. The limit is only a cap, so
        # headroom costs nothing unless used.
        max_tokens=max_tokens,
        temperature=0.2,
    )
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason


def caption_image_with_status(image_bytes: bytes) -> Caption | None:
    """Caption an image, detecting and retrying a length-truncated response.

    Returns ``None`` if captioning ultimately failed (transient errors
    exhausted, a bad-request/4xx, or an empty response). A response that stops
    on the length limit is retried once with a larger token budget; if it is
    STILL cut off, the partial text is returned with ``truncated=True`` (and a
    warning logged) rather than being stored as if it were complete. Never
    raises.
    """
    mime = _detect_mime(image_bytes)
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    content: str | None = None
    finish_reason: str | None = None
    try:
        for max_tokens in (_CAPTION_MAX_TOKENS, _CAPTION_RETRY_MAX_TOKENS):
            content, finish_reason = _caption_with_retry(data_url, max_tokens)
            if finish_reason != "length":
                break
            logger.warning(
                "Caption hit the %d-token limit (finish_reason=length)%s.",
                max_tokens,
                "; retrying with more headroom"
                if max_tokens == _CAPTION_MAX_TOKENS else "",
            )
    except Exception as exc:  # noqa: BLE001 - deliberately swallow; caller skips image
        logger.warning(
            "Vision captioning failed (%s): %s", type(exc).__name__, exc
        )
        return None

    text = (content or "").strip()
    if not text:
        logger.warning("Vision captioning returned an empty response.")
        return None
    truncated = finish_reason == "length"
    if truncated:
        logger.warning(
            "Caption is still truncated after the retry (%d chars); storing it "
            "flagged caption_truncated.", len(text),
        )
    return Caption(text=text, truncated=truncated)


def caption_image(image_bytes: bytes) -> str | None:
    """Caption text only (see :func:`caption_image_with_status`), or ``None``."""
    result = caption_image_with_status(image_bytes)
    return result.text if result else None
