"""Shared DeepSeek chat-completion client.

One place for the "call DeepSeek's OpenAI-compatible chat endpoint, retry
transient failures, degrade gracefully" pattern. Contextual prefixing uses it
today; generation and the Gate 1 / Gate 2 critics use it in later phases.

``chat_completion`` never raises: it returns ``None`` on a non-retryable 4xx,
on exhausted retries, or on an empty response, so callers can always fall
back instead of aborting a batch.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

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

# Only transient failures are retried. A BadRequestError / 4xx (other than
# 429) means the request itself is wrong and will never succeed.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


@lru_cache(maxsize=1)
def get_deepseek_client() -> OpenAI:
    """Return the lazily-built, process-wide DeepSeek client singleton."""
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
)
def _create_completion(
    messages: list[dict],
    model: str,
    temperature: float,
    **kwargs: Any,
) -> str | None:
    """One chat-completion call; retried by tenacity on transient errors."""
    response = get_deepseek_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content


def chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    **kwargs: Any,
) -> str | None:
    """Run a chat completion, returning the text or None on failure.

    ``model`` defaults to ``settings.deepseek_chat_model``. Extra keyword
    arguments (e.g. ``max_tokens``) are forwarded to the API call.
    """
    model = model or settings.deepseek_chat_model
    try:
        content = _create_completion(messages, model, temperature, **kwargs)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never raise
        logger.warning(
            "DeepSeek chat completion failed (model=%s, %s): %s",
            model, type(exc).__name__, exc,
        )
        return None

    if not content or not content.strip():
        logger.warning(
            "DeepSeek chat completion returned an empty response (model=%s).",
            model,
        )
        return None
    return content
