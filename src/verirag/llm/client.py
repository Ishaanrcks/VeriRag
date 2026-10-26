"""Shared DeepSeek chat-completion client.

One place for the "call DeepSeek's OpenAI-compatible chat endpoint, retry
transient failures, degrade gracefully" pattern. Contextual prefixing uses it
today; generation and the Gate 1 / Gate 2 critics use it in later phases.

``chat_completion`` never raises: it returns ``None`` on a non-retryable 4xx,
on exhausted retries, or on an empty response, so callers can always fall
back instead of aborting a batch.
"""

from __future__ import annotations

import contextvars
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


class QuotaExceededError(RuntimeError):
    """DeepSeek rate-limit/quota exhausted after retries.

    Raised by ``chat_completion`` ONLY when quota surfacing is enabled for the
    current context (``set_surface_quota(True)`` -- the Streamlit UI does this
    around a query). Every other caller keeps the historical behavior of a
    graceful ``None``, so the eval harness and CLI never start crashing on a
    429. It is deliberately distinct from a Gate 2 abstention: this is an
    infrastructure error, not an ungrounded answer.
    """


# Per-request overrides, set by the UI before a query and reset after. Context
# variables (not globals) so concurrent Streamlit sessions in one process never
# clobber each other's key or quota policy -- each ScriptRunner thread carries
# its own value. The pipeline is synchronous, so a value set before run_query
# is visible to every chat_completion it triggers in that thread.
_session_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "verirag_session_api_key", default=None
)
_surface_quota: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "verirag_surface_quota", default=False
)


def set_session_api_key(api_key: str | None) -> None:
    """Override the DeepSeek key for calls in this context (None = backend key)."""
    _session_api_key.set((api_key or "").strip() or None)


def set_surface_quota(enabled: bool) -> None:
    """When True, an exhausted rate limit raises QuotaExceededError instead of
    returning None, so the UI can show a distinct 'high traffic' message."""
    _surface_quota.set(bool(enabled))


@lru_cache(maxsize=8)
def _client_for_key(api_key: str) -> OpenAI:
    """One cached client per distinct key (backend key + any user keys)."""
    return OpenAI(api_key=api_key, base_url=settings.deepseek_base_url)


def get_deepseek_client() -> OpenAI:
    """Return the DeepSeek client for the current context.

    Uses the per-session override key when one is set (a demo user who supplied
    their own key after the free-query limit), otherwise the backend key.
    """
    api_key = _session_api_key.get() or settings.deepseek_api_key
    return _client_for_key(api_key)


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
    except RateLimitError as exc:
        # Quota/rate limit exhausted after retries. Surface it distinctly to the
        # UI when asked; otherwise degrade to None like any other failure.
        logger.warning(
            "DeepSeek rate limit / quota exhausted (model=%s): %s", model, exc
        )
        if _surface_quota.get():
            raise QuotaExceededError(
                "DeepSeek quota or rate limit exhausted"
            ) from exc
        return None
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
