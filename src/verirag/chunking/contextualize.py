"""Contextual prefixing of chunks (Anthropic's contextual-retrieval pattern).

For each chunk we ask the DeepSeek chat model to write a short (1-2 sentence)
description situating the chunk within its source document, then prepend that
context to the chunk's text before embedding. The bare context is also stored
in ``metadata['context_prefix']`` and the untouched text stays in
``metadata['raw_text']`` (so later phases cite the real text, not the prefix).

Failures never abort the batch: on any API error, empty response, or exception
(after retries) we log a warning, leave the chunk as unprefixed raw_text, and
set ``metadata['context_prefix'] = None``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
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

# MVP: use the first ~3000 chars of the source doc as context. A proper
# document summary would scale better for very long documents (future work).
_DOC_TEXT_TRUNCATE_CHARS = 3000

# Only transient errors are retried; a 4xx/bad request falls straight through.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

_CONTEXT_PROMPT = """\
<document>
{doc_text}
</document>

Here is a chunk extracted from the document above:
<chunk>
{chunk_text}
</chunk>

Write a short, 1-2 sentence context that situates this chunk within the overall
document, to improve search retrieval of the chunk. Answer with ONLY that
succinct context and nothing else."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Lazily build (and cache) the DeepSeek-pointed OpenAI client."""
    return OpenAI(
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
)
def _request_context(doc_text: str, chunk_text: str) -> str | None:
    """Single chat call for a chunk's context; retried on transient errors."""
    response = _client().chat.completions.create(
        model=settings.deepseek_chat_model,
        messages=[
            {
                "role": "user",
                "content": _CONTEXT_PROMPT.format(
                    doc_text=doc_text, chunk_text=chunk_text
                ),
            }
        ],
        max_tokens=150,
        temperature=0.2,
    )
    return response.choices[0].message.content


def add_context(chunk: Document, doc_text: str) -> Document:
    """Prepend a contextual description to ``chunk`` (in place) and return it.

    ``doc_text`` is the full source document text; it is truncated to roughly
    the first ``_DOC_TEXT_TRUNCATE_CHARS`` characters as a cheap MVP context.
    On failure, the chunk is left as its unprefixed ``raw_text`` and
    ``context_prefix`` is set to None.
    """
    raw_text = chunk.metadata.get("raw_text", chunk.page_content)
    truncated_doc = doc_text[:_DOC_TEXT_TRUNCATE_CHARS]
    chunk_id = chunk.metadata.get("chunk_id")

    context: str | None = None
    try:
        result = _request_context(truncated_doc, raw_text)
        context = (result or "").strip() or None
        if context is None:
            logger.warning(
                "Contextualization returned an empty response for chunk_id=%s",
                chunk_id,
            )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never abort
        logger.warning(
            "Contextualization failed for chunk_id=%s (%s): %s",
            chunk_id, type(exc).__name__, exc,
        )
        context = None

    if context is None:
        chunk.metadata["context_prefix"] = None
        chunk.page_content = raw_text
    else:
        chunk.metadata["context_prefix"] = context
        chunk.page_content = f"{context}\n\n{raw_text}"
    return chunk


def add_context_to_chunks(chunks: list[Document], doc_text: str) -> list[Document]:
    """Contextualize a list of same-document chunks, sequentially.

    Sequential by design for the MVP; concurrent/batched requests would matter
    for large corpora (future optimization).
    """
    return [add_context(chunk, doc_text) for chunk in chunks]
