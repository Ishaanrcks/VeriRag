"""LangSmith tracing setup.

``init_tracing()`` is called once when the graph module is first imported, so
tracing is active before any query runs. It configures LangSmith purely through
the environment variables the LangChain/LangSmith runtime actually reads, which
means the ``@traceable`` decorators scattered across the pipeline become named
spans without any further wiring.

Tracing is strictly optional: with no usable API key this logs an info message
and returns. A missing or misconfigured tracing backend must never stop a query
from being answered.
"""

from __future__ import annotations

import logging
import os

from verirag.config import configured_secret, settings

logger = logging.getLogger(__name__)

_initialized = False


def init_tracing() -> None:
    """Activate LangSmith tracing if a real API key is configured.

    Idempotent: safe to call more than once. Never raises.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Shares the placeholder guard with the Tavily key, so a leftover
    # `lsv2_...` from .env.example counts as "not configured".
    api_key = configured_secret(settings.langsmith_api_key)
    if api_key is None:
        logger.info(
            "LANGSMITH_API_KEY not configured - tracing inactive "
            "(queries run normally)."
        )
        return

    try:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info(
            "LangSmith tracing active (project=%r).", settings.langsmith_project
        )
    except Exception as exc:  # noqa: BLE001 - tracing must never block a query
        logger.warning(
            "Failed to initialize LangSmith tracing (%s): %s; continuing "
            "without tracing.",
            type(exc).__name__, exc,
        )


def tracing_active() -> bool:
    """True if tracing was actually switched on (for reporting/tests)."""
    return os.environ.get("LANGSMITH_TRACING") == "true"
