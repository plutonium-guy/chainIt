from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

# Shared, read-only-ish context for steps to access during a pipeline run.
_PIPELINE_CONTEXT: contextvars.ContextVar[Optional[Dict[str, Any]]] = (
    contextvars.ContextVar("pipecraft_context", default=None)
)


def get_context() -> Dict[str, Any]:
    """Return the active pipeline context, or an empty dict outside a run."""
    ctx = _PIPELINE_CONTEXT.get()
    return ctx if ctx is not None else {}


def _set_context(ctx: Optional[Dict[str, Any]]) -> contextvars.Token:
    return _PIPELINE_CONTEXT.set(ctx)


def _reset_context(token: contextvars.Token) -> None:
    _PIPELINE_CONTEXT.reset(token)
