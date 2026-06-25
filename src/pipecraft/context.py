from __future__ import annotations

import contextvars
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional

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


@contextmanager
def activate_context(ctx: Optional[Dict[str, Any]]) -> Iterator[None]:
    """Make *ctx* visible to steps via ``get_context()`` for the duration."""
    if ctx is None:
        yield
        return
    token = _set_context(ctx)
    try:
        yield
    finally:
        _reset_context(token)


def _run_with_captured_context(
    ctx: Optional[Dict[str, Any]],
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run *fn* after setting pipeline context (for process-pool workers)."""
    if ctx is None:
        return fn(*args, **kwargs)
    token = _set_context(ctx)
    try:
        return fn(*args, **kwargs)
    finally:
        _reset_context(token)


def wrap_worker(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap *fn* so pool workers inherit the caller's pipeline context."""
    ctx = _PIPELINE_CONTEXT.get()
    return functools.partial(_run_with_captured_context, ctx, fn)


# Backward-compatible aliases for thread vs process dispatch sites.
wrap_thread_worker = wrap_worker
wrap_process_worker = wrap_worker
