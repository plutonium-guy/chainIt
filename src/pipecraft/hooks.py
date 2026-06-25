from __future__ import annotations

from typing import Any, Callable, Optional

from .constants import logger

# name, step_input, step_output, step_dt (seconds)
StepHook = Callable[[str, Any, Any, float], None]


def _call_hook(
    hook: Optional[StepHook],
    name: str,
    step_input: Any,
    step_output: Any,
    step_dt: float,
) -> None:
    """Invoke an optional step hook.

    Hook failures must never mask pipeline results: an exception raised by the
    hook is logged as a warning and swallowed so execution continues.
    """
    if hook is None:
        return
    try:
        hook(name, step_input, step_output, step_dt)
    except Exception as exc:  # observability must not break the pipeline
        logger.warning("on_step hook raised %r for step %r; continuing", exc, name)
