from __future__ import annotations

import sys
import sysconfig
from typing import Optional

from .constants import logger

# True when this interpreter was built with optional GIL disabled (3.13+).
HAS_FREE_THREADING: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def is_gil_enabled() -> bool:
    """Return whether the GIL is currently enabled in this process."""
    checker = getattr(sys, "_is_gil_enabled", None)
    if checker is None:
        return True
    return checker()


def threads_provide_true_parallelism() -> bool:
    """True when thread pools can run Python code in parallel across CPU cores."""
    return HAS_FREE_THREADING and not is_gil_enabled()


def resolve_parallel_kind(kind: Optional[str]) -> Optional[str]:
    """Normalize parallel mode for the active Python runtime."""
    if kind is None:
        return None

    if kind == "auto":
        resolved = "thread" if threads_provide_true_parallelism() else "process"
        logger.debug(
            "parallel='auto' resolved to %r (free-threading=%s, gil_enabled=%s)",
            resolved,
            HAS_FREE_THREADING,
            is_gil_enabled(),
        )
        return resolved

    if kind == "process" and threads_provide_true_parallelism():
        logger.info(
            "parallel='process' resolved to 'thread' on free-threaded Python "
            "with GIL disabled; threads provide true multi-core parallelism"
        )
        return "thread"

    if kind not in ("thread", "process"):
        raise ValueError(
            f"parallel must be 'thread', 'process', or 'auto', got {kind!r}"
        )
    return kind
