from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from typing import Dict, Optional, Union

from .runtime import resolve_parallel_kind

_POOLS: Dict[str, Union[ThreadPoolExecutor, ProcessPoolExecutor]] = {}

# Worker counts applied on the next pool creation (None => executor default).
_POOL_CONFIG: Dict[str, Optional[int]] = {
    "thread_workers": None,
    "process_workers": None,
}


def configure_pools(
    *,
    thread_workers: Optional[int] = None,
    process_workers: Optional[int] = None,
) -> None:
    """Set max worker counts for thread/process pools.

    Takes effect on the next pool creation. To apply to pools that already
    exist, call :func:`cleanup_pools` first so they are recreated.
    """
    _POOL_CONFIG["thread_workers"] = thread_workers
    _POOL_CONFIG["process_workers"] = process_workers


def _get_pool(kind: str) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
    """Get or create a thread/process pool."""
    kind = resolve_parallel_kind(kind) or kind
    if kind not in _POOLS:
        if kind == "process":
            ctx = get_context("spawn" if sys.platform in ("darwin", "win32") else "fork")
            _POOLS[kind] = ProcessPoolExecutor(
                max_workers=_POOL_CONFIG["process_workers"], mp_context=ctx
            )
        else:
            _POOLS[kind] = ThreadPoolExecutor(
                max_workers=_POOL_CONFIG["thread_workers"]
            )
    return _POOLS[kind]


def cleanup_pools() -> None:
    """Clean up thread/process pools."""
    for pool in _POOLS.values():
        pool.shutdown(wait=True)
    _POOLS.clear()
