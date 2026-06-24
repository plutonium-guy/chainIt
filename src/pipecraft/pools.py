from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from typing import Dict, Union

from .runtime import resolve_parallel_kind

_POOLS: Dict[str, Union[ThreadPoolExecutor, ProcessPoolExecutor]] = {}


def _get_pool(kind: str) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
    """Get or create a thread/process pool."""
    kind = resolve_parallel_kind(kind) or kind
    if kind not in _POOLS:
        if kind == "process":
            ctx = get_context("spawn" if sys.platform in ("darwin", "win32") else "fork")
            _POOLS[kind] = ProcessPoolExecutor(max_workers=None, mp_context=ctx)
        else:
            _POOLS[kind] = ThreadPoolExecutor(max_workers=None)
    return _POOLS[kind]


def cleanup_pools() -> None:
    """Clean up thread/process pools."""
    for pool in _POOLS.values():
        pool.shutdown(wait=True)
    _POOLS.clear()
