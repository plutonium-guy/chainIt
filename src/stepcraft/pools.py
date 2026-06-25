from __future__ import annotations

import atexit
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
            # Always use spawn: fork() in a multi-threaded parent (common after
            # thread-pool tests or library threads) triggers deadlocks on 3.13+.
            ctx = get_context("spawn")
            _POOLS[kind] = ProcessPoolExecutor(
                max_workers=_POOL_CONFIG["process_workers"], mp_context=ctx
            )
        else:
            _POOLS[kind] = ThreadPoolExecutor(
                max_workers=_POOL_CONFIG["thread_workers"]
            )
    return _POOLS[kind]


def cleanup_pools(*, wait: bool = True) -> None:
    """Clean up thread/process pools.

    Process pools are torn down without blocking and their worker processes
    are forcibly terminated: joining spawn-based workers can hang
    indefinitely on Linux (this froze CI for 6h), and merely calling
    ``shutdown(wait=False)`` leaks live workers that keep the parent's stdout
    pipe open — so a CI step hangs after the tests have already passed.
    Terminating the workers closes those handles. Thread pools honor *wait*
    since joining threads is cheap and safe.
    """
    for pool in list(_POOLS.values()):
        if isinstance(pool, ProcessPoolExecutor):
            # Capture worker handles before shutdown drops the references.
            procs = list(getattr(pool, "_processes", {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for proc in procs:
                try:
                    if proc.is_alive():
                        proc.terminate()
                except (ValueError, OSError):
                    pass
        else:
            pool.shutdown(wait=wait, cancel_futures=not wait)
    _POOLS.clear()


def _atexit_cleanup_pools() -> None:
    # Do not block process exit on workers left running by timeouts / parallel
    # tests; blocking here caused pytest to hang on CI after all tests passed.
    cleanup_pools(wait=False)


atexit.register(_atexit_cleanup_pools)
