from __future__ import annotations

from typing import Any, Callable, Iterable, List

from .execution import SyncPoolRunner, default_runner

__all__ = ["run_sync_in_pool", "map_sync_bounded", "SyncPoolRunner", "default_runner"]


async def run_sync_in_pool(
    fn: Callable[..., Any],
    /,
    *args: Any,
    pool_kind: str = "thread",
    **kwargs: Any,
) -> Any:
    """Offload a sync callable from the event loop via the shared thread pool."""
    runner = default_runner if pool_kind == "thread" else SyncPoolRunner(pool_kind)
    return await runner(fn, *args, **kwargs)


def map_sync_bounded(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    max_workers: int,
) -> List[Any]:
    """Apply *fn* to *items* with a bounded thread pool (sync auto-map)."""
    return default_runner.map_bounded(fn, items, max_workers=max_workers)
