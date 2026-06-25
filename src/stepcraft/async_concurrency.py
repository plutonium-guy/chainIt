from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Iterable, List, Optional, TypeVar

T = TypeVar("T")


async def gather_limited(
    awaitables: Iterable[Awaitable[T]],
    *,
    max_concurrency: Optional[int] = None,
) -> List[T]:
    """Run awaitables concurrently, optionally capped by a semaphore."""
    items = list(awaitables)
    if not items:
        return []
    if max_concurrency is None:
        return list(await asyncio.gather(*items))

    sem = asyncio.Semaphore(max_concurrency)

    async def _limited(coro: Awaitable[T]) -> T:
        async with sem:
            return await coro

    return list(await asyncio.gather(*(_limited(c) for c in items)))
