from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Iterable, List, Optional, TypeVar

T = TypeVar("T")


class ConcurrentScheduler:
    """Run awaitables concurrently with optional bounded lazy scheduling."""

    async def gather(
        self,
        awaitables: Iterable[Awaitable[T]],
        *,
        max_concurrency: Optional[int] = None,
    ) -> List[T]:
        if max_concurrency is None:
            items = list(awaitables)
            if not items:
                return []
            return list(await asyncio.gather(*items))
        return await self._gather_bounded_ordered(awaitables, max_concurrency)

    async def _gather_bounded_ordered(
        self,
        awaitables: Iterable[Awaitable[T]],
        max_concurrency: int,
    ) -> List[T]:
        iterator = iter(awaitables)
        sem = asyncio.Semaphore(max_concurrency)
        results: dict[int, T] = {}
        next_index = 0
        lock = asyncio.Lock()

        async def _run(idx: int, aw: Awaitable[T]) -> None:
            async with sem:
                results[idx] = await aw

        in_flight: set[asyncio.Task[None]] = set()

        async def _submit_next() -> bool:
            nonlocal next_index
            async with lock:
                try:
                    aw = next(iterator)
                except StopIteration:
                    return False
                idx = next_index
                next_index += 1
            in_flight.add(asyncio.create_task(_run(idx, aw)))
            return True

        for _ in range(max_concurrency):
            if not await _submit_next():
                break

        while in_flight:
            done, in_flight = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                task.result()
            while len(in_flight) < max_concurrency and await _submit_next():
                pass

        return [results[i] for i in range(next_index)]


_default_scheduler = ConcurrentScheduler()
gather_limited = _default_scheduler.gather
