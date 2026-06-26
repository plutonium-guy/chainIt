from __future__ import annotations

import asyncio
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, List, Optional, Tuple

from .async_concurrency import gather_limited
from .constants import R, T
from .context import wrap_worker
from .execution import default_runner
from .pools import _get_pool
from .runtime import resolve_parallel_kind


def _run_branch_on(branch: Any, value: Any) -> Any:
    """Module-level (picklable) helper for ProcessPoolExecutor fan-out."""
    return branch.run(value)


@dataclass
class FanOutStep(Generic[T, R]):
    """Execute multiple branches in parallel."""

    branches: Tuple[Any, ...]
    parallel: Optional[str] = None
    max_concurrency: Optional[int] = None

    def __post_init__(self):
        parallel = resolve_parallel_kind(self.parallel)
        if parallel != self.parallel:
            object.__setattr__(self, 'parallel', parallel)

    def run(self, value: T) -> Tuple[R, ...]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            runner = wrap_worker(_run_branch_on)
            return tuple(
                pool.map(runner, self.branches, itertools.repeat(value))
            )
        return tuple(branch.run(value) for branch in self.branches)

    async def async_run(self, value: T) -> Tuple[R, ...]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            loop = asyncio.get_running_loop()
            runner = wrap_worker(_run_branch_on)
            tasks = [
                loop.run_in_executor(pool, runner, branch, value)
                for branch in self.branches
            ]
            return tuple(await gather_limited(
                tasks, max_concurrency=self.max_concurrency,
            ))

        tasks = []
        for branch in self.branches:
            if hasattr(branch, 'async_run'):
                tasks.append(branch.async_run(value))
            else:
                tasks.append(default_runner(lambda b=branch, v=value: b.run(v)))
        return tuple(await gather_limited(tasks, max_concurrency=self.max_concurrency))

    @property
    def _func_name(self) -> str:
        return "FanOutStep"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other


@dataclass
class FanInStep(Generic[T, R]):
    """Combine multiple inputs into single output."""

    combiner: Callable[..., R]

    def run(self, values: Tuple[T, ...]) -> R:
        return self.combiner(*values)

    async def async_run(self, values: Tuple[T, ...]) -> R:
        result = self.combiner(*values)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @property
    def _func_name(self) -> str:
        return "FanInStep"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other


@dataclass
class MapReduceStep(Generic[T, R]):
    """Map-reduce with optional batching."""

    mapper: Callable[[T], Any]
    reducer: Callable[[Iterable[Any]], R]
    batch_size: int = 1
    parallel: Optional[str] = None
    max_concurrency: Optional[int] = None
    _mapper_is_async: bool = field(default=False, init=False)

    def __post_init__(self):
        parallel = resolve_parallel_kind(self.parallel)
        if parallel != self.parallel:
            object.__setattr__(self, 'parallel', parallel)
        object.__setattr__(
            self, '_mapper_is_async', inspect.iscoroutinefunction(self.mapper),
        )

    async def _map_item(self, item: T) -> Any:
        if self._mapper_is_async:
            return await self.mapper(item)
        result = await default_runner(self.mapper, item)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _map_batch(self, batch: List[T]) -> List[Any]:
        return list(await gather_limited(
            (self._map_item(x) for x in batch),
            max_concurrency=self.max_concurrency,
        ))

    def _map_batch_sync(self, batch: List[T]) -> List[Any]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            runner = wrap_worker(self.mapper)
            return list(pool.map(runner, batch))
        return list(map(self.mapper, batch))

    def run(self, items: Iterable[T]) -> R:
        results: List[Any] = []
        batch: List[T] = []
        for item in items:
            batch.append(item)
            if len(batch) == self.batch_size:
                results.extend(self._map_batch_sync(batch))
                batch.clear()
        if batch:
            results.extend(self._map_batch_sync(batch))
        return self.reducer(results)

    async def async_run(self, items: Iterable[T]) -> R:
        results: List[Any] = []
        batch: List[T] = []
        for item in items:
            batch.append(item)
            if len(batch) == self.batch_size:
                results.extend(await self._map_batch(batch))
                batch.clear()
        if batch:
            results.extend(await self._map_batch(batch))
        out = self.reducer(results)
        if asyncio.iscoroutine(out):
            return await out
        return out

    @property
    def _func_name(self) -> str:
        return "MapReduceStep"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
