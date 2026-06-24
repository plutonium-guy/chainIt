from __future__ import annotations

import asyncio
import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, List, Optional, Tuple

from .constants import R, T
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

    def __post_init__(self):
        parallel = resolve_parallel_kind(self.parallel)
        if parallel != self.parallel:
            object.__setattr__(self, 'parallel', parallel)

    def run(self, value: T) -> Tuple[R, ...]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            return tuple(
                pool.map(_run_branch_on, self.branches, itertools.repeat(value))
            )
        return tuple(branch.run(value) for branch in self.branches)

    async def async_run(self, value: T) -> Tuple[R, ...]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(pool, _run_branch_on, branch, value)
                for branch in self.branches
            ]
            return tuple(await asyncio.gather(*tasks))

        tasks = []
        for branch in self.branches:
            if hasattr(branch, 'async_run'):
                tasks.append(branch.async_run(value))
            else:
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, branch.run, value))
        return tuple(await asyncio.gather(*tasks))

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
    _mapper_is_async: bool = field(default=False, init=False)

    def __post_init__(self):
        object.__setattr__(
            self, '_mapper_is_async', inspect.iscoroutinefunction(self.mapper),
        )

    async def _map_item(self, item: T) -> Any:
        if self._mapper_is_async:
            return await self.mapper(item)
        result = self.mapper(item)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _map_batch(self, batch: List[T]) -> List[Any]:
        return list(await asyncio.gather(*(self._map_item(x) for x in batch)))

    def run(self, items: Iterable[T]) -> R:
        results: List[Any] = []
        batch: List[T] = []
        for item in items:
            batch.append(item)
            if len(batch) == self.batch_size:
                results.extend(map(self.mapper, batch))
                batch.clear()
        if batch:
            results.extend(map(self.mapper, batch))
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
