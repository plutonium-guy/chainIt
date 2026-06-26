from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Union

from .async_concurrency import gather_limited
from .constants import R, T
from .context import activate_context, wrap_worker
from .execution import default_step_runner
from .hooks import StepHook, _call_hook
from .node import Node
from .pools import _get_pool, cleanup_pools
from .result import ExecutionResult


class PipelineBuilder(Generic[T]):
    """Fluent builder for Pipeline construction."""

    def __init__(self):
        self._steps: List[Any] = []

    def add(self, step) -> 'PipelineBuilder':
        self._steps.append(step)
        return self

    def build(self) -> 'Pipeline':
        from .pipeline import Pipeline
        return Pipeline(self._steps)


class Pipeline(Generic[T, R]):
    """Linear pipeline with advanced execution modes."""

    __slots__ = ('steps', '_cancel_event', '_context')

    def __init__(
        self,
        steps: Sequence[Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.steps = tuple(steps)
        self._cancel_event = None
        self._context = context

    def _merge_context(self, other: 'Pipeline') -> Optional[Dict[str, Any]]:
        if self._context is None and other._context is None:
            return None
        return {**(self._context or {}), **(other._context or {})}

    def __or__(self, other) -> 'Pipeline':
        if isinstance(other, Pipeline):
            return Pipeline(
                [*self.steps, *other.steps], context=self._merge_context(other)
            )
        return Pipeline([*self.steps, other], context=self._context)

    def __call__(self, seed: Any = None) -> R:
        return self.run(seed)

    def __repr__(self) -> str:
        step_names = []
        for s in self.steps:
            name = getattr(s, '_func_name', None) or type(s).__name__
            step_names.append(name)
        return f"Pipeline({' | '.join(step_names)})"

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __iter__(self):
        return iter(self.steps)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._teardown_setup_once_nodes()
        cleanup_pools()

    def _ensure_cancel_event(self) -> asyncio.Event:
        if self._cancel_event is None:
            self._cancel_event = asyncio.Event()
        return self._cancel_event

    def cancel(self) -> None:
        self._ensure_cancel_event().set()

    def _is_cancelled(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _prepare_step(self, step: Any) -> None:
        if self._cancel_event is not None and hasattr(step, '_cancel_event'):
            object.__setattr__(step, '_cancel_event', self._cancel_event)
        if self._is_cancelled():
            raise asyncio.CancelledError()

    @staticmethod
    def _teardown_setup_once_step(step: Any) -> None:
        if isinstance(step, Node) and step.setup_once and getattr(step, '_setup_active', False):
            step._teardown_once()

    def _teardown_setup_once_nodes(self) -> None:
        for step in self.steps:
            self._teardown_setup_once_step(step)

    async def _await_step(self, step: Any, step_input: Any) -> Any:
        return await default_step_runner.run_async(step, step_input)

    def run(
        self,
        seed: Any = None,
        *,
        on_step: Optional[StepHook] = None,
        _finalize_setup_once: bool = True,
    ) -> R:
        value = seed
        try:
            with activate_context(self._context):
                for step in self.steps:
                    self._prepare_step(step)
                    step_input = value
                    t0 = time.perf_counter()
                    value = step.run(step_input)
                    if on_step is not None:
                        name = getattr(step, '_func_name', type(step).__name__)
                        _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
            return value
        finally:
            if _finalize_setup_once:
                self._teardown_setup_once_nodes()

    async def async_run(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> R:
        value = seed
        try:
            with activate_context(self._context):
                for step in self.steps:
                    self._prepare_step(step)
                    step_input = value
                    t0 = time.perf_counter()
                    value = await self._await_step(step, step_input)
                    if on_step is not None:
                        name = getattr(step, '_func_name', type(step).__name__)
                        _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
            return value
        finally:
            self._teardown_setup_once_nodes()

    def run_detailed(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> ExecutionResult[R]:
        history = []
        value = seed
        start_time = time.perf_counter()
        try:
            with activate_context(self._context):
                for step in self.steps:
                    self._prepare_step(step)
                    step_input = value
                    t0 = time.perf_counter()
                    value = step.run(step_input)
                    name = getattr(step, '_func_name', type(step).__name__)
                    history.append((name, value))
                    if on_step is not None:
                        _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
            execution_time = time.perf_counter() - start_time
            return ExecutionResult(
                value=value,
                history=tuple(history),
                dt=execution_time,
                n=len(self.steps),
            )
        finally:
            self._teardown_setup_once_nodes()

    async def async_run_detailed(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> ExecutionResult[R]:
        history = []
        value = seed
        start_time = time.perf_counter()
        try:
            with activate_context(self._context):
                for step in self.steps:
                    self._prepare_step(step)
                    step_input = value
                    t0 = time.perf_counter()
                    value = await self._await_step(step, step_input)
                    name = getattr(step, '_func_name', type(step).__name__)
                    history.append((name, value))
                    if on_step is not None:
                        _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
            execution_time = time.perf_counter() - start_time
            return ExecutionResult(
                value=value,
                history=tuple(history),
                dt=execution_time,
                n=len(self.steps),
            )
        finally:
            self._teardown_setup_once_nodes()

    def map(
        self,
        items: Iterable[Any],
        *,
        on_step: Optional[StepHook] = None,
        parallel: Union[bool, int] = False,
    ) -> List[Any]:
        """Apply pipeline to each item in a collection.

        When *parallel* is ``True``, use the shared thread pool. When an
        ``int``, cap concurrency with a dedicated bounded pool.
        """
        items_list = list(items)
        if not parallel:
            try:
                return [
                    self.run(item, on_step=on_step, _finalize_setup_once=False)
                    for item in items_list
                ]
            finally:
                self._teardown_setup_once_nodes()

        run_item = wrap_worker(
            lambda item: self.run(
                item, on_step=on_step, _finalize_setup_once=False,
            ),
        )
        try:
            if parallel is True:
                pool = _get_pool('thread')
                return list(pool.map(run_item, items_list))

            max_workers = int(parallel)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                return list(pool.map(run_item, items_list))
        finally:
            self._teardown_setup_once_nodes()

    async def async_map(
        self,
        items: Iterable[Any],
        *,
        on_step: Optional[StepHook] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """Apply pipeline to each item asynchronously."""
        return await gather_limited(
            (self.async_run(item, on_step=on_step) for item in items),
            max_concurrency=max_concurrency,
        )

    def run_async(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None,
    ) -> R:
        """Run the pipeline asynchronously using uvloop when available."""
        from .async_runtime import run_async as _run_async
        return _run_async(self.async_run(seed, on_step=on_step))

    def map_async(
        self,
        items: Iterable[Any],
        *,
        on_step: Optional[StepHook] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """Apply pipeline to each item via the async runtime (uvloop when available)."""
        from .async_runtime import run_async as _run_async
        return _run_async(
            self.async_map(items, on_step=on_step, max_concurrency=max_concurrency),
        )

    @classmethod
    def from_spec(
        cls,
        spec_file: str,
        *,
        registry: Optional[dict] = None,
    ) -> 'Pipeline':
        """Build a pipeline from a YAML or JSON spec file."""
        from .spec import load_pipeline_from_spec

        return load_pipeline_from_spec(spec_file, registry)
