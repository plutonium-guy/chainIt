from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from typing import Any, Dict, Generic, Iterable, Iterator, List, Optional, Sequence

from .constants import R, T
from .context import _reset_context, _set_context
from .hooks import StepHook, _call_hook
from .pools import cleanup_pools
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

    @contextmanager
    def _activate_context(self) -> Iterator[None]:
        """Make ``self._context`` visible to steps via ``get_context()``."""
        if self._context is None:
            yield
            return
        token = _set_context(self._context)
        try:
            yield
        finally:
            _reset_context(token)

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

    def run(self, seed: Any = None, *, on_step: Optional[StepHook] = None) -> R:
        value = seed
        with self._activate_context():
            for step in self.steps:
                self._prepare_step(step)
                step_input = value
                t0 = time.perf_counter()
                value = step.run(step_input)
                if on_step is not None:
                    name = getattr(step, '_func_name', type(step).__name__)
                    _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
        return value

    async def async_run(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> R:
        value = seed
        with self._activate_context():
            for step in self.steps:
                self._prepare_step(step)
                step_input = value
                t0 = time.perf_counter()
                if hasattr(step, 'async_run'):
                    value = await step.async_run(step_input)
                else:
                    value = step.run(step_input)
                if on_step is not None:
                    name = getattr(step, '_func_name', type(step).__name__)
                    _call_hook(on_step, name, step_input, value, time.perf_counter() - t0)
        return value

    def run_detailed(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> ExecutionResult[R]:
        history = []
        value = seed
        start_time = time.perf_counter()
        with self._activate_context():
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

    async def async_run_detailed(
        self, seed: Any = None, *, on_step: Optional[StepHook] = None
    ) -> ExecutionResult[R]:
        history = []
        value = seed
        start_time = time.perf_counter()
        with self._activate_context():
            for step in self.steps:
                self._prepare_step(step)
                step_input = value
                t0 = time.perf_counter()
                if hasattr(step, 'async_run'):
                    value = await step.async_run(step_input)
                else:
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

    def map(self, items: Iterable[Any]) -> List[Any]:
        """Apply pipeline to each item in a collection."""
        return [self.run(item) for item in items]

    async def async_map(self, items: Iterable[Any]) -> List[Any]:
        """Apply pipeline to each item asynchronously."""
        tasks = [self.async_run(item) for item in items]
        return list(await asyncio.gather(*tasks))

    def run_async(self, seed: Any = None) -> R:
        """Run the pipeline asynchronously using rsloop when available."""
        from .async_runtime import run_async as _run_async
        return _run_async(self.async_run(seed))

    def map_async(self, items: Iterable[Any]) -> List[Any]:
        """Apply pipeline to each item via the async runtime (rsloop when available)."""
        from .async_runtime import run_async as _run_async
        return _run_async(self.async_map(items))

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
