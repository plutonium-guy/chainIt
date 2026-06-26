from __future__ import annotations

import asyncio
import functools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

from .config import CircuitBreakerConfig, CircuitState
from .context import wrap_worker
from .exceptions import CircuitBreakerError
from .pools import _get_pool


class ExecutionMode(StrEnum):
    """How a PipeStep dispatches a single invocation."""

    PARALLEL = "parallel"
    BATCH = "batch"
    AUTO_MAP = "auto_map"
    DIRECT = "direct"


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Resolved dispatch mode for one PipeStep call."""

    mode: ExecutionMode
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @property
    def mapped(self) -> bool:
        return self.mode in (ExecutionMode.PARALLEL, ExecutionMode.AUTO_MAP)


@runtime_checkable
class Runnable(Protocol):
    """Any step that can be executed synchronously or asynchronously."""

    def run(self, input_value: Any = ...) -> Any: ...

    async def async_run(self, input_value: Any = ...) -> Any: ...


class SyncPoolRunner:
    """Offload sync callables from the event loop via the shared thread pool."""

    def __init__(self, pool_kind: str = "thread") -> None:
        self._pool_kind = pool_kind

    async def __call__(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        pool = _get_pool(self._pool_kind)
        if args or kwargs:
            fn = functools.partial(fn, *args, **kwargs)
        worker = wrap_worker(fn)
        return await loop.run_in_executor(pool, worker)

    def map_bounded(
        self,
        fn: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        max_workers: int,
    ) -> List[Any]:
        """Sync-side bounded parallelism (for max_concurrency on auto-map)."""
        wrapped = wrap_worker(fn)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(wrapped, items))


default_runner = SyncPoolRunner()


class StepRunner:
    """Run any :class:`Runnable`, offloading sync work via :class:`SyncPoolRunner`."""

    def __init__(self, pool: SyncPoolRunner | None = None) -> None:
        self._pool = pool or default_runner

    async def run_async(self, step: Any, value: Any) -> Any:
        if hasattr(step, "async_run"):
            return await step.async_run(value)
        if hasattr(step, "run"):
            return await self._pool(step.run, value)
        result = step(value)
        if asyncio.iscoroutine(result):
            return await result
        return result


@dataclass
class CircuitBreaker:
    """Circuit breaker state and lifecycle for a single step."""

    config: Optional[CircuitBreakerConfig]
    func_name: str
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    half_open_calls: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        object.__setattr__(self, "_lock", threading.Lock())

    def preflight(self, *, cancelled: bool) -> None:
        """Check cancel and breaker state before executing a call."""
        if cancelled:
            raise asyncio.CancelledError()
        if self._must_short_circuit():
            raise CircuitBreakerError(
                self.func_name, Exception("Circuit breaker is open"),
            )

    def record_failure(self) -> None:
        if not self.config:
            return
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if (
                self.state is CircuitState.HALF_OPEN
                or self.failure_count >= self.config.threshold
            ):
                self.state = CircuitState.OPEN
                self.half_open_calls = 0

    def finalize_success(
        self,
        result: Any,
        *,
        schema: Optional[type],
        mapped: bool,
        sync: bool,
    ) -> Any:
        """Validate output, then reset breaker on success."""
        if sync and asyncio.iscoroutine(result):
            raise RuntimeError(
                f"Cannot await coroutine {self.func_name} in synchronous context"
            )
        self._validate_schema(result, schema=schema, mapped=mapped)
        self._reset()
        return result

    def _validate_schema(
        self,
        result: Any,
        *,
        schema: Optional[type],
        mapped: bool,
    ) -> None:
        if not schema:
            return
        values = result if mapped and isinstance(result, list) else (result,)
        for value in values:
            if not isinstance(value, schema):
                raise TypeError(
                    f"Output of {self.func_name} does not match schema {schema}"
                )

    def _must_short_circuit(self) -> bool:
        if not self.config:
            return False

        with self._lock:
            if self.state is CircuitState.OPEN:
                if time.time() - self.last_failure_time >= self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                    self.half_open_calls = 0
                else:
                    return True

            if self.state is CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    return True
                self.half_open_calls += 1
                return False

            return False

    def _reset(self) -> None:
        if not self.config:
            return
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0


default_step_runner = StepRunner()
