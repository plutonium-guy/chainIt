from __future__ import annotations

import abc
import asyncio
import functools
import inspect
import itertools
import logging
import pickle
import time
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import get_context
from typing import (
    Any, Callable, Dict, Generic, Iterable, List,
    Optional, Sequence, Set, Tuple, TypeVar, Union,
)

# NumPy is optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

# Constants
PIPE: object = object()
T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------
_POOLS: Dict[str, Union[ThreadPoolExecutor, ProcessPoolExecutor]] = {}


def _get_pool(kind: str) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
    """Get or create a thread/process pool."""
    if kind not in _POOLS:
        if kind == "process":
            ctx = get_context("spawn" if sys.platform in ("darwin", "win32") else "fork")
            _POOLS[kind] = ProcessPoolExecutor(max_workers=None, mp_context=ctx)
        else:
            _POOLS[kind] = ThreadPoolExecutor(max_workers=None)
    return _POOLS[kind]


def cleanup_pools():
    """Clean up thread/process pools."""
    for pool in _POOLS.values():
        pool.shutdown(wait=True)
    _POOLS.clear()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Base pipeline exception with context."""

    def __init__(self, func_name: str, original_error: Exception):
        self.func_name = func_name
        self.original_error = original_error
        super().__init__(f"Pipeline step '{func_name}' failed: {original_error}")


class RetryExhaustedError(PipelineError):
    """Raised when retry attempts are exhausted."""

    def __str__(self):
        return f"RetryExhausted: {super().__str__()}"


class CircuitBreakerError(PipelineError):
    """Raised when circuit breaker is open."""
    pass


class GraphCycleError(Exception):
    """Raised when a cycle is detected in the DAG."""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    errors: Tuple[type, ...] = (Exception,)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    threshold: int = 5
    timeout: float = 60.0


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _is_pickleable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _get_func_name(func: Callable) -> str:
    return getattr(func, '__name__', getattr(func, 'func_name', str(func)))


def _run_branch_value(branch: Any, value: Any) -> Any:
    """Run a (possibly None) branch synchronously on a value.

    Shared stateless helper used by ConditionalStep and SwitchStep. A None
    branch passes the value through unchanged; a branch with a `run` method is
    invoked via it; otherwise the branch is called directly.
    """
    if branch is None:
        return value
    if hasattr(branch, 'run'):
        return branch.run(value)
    return branch(value)


async def _async_run_branch_value(branch: Any, value: Any) -> Any:
    """Run a (possibly None) branch asynchronously on a value.

    Shared stateless helper used by ConditionalStep and SwitchStep.
    """
    if branch is None:
        return value
    if hasattr(branch, 'async_run'):
        return await branch.async_run(value)
    if hasattr(branch, 'run'):
        return branch.run(value)
    result = branch(value)
    if asyncio.iscoroutine(result):
        return await result
    return result


def _is_iterable_collection(obj: Any) -> bool:
    """Check if obj is a list/tuple/ndarray (not str/bytes/dict)."""
    if isinstance(obj, (list, tuple)):
        return True
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return True
    return False


# ---------------------------------------------------------------------------
# PipeStep
# ---------------------------------------------------------------------------
@dataclass
class PipeStep(Generic[T, R]):
    """Pipeline step wrapping a callable with execution options."""

    func: Callable[..., R]
    _args: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs: Dict[str, Any] = field(default_factory=dict)

    # Performance
    batch_size: int = 1
    parallel: Optional[str] = None  # 'thread' or 'process'

    # Reliability
    retry_config: Optional[RetryConfig] = None
    circuit_config: Optional[CircuitBreakerConfig] = None

    # Extra
    timeout: Optional[float] = None
    schema: Optional[type] = None

    # Internal state
    _circuit_state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _func_name: str = field(default="", init=False)
    _is_async: bool = field(default=False, init=False)
    _is_pickleable: bool = field(default=True, init=False)
    _signature: inspect.Signature = field(default=None, init=False)

    def __post_init__(self):
        object.__setattr__(self, '_func_name', _get_func_name(self.func))
        object.__setattr__(self, '_is_async', inspect.iscoroutinefunction(self.func))
        object.__setattr__(
            self, '_is_pickleable',
            _is_pickleable(self.func) if self.parallel == 'process' else True,
        )
        try:
            object.__setattr__(self, '_signature', inspect.signature(self.func))
        except (ValueError, TypeError):
            object.__setattr__(self, '_signature', None)

        if self.parallel == 'process' and not self._is_pickleable:
            logger.warning(
                f"Function {self._func_name} not pickleable, falling back to threads"
            )
            object.__setattr__(self, 'parallel', 'thread')

    def __repr__(self) -> str:
        parts = [self._func_name]
        if self.parallel:
            parts.append(f"parallel={self.parallel!r}")
        if self.batch_size > 1:
            parts.append(f"batch={self.batch_size}")
        if self.retry_config:
            parts.append(f"retry={self.retry_config.attempts}")
        if self.circuit_config:
            parts.append(f"circuit={self.circuit_config.threshold}")
        if self.timeout:
            parts.append(f"timeout={self.timeout}")
        if self._is_async:
            parts.append("async")
        return f"PipeStep({', '.join(parts)})"

    # -- Argument preparation ------------------------------------------------

    def _prepare_args(self, input_value: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        args = list(self._args)
        kwargs = dict(self._kwargs)

        if PIPE in args:
            args = [input_value if arg is PIPE else arg for arg in args]
        elif PIPE in kwargs.values():
            kwargs = {k: (input_value if v is PIPE else v) for k, v in kwargs.items()}
        elif input_value is not PIPE and not args and not kwargs:
            if self._signature:
                params = list(self._signature.parameters.values())
                if params and any(
                    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
                    for p in params
                ):
                    args = [input_value]
            else:
                args = [input_value]

        return tuple(args), kwargs

    # -- Partial application -------------------------------------------------

    def __call__(self, *args, **kwargs) -> 'PipeStep[T, R]':
        merged_kwargs = dict(self._kwargs)
        merged_kwargs.update(kwargs)
        for k, v in self._kwargs.items():
            if v is PIPE and k not in kwargs:
                merged_kwargs[k] = PIPE
        merged_args = args if args else self._args

        return PipeStep(
            func=self.func,
            _args=merged_args,
            _kwargs=merged_kwargs,
            batch_size=self.batch_size,
            parallel=self.parallel,
            retry_config=self.retry_config,
            circuit_config=self.circuit_config,
            timeout=self.timeout,
            schema=self.schema,
        )

    # -- Execution -----------------------------------------------------------

    def run(self, input_value: Any = PIPE) -> R:
        return self._execute_sync(input_value)

    async def async_run(self, input_value: Any = PIPE) -> R:
        return await self._execute_async(input_value)

    def _preflight_check(self) -> None:
        """Shared pre-flight: reject if circuit is open or execution cancelled.

        Identical for the sync and async execution paths.
        """
        if self._check_circuit_breaker():
            raise CircuitBreakerError(self._func_name, Exception("Circuit breaker is open"))

        cancel_event = getattr(self, '_cancel_event', None)
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

    def _record_and_raise(self, last_exception: BaseException) -> None:
        """Shared terminal step after the retry loop exhausts.

        Records exactly ONE circuit-breaker failure for the logical call (not one
        per retry attempt) and raises the appropriate error. Identical for the
        sync and async execution paths.
        """
        self._record_failure()
        if self.retry_config and isinstance(last_exception, self.retry_config.errors):
            raise RetryExhaustedError(self._func_name, last_exception)
        raise PipelineError(self._func_name, last_exception)

    def _execute_sync(self, input_value: Any) -> Any:
        self._preflight_check()

        last_exception = None
        delay = self.retry_config.delay if self.retry_config else 0

        for attempt in range(self.retry_config.attempts if self.retry_config else 1):
            try:
                if self.timeout is not None:
                    # Submit to the shared cached thread pool instead of spinning
                    # up a throwaway one. No `with` block: that would shut down the
                    # shared pool. If the timeout fires, fut.result(timeout=...)
                    # raises concurrent.futures.TimeoutError exactly as before and
                    # the worker thread keeps running in the background either way.
                    pool = _get_pool('thread')
                    fut = pool.submit(self._invoke_function, input_value)
                    result = fut.result(timeout=self.timeout)
                else:
                    result = self._invoke_function(input_value)

                if asyncio.iscoroutine(result):
                    raise RuntimeError(
                        f"Cannot await coroutine {self._func_name} in synchronous context"
                    )

                self._reset_circuit_breaker()
                if self.schema and not isinstance(result, self.schema):
                    raise TypeError(
                        f"Output of {self._func_name} does not match schema {self.schema}"
                    )
                return result

            except Exception as e:
                last_exception = e
                if (
                    self.retry_config
                    and attempt < self.retry_config.attempts - 1
                    and isinstance(e, self.retry_config.errors)
                ):
                    if delay > 0:
                        time.sleep(delay)
                        delay *= self.retry_config.backoff
                    continue
                break

        self._record_and_raise(last_exception)

    async def _execute_async(self, input_value: Any) -> Any:
        self._preflight_check()

        last_exception = None
        delay = self.retry_config.delay if self.retry_config else 0

        for attempt in range(self.retry_config.attempts if self.retry_config else 1):
            try:
                coro = self._invoke_function_async(input_value)
                if self.timeout is not None:
                    result = await asyncio.wait_for(coro, timeout=self.timeout)
                else:
                    result = await coro

                self._reset_circuit_breaker()
                if self.schema and not isinstance(result, self.schema):
                    raise TypeError(
                        f"Output of {self._func_name} does not match schema {self.schema}"
                    )
                return result

            except Exception as e:
                last_exception = e
                if (
                    self.retry_config
                    and attempt < self.retry_config.attempts - 1
                    and isinstance(e, self.retry_config.errors)
                ):
                    if delay > 0:
                        await asyncio.sleep(delay)  # properly awaited
                        delay *= self.retry_config.backoff
                    continue
                break

        self._record_and_raise(last_exception)

    def _invoke_function(self, input_value: Any) -> Any:
        args, kwargs = self._prepare_args(input_value)
        # Precedence is deterministic and documented: when both `parallel` and
        # `batch_size > 1` are configured, `parallel` wins (each item is mapped
        # individually across the pool). Batching only runs when `parallel` is
        # not set, so the two modes never silently interfere.
        if self.parallel and self._should_parallelize(args):
            return self._execute_parallel(args, kwargs)
        if self.batch_size > 1 and self._should_batch(args):
            return self._execute_batched(args, kwargs)
        return self.func(*args, **kwargs)

    async def _invoke_function_async(self, input_value: Any) -> Any:
        args, kwargs = self._prepare_args(input_value)
        # Same documented precedence as the sync path: `parallel` wins over
        # `batch_size` when both are configured.
        if self.parallel and self._should_parallelize(args):
            return await self._execute_parallel_async(args, kwargs)
        if self.batch_size > 1 and self._should_batch(args):
            return self._execute_batched(args, kwargs)
        if self._is_async:
            return await self.func(*args, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(self.func, *args, **kwargs))

    # -- Parallel / batch helpers -------------------------------------------

    def _should_parallelize(self, args: Tuple[Any, ...]) -> bool:
        return len(args) == 1 and _is_iterable_collection(args[0]) and len(args[0]) > 1

    def _should_batch(self, args: Tuple[Any, ...]) -> bool:
        return len(args) == 1 and _is_iterable_collection(args[0]) and len(args[0]) > self.batch_size

    def _execute_parallel(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        pool = _get_pool(self.parallel)
        # Forward any partially-applied kwargs to each call (consistent with
        # _execute_batched). functools.partial keeps the callable picklable for
        # process pools.
        call = functools.partial(self.func, **kwargs) if kwargs else self.func
        results = list(pool.map(call, items))
        return results[0] if len(results) == 1 else results

    async def _execute_parallel_async(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        pool = _get_pool(self.parallel)
        loop = asyncio.get_running_loop()
        # Forward any partially-applied kwargs to each call (see _execute_parallel).
        call = functools.partial(self.func, **kwargs) if kwargs else self.func
        tasks = [loop.run_in_executor(pool, call, item) for item in items]
        results = list(await asyncio.gather(*tasks))
        return results[0] if len(results) == 1 else results

    def _execute_batched(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            result = self.func(batch, **kwargs)
            results.append(result)
        if not results:
            return []
        if isinstance(results[0], (list, tuple)):
            return [item for sublist in results for item in sublist]
        # Genuine numeric scalars are reduced with sum() across batches (e.g. a
        # per-batch sum/mean reducer). `bool` is intentionally excluded even
        # though it subclasses int -- summing booleans across batches is a
        # surprising footgun, so boolean per-batch results are returned as a
        # list for the caller to reduce explicitly. numpy scalar types are
        # recognised as numeric when numpy is available, consistent with the
        # rest of the module.
        numeric_types: Tuple[type, ...] = (int, float)
        if HAS_NUMPY:
            numeric_types = (int, float, np.number)
        if isinstance(results[0], numeric_types) and not isinstance(results[0], bool):
            return sum(results)
        return results

    # -- Circuit breaker ----------------------------------------------------

    def _check_circuit_breaker(self) -> bool:
        if not self.circuit_config:
            return False
        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.circuit_config.timeout:
                object.__setattr__(self, '_circuit_state', CircuitState.HALF_OPEN)
                object.__setattr__(self, '_failure_count', 0)
                return False
            return True
        return False

    def _record_failure(self):
        if not self.circuit_config:
            return
        object.__setattr__(self, '_failure_count', self._failure_count + 1)
        object.__setattr__(self, '_last_failure_time', time.time())
        if self._failure_count >= self.circuit_config.threshold:
            object.__setattr__(self, '_circuit_state', CircuitState.OPEN)

    def _reset_circuit_breaker(self):
        if self.circuit_config:
            object.__setattr__(self, '_circuit_state', CircuitState.CLOSED)
            object.__setattr__(self, '_failure_count', 0)

    def __or__(self, other) -> 'Pipeline[T, R]':
        return Pipeline([self]) | other


# ---------------------------------------------------------------------------
# OOP Node base class
# ---------------------------------------------------------------------------
class Node(abc.ABC):
    """Abstract base class for OOP-style pipeline steps.

    Subclass and implement `process()` to create reusable pipeline components
    with setup/teardown lifecycle and internal state.

    Usage::

        class Doubler(Node):
            def process(self, x):
                return x * 2

        class Adder(Node):
            def __init__(self, n):
                self.n = n

            def process(self, x):
                return x + self.n

        pipeline = Doubler() | Adder(10)
        pipeline.run(5)  # (5*2) + 10 = 20
    """

    def setup(self) -> None:
        """Called once before first execution. Override for initialization."""
        pass

    def teardown(self) -> None:
        """Called after execution completes. Override for cleanup."""
        pass

    @abc.abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Core processing logic. Must be implemented by subclasses."""
        ...

    def run(self, input_value: Any = PIPE) -> Any:
        self.setup()
        try:
            if input_value is PIPE:
                return self.process()
            return self.process(input_value)
        finally:
            self.teardown()

    async def async_run(self, input_value: Any = PIPE) -> Any:
        self.setup()
        try:
            if input_value is PIPE:
                result = self.process()
            else:
                result = self.process(input_value)
            if asyncio.iscoroutine(result):
                return await result
            return result
        finally:
            self.teardown()

    @property
    def _func_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __or__(self, other) -> 'Pipeline':
        return Pipeline([self]) | other


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult(Generic[R]):
    value: R
    history: Tuple[Tuple[str, Any], ...]
    dt: float
    n: int

    @property
    def execution_time(self) -> float:
        return self.dt

    @property
    def step_count(self) -> int:
        return self.n


# ---------------------------------------------------------------------------
# PipelineBuilder
# ---------------------------------------------------------------------------
class PipelineBuilder(Generic[T]):
    """Fluent builder for Pipeline construction."""

    def __init__(self):
        self._steps: List[Any] = []

    def add(self, step) -> 'PipelineBuilder[T]':
        self._steps.append(step)
        return self

    def build(self) -> 'Pipeline':
        return Pipeline(self._steps)


# ---------------------------------------------------------------------------
# ConditionalStep
# ---------------------------------------------------------------------------
@dataclass
class ConditionalStep(Generic[T, R]):
    """Route data through different branches based on a condition.

    Usage::

        cond = ConditionalStep(
            condition=lambda x: x > 0,
            if_true=double,
            if_false=negate,
        )
        cond.run(5)   # 10
        cond.run(-3)  # 3
    """

    condition: Callable[[T], bool]
    if_true: Any  # PipeStep, Node, Pipeline, or callable
    if_false: Any = None

    def run(self, value: T) -> R:
        if self.condition(value):
            return _run_branch_value(self.if_true, value)
        return _run_branch_value(self.if_false, value)

    async def async_run(self, value: T) -> R:
        if self.condition(value):
            return await _async_run_branch_value(self.if_true, value)
        return await _async_run_branch_value(self.if_false, value)

    @property
    def _func_name(self) -> str:
        return f"ConditionalStep({_get_func_name(self.condition)})"

    def __or__(self, other) -> 'Pipeline':
        return Pipeline([self]) | other


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline(Generic[T, R]):
    """Linear pipeline with advanced execution modes."""

    __slots__ = ('steps', '_cancel_event')

    def __init__(self, steps: Sequence[Any]):
        self.steps = tuple(steps)
        self._cancel_event = None

    def __or__(self, other) -> 'Pipeline[T, R]':
        if isinstance(other, Pipeline):
            return Pipeline([*self.steps, *other.steps])
        return Pipeline([*self.steps, other])

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

    def run(self, seed: Any = None) -> R:
        value = seed
        for step in self.steps:
            if self._cancel_event is not None and hasattr(step, '_cancel_event'):
                object.__setattr__(step, '_cancel_event', self._cancel_event)
            if self._is_cancelled():
                raise asyncio.CancelledError()
            value = step.run(value)
        return value

    async def async_run(self, seed: Any = None) -> R:
        value = seed
        for step in self.steps:
            if self._cancel_event is not None and hasattr(step, '_cancel_event'):
                object.__setattr__(step, '_cancel_event', self._cancel_event)
            if self._is_cancelled():
                raise asyncio.CancelledError()
            if hasattr(step, 'async_run'):
                value = await step.async_run(value)
            else:
                value = step.run(value)
        return value

    def run_detailed(self, seed: Any = None) -> ExecutionResult[R]:
        history = []
        value = seed
        start_time = time.perf_counter()
        for step in self.steps:
            if self._cancel_event is not None and hasattr(step, '_cancel_event'):
                object.__setattr__(step, '_cancel_event', self._cancel_event)
            if self._is_cancelled():
                raise asyncio.CancelledError()
            value = step.run(value)
            name = getattr(step, '_func_name', type(step).__name__)
            history.append((name, value))
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

    @classmethod
    def from_spec(cls, spec_file: str) -> 'Pipeline':
        raise NotImplementedError("from_spec not implemented in this version")


# ---------------------------------------------------------------------------
# Fan-out / Fan-in / MapReduce
# ---------------------------------------------------------------------------
def _run_branch_on(branch: Any, value: Any) -> Any:
    """Run a single branch on a value.

    Module-level (picklable) helper so FanOutStep can use a ProcessPoolExecutor.
    """
    return branch.run(value)


@dataclass
class FanOutStep(Generic[T, R]):
    """Execute multiple branches in parallel."""

    branches: Tuple[Any, ...]
    parallel: Optional[str] = None

    def run(self, value: T) -> Tuple[R, ...]:
        if self.parallel:
            pool = _get_pool(self.parallel)
            # Use a module-level picklable callable (not a lambda) so this works
            # with ProcessPoolExecutor (parallel='process') as well as threads.
            return tuple(
                pool.map(_run_branch_on, self.branches, itertools.repeat(value))
            )
        return tuple(branch.run(value) for branch in self.branches)

    async def async_run(self, value: T) -> Tuple[R, ...]:
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

    def __or__(self, other) -> Pipeline:
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

    def __or__(self, other) -> Pipeline:
        return Pipeline([self]) | other


@dataclass
class MapReduceStep(Generic[T, R]):
    """Map-reduce with optional batching."""

    mapper: Callable[[T], Any]
    reducer: Callable[[Iterable[Any]], R]
    batch_size: int = 1

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
                mapped = [self.mapper(x) for x in batch]
                results.extend(
                    [await r if asyncio.iscoroutine(r) else r for r in mapped]
                )
                batch.clear()
        if batch:
            mapped = [self.mapper(x) for x in batch]
            results.extend(
                [await r if asyncio.iscoroutine(r) else r for r in mapped]
            )
        out = self.reducer(results)
        if asyncio.iscoroutine(out):
            return await out
        return out

    @property
    def _func_name(self) -> str:
        return "MapReduceStep"

    def __or__(self, other) -> Pipeline:
        return Pipeline([self]) | other


# ---------------------------------------------------------------------------
# SwitchStep — multi-branch routing
# ---------------------------------------------------------------------------
@dataclass
class SwitchStep(Generic[T, R]):
    """Route data through one of many branches based on a key function.

    Usage::

        switch = SwitchStep(
            key=lambda x: "pos" if x > 0 else "neg",
            branches={
                "pos": double_step,
                "neg": negate_step,
            },
            default=identity_step,  # optional fallback
        )
        switch.run(5)   # double_step(5)
        switch.run(-3)  # negate_step(-3)
    """

    key: Callable[[T], str]
    branches: Dict[str, Any]
    default: Any = None

    def _get_branch(self, value: T) -> Any:
        k = self.key(value)
        return self.branches.get(k, self.default)

    def run(self, value: T) -> R:
        return _run_branch_value(self._get_branch(value), value)

    async def async_run(self, value: T) -> R:
        return await _async_run_branch_value(self._get_branch(value), value)

    @property
    def _func_name(self) -> str:
        return f"SwitchStep({_get_func_name(self.key)})"

    def __or__(self, other) -> 'Pipeline':
        return Pipeline([self]) | other


# ---------------------------------------------------------------------------
# Graph (DAG) execution
# ---------------------------------------------------------------------------
class Graph:
    """DAG-based pipeline for complex dependency graphs.

    Independent nodes at the same depth level execute concurrently in async mode.

    Usage::

        g = Graph()
        g.add_node("fetch", fetch_step)
        g.add_node("parse", parse_step)
        g.add_node("validate", validate_step)
        g.add_node("save", save_step)

        g.add_edge("fetch", "parse")
        g.add_edge("parse", "validate")
        g.add_edge("parse", "save")
        g.add_edge("validate", "save")

        results = g.run(seed=raw_data)
        print(results["save"])
    """

    def __init__(self):
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, Set[str]] = {}
        self._reverse: Dict[str, Set[str]] = {}

    def add_node(self, name: str, step: Any) -> 'Graph':
        self._nodes[name] = step
        self._edges.setdefault(name, set())
        self._reverse.setdefault(name, set())
        return self

    def add_edge(self, from_node: str, to_node: str) -> 'Graph':
        if from_node not in self._nodes:
            raise KeyError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise KeyError(f"Node '{to_node}' not found")
        self._edges[from_node].add(to_node)
        self._reverse.setdefault(to_node, set()).add(from_node)
        return self

    @property
    def roots(self) -> List[str]:
        """Nodes with no parents (in-degree 0)."""
        return sorted(n for n in self._nodes if not self._reverse.get(n))

    @property
    def leaves(self) -> List[str]:
        """Nodes with no children (out-degree 0)."""
        return sorted(n for n in self._nodes if not self._edges.get(n))

    def _topo_sort(self) -> List[str]:
        """Topological sort via Kahn's algorithm."""
        in_degree = {n: len(self._reverse.get(n, set())) for n in self._nodes}
        queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in sorted(self._edges.get(node, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._nodes):
            raise GraphCycleError("Cycle detected in graph")
        return order

    def _topo_levels(self) -> List[List[str]]:
        """Group nodes by dependency level for parallel execution."""
        order = self._topo_sort()
        depth: Dict[str, int] = {}
        for name in order:
            parents = self._reverse.get(name, set())
            if not parents:
                depth[name] = 0
            else:
                depth[name] = max(depth[p] for p in parents) + 1
        max_depth = max(depth.values()) if depth else 0
        levels: List[List[str]] = [[] for _ in range(max_depth + 1)]
        for name in order:
            levels[depth[name]].append(name)
        return levels

    def _run_node(self, name: str, input_value: Any) -> Any:
        step = self._nodes[name]
        if hasattr(step, 'run'):
            return step.run(input_value)
        return step(input_value)

    async def _async_run_node(self, name: str, input_value: Any) -> Any:
        step = self._nodes[name]
        if hasattr(step, 'async_run'):
            return await step.async_run(input_value)
        if hasattr(step, 'run'):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, step.run, input_value)
        result = step(input_value)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _get_node_input(self, name: str, results: Dict[str, Any], seed: Any) -> Any:
        parents = self._reverse.get(name, set())
        if not parents:
            return seed
        if len(parents) == 1:
            return results[next(iter(parents))]
        return tuple(results[p] for p in sorted(parents))

    def run(self, seed: Any = None, parallel: bool = False) -> Dict[str, Any]:
        """Execute graph synchronously.

        Args:
            seed: Initial input for root nodes.
            parallel: Execute independent nodes at same level concurrently (threads).
        """
        results: Dict[str, Any] = {}

        if parallel:
            pool = _get_pool('thread')
            for level in self._topo_levels():
                if len(level) == 1:
                    name = level[0]
                    results[name] = self._run_node(
                        name, self._get_node_input(name, results, seed)
                    )
                else:
                    futures = {}
                    for name in level:
                        inp = self._get_node_input(name, results, seed)
                        futures[name] = pool.submit(self._run_node, name, inp)
                    for name, fut in futures.items():
                        results[name] = fut.result()
        else:
            for name in self._topo_sort():
                results[name] = self._run_node(
                    name, self._get_node_input(name, results, seed)
                )

        return results

    async def async_run(self, seed: Any = None, parallel: bool = True) -> Dict[str, Any]:
        """Execute graph asynchronously.

        Independent nodes at same dependency level run concurrently by default.
        """
        results: Dict[str, Any] = {}

        if parallel:
            for level in self._topo_levels():
                if len(level) == 1:
                    name = level[0]
                    results[name] = await self._async_run_node(
                        name, self._get_node_input(name, results, seed)
                    )
                else:
                    tasks = {}
                    for name in level:
                        inp = self._get_node_input(name, results, seed)
                        tasks[name] = asyncio.create_task(
                            self._async_run_node(name, inp)
                        )
                    for name, task in tasks.items():
                        results[name] = await task
        else:
            for name in self._topo_sort():
                results[name] = await self._async_run_node(
                    name, self._get_node_input(name, results, seed)
                )

        return results

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = sum(len(children) for children in self._edges.values())
        return f"Graph(nodes={n}, edges={e})"

    def describe(self) -> str:
        """Text visualization of graph structure."""
        lines = []
        order = self._topo_sort()
        for name in order:
            children = sorted(self._edges.get(name, set()))
            if children:
                lines.append(f"  {name} -> {', '.join(children)}")
            else:
                lines.append(f"  {name} (leaf)")
        return "Graph:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def piped(
    func: Optional[Callable] = None,
    *,
    batch_size: int = 1,
    parallel: Optional[str] = None,
    timeout: Optional[float] = None,
    schema: Optional[type] = None,
    jit: bool = False,
    vectorize: bool = False,
) -> Union[PipeStep, Callable[[Callable], PipeStep]]:
    """Create a PipeStep from a function.

    Args:
        func: Function to wrap
        batch_size: Batch size for processing
        parallel: 'thread' or 'process'
        timeout: Per-step timeout in seconds
        schema: Expected output type
        jit: Apply Numba JIT (requires numba)
        vectorize: Apply NumPy vectorization (requires numba or numpy)
    """

    def decorator(f: Callable) -> PipeStep:
        optimized = f

        if jit:
            try:
                import numba
                optimized = numba.njit(fastmath=True, cache=True, nogil=True)(optimized)
            except ImportError:
                pass  # graceful degradation

        if vectorize:
            try:
                import numba
                optimized = numba.vectorize(
                    ['float64(float64)', 'float32(float32)', 'int64(int64)'],
                    nopython=True, cache=True,
                )(optimized)
            except ImportError:
                if HAS_NUMPY:
                    optimized = np.vectorize(optimized, cache=True)

        return PipeStep(
            func=optimized,
            batch_size=batch_size,
            parallel=parallel,
            timeout=timeout,
            schema=schema,
        )

    return decorator(func) if func else decorator


def retry(
    *,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    errors: Tuple[type, ...] = (Exception,),
) -> Callable[[PipeStep], PipeStep]:
    """Add retry capability to a pipeline step."""

    def decorator(step: PipeStep) -> PipeStep:
        step.retry_config = RetryConfig(
            attempts=max_attempts, delay=delay, backoff=backoff, errors=errors
        )
        return step

    return decorator


def circuit_breaker(
    *,
    threshold: int = 5,
    timeout: float = 60.0,
    failure_threshold: int = None,
    recovery_timeout: float = None,
) -> Callable[[PipeStep], PipeStep]:
    """Add circuit breaker capability to a pipeline step."""
    if failure_threshold is not None:
        threshold = failure_threshold
    if recovery_timeout is not None:
        timeout = recovery_timeout

    def decorator(step: PipeStep) -> PipeStep:
        step.circuit_config = CircuitBreakerConfig(threshold=threshold, timeout=timeout)
        return step

    return decorator


def node(func: Optional[Callable] = None, *, setup: Optional[Callable] = None,
         teardown: Optional[Callable] = None) -> Union[Node, Callable[[Callable], Node]]:
    """Decorator to create a Node from a function.

    Usage::

        @node
        def double(x):
            return x * 2

        @node(setup=lambda self: print("init"), teardown=lambda self: print("done"))
        def process(x):
            return x + 1

        pipeline = double | process
        pipeline.run(5)  # 11
    """

    def decorator(f: Callable) -> Node:
        class FuncNode(Node):
            def process(self, *args, **kwargs):
                return f(*args, **kwargs)

        if setup:
            FuncNode.setup = setup
        if teardown:
            FuncNode.teardown = teardown

        inst = FuncNode()
        # Preserve original function metadata
        inst.__class__.__name__ = _get_func_name(f)
        inst.__class__.__qualname__ = _get_func_name(f)
        return inst

    return decorator(func) if func else decorator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Core
    'PIPE', 'piped', 'retry', 'circuit_breaker',
    'PipeStep', 'Pipeline', 'PipelineBuilder',
    # OOP
    'Node', 'node',
    # Graph / DAG
    'Graph', 'ConditionalStep', 'SwitchStep',
    # Fan-out/in
    'FanOutStep', 'FanInStep', 'MapReduceStep',
    # Results
    'ExecutionResult',
    # Errors
    'PipelineError', 'RetryExhaustedError', 'CircuitBreakerError', 'GraphCycleError',
    # Utilities
    'cleanup_pools', 'HAS_NUMPY',
]
