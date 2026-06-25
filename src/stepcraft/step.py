from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple

from .async_concurrency import gather_limited
from .config import CircuitBreakerConfig, CircuitState, RetryConfig
from .context import wrap_worker
from .constants import HAS_NUMPY, PIPE, R, T, get_numpy, logger
from .exceptions import CircuitBreakerError, PipelineError, RetryExhaustedError
from .pools import _get_pool
from .runtime import resolve_parallel_kind
from .utils import (
    _get_func_name,
    _is_auto_map_collection,
    _is_iterable_collection,
    _is_pickleable,
)


@dataclass
class PipeStep(Generic[T, R]):
    """Pipeline step wrapping a callable with execution options."""

    func: Callable[..., R]
    _args: Tuple[Any, ...] = field(default_factory=tuple)
    _kwargs: Dict[str, Any] = field(default_factory=dict)

    # Performance
    batch_size: int = 1
    parallel: Optional[str] = None  # 'thread', 'process', or 'auto'
    auto_map: bool = True
    max_concurrency: Optional[int] = None

    # Reliability
    retry_config: Optional[RetryConfig] = None
    circuit_config: Optional[CircuitBreakerConfig] = None

    # Extra
    timeout: Optional[float] = None
    cancel_on_timeout: bool = False
    schema: Optional[type] = None

    # Internal state
    _circuit_state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _func_name: str = field(default="", init=False)
    _is_async: bool = field(default=False, init=False)
    _is_pickleable: bool = field(default=True, init=False)
    _signature: inspect.Signature = field(default=None, init=False)

    def __post_init__(self):
        object.__setattr__(self, '_func_name', _get_func_name(self.func))
        object.__setattr__(self, '_is_async', inspect.iscoroutinefunction(self.func))
        parallel = resolve_parallel_kind(self.parallel)
        if parallel != self.parallel:
            object.__setattr__(self, 'parallel', parallel)
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
        object.__setattr__(self, '_breaker_lock', threading.Lock())

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_breaker_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        object.__setattr__(self, '_breaker_lock', threading.Lock())

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

    def __call__(self, *args, **kwargs) -> 'PipeStep':
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
            auto_map=self.auto_map,
            retry_config=self.retry_config,
            circuit_config=self.circuit_config,
            timeout=self.timeout,
            cancel_on_timeout=self.cancel_on_timeout,
            schema=self.schema,
        )

    def copy(self, **overrides: Any) -> 'PipeStep':
        """Return a new step with fresh breaker state (for decorators / reuse)."""
        fields = {
            'func': self.func,
            '_args': self._args,
            '_kwargs': self._kwargs,
            'batch_size': self.batch_size,
            'parallel': self.parallel,
            'auto_map': self.auto_map,
            'max_concurrency': self.max_concurrency,
            'retry_config': self.retry_config,
            'circuit_config': self.circuit_config,
            'timeout': self.timeout,
            'cancel_on_timeout': self.cancel_on_timeout,
            'schema': self.schema,
        }
        fields.update(overrides)
        return PipeStep(**fields)

    def run(self, input_value: Any = PIPE) -> R:
        return self._execute_sync(input_value)

    async def async_run(self, input_value: Any = PIPE) -> R:
        return await self._execute_async(input_value)

    def _preflight_check(self) -> None:
        cancel_event = getattr(self, '_cancel_event', None)
        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError()

        if self._check_circuit_breaker():
            raise CircuitBreakerError(self._func_name, Exception("Circuit breaker is open"))

    def _record_and_raise(self, last_exception: BaseException) -> None:
        self._record_failure()
        if self.retry_config and isinstance(last_exception, self.retry_config.errors):
            raise RetryExhaustedError(self._func_name, last_exception)
        raise PipelineError(self._func_name, last_exception)

    def _finalize_success(self, result: Any, *, sync: bool, mapped: bool = False) -> Any:
        if sync and asyncio.iscoroutine(result):
            raise RuntimeError(
                f"Cannot await coroutine {self._func_name} in synchronous context"
            )
        self._validate_schema(result, mapped=mapped)
        self._reset_circuit_breaker()
        return result

    def _validate_schema(self, result: Any, *, mapped: bool) -> None:
        """Validate the step output against an explicit ``schema=``.

        When the step mapped over a collection (auto-map / parallel), the
        schema describes each element, so validate them individually rather
        than the aggregated list.
        """
        if not self.schema:
            return
        values = result if mapped and isinstance(result, list) else (result,)
        for value in values:
            if not isinstance(value, self.schema):
                raise TypeError(
                    f"Output of {self._func_name} does not match schema {self.schema}"
                )

    def _should_retry(self, attempt: int, error: Exception) -> bool:
        return (
            self.retry_config is not None
            and attempt < self.retry_config.attempts - 1
            and isinstance(error, self.retry_config.errors)
        )

    def _execute_sync(self, input_value: Any) -> Any:
        self._preflight_check()

        last_exception = None
        delay = self.retry_config.delay if self.retry_config else 0

        for attempt in range(self.retry_config.attempts if self.retry_config else 1):
            try:
                if self.timeout is not None:
                    pool = _get_pool('thread')
                    fut = pool.submit(
                        self._wrap_worker(self._invoke_function), input_value,
                    )
                    try:
                        result, mapped = fut.result(timeout=self.timeout)
                    except Exception:
                        # Best-effort: cancel only succeeds if the worker has
                        # not started; a running thread cannot be interrupted.
                        if self.cancel_on_timeout:
                            fut.cancel()
                        if not fut.done():
                            logger.warning(
                                "%s timed out after %ss; the worker thread may "
                                "still be running",
                                self._func_name, self.timeout,
                            )
                        raise
                else:
                    result, mapped = self._invoke_function(input_value)

                return self._finalize_success(result, sync=True, mapped=mapped)

            except Exception as e:
                last_exception = e
                if self._should_retry(attempt, e):
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
                    result, mapped = await asyncio.wait_for(coro, timeout=self.timeout)
                else:
                    result, mapped = await coro

                return self._finalize_success(result, sync=False, mapped=mapped)

            except Exception as e:
                last_exception = e
                if self._should_retry(attempt, e):
                    if delay > 0:
                        await asyncio.sleep(delay)
                        delay *= self.retry_config.backoff
                    continue
                break

        self._record_and_raise(last_exception)

    def _invoke_function(self, input_value: Any) -> Tuple[Any, bool]:
        args, kwargs = self._prepare_args(input_value)
        # parallel wins over batch; batch wins over per-element auto-map.
        # `mapped` marks element-wise execution so schema is checked per item.
        if self.parallel and self._should_parallelize(args):
            result, mapped = self._execute_parallel(args, kwargs), True
        elif self.batch_size > 1 and self._should_batch(args):
            result, mapped = self._execute_batched(args, kwargs), False
        elif self._should_auto_map(args):
            result, mapped = self._execute_auto_map(args, kwargs), True
        else:
            result, mapped = self.func(*args, **kwargs), False
        return result, mapped

    async def _invoke_function_async(self, input_value: Any) -> Tuple[Any, bool]:
        args, kwargs = self._prepare_args(input_value)
        if self.parallel and self._should_parallelize(args):
            result, mapped = await self._execute_parallel_async(args, kwargs), True
        elif self.batch_size > 1 and self._should_batch(args):
            result, mapped = await self._execute_batched_async(args, kwargs), False
        elif self._should_auto_map(args):
            result, mapped = await self._execute_auto_map_async(args, kwargs), True
        elif self._is_async:
            result, mapped = await self.func(*args, **kwargs), False
        else:
            loop = asyncio.get_running_loop()
            call = self._wrap_worker(functools.partial(self.func, *args, **kwargs))
            result, mapped = await loop.run_in_executor(None, call), False
        return result, mapped

    def _should_parallelize(self, args: Tuple[Any, ...]) -> bool:
        return len(args) == 1 and _is_iterable_collection(args[0]) and len(args[0]) > 1

    def _should_batch(self, args: Tuple[Any, ...]) -> bool:
        return len(args) == 1 and _is_iterable_collection(args[0]) and len(args[0]) > self.batch_size

    def _wrap_worker(self, fn: Callable[..., Any], pool_kind: Optional[str] = None) -> Callable[..., Any]:
        return wrap_worker(fn)

    def _should_auto_map(self, args: Tuple[Any, ...]) -> bool:
        return (
            self.auto_map
            and self.parallel is None
            and self.batch_size == 1
            and len(args) == 1
            and _is_auto_map_collection(args[0])
            and len(args[0]) != 1
        )

    def _execute_auto_map(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        if not items:
            return []
        call = functools.partial(self.func, **kwargs) if kwargs else self.func
        return [call(item) for item in items]

    async def _execute_auto_map_async(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any],
    ) -> Any:
        items = args[0]
        if not items:
            return []
        if self._is_async:
            call = functools.partial(self.func, **kwargs) if kwargs else self.func
            return list(await gather_limited(
                (call(item) for item in items),
                max_concurrency=self.max_concurrency,
            ))
        call = functools.partial(self.func, **kwargs) if kwargs else self.func
        call = self._wrap_worker(call)
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(None, call, item) for item in items]
        return list(await gather_limited(tasks, max_concurrency=self.max_concurrency))

    def _execute_parallel(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        pool = _get_pool(self.parallel)
        call = functools.partial(self.func, **kwargs) if kwargs else self.func
        call = self._wrap_worker(call, self.parallel)
        results = list(pool.map(call, items))
        return results[0] if len(results) == 1 else results

    async def _execute_parallel_async(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        if self._is_async:
            call = functools.partial(self.func, **kwargs) if kwargs else self.func
            results = list(await gather_limited(
                (call(item) for item in items),
                max_concurrency=self.max_concurrency,
            ))
        else:
            pool = _get_pool(self.parallel)
            loop = asyncio.get_running_loop()
            call = functools.partial(self.func, **kwargs) if kwargs else self.func
            call = self._wrap_worker(call, self.parallel)
            tasks = [loop.run_in_executor(pool, call, item) for item in items]
            results = list(await gather_limited(
                tasks, max_concurrency=self.max_concurrency,
            ))
        return results[0] if len(results) == 1 else results

    def _execute_batched(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        items = args[0]
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            result = self.func(batch, **kwargs)
            results.append(result)
        return self._aggregate_batch_results(results)

    async def _execute_batched_async(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any],
    ) -> Any:
        items = args[0]
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            if self._is_async:
                result = await self.func(batch, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                call = self._wrap_worker(functools.partial(self.func, batch, **kwargs))
                result = await loop.run_in_executor(None, call)
            results.append(result)
        return self._aggregate_batch_results(results)

    def _aggregate_batch_results(self, results: list) -> Any:
        if not results:
            return []
        if isinstance(results[0], (list, tuple)):
            return [item for sublist in results for item in sublist]
        numeric_types: Tuple[type, ...] = (int, float)
        if HAS_NUMPY:
            np_mod = get_numpy()
            if np_mod is not None:
                numeric_types = (int, float, np_mod.number)
        if isinstance(results[0], numeric_types) and not isinstance(results[0], bool):
            return sum(results)
        return results

    def _check_circuit_breaker(self) -> bool:
        """Return True if the call must be short-circuited (breaker open)."""
        if not self.circuit_config:
            return False

        with self._breaker_lock:
            if self._circuit_state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.circuit_config.timeout:
                    # Recovery window elapsed: move to HALF_OPEN for limited probing.
                    object.__setattr__(self, '_circuit_state', CircuitState.HALF_OPEN)
                    object.__setattr__(self, '_failure_count', 0)
                    object.__setattr__(self, '_half_open_calls', 0)
                else:
                    return True

            if self._circuit_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.circuit_config.half_open_max_calls:
                    return True  # probe quota exhausted; keep blocking until verdict
                object.__setattr__(self, '_half_open_calls', self._half_open_calls + 1)
                return False

            return False

    def _record_failure(self):
        if not self.circuit_config:
            return
        with self._breaker_lock:
            object.__setattr__(self, '_failure_count', self._failure_count + 1)
            object.__setattr__(self, '_last_failure_time', time.time())
            # A failed half-open probe re-opens immediately; otherwise trip on threshold.
            if (
                self._circuit_state == CircuitState.HALF_OPEN
                or self._failure_count >= self.circuit_config.threshold
            ):
                object.__setattr__(self, '_circuit_state', CircuitState.OPEN)
                object.__setattr__(self, '_half_open_calls', 0)

    def _reset_circuit_breaker(self):
        if self.circuit_config:
            with self._breaker_lock:
                object.__setattr__(self, '_circuit_state', CircuitState.CLOSED)
                object.__setattr__(self, '_failure_count', 0)
                object.__setattr__(self, '_half_open_calls', 0)

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
