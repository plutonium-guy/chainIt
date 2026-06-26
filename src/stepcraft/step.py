from __future__ import annotations

import asyncio
import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple

from .async_concurrency import gather_limited
from .config import RetryConfig
from .constants import HAS_NUMPY, PIPE, R, T, get_numpy, logger
from .context import wrap_worker
from .exceptions import PipelineError, RetryExhaustedError
from .execution import (
    CircuitBreaker,
    ExecutionMode,
    ExecutionPlan,
    SyncPoolRunner,
)
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
    circuit_config: Optional[Any] = None

    # Extra
    timeout: Optional[float] = None
    cancel_on_timeout: bool = False
    schema: Optional[type] = None

    # Internal state
    _func_name: str = field(default="", init=False)
    _is_async: bool = field(default=False, init=False)
    _is_pickleable: bool = field(default=True, init=False)
    _signature: inspect.Signature = field(default=None, init=False)
    _breaker: CircuitBreaker = field(init=False, repr=False)
    _pool_runner: SyncPoolRunner = field(default_factory=SyncPoolRunner, init=False, repr=False)

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
        object.__setattr__(
            self,
            '_breaker',
            CircuitBreaker(self.circuit_config, self._func_name),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_breaker' not in self.__dict__:
            object.__setattr__(
                self,
                '_breaker',
                CircuitBreaker(self.circuit_config, self._func_name),
            )

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
            max_concurrency=self.max_concurrency,
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

    def _is_cancelled(self) -> bool:
        cancel_event = getattr(self, '_cancel_event', None)
        return cancel_event is not None and cancel_event.is_set()

    def _preflight_check(self) -> None:
        self._breaker.preflight(cancelled=self._is_cancelled())

    def _record_and_raise(self, last_exception: BaseException) -> None:
        self._breaker.record_failure()
        if self.retry_config and isinstance(last_exception, self.retry_config.errors):
            raise RetryExhaustedError(self._func_name, last_exception)
        raise PipelineError(self._func_name, last_exception)

    def _finalize_success(self, result: Any, *, sync: bool, mapped: bool = False) -> Any:
        return self._breaker.finalize_success(
            result, schema=self.schema, mapped=mapped, sync=sync,
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

    def _plan_execution(self, input_value: Any) -> ExecutionPlan:
        """Resolve dispatch mode once for sync and async paths."""
        args, kwargs = self._prepare_args(input_value)
        if self.parallel and self._should_parallelize(args):
            mode = ExecutionMode.PARALLEL
        elif self.batch_size > 1 and self._should_batch(args):
            mode = ExecutionMode.BATCH
        elif self._should_auto_map(args):
            mode = ExecutionMode.AUTO_MAP
        else:
            mode = ExecutionMode.DIRECT
        return ExecutionPlan(mode, args, kwargs)

    def _invoke_function(self, input_value: Any) -> Tuple[Any, bool]:
        plan = self._plan_execution(input_value)
        executors = {
            ExecutionMode.PARALLEL: lambda: self._execute_parallel(plan.args, plan.kwargs),
            ExecutionMode.BATCH: lambda: self._execute_batched(plan.args, plan.kwargs),
            ExecutionMode.AUTO_MAP: lambda: self._execute_auto_map(plan.args, plan.kwargs),
            ExecutionMode.DIRECT: lambda: self.func(*plan.args, **plan.kwargs),
        }
        return executors[plan.mode](), plan.mapped

    async def _invoke_function_async(self, input_value: Any) -> Tuple[Any, bool]:
        plan = self._plan_execution(input_value)
        if plan.mode is ExecutionMode.PARALLEL:
            result = await self._execute_parallel_async(plan.args, plan.kwargs)
        elif plan.mode is ExecutionMode.BATCH:
            result = await self._execute_batched_async(plan.args, plan.kwargs)
        elif plan.mode is ExecutionMode.AUTO_MAP:
            result = await self._execute_auto_map_async(plan.args, plan.kwargs)
        elif self._is_async:
            result = await self.func(*plan.args, **plan.kwargs)
        else:
            result = await self._pool_runner(self.func, *plan.args, **plan.kwargs)
        return result, plan.mapped

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
        if self.max_concurrency is not None:
            return self._pool_runner.map_bounded(
                call, items, max_workers=self.max_concurrency,
            )
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
        pool = _get_pool('thread')
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(pool, call, item) for item in items]
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
                result = await self._pool_runner(self.func, batch, **kwargs)
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

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
