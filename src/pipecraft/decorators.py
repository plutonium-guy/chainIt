from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

from .config import CircuitBreakerConfig, RetryConfig
from .constants import HAS_NUMPY, logger, np
from .node import Node
from .step import PipeStep
from .utils import _get_func_name


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
    """Create a PipeStep from a function."""

    def decorator(f: Callable) -> PipeStep:
        optimized = f

        if jit:
            try:
                import numba
                optimized = numba.njit(fastmath=True, cache=True, nogil=True)(optimized)
            except ImportError:
                logger.warning(
                    "jit=True on %s but numba is not installed; running without JIT. "
                    "Install with: pip install pipecraft[numba]",
                    _get_func_name(f),
                )

        if vectorize:
            vectorized = False
            try:
                import numba
                optimized = numba.vectorize(
                    ['float64(float64)', 'float32(float32)', 'int64(int64)'],
                    nopython=True, cache=True,
                )(optimized)
                vectorized = True
            except ImportError:
                if HAS_NUMPY:
                    optimized = np.vectorize(optimized, cache=True)
                    vectorized = True
            if not vectorized:
                logger.warning(
                    "vectorize=True on %s but neither numba nor numpy is available; "
                    "running without vectorization. "
                    "Install with: pip install pipecraft[numba] or pipecraft[numpy]",
                    _get_func_name(f),
                )

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


def node(
    func: Optional[Callable] = None,
    *,
    setup: Optional[Callable] = None,
    teardown: Optional[Callable] = None,
) -> Union[Node, Callable[[Callable], Node]]:
    """Decorator to create a Node from a function."""

    def decorator(f: Callable) -> Node:
        class FuncNode(Node):
            def process(self, *args, **kwargs):
                return f(*args, **kwargs)

        if setup:
            FuncNode.setup = setup
        if teardown:
            FuncNode.teardown = teardown

        inst = FuncNode()
        inst.__class__.__name__ = _get_func_name(f)
        inst.__class__.__qualname__ = _get_func_name(f)
        return inst

    return decorator(func) if func else decorator
