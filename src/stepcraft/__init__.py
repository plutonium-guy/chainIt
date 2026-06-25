"""Composable function pipeline framework for Python."""

from .async_runtime import (
    HAS_UVLOOP,
    install_uvloop,
    run_async,
    uninstall_uvloop,
    uvloop_policy,
)
from .branching import ConditionalStep, SwitchStep
from .constants import HAS_NUMPY, PIPE, get_numpy
from .context import get_context
from .decorators import circuit_breaker, node, piped, retry
from .exceptions import (
    CircuitBreakerError,
    GraphCycleError,
    MissingAnnotationError,
    PipelineError,
    RetryExhaustedError,
)
from .fan import FanInStep, FanOutStep, MapReduceStep
from .graph import Graph
from .hooks import StepHook
from .node import Node
from .pipeline import Pipeline, PipelineBuilder
from .pools import _POOLS, cleanup_pools, configure_pools
from .result import ExecutionResult
from .runtime import (
    HAS_FREE_THREADING,
    is_gil_enabled,
    threads_provide_true_parallelism,
)
from .step import PipeStep
from .typecheck import apply_step_beartype, beartype_enabled

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
    # Observability
    'StepHook',
    # Shared context
    'get_context',
    # Errors
    'PipelineError', 'RetryExhaustedError', 'CircuitBreakerError', 'GraphCycleError',
    'MissingAnnotationError',
    # Utilities
    'cleanup_pools', 'configure_pools', 'HAS_NUMPY', 'get_numpy', 'HAS_UVLOOP', 'HAS_FREE_THREADING',
    'is_gil_enabled', 'threads_provide_true_parallelism',
    'run_async', 'install_uvloop', 'uninstall_uvloop', 'uvloop_policy',
    'apply_step_beartype', 'beartype_enabled',
]
