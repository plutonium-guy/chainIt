"""Composable function pipeline framework for Python."""

from .async_runtime import (
    HAS_RSLOOP,
    install_rsloop,
    rsloop_policy,
    run_async,
    uninstall_rsloop,
)
from .branching import ConditionalStep, SwitchStep
from .constants import HAS_NUMPY, PIPE
from .decorators import circuit_breaker, node, piped, retry
from .exceptions import (
    CircuitBreakerError,
    GraphCycleError,
    PipelineError,
    RetryExhaustedError,
)
from .fan import FanInStep, FanOutStep, MapReduceStep
from .graph import Graph
from .node import Node
from .pipeline import Pipeline, PipelineBuilder
from .pools import _POOLS, cleanup_pools
from .result import ExecutionResult
from .step import PipeStep

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
    'cleanup_pools', 'HAS_NUMPY', 'HAS_RSLOOP',
    'run_async', 'install_rsloop', 'uninstall_rsloop', 'rsloop_policy',
]
