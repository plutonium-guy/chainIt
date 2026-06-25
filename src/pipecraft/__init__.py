"""Composable function pipeline framework for Python."""

# Runtime type-checking: beartype decorates every function/method in this
# package via its import hook. Installed before submodules are imported so they
# are hooked as they load. Degrades to a no-op if beartype is unavailable.
#
# - is_pep484_tower=True accepts int where float is annotated (Python's implicit
#   numeric tower), matching how callers pass delay=0, timeout=5, etc.
# - PEP 585 deprecation warnings (typing.List vs list) are silenced; the hints
#   still work on supported Python versions.
try:
    import warnings as _warnings

    from beartype import BeartypeConf as _BeartypeConf
    from beartype.claw import beartype_this_package as _beartype_this_package
    from beartype.roar import (
        BeartypeDecorHintPep585DeprecationWarning as _Pep585DeprecationWarning,
    )

    _warnings.filterwarnings("ignore", category=_Pep585DeprecationWarning)
    _beartype_this_package(conf=_BeartypeConf(is_pep484_tower=True))
except ImportError:  # pragma: no cover - beartype is a declared dependency
    pass

from .async_runtime import (
    HAS_RSLOOP,
    install_rsloop,
    rsloop_policy,
    run_async,
    uninstall_rsloop,
)
from .branching import ConditionalStep, SwitchStep
from .constants import HAS_NUMPY, PIPE
from .context import get_context
from .decorators import circuit_breaker, node, piped, retry
from .exceptions import (
    CircuitBreakerError,
    GraphCycleError,
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
    # Utilities
    'cleanup_pools', 'configure_pools', 'HAS_NUMPY', 'HAS_RSLOOP', 'HAS_FREE_THREADING',
    'is_gil_enabled', 'threads_provide_true_parallelism',
    'run_async', 'install_rsloop', 'uninstall_rsloop', 'rsloop_policy',
]
