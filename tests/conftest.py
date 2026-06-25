"""Shared fixtures and test helpers for the stepcraft test suite."""

import pytest
import asyncio
from pathlib import Path
from pipeline import (
    PIPE, Pipeline, piped, retry, circuit_breaker,
    FanOutStep, FanInStep, PipelineError, ExecutionResult,
    PipelineBuilder, MapReduceStep, Node, node, ConditionalStep,
    SwitchStep, Graph, GraphCycleError, MissingAnnotationError,
    HAS_UVLOOP, run_async, install_uvloop, uninstall_uvloop, uvloop_policy,
)

# stepcraft requires full type annotations on @piped/@node/Node by default.
# These pre-existing tests exercise *other* behaviour with unannotated
# functions, so we centrally opt them out here.
strict_piped, strict_node, StrictNode = piped, node, Node


def piped(func=None, **kwargs):  # noqa: F811 - intentional test-wide shim
    kwargs.setdefault("require_annotations", False)
    return strict_piped(func, **kwargs) if func is not None else strict_piped(**kwargs)


def node(func=None, **kwargs):  # noqa: F811 - intentional test-wide shim
    kwargs.setdefault("require_annotations", False)
    return strict_node(func, **kwargs) if func is not None else strict_node(**kwargs)


class Node(StrictNode):  # noqa: F811 - intentional test-wide shim
    require_annotations = False


try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


FIXTURES = Path(__file__).parent / "fixtures"


class MockException(Exception):
    pass


class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count
