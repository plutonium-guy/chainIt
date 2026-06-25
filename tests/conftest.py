"""Pytest hooks for stepcraft test suite."""

import os
import sys

import pytest

from stepcraft.pools import cleanup_pools

_EXIT_STATUS = 0


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Release executors (terminating process workers) once tests finish."""
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)
    cleanup_pools(wait=False)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Hard-exit after the summary so a leaked thread/process cannot stall.

    Even with all tests passing, the process can hang at interpreter exit on
    Linux: leaked spawn-based ProcessPoolExecutor workers and native-extension
    runtime threads (e.g. rsloop's) are not always joined, leaving a CI step
    hanging for hours after a green run. Running last — after the terminal
    summary is printed — we flush and hard-exit with the pytest status code.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
