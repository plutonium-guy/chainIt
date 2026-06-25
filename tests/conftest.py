"""Pytest hooks for stepcraft test suite."""

from stepcraft.pools import cleanup_pools


def pytest_sessionfinish(session, exitstatus):
    """Release global executors so the test process can exit promptly."""
    cleanup_pools(wait=False)
