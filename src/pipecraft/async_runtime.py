from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Coroutine, Generator, TypeVar

T = TypeVar("T")

try:
    import rsloop

    HAS_RSLOOP = True
except ImportError:
    rsloop = None  # type: ignore[assignment]
    HAS_RSLOOP = False


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine with rsloop when installed, otherwise stdlib asyncio."""
    if HAS_RSLOOP:
        return rsloop.run(coro)
    return asyncio.run(coro)


def install_rsloop() -> None:
    """Install rsloop as the default asyncio event loop policy."""
    if not HAS_RSLOOP:
        raise ImportError(
            "rsloop is not installed. Install with: pip install pipecraft[rsloop]"
        )
    rsloop.install()


def uninstall_rsloop() -> None:
    """Restore the previous asyncio event loop policy."""
    if not HAS_RSLOOP:
        raise ImportError(
            "rsloop is not installed. Install with: pip install pipecraft[rsloop]"
        )
    rsloop.uninstall()


@contextmanager
def rsloop_policy() -> Generator[None, None, None]:
    """Context manager that temporarily installs rsloop as the event loop policy."""
    install_rsloop()
    try:
        yield
    finally:
        uninstall_rsloop()
