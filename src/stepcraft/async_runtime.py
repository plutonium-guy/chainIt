from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any, Coroutine, Generator, TypeVar

T = TypeVar("T")

try:
    import uvloop

    HAS_UVLOOP = True
except ImportError:  # uvloop is unavailable on Windows / some platforms
    uvloop = None  # type: ignore[assignment]
    HAS_UVLOOP = False


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine with uvloop when available, otherwise stdlib asyncio."""
    if HAS_UVLOOP:
        return uvloop.run(coro)
    return asyncio.run(coro)


def install_uvloop() -> None:
    """Install uvloop as the default asyncio event loop policy."""
    if not HAS_UVLOOP:
        raise ImportError(
            "uvloop is not installed (it is unavailable on Windows). "
            "Install with: pip install stepcraft[uvloop]"
        )
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def uninstall_uvloop() -> None:
    """Restore the default asyncio event loop policy."""
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


@contextmanager
def uvloop_policy() -> Generator[None, None, None]:
    """Context manager that temporarily installs uvloop as the loop policy."""
    install_uvloop()
    try:
        yield
    finally:
        uninstall_uvloop()
