from __future__ import annotations

import asyncio
import pickle
from typing import Any, Callable

from .constants import HAS_NUMPY, np


def _is_pickleable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _get_func_name(func: Callable) -> str:
    return getattr(func, '__name__', getattr(func, 'func_name', str(func)))


def _run_branch_value(branch: Any, value: Any) -> Any:
    """Run a (possibly None) branch synchronously on a value."""
    if branch is None:
        return value
    if hasattr(branch, 'run'):
        return branch.run(value)
    return branch(value)


async def _async_run_branch_value(branch: Any, value: Any) -> Any:
    """Run a (possibly None) branch asynchronously on a value."""
    if branch is None:
        return value
    if hasattr(branch, 'async_run'):
        return await branch.async_run(value)
    if hasattr(branch, 'run'):
        result = branch.run(value)
        if asyncio.iscoroutine(result):
            return await result
        return result
    result = branch(value)
    if asyncio.iscoroutine(result):
        return await result
    return result


def _is_iterable_collection(obj: Any) -> bool:
    """Check if obj is a list/tuple/ndarray (not str/bytes/dict)."""
    if isinstance(obj, (list, tuple)):
        return True
    if HAS_NUMPY and isinstance(obj, np.ndarray):
        return True
    return False


def _is_auto_map_collection(obj: Any) -> bool:
    """List inputs that default @piped steps map over element-wise."""
    return isinstance(obj, list)
