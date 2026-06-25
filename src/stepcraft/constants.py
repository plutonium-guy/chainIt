from __future__ import annotations

import logging
from typing import Any, TypeVar

PIPE: object = object()
T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)

_np = None
_np_checked = False


def get_numpy() -> Any:
    """Return the numpy module when installed, else None (lazy import)."""
    global _np, _np_checked
    if not _np_checked:
        _np_checked = True
        try:
            import numpy as numpy_mod

            _np = numpy_mod
        except ImportError:
            _np = None
    return _np


class _HasNumpy:
    """Lazy truthiness check for numpy availability."""

    def __bool__(self) -> bool:
        return get_numpy() is not None


class _NumpyProxy:
    """Lazy proxy so ``from constants import np`` does not import numpy eagerly."""

    def __getattr__(self, name: str) -> Any:
        mod = get_numpy()
        if mod is None:
            raise ImportError(
                "numpy is not installed; install with: pip install stepcraft[numpy]"
            )
        return getattr(mod, name)


HAS_NUMPY = _HasNumpy()
np = _NumpyProxy()
