from __future__ import annotations

import logging
from typing import TypeVar

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

PIPE: object = object()
T = TypeVar("T")
R = TypeVar("R")
logger = logging.getLogger(__name__)
