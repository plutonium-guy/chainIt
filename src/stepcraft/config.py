from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    errors: Tuple[type, ...] = (Exception,)


@dataclass(frozen=True)
class CircuitBreakerConfig:
    threshold: int = 5
    timeout: float = 60.0
    half_open_max_calls: int = 1


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2
