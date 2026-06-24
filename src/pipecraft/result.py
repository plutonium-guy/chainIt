from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Tuple, TypeVar

from .constants import R

T = TypeVar("T")


@dataclass
class ExecutionResult(Generic[R]):
    value: R
    history: Tuple[Tuple[str, Any], ...]
    dt: float
    n: int

    @property
    def execution_time(self) -> float:
        return self.dt

    @property
    def step_count(self) -> int:
        return self.n
