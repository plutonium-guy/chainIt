from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic

from .constants import R, T
from .utils import _async_run_branch_value, _get_func_name, _run_branch_value


@dataclass
class ConditionalStep(Generic[T, R]):
    """Route data through different branches based on a condition."""

    condition: Callable[[T], bool]
    if_true: Any
    if_false: Any = None

    def run(self, value: T) -> R:
        if self.condition(value):
            return _run_branch_value(self.if_true, value)
        return _run_branch_value(self.if_false, value)

    async def async_run(self, value: T) -> R:
        if self.condition(value):
            return await _async_run_branch_value(self.if_true, value)
        return await _async_run_branch_value(self.if_false, value)

    @property
    def _func_name(self) -> str:
        return f"ConditionalStep({_get_func_name(self.condition)})"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other


@dataclass
class SwitchStep(Generic[T, R]):
    """Route data through one of many branches based on a key function."""

    key: Callable[[T], str]
    branches: Dict[str, Any]
    default: Any = None

    def _get_branch(self, value: T) -> Any:
        k = self.key(value)
        return self.branches.get(k, self.default)

    def run(self, value: T) -> R:
        return _run_branch_value(self._get_branch(value), value)

    async def async_run(self, value: T) -> R:
        return await _async_run_branch_value(self._get_branch(value), value)

    @property
    def _func_name(self) -> str:
        return f"SwitchStep({_get_func_name(self.key)})"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
