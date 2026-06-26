from __future__ import annotations

import abc
import asyncio
import inspect
from typing import Any

from .constants import PIPE
from .execution import default_runner
from .lifecycle import StepLifecycle


class Node(StepLifecycle, abc.ABC):
    """Abstract base class for OOP-style pipeline steps."""

    require_annotations: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        process = cls.__dict__.get("process")
        if process is None or not getattr(cls, "require_annotations", True):
            return
        from .typecheck import assert_fully_annotated

        assert_fully_annotated(process, name=f"{cls.__name__}.process")

    @abc.abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Core processing logic. Must be implemented by subclasses."""
        ...

    def run(self, input_value: Any = PIPE) -> Any:
        return self._run_lifecycle(self._invoke_process, input_value)

    async def async_run(self, input_value: Any = PIPE) -> Any:
        return await self._run_lifecycle_async(self._invoke_process_async, input_value)

    def _invoke_process(self, input_value: Any) -> Any:
        if input_value is PIPE:
            return self.process()
        return self.process(input_value)

    async def _invoke_process_async(self, input_value: Any) -> Any:
        if inspect.iscoroutinefunction(self.process):
            if input_value is PIPE:
                return await self.process()
            return await self.process(input_value)
        result = await default_runner(self._invoke_process, input_value)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @property
    def _func_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
