from __future__ import annotations

import abc
import asyncio
from typing import Any

from .constants import PIPE


class Node(abc.ABC):
    """Abstract base class for OOP-style pipeline steps."""

    # Require every subclass that defines `process` to fully annotate it.
    # Set ``require_annotations = False`` on a subclass to opt it out.
    require_annotations: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        process = cls.__dict__.get("process")
        if process is None or not getattr(cls, "require_annotations", True):
            return
        from .typecheck import assert_fully_annotated

        assert_fully_annotated(process, name=f"{cls.__name__}.process")

    def setup(self) -> None:
        """Called once before first execution. Override for initialization."""
        pass

    def teardown(self) -> None:
        """Called after execution completes. Override for cleanup."""
        pass

    @abc.abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Core processing logic. Must be implemented by subclasses."""
        ...

    def run(self, input_value: Any = PIPE) -> Any:
        self.setup()
        try:
            if input_value is PIPE:
                return self.process()
            return self.process(input_value)
        finally:
            self.teardown()

    async def async_run(self, input_value: Any = PIPE) -> Any:
        self.setup()
        try:
            if input_value is PIPE:
                result = self.process()
            else:
                result = self.process(input_value)
            if asyncio.iscoroutine(result):
                return await result
            return result
        finally:
            self.teardown()

    @property
    def _func_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __or__(self, other):
        from .pipeline import Pipeline
        return Pipeline([self]) | other
