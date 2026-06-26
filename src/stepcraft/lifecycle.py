from __future__ import annotations

from typing import Any, Awaitable, Callable


class StepLifecycle:
    """Shared setup/teardown lifecycle for OOP pipeline steps."""

    setup_once: bool = False

    def setup(self) -> None:
        """Called before execution. Override for initialization."""

    def teardown(self) -> None:
        """Called after execution completes. Override for cleanup."""

    def _ensure_setup(self) -> None:
        if self.setup_once:
            if not getattr(self, '_setup_active', False):
                self.setup()
                self._setup_active = True
        else:
            self.setup()

    def _teardown_once(self) -> None:
        if getattr(self, '_setup_active', False):
            self.teardown()
            self._setup_active = False

    def _run_lifecycle(
        self,
        invoke: Callable[[Any], Any],
        input_value: Any,
    ) -> Any:
        self._ensure_setup()
        try:
            result = invoke(input_value)
            if not self.setup_once:
                self.teardown()
            return result
        except Exception:
            if not self.setup_once:
                self.teardown()
            raise

    async def _run_lifecycle_async(
        self,
        invoke: Callable[[Any], Awaitable[Any]],
        input_value: Any,
    ) -> Any:
        self._ensure_setup()
        try:
            result = await invoke(input_value)
            if not self.setup_once:
                self.teardown()
            return result
        except Exception:
            if not self.setup_once:
                self.teardown()
            raise
