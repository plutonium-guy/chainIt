from __future__ import annotations


class PipelineError(Exception):
    """Base pipeline exception with context."""

    def __init__(self, func_name: str, original_error: Exception):
        self.func_name = func_name
        self.original_error = original_error
        super().__init__(f"Pipeline step '{func_name}' failed: {original_error}")


class RetryExhaustedError(PipelineError):
    """Raised when retry attempts are exhausted."""

    def __str__(self):
        return f"RetryExhausted: {super().__str__()}"


class CircuitBreakerError(PipelineError):
    """Raised when circuit breaker is open."""
    pass


class GraphCycleError(Exception):
    """Raised when a cycle is detected in the DAG."""
    pass


class MissingAnnotationError(TypeError):
    """Raised when a @piped/@node callable or Node is not fully annotated.

    stepcraft requires type annotations on every parameter and the return value
    by default. Pass ``require_annotations=False`` (or set it on a Node subclass)
    to opt a specific step out.
    """

    def __init__(self, name: str, missing: list[str]):
        self.name = name
        self.missing = missing
        super().__init__(
            f"{name} is missing type annotations for: {', '.join(missing)}. "
            f"stepcraft requires fully annotated steps by default; "
            f"pass require_annotations=False to opt out."
        )
