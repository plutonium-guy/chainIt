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
