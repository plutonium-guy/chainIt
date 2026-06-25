"""Resilience tests: retry, circuit breaker, pool configuration."""

import asyncio
import time

import pytest

from conftest import Counter, MockException, Pipeline, PipelineError, circuit_breaker, piped, retry


def test_circuit_breaker_decorated_copies_are_independent():
    @piped
    def faulty(x):
        raise MockException("fail")

    a = circuit_breaker(failure_threshold=2)(faulty)
    b = circuit_breaker(failure_threshold=2)(faulty)
    assert a is not b

    with pytest.raises(PipelineError):
        a.run(1)
    with pytest.raises(PipelineError):
        a.run(2)
    with pytest.raises(PipelineError):
        a.run(3)

    with pytest.raises(PipelineError):
        b.run(1)


def test_retry_success():
    counter = Counter()

    @retry(max_attempts=3)
    @piped
    def flaky(x):
        counter.increment()
        if counter.count < 3:
            raise MockException("Flaky")
        return x * 2

    result = flaky.run(5)
    assert result == 10
    assert counter.count == 3


def test_retry_exhaustion():
    counter = Counter()

    @retry(max_attempts=3)
    @piped
    def always_fails(x):
        counter.increment()
        raise MockException("Always fails")

    with pytest.raises(PipelineError) as exc_info:
        always_fails.run(5)
    assert "RetryExhausted" in str(exc_info.value)
    assert counter.count == 3


def test_circuit_breaker_trip():
    counter = Counter()

    @circuit_breaker(failure_threshold=2)
    @piped
    def faulty(x):
        counter.increment()
        raise MockException("Faulty")

    with pytest.raises(PipelineError):
        faulty.run(1)
    with pytest.raises(PipelineError):
        faulty.run(2)
    with pytest.raises(PipelineError):
        faulty.run(3)
    assert counter.count == 2


def test_circuit_breaker_recovery():
    counter = Counter()

    @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
    @piped
    def sometimes_fails(x):
        counter.increment()
        if x % 2 == 0:
            raise MockException("Failed on even")
        return x * 2

    with pytest.raises(PipelineError):
        sometimes_fails.run(2)
    with pytest.raises(PipelineError):
        sometimes_fails.run(4)
    with pytest.raises(PipelineError):
        sometimes_fails.run(6)

    time.sleep(0.15)
    result = sometimes_fails.run(3)
    assert result == 6
    result = sometimes_fails.run(5)
    assert result == 10


def test_circuit_breaker_records_one_failure_per_logical_call():
    counter = Counter()

    @circuit_breaker(failure_threshold=2)
    @retry(max_attempts=3, delay=0)
    @piped
    def flaky(x):
        counter.increment()
        raise MockException("always fails")

    with pytest.raises(PipelineError):
        flaky.run(1)
    with pytest.raises(PipelineError):
        flaky.run(2)
    with pytest.raises(PipelineError):
        flaky.run(3)
    assert counter.count == 6


def test_circuit_breaker_half_open_single_probe():
    counter = Counter()

    @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
    @piped
    def always_fails(x):
        counter.increment()
        raise MockException("fail")

    for i in range(2):
        with pytest.raises(PipelineError):
            always_fails.run(i)
    assert counter.count == 2

    with pytest.raises(PipelineError):
        always_fails.run(99)
    assert counter.count == 2

    time.sleep(0.15)

    with pytest.raises(PipelineError):
        always_fails.run(1)
    assert counter.count == 3

    with pytest.raises(PipelineError):
        always_fails.run(2)
    assert counter.count == 3


def test_circuit_breaker_half_open_probe_success_closes():
    counter = Counter()

    @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
    @piped
    def sometimes(x):
        counter.increment()
        if x < 0:
            raise MockException("neg")
        return x * 2

    for x in (-1, -2):
        with pytest.raises(PipelineError):
            sometimes.run(x)

    time.sleep(0.15)
    assert sometimes.run(5) == 10
    assert sometimes.run(7) == 14


def test_circuit_breaker_half_open_cancel_preserves_probe():
    counter = Counter()

    @circuit_breaker(failure_threshold=1, recovery_timeout=0.1)
    @piped
    def sometimes(x):
        counter.increment()
        if x < 0:
            raise MockException("fail")
        return x * 2

    with pytest.raises(PipelineError):
        sometimes.run(-1)
    assert counter.count == 1

    time.sleep(0.15)

    cancelled = Pipeline([sometimes])
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        cancelled.run(5)
    assert counter.count == 1

    assert Pipeline([sometimes]).run(5) == 10
    assert counter.count == 2


def test_circuit_breaker_schema_failure_reopens_half_open():
    counter = Counter()

    @circuit_breaker(failure_threshold=1, recovery_timeout=0.1)
    @piped(schema=int)
    def probe(x):
        counter.increment()
        if x < 0:
            raise MockException("fail")
        return "wrong-type"

    with pytest.raises(PipelineError):
        probe.run(-1)
    assert counter.count == 1

    time.sleep(0.15)

    with pytest.raises(PipelineError):
        probe.run(5)
    assert counter.count == 2

    with pytest.raises(PipelineError):
        probe.run(99)
    assert counter.count == 2


def test_configure_pools_respects_max_workers():
    from pipeline import configure_pools, cleanup_pools
    from stepcraft.pools import _get_pool

    cleanup_pools()
    configure_pools(thread_workers=2)
    try:
        pool = _get_pool('thread')
        assert pool._max_workers == 2
    finally:
        cleanup_pools()
        configure_pools(thread_workers=None, process_workers=None)
