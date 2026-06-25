import pytest
import time
import asyncio
import logging

from conftest import (
    FIXTURES,
    Counter,
    Graph,
    GraphCycleError,
    HAS_NUMPY,
    HAS_UVLOOP,
    MockException,
    MissingAnnotationError,
    Node,
    PIPE,
    Pipeline,
    PipelineBuilder,
    PipelineError,
    ExecutionResult,
    FanInStep,
    FanOutStep,
    MapReduceStep,
    ConditionalStep,
    SwitchStep,
    install_uvloop,
    node,
    piped,
    retry,
    circuit_breaker,
    run_async,
    strict_node,
    strict_piped,
    StrictNode,
    uninstall_uvloop,
    uvloop_policy,
    np,
)

# =============================================================================
# Basic Pipeline Tests
# =============================================================================

def test_basic_pipeline():
    @piped
    def add_one(x):
        return x + 1

    @piped
    def double(x):
        return x * 2

    pipeline = add_one | double
    result = pipeline.run(5)
    assert result == 12  # (5+1)*2


def test_pipe_injection():
    @piped
    def add(a, b):
        return a + b

    result = add(3, PIPE).run(5)
    assert result == 8


def test_kwarg_injection():
    @piped
    def multiply(a, b):
        return a * b

    step = multiply(a=3, b=PIPE)
    result = step.run(5)
    assert result == 15


def test_auto_map_collection_on_step():
    """AUDIT: default @piped steps map list inputs per element."""
    @piped
    def double(x):
        return x * 2

    assert double.run([1, 2, 3]) == [2, 4, 6]
    assert double.run([]) == []


def test_auto_map_does_not_split_tuples():
    """Tuples are structural values (e.g. graph fan-in), not auto-mapped."""
    @piped
    def add_pair(vals):
        return vals[0] + vals[1]

    assert add_pair.run((12, 18)) == 30


def test_auto_map_collection_in_pipeline():
    @piped
    def add_one(x):
        return x + 1

    pipeline = add_one | add_one
    assert pipeline.run([1, 2, 3]) == [3, 4, 5]


def test_auto_map_async():
    @piped
    async def double(x):
        return x * 2

    result = asyncio.run(double.async_run([1, 2, 3]))
    assert result == [2, 4, 6]


def test_auto_map_single_element_list_passes_through():
    """A one-item list is treated as a single value, not auto-mapped."""
    @piped
    def identity(x):
        return x

    assert identity.run([42]) == [42]


def test_auto_map_respects_batch_and_parallel():
    @piped(batch_size=2)
    def batch_sum(batch):
        return sum(batch)

    assert batch_sum.run([1, 2, 3, 4]) == 10

    @piped(parallel='thread')
    def slow_square(x):
        return x * x

    assert slow_square.run([2, 3]) == [4, 9]


def test_auto_map_can_be_disabled():
    @piped(auto_map=False)
    def length(xs):
        return len(xs)

    assert length.run([1, 2, 3]) == 3


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_error_propagation():
    @piped
    def fails(x):
        raise ValueError("Test error")

    pipeline = fails
    with pytest.raises(PipelineError) as exc_info:
        pipeline.run(5)
    assert "fails" in str(exc_info.value)
    assert "Test error" in str(exc_info.value)


def test_error_history():
    counter = Counter()

    @piped
    def step1(x):
        counter.increment()
        return x + 1

    @piped
    def step2(x):
        counter.increment()
        raise MockException("Failed")

    pipeline = step1 | step2
    with pytest.raises(PipelineError):
        pipeline.run(5)
    assert counter.count == 2


# =============================================================================
# JIT / Vectorize Tests (graceful without numba)
# =============================================================================

def test_jit_compilation():
    @piped(jit=True)
    def sum_squares(n):
        total = 0
        for i in range(n):
            total += i ** 2
        return total

    result = sum_squares.run(100)
    expected = sum(i ** 2 for i in range(100))
    assert result == expected


def test_jit_warns_without_numba(caplog, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.WARNING, logger="stepcraft.constants"):
        step = piped(jit=True)(lambda x: x + 1)
        assert step.run(2) == 3
    assert any("numba" in r.message.lower() for r in caplog.records)


def test_vectorize_warns_without_numba_or_numpy(caplog, monkeypatch):
    import builtins
    import stepcraft.decorators as dec

    monkeypatch.setattr(dec, "HAS_NUMPY", False)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.WARNING, logger="stepcraft.constants"):
        step = piped(vectorize=True)(lambda x: x * 2)
        assert step.run(3) == 6
    assert any("vectorize" in r.message.lower() for r in caplog.records)


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_vectorized_operations():
    @piped(vectorize=True)
    def double(x):
        return x * 2

    data = np.array([1, 2, 3, 4])
    result = double.run(data)
    assert np.array_equal(result, np.array([2, 4, 6, 8]))


# =============================================================================
# Parallel Execution Tests
# =============================================================================

def test_thread_parallel():
    numbers = list(range(5))
    sleep_s = 0.03

    @piped(parallel='thread')
    def slow_square(x):
        time.sleep(sleep_s)
        return x * x

    start = time.perf_counter()
    results = slow_square.run(numbers)
    parallel_time = time.perf_counter() - start

    assert results == [x * x for x in numbers]

    start = time.perf_counter()
    for _ in numbers:
        time.sleep(sleep_s)
    sequential_time = time.perf_counter() - start

    # Parallel should beat a sequential baseline; ratio avoids flaky absolute bounds on CI.
    assert parallel_time < sequential_time * 0.75


def test_process_parallel():
    @piped(parallel='process')
    def cpu_intensive(x):
        return sum(i * i for i in range(x))

    numbers = list(range(100, 105))
    results = cpu_intensive.run(numbers)
    assert results == [sum(i * i for i in range(n)) for n in numbers]


def test_parallel_process_uses_threads_on_free_threading(monkeypatch):
    """On 3.14+ free-threaded builds, process parallel maps to thread pools."""
    import stepcraft.runtime as runtime

    monkeypatch.setattr(runtime, "HAS_FREE_THREADING", True)
    monkeypatch.setattr(runtime, "is_gil_enabled", lambda: False)

    @piped(parallel='process')
    def square(x):
        return x * x

    assert square.parallel == 'thread'
    assert square.run([2, 3, 4]) == [4, 9, 16]


def test_parallel_auto_prefers_threads_on_free_threading(monkeypatch):
    import stepcraft.runtime as runtime

    monkeypatch.setattr(runtime, "HAS_FREE_THREADING", True)
    monkeypatch.setattr(runtime, "is_gil_enabled", lambda: False)

    @piped(parallel='auto')
    def double(x):
        return x * 2

    assert double.parallel == 'thread'


def test_parallel_auto_uses_process_with_gil(monkeypatch):
    import stepcraft.runtime as runtime

    monkeypatch.setattr(runtime, "HAS_FREE_THREADING", False)
    monkeypatch.setattr(runtime, "is_gil_enabled", lambda: True)

    step = piped(parallel='auto')(_fanout_add_one)
    assert step.parallel == 'process'


def test_free_threading_runtime_api():
    from stepcraft import (
        HAS_FREE_THREADING,
        is_gil_enabled,
        threads_provide_true_parallelism,
    )

    assert isinstance(HAS_FREE_THREADING, bool)
    assert isinstance(is_gil_enabled(), bool)
    assert isinstance(threads_provide_true_parallelism(), bool)


# =============================================================================
# Batch Processing Tests
# =============================================================================

def test_batch_processing():
    counter = Counter()

    @piped(batch_size=3)
    def batch_sum(batch):
        counter.increment()
        return sum(batch)

    data = list(range(10))
    result = batch_sum.run(data)
    assert counter.count == 4  # ceil(10/3) = 4 batches
    assert result == sum(data)


# =============================================================================
# Fan-out / Fan-in Tests
# =============================================================================

def test_fan_out_fan_in():
    @piped
    def branch1(x):
        return x * 2

    @piped
    def branch2(x):
        return x + 3

    @piped
    def combine(a, b):
        return a + b

    fan_out = FanOutStep((branch1, branch2))
    fan_in = FanInStep(combine.func)

    pipeline = fan_out | fan_in
    result = pipeline.run(5)
    assert result == 18  # (5*2) + (5+3)


def test_async_fan_out():
    @piped
    async def async_branch(x):
        await asyncio.sleep(0.01)
        return x * 2

    fan_out = FanOutStep((async_branch, async_branch))

    async def run_it():
        return await fan_out.async_run(5)

    result = asyncio.run(run_it())
    assert result == (10, 10)


# =============================================================================
# Async Pipeline Tests
# =============================================================================

def test_async_pipeline():
    @piped
    async def async_add_one(x):
        await asyncio.sleep(0.01)
        return x + 1

    @piped
    async def async_double(x):
        await asyncio.sleep(0.01)
        return x * 2

    pipeline = async_add_one | async_double
    result = asyncio.run(pipeline.async_run(5))
    assert result == 12


# =============================================================================
# Node (OOP) Tests
# =============================================================================

def test_node_basic():
    class Doubler(Node):
        def process(self, x):
            return x * 2

    result = Doubler().run(5)
    assert result == 10


def test_node_with_state():
    class Accumulator(Node):
        def __init__(self, offset):
            self.offset = offset

        def process(self, x):
            return x + self.offset

    result = Accumulator(10).run(5)
    assert result == 15


def test_node_lifecycle():
    events = []

    class Tracked(Node):
        def setup(self):
            events.append("setup")

        def teardown(self):
            events.append("teardown")

        def process(self, x):
            events.append("process")
            return x

    Tracked().run(1)
    assert events == ["setup", "process", "teardown"]


def test_node_teardown_on_error():
    events = []

    class FailNode(Node):
        def setup(self):
            events.append("setup")

        def teardown(self):
            events.append("teardown")

        def process(self, x):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        FailNode().run(1)
    assert "teardown" in events  # teardown still called


def test_node_pipeline_composition():
    class Increment(Node):
        def process(self, x):
            return x + 1

    class Double(Node):
        def process(self, x):
            return x * 2

    pipeline = Increment() | Double()
    result = pipeline.run(5)
    assert result == 12  # (5+1)*2


def test_node_mixed_with_pipestep():
    @piped
    def add_one(x):
        return x + 1

    class Triple(Node):
        def process(self, x):
            return x * 3

    pipeline = add_one | Triple()
    result = pipeline.run(5)
    assert result == 18  # (5+1)*3


def test_node_inheritance():
    class ScaleNode(Node):
        def __init__(self, factor):
            self.factor = factor

        def process(self, x):
            return x * self.factor

    class DoubleNode(ScaleNode):
        def __init__(self):
            super().__init__(2)

    class TripleNode(ScaleNode):
        def __init__(self):
            super().__init__(3)

    pipeline = DoubleNode() | TripleNode()
    result = pipeline.run(5)
    assert result == 30  # 5*2*3


# =============================================================================
# ConditionalStep Tests
# =============================================================================

def test_conditional_basic():
    @piped
    def double(x):
        return x * 2

    @piped
    def negate(x):
        return -x

    cond = ConditionalStep(
        condition=lambda x: x > 0,
        if_true=double,
        if_false=negate,
    )
    assert cond.run(5) == 10
    assert cond.run(-3) == 3


def test_conditional_no_false_branch():
    @piped
    def double(x):
        return x * 2

    cond = ConditionalStep(condition=lambda x: x > 0, if_true=double)
    assert cond.run(5) == 10
    assert cond.run(-3) == -3  # passthrough


def test_conditional_with_nodes():
    class Positive(Node):
        def process(self, x):
            return abs(x)

    class Negative(Node):
        def process(self, x):
            return -abs(x)

    cond = ConditionalStep(
        condition=lambda x: x > 0,
        if_true=Positive(),
        if_false=Negative(),
    )
    assert cond.run(5) == 5
    assert cond.run(-3) == -3


def test_conditional_in_pipeline():
    @piped
    def add_one(x):
        return x + 1

    cond = ConditionalStep(
        condition=lambda x: x > 5,
        if_true=piped(lambda x: x * 10),
        if_false=piped(lambda x: x * 2),
    )

    pipeline = add_one | cond
    assert pipeline.run(5) == 60   # 6 > 5 -> 6*10
    assert pipeline.run(3) == 8    # 4 <= 5 -> 4*2


def test_conditional_async():
    @piped
    async def async_double(x):
        return x * 2

    cond = ConditionalStep(
        condition=lambda x: x > 0,
        if_true=async_double,
    )
    result = asyncio.run(cond.async_run(5))
    assert result == 10


class _RunReturnsCoroutine:
    """Custom branch with only run() that returns a coroutine (no async_run)."""

    def run(self, value):
        async def _work():
            return value * 2

        return _work()


def test_conditional_async_run_branch_sync_run_coroutine():
    """AUDIT bug 3: async path must await coroutines from branch.run()."""
    cond = ConditionalStep(
        condition=lambda x: x > 0,
        if_true=_RunReturnsCoroutine(),
    )
    result = asyncio.run(cond.async_run(5))
    assert result == 10


def test_switch_async_run_branch_sync_run_coroutine():
    """AUDIT bug 3: SwitchStep async path must await coroutines from branch.run()."""
    switch = SwitchStep(
        key=lambda x: "go",
        branches={"go": _RunReturnsCoroutine()},
    )
    result = asyncio.run(switch.async_run(5))
    assert result == 10


# =============================================================================
# Graph (DAG) Tests
# =============================================================================

def test_graph_linear():
    g = Graph()
    g.add_node("a", piped(lambda x: x + 1))
    g.add_node("b", piped(lambda x: x * 2))
    g.add_edge("a", "b")

    results = g.run(seed=5)
    assert results["a"] == 6
    assert results["b"] == 12


def test_graph_diamond():
    """Diamond dependency: a -> b, a -> c, b+c -> d"""
    g = Graph()
    g.add_node("a", piped(lambda x: x + 1))
    g.add_node("b", piped(lambda x: x * 2))
    g.add_node("c", piped(lambda x: x * 3))
    g.add_node("d", piped(lambda vals: vals[0] + vals[1]))

    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")

    results = g.run(seed=5)
    assert results["a"] == 6
    assert results["b"] == 12
    assert results["c"] == 18
    assert results["d"] == 30  # 12 + 18


def test_graph_with_nodes():
    class Double(Node):
        def process(self, x):
            return x * 2

    g = Graph()
    g.add_node("start", piped(lambda x: x + 1))
    g.add_node("double", Double())
    g.add_edge("start", "double")

    results = g.run(seed=5)
    assert results["double"] == 12


def test_graph_cycle_detection():
    g = Graph()
    g.add_node("a", piped(lambda x: x))
    g.add_node("b", piped(lambda x: x))
    g.add_edge("a", "b")
    g.add_edge("b", "a")

    with pytest.raises(GraphCycleError):
        g.run(seed=1)


def test_graph_missing_node():
    g = Graph()
    g.add_node("a", piped(lambda x: x))

    with pytest.raises(KeyError):
        g.add_edge("a", "nonexistent")


def test_graph_chaining():
    """Test fluent API."""
    g = (
        Graph()
        .add_node("a", piped(lambda x: x + 1))
        .add_node("b", piped(lambda x: x * 2))
        .add_edge("a", "b")
    )
    results = g.run(seed=5)
    assert results["b"] == 12


def test_graph_independent_nodes():
    """Nodes with no edges run independently with seed."""
    g = Graph()
    g.add_node("a", piped(lambda x: x + 1))
    g.add_node("b", piped(lambda x: x * 2))

    results = g.run(seed=5)
    assert results["a"] == 6
    assert results["b"] == 10


def test_graph_async():
    g = Graph()
    g.add_node("a", piped(lambda x: x + 1))
    g.add_node("b", piped(lambda x: x * 2))
    g.add_edge("a", "b")

    results = asyncio.run(g.async_run(seed=5))
    assert results["b"] == 12


# =============================================================================
# Performance / Edge Case Tests
# =============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_large_data_throughput():
    @piped(batch_size=10000)
    def process_chunk(chunk):
        return np.mean(chunk)

    data = np.random.rand(100_000)
    start = time.perf_counter()
    result = process_chunk.run(data)
    duration = time.perf_counter() - start

    n_batches = (len(data) + 9999) // 10000
    expected = sum(np.mean(data[i:i + 10000]) for i in range(0, len(data), 10000))
    assert abs(result - expected) < 1e-6
    assert duration < 1.0


def test_empty_pipeline():
    pipeline = Pipeline([])
    result = pipeline.run(5)
    assert result == 5


def test_single_step_pipeline():
    @piped
    def identity(x):
        return x

    result = identity.run(5)
    assert result == 5


def test_none_handling():
    @piped
    def handle_none(x):
        return x is None

    result = handle_none.run(None)
    assert result is True


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_integration():
    counter = Counter()

    @piped
    def fetch_data():
        return list(range(1, 6))

    @piped(parallel='thread')
    def process_item(x):
        time.sleep(0.01)
        return x * 2

    @circuit_breaker(failure_threshold=2)
    @retry(max_attempts=3)
    @piped(parallel='thread')
    def flaky_operation(x):
        counter.increment()
        if x == 4 and counter.count < 3:
            raise MockException("Flaky on 4")
        return x + 1

    @piped(batch_size=2)
    def batch_sum(items):
        return sum(items)

    pipeline = fetch_data | process_item | flaky_operation | batch_sum
    result = pipeline.run()
    expected = sum([3, 5, 7, 9, 11])
    assert result == expected


# =============================================================================
# Execution Result Tests
# =============================================================================

def test_execution_result():
    @piped
    def step1(x):
        return x + 1

    @piped
    def step2(x):
        return x * 2

    pipeline = step1 | step2
    result = pipeline.run_detailed(5)

    assert result.value == 12
    assert len(result.history) == 2
    assert result.history[0] == ("step1", 6)
    assert result.history[1] == ("step2", 12)
    assert result.execution_time > 0
    assert result.step_count == 2


def test_async_execution_result():
    @piped
    async def step1(x):
        return x + 1

    @piped
    async def step2(x):
        return x * 2

    pipeline = step1 | step2
    result = asyncio.run(pipeline.async_run_detailed(5))

    assert result.value == 12
    assert len(result.history) == 2
    assert result.history[0] == ("step1", 6)
    assert result.history[1] == ("step2", 12)
    assert result.execution_time >= 0
    assert result.step_count == 2


# =============================================================================
# Misc Tests
# =============================================================================

def test_declarative_pipeline_yaml():
    pipeline = Pipeline.from_spec(str(FIXTURES / "inc_double.yaml"))
    assert pipeline.run(5) == 12


def test_declarative_pipeline_json():
    pipeline = Pipeline.from_spec(str(FIXTURES / "inc_double.json"))
    assert pipeline.run(5) == 12


def test_declarative_pipeline_registry(tmp_path):
    spec = tmp_path / "pipe.yaml"
    spec.write_text(
        "steps:\n"
        "  - import: custom:inc\n"
        "  - import: tests.spec_fixtures:double\n"
    )
    registry = {"custom:inc": piped(lambda x: x + 10)}
    pipeline = Pipeline.from_spec(str(spec), registry=registry)
    assert pipeline.run(5) == 30


def test_declarative_pipeline_missing_file():
    with pytest.raises(FileNotFoundError):
        Pipeline.from_spec("missing.yaml")


def test_builder_dsl():
    @piped
    def inc(x):
        return x + 1

    builder = PipelineBuilder()
    builder.add(inc)
    pipeline = builder.build()
    assert pipeline.run(1) == 2


def test_context_manager_cleanup():
    @piped(parallel='thread')
    def work(x):
        return x + 1

    with Pipeline([work]) as p:
        assert p.run(1) == 2
    from pipeline import _POOLS
    assert _POOLS == {}


def test_timeout_enforced():
    # Large margin (0.1s timeout vs 2s work) so a loaded CI runner cannot let
    # the call finish before the timeout fires.
    @piped(timeout=0.1)
    def slow(x):
        time.sleep(2.0)
        return x

    with pytest.raises(PipelineError):
        slow.run(1)


def test_map_reduce():
    step = MapReduceStep(mapper=lambda x: x * 2, reducer=sum, batch_size=2)
    assert step.run([1, 2, 3]) == 12


def test_schema_enforcement():
    @piped(schema=int)
    def to_str(x):
        return str(x)

    with pytest.raises(PipelineError):
        to_str.run(1)


def test_graceful_cancellation():
    @piped
    async def slow(x):
        await asyncio.sleep(0.5)
        return x

    pipeline = Pipeline([slow])

    async def run_and_cancel():
        pipeline.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pipeline.async_run(1)

    asyncio.run(run_and_cancel())


# =============================================================================
# @node Decorator Tests
# =============================================================================

def test_node_decorator_basic():
    @node
    def double(x):
        return x * 2

    result = double.run(5)
    assert result == 10


def test_node_decorator_pipeline():
    @node
    def add_one(x):
        return x + 1

    @node
    def triple(x):
        return x * 3

    pipeline = add_one | triple
    result = pipeline.run(5)
    assert result == 18  # (5+1)*3


def test_node_decorator_with_lifecycle():
    events = []

    @node(setup=lambda self: events.append("setup"),
          teardown=lambda self: events.append("teardown"))
    def process(x):
        events.append("process")
        return x * 2

    result = process.run(5)
    assert result == 10
    assert events == ["setup", "process", "teardown"]


def test_node_decorator_mixed_with_piped():
    @piped
    def add_one(x):
        return x + 1

    @node
    def double(x):
        return x * 2

    pipeline = add_one | double
    result = pipeline.run(5)
    assert result == 12


def test_node_decorator_preserves_display_name():
    """@node must not mutate __class__.__name__ for display."""
    @node
    def my_step(x):
        return x + 1

    assert my_step._func_name == "my_step"
    assert repr(my_step) == "my_step()"
    assert my_step.__class__.__name__ == "FuncNode"


def test_piped_beartype_validates_input():
    @piped
    def only_int(x: int) -> int:
        return x + 1

    assert only_int.run(1) == 2
    with pytest.raises(PipelineError):
        only_int.run("bad")


def test_piped_beartype_validates_return():
    @piped
    def must_return_int(x: int) -> int:
        return str(x)  # type: ignore[return-value]

    with pytest.raises(PipelineError):
        must_return_int.run(1)


def test_piped_return_annotation_enforces_output():
    @piped
    def via_annotation(x) -> int:
        return "not-int"

    with pytest.raises(PipelineError):
        via_annotation.run(1)


def test_piped_beartype_disabled_by_env(monkeypatch):
    monkeypatch.setenv("STEPCRAFT_NO_BEARTYPE", "1")

    @piped
    def loose(x: int):
        return x

    assert loose.run("not-checked") == "not-checked"


def test_return_annotation_does_not_break_auto_map():
    """An annotated @piped step auto-mapped over a list returns a list and is
    not wrongly checked against the per-element return annotation."""
    @piped
    def to_pair(x: int) -> dict:
        return {"v": x}

    assert to_pair.run([1, 2, 3]) == [{"v": 1}, {"v": 2}, {"v": 3}]
    # beartype still enforces the per-element return type.
    @piped
    def bad(x: int) -> dict:
        return x  # type: ignore[return-value]

    with pytest.raises(PipelineError):
        bad.run([1, 2])


def test_explicit_schema_validates_per_element_under_auto_map():
    @piped(schema=int)
    def inc(x):
        return x + 1

    assert inc.run([1, 2, 3]) == [2, 3, 4]  # each element is an int

    @piped(schema=int)
    def to_str(x):
        return str(x)

    with pytest.raises(PipelineError):
        to_str.run([1, 2, 3])  # each element is a str -> fails per element


def test_explicit_schema_validates_per_element_under_parallel():
    @piped(parallel="thread", schema=int)
    def inc(x):
        return x + 1

    assert inc.run([1, 2, 3]) == [2, 3, 4]


def test_piped_parallel_batch_warns(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="stepcraft"):
        @piped(parallel="thread", batch_size=4)
        def work(x):
            return x

    assert any("batch_size" in r.message for r in caplog.records)


# =============================================================================
# SwitchStep Tests
# =============================================================================

def test_switch_basic():
    switch = SwitchStep(
        key=lambda x: "pos" if x > 0 else "neg",
        branches={
            "pos": piped(lambda x: x * 2),
            "neg": piped(lambda x: -x),
        },
    )
    assert switch.run(5) == 10
    assert switch.run(-3) == 3


def test_switch_default():
    switch = SwitchStep(
        key=lambda x: "a" if x == 1 else "unknown",
        branches={"a": piped(lambda x: x * 10)},
        default=piped(lambda x: x),
    )
    assert switch.run(1) == 10
    assert switch.run(99) == 99


def test_switch_no_match_no_default():
    switch = SwitchStep(
        key=lambda x: "missing",
        branches={"a": piped(lambda x: x * 10)},
    )
    # No match, no default -> passthrough
    assert switch.run(5) == 5


def test_switch_with_nodes():
    class Doubler(Node):
        def process(self, x):
            return x * 2

    class Negator(Node):
        def process(self, x):
            return -x

    switch = SwitchStep(
        key=lambda x: "double" if x > 0 else "negate",
        branches={"double": Doubler(), "negate": Negator()},
    )
    assert switch.run(5) == 10
    assert switch.run(-3) == 3


def test_switch_in_pipeline():
    switch = SwitchStep(
        key=lambda x: "big" if x > 10 else "small",
        branches={
            "big": piped(lambda x: x * 100),
            "small": piped(lambda x: x * 2),
        },
    )
    pipeline = piped(lambda x: x + 5) | switch
    assert pipeline.run(10) == 1500  # 15 > 10 -> 15*100
    assert pipeline.run(3) == 16     # 8 <= 10 -> 8*2


def test_switch_async():
    switch = SwitchStep(
        key=lambda x: "double" if x > 0 else "negate",
        branches={
            "double": piped(lambda x: x * 2),
            "negate": piped(lambda x: -x),
        },
    )
    result = asyncio.run(switch.async_run(5))
    assert result == 10


# =============================================================================
# Graph Parallel Execution Tests
# =============================================================================

def test_graph_parallel_sync():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x + 1))
        .add_node("b", piped(lambda x: x * 2))
        .add_node("c", piped(lambda x: x * 3))
        .add_node("d", piped(lambda vals: vals[0] + vals[1]))
        .add_edge("a", "b")
        .add_edge("a", "c")
        .add_edge("b", "d")
        .add_edge("c", "d")
    )
    results = g.run(seed=5, parallel=True)
    assert results["a"] == 6
    assert results["b"] == 12
    assert results["c"] == 18
    assert results["d"] == 30


def test_graph_parallel_async():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x + 1))
        .add_node("b", piped(lambda x: x * 2))
        .add_node("c", piped(lambda x: x * 3))
        .add_edge("a", "b")
        .add_edge("a", "c")
    )
    results = asyncio.run(g.async_run(seed=5, parallel=True))
    assert results["a"] == 6
    assert results["b"] == 12
    assert results["c"] == 18


def test_graph_roots_and_leaves():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x))
        .add_node("b", piped(lambda x: x))
        .add_node("c", piped(lambda x: x))
        .add_edge("a", "b")
        .add_edge("b", "c")
    )
    assert g.roots == ["a"]
    assert g.leaves == ["c"]


def test_graph_multiple_roots():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x))
        .add_node("b", piped(lambda x: x))
        .add_node("c", piped(lambda x: x))
        .add_edge("a", "c")
        .add_edge("b", "c")
    )
    assert g.roots == ["a", "b"]
    assert g.leaves == ["c"]


# =============================================================================
# Graph topological sort
# =============================================================================

def test_graph_topo_sort():
    g = (
        Graph()
        .add_node("x", piped(lambda v: v + 1))
        .add_node("y", piped(lambda v: v * 2))
        .add_edge("x", "y")
    )
    results = g.run(seed=10)
    assert results["y"] == 22  # (10+1)*2


def test_graph_topo_sort_cycle():
    g = (
        Graph()
        .add_node("a", piped(lambda v: v))
        .add_node("b", piped(lambda v: v))
        .add_edge("a", "b")
        .add_edge("b", "a")
    )
    with pytest.raises(GraphCycleError):
        g.run(seed=1)


# =============================================================================
# Developer Experience Tests
# =============================================================================

def test_pipeline_repr():
    @piped
    def add(x):
        return x + 1

    @piped
    def double(x):
        return x * 2

    p = add | double
    assert "Pipeline(" in repr(p)
    assert "add" in repr(p)
    assert "double" in repr(p)


def test_pipeline_len():
    @piped
    def a(x):
        return x

    @piped
    def b(x):
        return x

    p = a | b
    assert len(p) == 2


def test_pipeline_getitem():
    @piped
    def a(x):
        return x + 1

    @piped
    def b(x):
        return x * 2

    p = a | b
    assert p[0]._func_name == "a"
    assert p[1]._func_name == "b"


def test_pipeline_iter():
    @piped
    def a(x):
        return x

    @piped
    def b(x):
        return x

    p = a | b
    names = [s._func_name for s in p]
    assert names == ["a", "b"]


def test_pipeline_map():
    @piped
    def double(x):
        return x * 2

    p = Pipeline([double])
    result = p.map([1, 2, 3, 4])
    assert result == [2, 4, 6, 8]


def test_pipeline_async_map():
    @piped
    async def double(x):
        return x * 2

    p = Pipeline([double])
    result = asyncio.run(p.async_map([1, 2, 3]))
    assert result == [2, 4, 6]


def test_pipestep_repr():
    @piped(parallel='thread', batch_size=10)
    def work(x):
        return x

    r = repr(work)
    assert "PipeStep(" in r
    assert "work" in r
    assert "thread" in r
    assert "batch=10" in r


def test_node_repr():
    class MyNode(Node):
        def process(self, x):
            return x

    assert "MyNode()" == repr(MyNode())


def test_graph_repr():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x))
        .add_node("b", piped(lambda x: x))
        .add_edge("a", "b")
    )
    assert "Graph(nodes=2, edges=1)" == repr(g)


def test_graph_describe():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x))
        .add_node("b", piped(lambda x: x))
        .add_node("c", piped(lambda x: x))
        .add_edge("a", "b")
        .add_edge("a", "c")
    )
    desc = g.describe()
    assert "Graph:" in desc
    assert "a -> b, c" in desc
    assert "b (leaf)" in desc
    assert "c (leaf)" in desc


# =============================================================================
# Regression tests for verified correctness bugs
# =============================================================================

def test_async_timeout_zero_is_enforced():
    """BUG 2: async path must honor timeout=0.0 (is not None), not skip it."""
    @piped(timeout=0.0)
    async def slow(x):
        await asyncio.sleep(0.05)
        return x

    with pytest.raises(PipelineError):
        asyncio.run(slow.async_run(1))


def test_parallel_and_batch_precedence_is_deterministic():
    """BUG 3: when both parallel and batch_size are set, behavior must be
    explicit and deterministic (parallel wins)."""
    @piped(parallel='thread', batch_size=3)
    def identity(x):
        return x

    # parallel wins: each item mapped individually -> per-item results.
    result = identity.run([1, 2, 3, 4, 5])
    assert result == [1, 2, 3, 4, 5]


def test_parallel_forwards_kwargs_sync():
    """BUG 4: keyword args bound by partial application must reach the parallel
    workers (sync path)."""
    @piped(parallel='thread')
    def add(x, *, offset=0):
        return x + offset

    bound = add(PIPE, offset=10)
    result = bound.run([1, 2, 3])
    assert result == [11, 12, 13]


def test_parallel_forwards_kwargs_async():
    """BUG 4: keyword args must reach the parallel workers (async path)."""
    @piped(parallel='thread')
    def add(x, *, offset=0):
        return x + offset

    bound = add(PIPE, offset=100)
    result = asyncio.run(bound.async_run([1, 2, 3]))
    assert result == [101, 102, 103]


def test_batched_does_not_sum_booleans():
    """BUG 5: per-batch boolean results must NOT be silently summed; they are
    returned as a list for explicit reduction."""
    @piped(batch_size=2)
    def all_positive(batch):
        return all(v > 0 for v in batch)

    result = all_positive.run([1, 2, 3, 4, -5, 6])
    # 3 batches -> list of bools, not an int sum.
    assert result == [True, True, False]


def test_async_batched_awaits_coroutines():
    """AUDIT bug 1: async @piped with batch_size must await each batch."""
    @piped(batch_size=2)
    async def process_batch(batch):
        return [x * 2 for x in batch]

    result = asyncio.run(process_batch.async_run([1, 2, 3, 4]))
    assert result == [2, 4, 6, 8]


def test_async_batched_sync_func_in_executor():
    """AUDIT bug 1: sync func with batch_size on async_run uses executor."""
    @piped(batch_size=2)
    def batch_double(batch):
        return [x * 2 for x in batch]

    result = asyncio.run(batch_double.async_run([1, 2, 3, 4]))
    assert result == [2, 4, 6, 8]


def test_async_parallel_thread_awaits_coroutines():
    """AUDIT bug 2: async @piped with parallel must await each item."""
    @piped(parallel='thread')
    async def double(x):
        return x * 2

    result = asyncio.run(double.async_run([1, 2, 3]))
    assert result == [2, 4, 6]


def test_async_parallel_forwards_kwargs():
    """AUDIT bug 2: async parallel path must forward keyword args."""
    @piped(parallel='thread')
    async def add(x, *, offset=0):
        return x + offset

    bound = add(PIPE, offset=100)
    result = asyncio.run(bound.async_run([1, 2, 3]))
    assert result == [101, 102, 103]


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_batched_recognizes_numpy_scalars():
    """BUG 6: numpy scalar results from batches are recognized as numeric and
    summed (consistent with the numpy-aware rest of the module)."""
    @piped(batch_size=2)
    def batch_sum(batch):
        return np.sum(np.array(batch))  # returns an np.integer scalar

    result = batch_sum.run([1, 2, 3, 4])
    assert result == 10


def test_fanout_thread_parallel():
    """BUG 7: FanOutStep with parallel='thread' must run without a lambda."""
    branch1 = piped(lambda x: x + 1)
    branch2 = piped(lambda x: x * 2)
    fan_out = FanOutStep((branch1, branch2), parallel='thread')
    assert fan_out.run(5) == (6, 10)


def _fanout_add_one(x):
    return x + 1


def _fanout_times_two(x):
    return x * 2


def test_fanout_process_parallel():
    """BUG 7: FanOutStep with parallel='process' must not crash on an
    unpicklable lambda; branches over module-level functions are picklable."""
    branch1 = piped(_fanout_add_one)
    branch2 = piped(_fanout_times_two)
    fan_out = FanOutStep((branch1, branch2), parallel='process')
    assert fan_out.run(5) == (6, 10)


def test_fanout_async_parallel_thread():
    """AUDIT: FanOutStep.async_run honors parallel='thread'."""
    branch1 = piped(_fanout_add_one)
    branch2 = piped(_fanout_times_two)
    fan_out = FanOutStep((branch1, branch2), parallel='thread')
    result = asyncio.run(fan_out.async_run(5))
    assert result == (6, 10)


def test_fanout_async_parallel_process():
    branch1 = piped(_fanout_add_one)
    branch2 = piped(_fanout_times_two)
    fan_out = FanOutStep((branch1, branch2), parallel='process')
    result = asyncio.run(fan_out.async_run(5))
    assert result == (6, 10)


def test_mapreduce_async_mapper_parallel():
    """AUDIT: MapReduceStep.async_run maps batch items concurrently."""
    delays = []

    async def mapper(x):
        delays.append(x)
        await asyncio.sleep(0.02)
        return x * 2

    step = MapReduceStep(mapper=mapper, reducer=sum, batch_size=4)
    start = time.perf_counter()
    result = asyncio.run(step.async_run(range(4)))
    elapsed = time.perf_counter() - start

    assert result == sum(x * 2 for x in range(4))
    assert elapsed < 0.08  # parallel ~20ms, sequential would be ~80ms


def test_mapreduce_async_reducer_coroutine():
    async def mapper(x):
        return x * 2

    async def reducer(values):
        return sum(values)

    step = MapReduceStep(mapper=mapper, reducer=reducer)
    result = asyncio.run(step.async_run([1, 2, 3]))
    assert result == 12


def test_pipeline_shim_reexports():
    """BUG 8: the `pipeline` shim must re-export everything the tests use."""
    import pipeline
    for name in (
        "PIPE", "Pipeline", "piped", "retry", "circuit_breaker",
        "FanOutStep", "FanInStep", "PipelineError", "ExecutionResult",
        "PipelineBuilder", "MapReduceStep", "Node", "node", "ConditionalStep",
        "SwitchStep", "Graph", "GraphCycleError", "_POOLS",
    ):
        assert hasattr(pipeline, name), f"pipeline shim missing {name}"


# =============================================================================
# uvloop async runtime
# =============================================================================

def test_run_async_fallback_without_uvloop(monkeypatch):
    """run_async falls back to asyncio.run when uvloop is unavailable."""
    import stepcraft.async_runtime as ar

    monkeypatch.setattr(ar, "HAS_UVLOOP", False)

    async def coro():
        return 7

    assert ar.run_async(coro()) == 7


@pytest.mark.skipif(not HAS_UVLOOP, reason="uvloop not installed")
@pytest.mark.uvloop
def test_run_async_with_uvloop():
    async def coro():
        await asyncio.sleep(0.01)
        return 99

    assert run_async(coro()) == 99


@pytest.mark.skipif(not HAS_UVLOOP, reason="uvloop not installed")
@pytest.mark.uvloop
def test_pipeline_run_async():
    @piped
    async def add_one(x):
        await asyncio.sleep(0.01)
        return x + 1

    @piped
    async def double(x):
        return x * 2

    pipeline = add_one | double
    assert pipeline.run_async(5) == 12


@pytest.mark.skipif(not HAS_UVLOOP, reason="uvloop not installed")
@pytest.mark.uvloop
def test_pipeline_map_async():
    @piped
    async def double(x):
        return x * 2

    p = Pipeline([double])
    assert p.map_async([1, 2, 3]) == [2, 4, 6]


@pytest.mark.skipif(not HAS_UVLOOP, reason="uvloop not installed")
@pytest.mark.uvloop
def test_graph_run_async():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x + 1))
        .add_node("b", piped(lambda x: x * 2))
        .add_edge("a", "b")
    )
    # parallel=False avoids thread-pool + uvloop deadlocks on Linux CI.
    results = g.run_async(seed=5, parallel=False)
    assert results["a"] == 6
    assert results["b"] == 12


@pytest.mark.skipif(not HAS_UVLOOP, reason="uvloop not installed")
@pytest.mark.uvloop
def test_uvloop_policy_context():
    async def coro():
        return asyncio.get_running_loop().__class__.__module__.startswith("uvloop")

    with uvloop_policy():
        assert run_async(coro()) is True


# =============================================================================
# map= alias (README-friendly auto_map)
# =============================================================================

def test_map_alias_disables_auto_map():
    @piped(map=False)
    def length(xs):
        return len(xs)

    assert length.run([1, 2, 3]) == 3


def test_map_alias_true_keeps_auto_map():
    @piped(map=True)
    def double(x):
        return x * 2

    assert double.run([1, 2, 3]) == [2, 4, 6]


# =============================================================================
# Step hooks / observability
# =============================================================================

def test_on_step_hook_called():
    @piped
    def add_one(x):
        return x + 1

    @piped
    def double(x):
        return x * 2

    events = []

    def hook(name, inp, out, dt):
        events.append((name, inp, out))
        assert isinstance(dt, float) and dt >= 0

    pipeline = add_one | double
    result = pipeline.run(5, on_step=hook)
    assert result == 12
    assert events == [("add_one", 5, 6), ("double", 6, 12)]


def test_on_step_hook_run_detailed():
    @piped
    def add_one(x):
        return x + 1

    calls = []
    pipeline = Pipeline([add_one])
    result = pipeline.run_detailed(5, on_step=lambda *a: calls.append(a))
    assert result.value == 6
    assert len(calls) == 1
    assert calls[0][0] == "add_one"


def test_on_step_hook_async():
    @piped
    async def add_one(x):
        return x + 1

    events = []
    pipeline = Pipeline([add_one])
    result = asyncio.run(pipeline.async_run(5, on_step=lambda *a: events.append(a)))
    assert result == 6
    assert events[0][0] == "add_one"


def test_map_on_step_hook():
    """CR-5: map() must forward on_step to each per-item run."""
    calls = []

    @piped
    def add_one(x):
        return x + 1

    pipeline = Pipeline([add_one])
    result = pipeline.map([1, 2, 3], on_step=lambda *a: calls.append(a))
    assert result == [2, 3, 4]
    assert len(calls) == 3


def test_run_async_on_step_hook():
    calls = []

    @piped
    def add_one(x):
        return x + 1

    pipeline = Pipeline([add_one])
    result = pipeline.run_async(5, on_step=lambda *a: calls.append(a))
    assert result == 6
    assert len(calls) == 1


def test_on_step_hook_error_does_not_break_pipeline(caplog):
    @piped
    def add_one(x):
        return x + 1

    def bad_hook(name, inp, out, dt):
        raise RuntimeError("hook boom")

    with caplog.at_level(logging.WARNING, logger="stepcraft.hooks"):
        result = Pipeline([add_one]).run(5, on_step=bad_hook)
    assert result == 6
    assert any("hook" in r.message.lower() for r in caplog.records)


def test_graph_on_step_hook():
    g = (
        Graph()
        .add_node("a", piped(lambda x: x + 1))
        .add_node("b", piped(lambda x: x * 2))
        .add_edge("a", "b")
    )
    seen = []
    results = g.run(seed=5, on_step=lambda name, *_: seen.append(name))
    assert results["b"] == 12
    assert set(seen) == {"a", "b"}


# =============================================================================
# cancel_on_timeout
# =============================================================================

def test_cancel_on_timeout_still_raises():
    @piped(timeout=0.05, cancel_on_timeout=True)
    def slow(x):
        time.sleep(0.3)
        return x

    with pytest.raises(PipelineError):
        slow.run(1)


# =============================================================================
# Graph from_spec
# =============================================================================

def test_graph_from_spec_yaml():
    g = Graph.from_spec(str(FIXTURES / "graph_diamond.yaml"))
    results = g.run(seed=5)
    assert results["a"] == 6
    assert results["b"] == 12
    assert results["c"] == 7
    assert results["d"] == 19  # sum((12, 7))


def test_graph_from_spec_registry():
    g = Graph.from_spec(
        str(FIXTURES / "graph_diamond.yaml"),
        registry={"tests.spec_fixtures:inc": piped(lambda x: x + 1)},
    )
    results = g.run(seed=5)
    assert results["a"] == 6


def test_pipeline_from_spec_rejects_graph(tmp_path):
    spec = tmp_path / "g.yaml"
    spec.write_text(
        "graph:\n"
        "  nodes:\n"
        "    a:\n"
        "      import: tests.spec_fixtures:inc\n"
    )
    with pytest.raises(ValueError):
        Pipeline.from_spec(str(spec))


def test_pipeline_from_spec_rejects_both_graph_and_steps(tmp_path):
    """CR-6: ambiguous specs with both keys must raise."""
    spec = tmp_path / "ambiguous.json"
    spec.write_text(
        '{"steps": [{"import": "tests.spec_fixtures:inc"}], '
        '"graph": {"nodes": {"a": {"import": "tests.spec_fixtures:inc"}}}}'
    )
    with pytest.raises(ValueError, match="both 'graph' and 'steps'"):
        Pipeline.from_spec(str(spec))


def test_graph_from_spec_rejects_both_graph_and_steps(tmp_path):
    spec = tmp_path / "ambiguous.json"
    spec.write_text(
        '{"steps": [{"import": "tests.spec_fixtures:inc"}], '
        '"graph": {"nodes": {"a": {"import": "tests.spec_fixtures:inc"}}}}'
    )
    with pytest.raises(ValueError, match="both 'graph' and 'steps'"):
        Graph.from_spec(str(spec))


def test_shim_exports_new_apis():
    import pipeline
    for name in ("get_context", "configure_pools", "StepHook"):
        assert hasattr(pipeline, name), f"pipeline shim missing {name}"


# =============================================================================
# Annotation enforcement (strict by default)
# =============================================================================

def test_piped_requires_annotations_by_default():
    with pytest.raises(MissingAnnotationError):
        @strict_piped
        def no_anno(x):
            return x


def test_piped_requires_return_annotation():
    with pytest.raises(MissingAnnotationError) as exc:
        @strict_piped
        def no_return(x: int):
            return x
    assert "return" in str(exc.value)


def test_piped_fully_annotated_ok():
    @strict_piped
    def good(x: int) -> int:
        return x + 1

    assert good.run(1) == 2


def test_piped_require_annotations_false_opts_out():
    @strict_piped(require_annotations=False)
    def loose(x):
        return x

    assert loose.run(5) == 5


def test_piped_lambda_is_exempt():
    step = strict_piped(lambda x: x * 2)  # lambdas cannot be annotated
    assert step.run(3) == 6


def test_piped_varargs_need_no_annotation():
    # *args / **kwargs need no annotation; only named params and return do.
    @strict_piped
    def variadic(*args, **kwargs) -> int:
        return sum(args)

    assert variadic.run(5) == 5  # variadic(5) -> sum((5,))


def test_node_decorator_requires_annotations():
    with pytest.raises(MissingAnnotationError):
        @strict_node
        def bad(x):
            return x


def test_node_decorator_annotated_ok():
    @strict_node
    def good(x: int) -> int:
        return x * 2

    assert good.run(4) == 8


def test_node_subclass_requires_annotations():
    with pytest.raises(MissingAnnotationError):
        class Bad(StrictNode):
            def process(self, x):
                return x


def test_node_subclass_opt_out():
    class Loose(StrictNode):
        require_annotations = False

        def process(self, x):
            return x

    assert Loose().run(7) == 7


def test_node_subclass_annotated_ok():
    class Good(StrictNode):
        def process(self, x: int) -> int:
            return x + 1

    assert Good().run(1) == 2


if __name__ == "__main__":
    pytest.main(["-v", "-s", "--durations=0"])
