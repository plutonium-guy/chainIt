"""Shared context propagation tests."""

import asyncio

from conftest import Graph, Node, Pipeline, piped


def test_pipeline_shared_context():
    from pipeline import get_context

    seen = {}

    @piped
    def step(x):
        seen.update(get_context())
        return x + 1

    p = Pipeline([step], context={"request_id": "abc"})
    assert p.run(1) == 2
    assert seen["request_id"] == "abc"


def test_context_reset_after_run():
    from pipeline import get_context

    p = Pipeline([piped(lambda x: x)], context={"k": "v"})
    p.run(1)
    assert get_context() == {}


def test_node_setup_reads_context():
    from pipeline import get_context

    captured = {}

    class N(Node):
        def setup(self):
            captured.update(get_context())

        def process(self, x):
            return x

    Pipeline([N()], context={"u": 1}).run(5)
    assert captured["u"] == 1


def test_context_merged_on_compose():
    from pipeline import get_context

    seen = {}

    @piped
    def step(x):
        seen.update(get_context())
        return x

    left = Pipeline([piped(lambda x: x)], context={"a": 1})
    right = Pipeline([step], context={"b": 2})
    (left | right).run(0)
    assert seen == {"a": 1, "b": 2}


def test_context_async():
    from pipeline import get_context

    seen = {}

    @piped
    async def step(x):
        seen.update(get_context())
        return x

    p = Pipeline([step], context={"trace": "xyz"})
    asyncio.run(p.async_run(0))
    assert seen["trace"] == "xyz"


def test_context_parallel_thread():
    from pipeline import get_context

    seen = []

    @piped(parallel='thread')
    def step(x):
        seen.append(dict(get_context()))
        return x * 2

    p = Pipeline([step], context={"request_id": "par"})
    assert p.run([1, 2, 3]) == [2, 4, 6]
    assert all(item["request_id"] == "par" for item in seen)


def test_context_timeout_thread_pool():
    from pipeline import get_context

    seen = {}

    @piped(timeout=1.0)
    def step(x):
        seen.update(get_context())
        return x + 1

    p = Pipeline([step], context={"tid": "timeout"})
    assert p.run(1) == 2
    assert seen["tid"] == "timeout"


def test_context_async_sync_step_in_executor():
    from pipeline import get_context

    seen = {}

    @piped
    def step(x):
        seen.update(get_context())
        return x + 1

    p = Pipeline([step], context={"async_exec": "yes"})
    asyncio.run(p.async_run(4))
    assert seen["async_exec"] == "yes"


def test_graph_shared_context():
    from pipeline import get_context

    seen = {}

    @piped
    def inc(x):
        seen.update(get_context())
        return x + 1

    g = Graph(context={"graph_id": "dag"})
    g.add_node("a", inc)
    results = g.run(seed=5)
    assert results["a"] == 6
    assert seen["graph_id"] == "dag"


def test_graph_shared_context_async():
    from pipeline import get_context

    seen = {}

    @piped
    async def inc(x):
        seen.update(get_context())
        return x + 1

    g = Graph(context={"mode": "async"})
    g.add_node("a", inc)
    results = asyncio.run(g.async_run(seed=3, parallel=False))
    assert results["a"] == 4
    assert seen["mode"] == "async"
