"""Tests for optimization roadmap features."""

import asyncio
import threading
import time

from conftest import Graph, Node, Pipeline, piped, strict_piped
from pipeline import FanOutStep, get_context


def test_pipeline_map_parallel():
    @piped
    def double(x):
        return x * 2

    p = Pipeline([double])
    assert p.map([1, 2, 3], parallel=True) == [2, 4, 6]


def test_pipeline_map_parallel_bounded():
    active = []
    lock = threading.Lock()
    peak = [0]

    @piped
    def work(x):
        with lock:
            active.append(x)
            peak[0] = max(peak[0], len(active))
        time.sleep(0.02)
        with lock:
            active.remove(x)
        return x

    p = Pipeline([work])
    result = p.map(range(6), parallel=2)
    assert result == list(range(6))
    assert peak[0] <= 2


def test_pipeline_async_map_max_concurrency():
    active = []
    lock = threading.Lock()
    peak = [0]

    @piped
    async def work(x):
        with lock:
            active.append(x)
            peak[0] = max(peak[0], len(active))
        await asyncio.sleep(0.02)
        with lock:
            active.remove(x)
        return x

    p = Pipeline([work])
    result = asyncio.run(p.async_map(range(6), max_concurrency=2))
    assert result == list(range(6))
    assert peak[0] <= 2


def test_piped_max_concurrency_limits_async_auto_map():
    active = []
    lock = threading.Lock()
    peak = [0]

    @piped(max_concurrency=2)
    async def work(x):
        with lock:
            active.append(x)
            peak[0] = max(peak[0], len(active))
        await asyncio.sleep(0.02)
        with lock:
            active.remove(x)
        return x * 2

    result = asyncio.run(work.async_run([1, 2, 3, 4, 5]))
    assert result == [2, 4, 6, 8, 10]
    assert peak[0] <= 2


def test_node_setup_once_runs_setup_once_per_pipeline_run():
    setup_count = [0]
    teardown_count = [0]

    class ResourceNode(Node):
        setup_once = True

        def setup(self):
            setup_count[0] += 1

        def teardown(self):
            teardown_count[0] += 1

        def process(self, x):
            return x + 1

    node_step = ResourceNode()
    p = Pipeline([node_step])
    assert p.map([1, 2, 3]) == [2, 3, 4]
    assert setup_count[0] == 1
    assert teardown_count[0] == 1


def test_piped_typecheck_false_skips_beartype():
    @strict_piped(typecheck=False)
    def loose(x: int) -> int:
        return x

    assert loose.run("not-checked") == "not-checked"


def test_graph_parallel_context_in_thread_pool():
    seen = []

    @piped(parallel='thread')
    def step(x):
        seen.append(dict(get_context()))
        return x * 2

    g = Graph(context={"graph_parallel": "yes"})
    g.add_node("a", step)
    results = g.run(seed=[1, 2], parallel=True)
    assert results["a"] == [2, 4]
    assert all(item["graph_parallel"] == "yes" for item in seen)


def _fanout_add_one(x):
    return x + 1


def _fanout_times_two(x):
    return x * 2


def test_fanout_async_max_concurrency():
    branch1 = piped(_fanout_add_one)
    branch2 = piped(_fanout_times_two)
    fan_out = FanOutStep((branch1, branch2), max_concurrency=1)
    result = asyncio.run(fan_out.async_run(5))
    assert result == (6, 10)
