from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

from .context import activate_context, wrap_worker
from .exceptions import GraphCycleError
from .hooks import StepHook, _call_hook
from .pools import _get_pool


class Graph:
    """DAG-based pipeline for complex dependency graphs."""

    __slots__ = ('_nodes', '_edges', '_reverse', '_topo_order', '_topo_levels_cache', '_context')

    def __init__(self, *, context: Optional[Dict[str, Any]] = None):
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, Set[str]] = {}
        self._reverse: Dict[str, Set[str]] = {}
        self._topo_order: Optional[List[str]] = None
        self._topo_levels_cache: Optional[List[List[str]]] = None
        self._context = context

    def _invalidate_topo_cache(self) -> None:
        self._topo_order = None
        self._topo_levels_cache = None

    def add_node(self, name: str, step: Any) -> 'Graph':
        self._nodes[name] = step
        self._edges.setdefault(name, set())
        self._reverse.setdefault(name, set())
        self._invalidate_topo_cache()
        return self

    def add_edge(self, from_node: str, to_node: str) -> 'Graph':
        if from_node not in self._nodes:
            raise KeyError(f"Node '{from_node}' not found")
        if to_node not in self._nodes:
            raise KeyError(f"Node '{to_node}' not found")
        self._edges[from_node].add(to_node)
        self._reverse.setdefault(to_node, set()).add(from_node)
        self._invalidate_topo_cache()
        return self

    @property
    def roots(self) -> List[str]:
        """Nodes with no parents (in-degree 0)."""
        return sorted(n for n in self._nodes if not self._reverse.get(n))

    @property
    def leaves(self) -> List[str]:
        """Nodes with no children (out-degree 0)."""
        return sorted(n for n in self._nodes if not self._edges.get(n))

    def _topo_sort(self) -> List[str]:
        """Topological sort via Kahn's algorithm (cached until graph changes)."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree = {n: len(self._reverse.get(n, set())) for n in self._nodes}
        queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in sorted(self._edges.get(node, set())):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._nodes):
            raise GraphCycleError("Cycle detected in graph")
        self._topo_order = order
        return order

    def _topo_levels(self) -> List[List[str]]:
        """Group nodes by dependency level for parallel execution (cached)."""
        if self._topo_levels_cache is not None:
            return self._topo_levels_cache

        order = self._topo_sort()
        depth: Dict[str, int] = {}
        for name in order:
            parents = self._reverse.get(name, set())
            if not parents:
                depth[name] = 0
            else:
                depth[name] = max(depth[p] for p in parents) + 1
        max_depth = max(depth.values()) if depth else 0
        levels: List[List[str]] = [[] for _ in range(max_depth + 1)]
        for name in order:
            levels[depth[name]].append(name)
        self._topo_levels_cache = levels
        return levels

    def _run_node(self, name: str, input_value: Any) -> Any:
        step = self._nodes[name]
        if hasattr(step, 'run'):
            return step.run(input_value)
        return step(input_value)

    async def _async_run_node(self, name: str, input_value: Any) -> Any:
        step = self._nodes[name]
        if hasattr(step, 'async_run'):
            return await step.async_run(input_value)
        if hasattr(step, 'run'):
            loop = asyncio.get_running_loop()
            call = wrap_worker(lambda: step.run(input_value))
            return await loop.run_in_executor(None, call)
        result = step(input_value)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _get_node_input(self, name: str, results: Dict[str, Any], seed: Any) -> Any:
        parents = self._reverse.get(name, set())
        if not parents:
            return seed
        if len(parents) == 1:
            return results[next(iter(parents))]
        return tuple(results[p] for p in sorted(parents))

    def run(
        self,
        seed: Any = None,
        parallel: bool = False,
        *,
        on_step: Optional[StepHook] = None,
    ) -> Dict[str, Any]:
        """Execute graph synchronously."""
        results: Dict[str, Any] = {}

        with activate_context(self._context):
            if parallel:
                pool = _get_pool('thread')
                run_node = wrap_worker(self._run_node)
                for level in self._topo_levels():
                    if len(level) == 1:
                        name = level[0]
                        inp = self._get_node_input(name, results, seed)
                        t0 = time.perf_counter()
                        results[name] = self._run_node(name, inp)
                        _call_hook(on_step, name, inp, results[name], time.perf_counter() - t0)
                    else:
                        futures = {}
                        inputs = {}
                        starts = {}
                        for name in level:
                            inp = self._get_node_input(name, results, seed)
                            inputs[name] = inp
                            starts[name] = time.perf_counter()
                            futures[name] = pool.submit(run_node, name, inp)
                        for name, fut in futures.items():
                            results[name] = fut.result()
                            _call_hook(
                                on_step, name, inputs[name], results[name],
                                time.perf_counter() - starts[name],
                            )
            else:
                for name in self._topo_sort():
                    inp = self._get_node_input(name, results, seed)
                    t0 = time.perf_counter()
                    results[name] = self._run_node(name, inp)
                    _call_hook(on_step, name, inp, results[name], time.perf_counter() - t0)

        return results

    async def async_run(
        self,
        seed: Any = None,
        parallel: bool = True,
        *,
        on_step: Optional[StepHook] = None,
    ) -> Dict[str, Any]:
        """Execute graph asynchronously."""
        results: Dict[str, Any] = {}

        with activate_context(self._context):
            if parallel:
                for level in self._topo_levels():
                    if len(level) == 1:
                        name = level[0]
                        inp = self._get_node_input(name, results, seed)
                        t0 = time.perf_counter()
                        results[name] = await self._async_run_node(name, inp)
                        _call_hook(on_step, name, inp, results[name], time.perf_counter() - t0)
                    else:
                        tasks = {}
                        inputs = {}
                        starts = {}
                        for name in level:
                            inp = self._get_node_input(name, results, seed)
                            inputs[name] = inp
                            starts[name] = time.perf_counter()
                            tasks[name] = asyncio.create_task(
                                self._async_run_node(name, inp)
                            )
                        for name, task in tasks.items():
                            results[name] = await task
                            _call_hook(
                                on_step, name, inputs[name], results[name],
                                time.perf_counter() - starts[name],
                            )
            else:
                for name in self._topo_sort():
                    inp = self._get_node_input(name, results, seed)
                    t0 = time.perf_counter()
                    results[name] = await self._async_run_node(name, inp)
                    _call_hook(on_step, name, inp, results[name], time.perf_counter() - t0)

        return results

    def run_async(self, seed: Any = None, parallel: bool = True) -> Dict[str, Any]:
        """Execute graph asynchronously using uvloop when available."""
        from .async_runtime import run_async as _run_async
        return _run_async(self.async_run(seed, parallel=parallel))

    @classmethod
    def from_spec(
        cls,
        spec_file: str,
        *,
        registry: Optional[dict] = None,
    ) -> 'Graph':
        """Build a graph from a YAML or JSON spec file with a ``graph:`` block."""
        from .spec import load_graph_from_spec

        return load_graph_from_spec(spec_file, registry)

    def __repr__(self) -> str:
        n = len(self._nodes)
        e = sum(len(children) for children in self._edges.values())
        return f"Graph(nodes={n}, edges={e})"

    def describe(self) -> str:
        """Text visualization of graph structure."""
        lines = []
        order = self._topo_sort()
        for name in order:
            children = sorted(self._edges.get(name, set()))
            if children:
                lines.append(f"  {name} -> {', '.join(children)}")
            else:
                lines.append(f"  {name} (leaf)")
        return "Graph:\n" + "\n".join(lines)
