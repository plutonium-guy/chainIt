from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .branching import ConditionalStep, SwitchStep
from .decorators import piped
from .fan import FanInStep, FanOutStep, MapReduceStep
from .step import PipeStep


def _load_spec_file(spec_file: str | Path) -> dict:
    path = Path(spec_file)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline spec not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML pipeline specs. "
                "Install with: pip install pipecraft[spec]"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported pipeline spec format {suffix!r}; use .yaml, .yml, or .json"
        )

    if not isinstance(data, dict):
        raise ValueError("Pipeline spec must be a mapping at the top level")
    return data


def _resolve_callable(
    ref: str,
    registry: Optional[Mapping[str, Any]] = None,
) -> Any:
    if registry and ref in registry:
        return registry[ref]

    if ":" not in ref:
        raise ValueError(
            f"Invalid import ref {ref!r}; expected 'module.path:callable'"
        )
    module_path, attr = ref.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _piped_options(entry: dict) -> dict:
    opts = dict(entry.get("piped") or {})
    for key in (
        "batch_size", "parallel", "auto_map", "map", "timeout",
        "cancel_on_timeout", "schema", "jit", "vectorize",
    ):
        if key in entry and key not in opts:
            opts[key] = entry[key]
    return opts


def _build_piped_step(target: Any, entry: dict) -> PipeStep:
    opts = _piped_options(entry)
    if isinstance(target, PipeStep):
        if opts:
            raise ValueError("Cannot apply piped options to an existing PipeStep")
        return target
    if not callable(target):
        raise TypeError(f"Step target must be callable, got {type(target)!r}")
    return piped(target, **opts)


def _build_step(
    entry: Any,
    registry: Optional[Mapping[str, Any]] = None,
) -> Any:
    if isinstance(entry, str):
        entry = {"import": entry}
    if not isinstance(entry, dict):
        raise TypeError(f"Each step must be a mapping or import string, got {type(entry)!r}")

    step_type = entry.get("type", "piped")

    if step_type == "piped":
        if "import" not in entry:
            raise ValueError("piped steps require an 'import' field")
        return _build_piped_step(_resolve_callable(entry["import"], registry), entry)

    if step_type == "conditional":
        return ConditionalStep(
            condition=_resolve_callable(entry["condition"], registry),
            if_true=_build_step(entry["if_true"], registry),
            if_false=(
                _build_step(entry["if_false"], registry)
                if entry.get("if_false") is not None
                else None
            ),
        )

    if step_type == "switch":
        branches = {
            key: _build_step(branch, registry)
            for key, branch in entry["branches"].items()
        }
        default = (
            _build_step(entry["default"], registry)
            if entry.get("default") is not None
            else None
        )
        return SwitchStep(
            key=_resolve_callable(entry["key"], registry),
            branches=branches,
            default=default,
        )

    if step_type == "fan_out":
        branches = tuple(_build_step(branch, registry) for branch in entry["branches"])
        return FanOutStep(branches=branches, parallel=entry.get("parallel"))

    if step_type == "fan_in":
        return FanInStep(combiner=_resolve_callable(entry["combiner"], registry))

    if step_type == "map_reduce":
        return MapReduceStep(
            mapper=_resolve_callable(entry["mapper"], registry),
            reducer=_resolve_callable(entry["reducer"], registry),
            batch_size=entry.get("batch_size", 1),
        )

    raise ValueError(f"Unknown pipeline step type: {step_type!r}")


def build_pipeline_from_spec(
    spec: dict,
    registry: Optional[Mapping[str, Any]] = None,
):
    if "graph" in spec and "steps" not in spec:
        raise ValueError(
            "Spec defines a 'graph:' block; load it with Graph.from_spec(), "
            "not Pipeline.from_spec()"
        )
    steps_spec = spec.get("steps")
    if not isinstance(steps_spec, list) or not steps_spec:
        raise ValueError("Pipeline spec must include a non-empty 'steps' list")

    from .pipeline import Pipeline

    steps = [_build_step(entry, registry) for entry in steps_spec]
    context = spec.get("context")
    if context is not None and not isinstance(context, dict):
        raise ValueError("Pipeline spec 'context' must be a mapping")
    return Pipeline(steps, context=context)


def build_graph_from_spec(
    spec: dict,
    registry: Optional[Mapping[str, Any]] = None,
):
    graph_spec = spec.get("graph")
    if not isinstance(graph_spec, dict):
        raise ValueError("Graph spec must include a 'graph' mapping")

    nodes = graph_spec.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("Graph spec must include a non-empty 'nodes' mapping")

    from .graph import Graph

    graph = Graph()
    for name, entry in nodes.items():
        graph.add_node(name, _build_step(entry, registry))

    for edge in graph_spec.get("edges", []) or []:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f"Each edge must be a [from, to] pair, got {edge!r}")
        graph.add_edge(edge[0], edge[1])

    return graph


def load_pipeline_from_spec(
    spec_file: str | Path,
    registry: Optional[Mapping[str, Any]] = None,
):
    spec = _load_spec_file(spec_file)
    return build_pipeline_from_spec(spec, registry)


def load_graph_from_spec(
    spec_file: str | Path,
    registry: Optional[Mapping[str, Any]] = None,
):
    spec = _load_spec_file(spec_file)
    return build_graph_from_spec(spec, registry)
