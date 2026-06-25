# stepcraft

Composable function pipeline framework for Python. Pipe functions with `|`, build DAGs, branch conditionally, and run in parallel.

```python
from stepcraft import piped

@piped
def add_one(x: int) -> int:
    return x + 1

@piped
def double(x: int) -> int:
    return x * 2

result = (add_one | double).run(5)  # 12
```

> stepcraft requires every `@piped`/`@node` function (and `Node` subclass) to be
> fully type-annotated by default — see [Type annotations](#type-annotations).

## Installation

```bash
pip install stepcraft
```

For optional dependencies:

```bash
pip install stepcraft[numpy]    # numpy support
pip install stepcraft[uvloop]   # fast libuv-based event loop (Unix)
pip install stepcraft[all]      # numpy + numba + uvloop
```

## Features

- **`|` operator** to compose functions into pipelines
- **OOP nodes** with setup/teardown lifecycle
- **DAG execution** with topological sort and parallel scheduling
- **Conditional routing** (`ConditionalStep`, `SwitchStep`)
- **Fan-out/fan-in** for parallel branches
- **Retry + circuit breaker** for reliability
- **Async support** across every component

## Usage

### Function Pipelines

```python
from stepcraft import piped, PIPE

@piped
def fetch(url: str) -> dict:
    return requests.get(url).json()

@piped
def extract(data: dict) -> list:
    return data["results"]

@piped(parallel='thread')
def process(items: list) -> list:
    return [transform(i) for i in items]

pipeline = fetch | extract | process
result = pipeline.run("https://api.example.com/data")

# Apply to multiple inputs
results = pipeline.map(["https://api.example.com/1", "https://api.example.com/2"])
```

Options for `@piped`:

```python
@piped(parallel='thread')         # thread pool for collections
@piped(parallel='process')        # process pool (pickleable functions)
@piped(parallel='auto')           # threads on free-threaded 3.14t, else process
@piped(batch_size=1024)           # batch incoming iterables
@piped(timeout=5.0)               # per-step timeout in seconds
@piped(cancel_on_timeout=True)    # best-effort cancel on timeout (see Parallelism)
@piped(map=False)                 # disable per-element auto-mapping of list inputs
@piped(schema=int)                # validate output type
@piped(jit=True)                  # numba JIT compilation
@piped(vectorize=True)            # numpy vectorization
```

### Auto-mapping list inputs

A default `@piped` step applied to a **list** maps element-wise; pass `map=False`
(alias of `auto_map=False`) to receive the whole list as one argument. Tuples are
treated as structural values and are never split (e.g. graph fan-in).

```python
@piped
def double(x: int) -> int:
    return x * 2

double.run([1, 2, 3])              # [2, 4, 6]  — auto-mapped

@piped(map=False)
def total(xs: list) -> int:
    return sum(xs)

total.run([1, 2, 3])              # 6  — whole list
```

### OOP Nodes

Subclass `Node` for reusable components with lifecycle hooks:

```python
from stepcraft import Node

class DatabaseWriter(Node):
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.conn = None

    def setup(self):
        self.conn = connect(self.conn_str)

    def teardown(self):
        self.conn.close()

    def process(self, records: list) -> int:
        self.conn.insert_many(records)
        return len(records)

pipeline = fetch | extract | DatabaseWriter("postgres://...")
pipeline.run("https://api.example.com/data")
```

Quick nodes with the `@node` decorator:

```python
from stepcraft import node

@node
def double(x: int) -> int:
    return x * 2
```

### Argument Injection with PIPE

```python
from stepcraft import PIPE

@piped
def add(a: int, b: int) -> int:
    return a + b

add(3, PIPE).run(5)       # 8 -> add(3, 5)
add(a=PIPE, b=10).run(5)  # 15 -> add(5, 10)
```

### Conditional Branching

```python
from stepcraft import ConditionalStep, SwitchStep

# If/else
cond = ConditionalStep(
    condition=lambda x: x > 0,
    if_true=piped(lambda x: x * 2),
    if_false=piped(lambda x: -x),
)

# Multi-branch
switch = SwitchStep(
    key=lambda x: "high" if x > 100 else "low",
    branches={
        "high": piped(lambda x: x * 0.9),   # discount
        "low": piped(lambda x: x * 1.1),    # markup
    },
    default=piped(lambda x: x),
)

pipeline = normalize | switch | format_output
```

### DAG Execution

```python
from stepcraft import Graph

g = (
    Graph()
    .add_node("fetch", fetch)
    .add_node("parse", parse)
    .add_node("validate", validate)
    .add_node("save", save)
    .add_edge("fetch", "parse")
    .add_edge("parse", "validate")
    .add_edge("parse", "save")
    .add_edge("validate", "save")
)

# Sequential
results = g.run(seed=url)

# Parallel (independent nodes run concurrently)
results = g.run(seed=url, parallel=True)

# Async (parallel by default)
results = await g.async_run(seed=url)

# Inspect
print(g.roots)     # ['fetch']
print(g.leaves)    # ['save']
print(g.describe())
# Graph:
#   fetch -> parse
#   parse -> save, validate
#   validate -> save
#   save (leaf)
```

### Fan-out / Fan-in

```python
from stepcraft import FanOutStep, FanInStep

pipeline = (
    FanOutStep((branch_a, branch_b), parallel='thread')
    | FanInStep(lambda a, b: {"a": a, "b": b})
)
result = pipeline.run(input_data)
```

### Reliability

```python
from stepcraft import retry, circuit_breaker

@circuit_breaker(failure_threshold=3, recovery_timeout=60)
@retry(max_attempts=5, delay=0.5, backoff=2)
@piped
def call_api(data: dict) -> dict:
    return requests.post(url, json=data).json()
```

### Async

Every component supports async. For better performance, install [uvloop](https://github.com/MagicStack/uvloop) (libuv-based asyncio event loop):

```bash
pip install stepcraft[uvloop]
```

```python
from stepcraft import piped, run_async

@piped
async def fetch(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        return await (await s.get(url)).json()

pipeline = fetch | process | save

# Uses uvloop when installed, falls back to asyncio.run otherwise
result = pipeline.run_async("https://api.example.com")
results = pipeline.map_async(urls)

# Or run any coroutine directly
result = run_async(pipeline.async_run("https://api.example.com"))
```

You can also install uvloop as the default event loop policy:

```python
from stepcraft import uvloop_policy

with uvloop_policy():
    result = run_async(pipeline.async_run(seed))
```

### Declarative specs (YAML / JSON)

Build pipelines and graphs from a spec file (`pip install stepcraft[spec]`):

```python
from stepcraft import Pipeline, Graph

pipe = Pipeline.from_spec("pipeline.yaml")   # { steps: [...] }
graph = Graph.from_spec("graph.yaml")        # { graph: { nodes:, edges: } }
```

```yaml
# pipeline.yaml
steps:
  - import: mypkg:fetch
  - import: mypkg:parse
    parallel: thread
context:
  request_id: abc          # optional shared context

# graph.yaml
graph:
  nodes:
    a: { import: mypkg:fetch }
    b: { import: mypkg:parse }
  edges:
    - [a, b]
```

### Step hooks (observability)

Pass `on_step` to any run method to observe each step without wrapping functions.
Hook errors are logged and never break the run.

```python
def on_step(name, step_input, step_output, step_dt):
    print(f"{name} took {step_dt*1000:.1f}ms")

pipeline.run(seed, on_step=on_step)
await pipeline.async_run(seed, on_step=on_step)
graph.run(seed, on_step=on_step)
```

### Shared context

Attach a context dict to a pipeline; steps (and `Node.setup`) read it via `get_context()`.
Context is set on run entry and reset on exit.

```python
from stepcraft import Pipeline, piped, get_context

@piped
def step(x):
    rid = get_context().get("request_id")
    return x

Pipeline([step], context={"request_id": "abc"}).run(1)
```

### Parallelism: GIL vs free-threading vs processes

- `parallel='thread'` — best for I/O-bound work. On the standard (GIL) build,
  threads do **not** give CPU parallelism for pure-Python code.
- `parallel='process'` — true multi-core for CPU-bound work; functions must be
  pickleable (use module-level functions, not lambdas).
- `parallel='auto'` — resolves to `'thread'` on free-threaded CPython 3.14t
  (no-GIL, true multi-core threads) and to `'process'` otherwise.

On a free-threaded 3.14t interpreter, `parallel='process'` is automatically
downgraded to `'thread'`, since threads already provide true parallelism.

```python
from stepcraft import (
    HAS_FREE_THREADING, is_gil_enabled, threads_provide_true_parallelism,
    configure_pools, cleanup_pools,
)

# Tune pool sizes (applies on next pool creation; call cleanup_pools() to recreate)
configure_pools(thread_workers=8, process_workers=4)
```

### Type annotations

**Annotations are required by default.** Every `@piped`/`@node` function and
every `Node` subclass's `process` must annotate all parameters and the return
value, or stepcraft raises `MissingAnnotationError` at decoration time:

```python
@piped
def add_tax(amount: float, rate: float = 0.1) -> float:
    return amount * (1 + rate)
```

Those annotations are then enforced at runtime by
[beartype](https://beartype.readthedocs.io/): arguments and the return value are
checked on every call (including each element of an auto-mapped / parallel run).
The implicit numeric tower is enabled, so `int` is accepted where `float` is
annotated.

Exemptions and opt-outs:

- **Lambdas are exempt** — they cannot carry annotations.
- `@piped(require_annotations=False)` / `@node(require_annotations=False)` opts a
  single step out.
- `@piped(typecheck=False)` skips beartype on that step while keeping annotations
  required (unless you also opt out of annotations).
- On a `Node` subclass, set `require_annotations = False` as a class attribute.

For explicit, beartype-independent output validation use `@piped(schema=int)`;
under auto-map/parallel the schema is checked per element.

Disable runtime type-checking entirely when needed (annotations are still
required unless you also opt out per step):

```bash
STEPCRAFT_NO_BEARTYPE=1 python my_app.py
```

Internal stepcraft machinery is **not** beartype-decorated — only your step
functions are, keeping overhead on your pipeline logic rather than the framework.

## API Reference

| Function / Class | Description |
|---|---|
| `piped(func, **opts)` | Wrap function as pipeline step |
| `node(func)` | Wrap function as OOP node |
| `retry(**opts)` | Add retry to a step |
| `circuit_breaker(**opts)` | Add circuit breaker to a step |
| `Pipeline(steps)` | Linear pipeline (usually built with `\|`) |
| `Node` | Abstract base class for OOP steps |
| `Graph` | DAG-based pipeline |
| `ConditionalStep` | If/else branching |
| `SwitchStep` | Multi-branch routing |
| `FanOutStep` | Broadcast to parallel branches |
| `FanInStep` | Merge branch outputs |
| `MapReduceStep` | Batched map-reduce |
| `Pipeline.from_spec(file)` | Build a pipeline from a YAML/JSON spec |
| `Graph.from_spec(file)` | Build a graph from a YAML/JSON `graph:` spec |
| `get_context()` | Read the active pipeline's shared context |
| `configure_pools(...)` | Set thread/process pool worker counts |
| `StepHook` | `(name, input, output, dt)` callback type for `on_step` |
| `run_async(coro)` | Run coroutine via uvloop (or asyncio fallback) |
| `pipeline.run_async(seed)` | Sync wrapper around `async_run` |
| `pipeline.map_async(items)` | Sync wrapper around `async_map` |
| `Graph.run_async(seed)` | Sync wrapper around graph `async_run` |
| `ExecutionResult` | Detailed run result (value, history, timing) |

### Pipeline Methods

```python
pipeline.run(seed, on_step=hook)          # Execute synchronously
pipeline.async_run(seed, on_step=hook)    # Execute asynchronously
pipeline.run_async(seed)                  # async_run via uvloop (when installed)
pipeline.run_detailed(seed, on_step=hook) # Execute with timing/history
pipeline.async_run_detailed(seed)         # Async timing/history
pipeline.map(items)                       # Apply to each item
pipeline.map(items, parallel=True)        # Thread-pool parallel map
pipeline.map(items, parallel=4)           # Bounded thread concurrency
pipeline.async_map(items)                 # Apply to each item (async)
pipeline.async_map(items, max_concurrency=32)  # Cap in-flight async runs
pipeline.map_async(items)                 # async_map via uvloop (when installed)
pipeline.cancel()            # Cancel running pipeline
len(pipeline)                # Number of steps
pipeline[0]                  # Access step by index
```

### Production performance tuning

Recommended install for throughput workloads:

```bash
pip install "stepcraft[all]"   # numpy, numba, uvloop, pyyaml
```

Tuning checklist:

| Workload | Setting |
|---|---|
| I/O-bound list processing | `@piped(parallel='thread')` or `pipeline.map(items, parallel=True)` |
| CPU-bound, pickleable funcs | `@piped(parallel='process')` with module-level functions |
| Free-threaded CPython 3.14t | `parallel='auto'` (threads already parallelize CPU work) |
| Large async fan-out | `pipeline.async_map(..., max_concurrency=32)` or `@piped(max_concurrency=32)` |
| Expensive one-time setup | `class MyNode(Node): setup_once = True` |
| Hot inner loops | `STEPCRAFT_NO_BEARTYPE=1` or `@piped(typecheck=False)` |
| Pool sizing | `configure_pools(thread_workers=8, process_workers=4)` then `cleanup_pools()` to apply |
| DAG parallelism | `Graph.run(parallel=True)` / `Graph.async_run(parallel=True)` |
| Chunked transforms | `@piped(batch_size=N)` (do not combine with `parallel`) |

```python
from stepcraft import configure_pools, cleanup_pools

configure_pools(thread_workers=8, process_workers=4)
# Recreate pools after changing worker counts:
cleanup_pools()
```

## Development

```bash
git clone https://github.com/amiyamandal-dev/chainIt.git
cd chainIt
python -m venv .venv && source .venv/bin/activate
pip install pytest numpy pyyaml beartype
pip install -e .
pytest tests/ -v
```

## License

MIT
