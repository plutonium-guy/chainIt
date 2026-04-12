# pipecraft

Composable function pipeline framework for Python. Pipe functions with `|`, build DAGs, branch conditionally, and run in parallel - with optional Rust acceleration.

```python
from pipecraft import piped

@piped
def add_one(x):
    return x + 1

@piped
def double(x):
    return x * 2

result = (add_one | double).run(5)  # 12
```

## Installation

```bash
pip install pipecraft
```

Rust acceleration is included automatically when installing from a wheel. For optional dependencies:

```bash
pip install pipecraft[numpy]    # numpy support
pip install pipecraft[all]      # numpy + numba JIT
```

## Features

- **`|` operator** to compose functions into pipelines
- **OOP nodes** with setup/teardown lifecycle
- **DAG execution** with topological sort and parallel scheduling
- **Conditional routing** (`ConditionalStep`, `SwitchStep`)
- **Fan-out/fan-in** for parallel branches
- **Retry + circuit breaker** for reliability
- **Async support** across every component
- **Rust-accelerated** graph operations (auto-detected)

## Usage

### Function Pipelines

```python
from pipecraft import piped, PIPE

@piped
def fetch(url):
    return requests.get(url).json()

@piped
def extract(data):
    return data["results"]

@piped(parallel='thread')
def process(items):
    return [transform(i) for i in items]

pipeline = fetch | extract | process
result = pipeline.run("https://api.example.com/data")

# Apply to multiple inputs
results = pipeline.map(["https://api.example.com/1", "https://api.example.com/2"])
```

Options for `@piped`:

```python
@piped(parallel='thread')     # thread pool for collections
@piped(parallel='process')    # process pool (pickleable functions)
@piped(batch_size=1024)       # batch incoming iterables
@piped(timeout=5.0)           # per-step timeout in seconds
@piped(schema=int)            # validate output type
@piped(jit=True)              # numba JIT compilation
@piped(vectorize=True)        # numpy vectorization
```

### OOP Nodes

Subclass `Node` for reusable components with lifecycle hooks:

```python
from pipecraft import Node

class DatabaseWriter(Node):
    def __init__(self, connection_string):
        self.conn_str = connection_string
        self.conn = None

    def setup(self):
        self.conn = connect(self.conn_str)

    def teardown(self):
        self.conn.close()

    def process(self, records):
        self.conn.insert_many(records)
        return len(records)

pipeline = fetch | extract | DatabaseWriter("postgres://...")
pipeline.run("https://api.example.com/data")
```

Quick nodes with the `@node` decorator:

```python
from pipecraft import node

@node
def double(x):
    return x * 2
```

### Argument Injection with PIPE

```python
from pipecraft import PIPE

@piped
def add(a, b):
    return a + b

add(3, PIPE).run(5)       # 8 -> add(3, 5)
add(a=PIPE, b=10).run(5)  # 15 -> add(5, 10)
```

### Conditional Branching

```python
from pipecraft import ConditionalStep, SwitchStep

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
from pipecraft import Graph

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
from pipecraft import FanOutStep, FanInStep

pipeline = (
    FanOutStep((branch_a, branch_b), parallel='thread')
    | FanInStep(lambda a, b: {"a": a, "b": b})
)
result = pipeline.run(input_data)
```

### Reliability

```python
from pipecraft import retry, circuit_breaker

@circuit_breaker(failure_threshold=3, recovery_timeout=60)
@retry(max_attempts=5, delay=0.5, backoff=2)
@piped
def call_api(data):
    return requests.post(url, json=data).json()
```

### Async

Every component supports async:

```python
import asyncio

@piped
async def fetch(url):
    async with aiohttp.ClientSession() as s:
        return await (await s.get(url)).json()

pipeline = fetch | process | save
result = asyncio.run(pipeline.async_run("https://api.example.com"))

# Parallel map
results = asyncio.run(pipeline.async_map(urls))
```

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
| `ExecutionResult` | Detailed run result (value, history, timing) |

### Pipeline Methods

```python
pipeline.run(seed)           # Execute synchronously
pipeline.async_run(seed)     # Execute asynchronously
pipeline.run_detailed(seed)  # Execute with timing/history
pipeline.map(items)          # Apply to each item
pipeline.async_map(items)    # Apply to each item (async)
pipeline.cancel()            # Cancel running pipeline
len(pipeline)                # Number of steps
pipeline[0]                  # Access step by index
```

## Rust Extension

The Rust extension is built automatically when installing from a wheel. It accelerates:

- `Graph._topo_sort()` - topological sort
- `Graph.roots` / `Graph.leaves` - graph introspection

Check availability:

```python
from pipecraft import HAS_RUST
print(HAS_RUST)  # True if Rust extension is loaded
```

Direct access:

```python
from pipecraft._rust import topo_sort, fast_map, batch_items, find_roots, find_leaves
```

## Development

```bash
git clone https://github.com/amiyamandal-dev/chainIt.git
cd chainIt
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
pytest tests/ -v
```

## License

MIT
