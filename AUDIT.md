# stepcraft Audit — Bugs & Incomplete Features

Audit date: 2026-06-24. Based on code review, runtime reproduction, and test coverage analysis.
Updated 2026-06-25 with findings from the multi-agent code-review pass on the
`working_barnch` feature diff (hooks, shared context, half-open breaker, beartype).

---

## Open findings — code-review pass (2026-06-25)

Verified findings against the feature diff. Ordered by severity. Each item lists
**what** is wrong and **how** to fix it.

### CR-1. Shared context never reaches pool workers (HIGH)

**What:** `Pipeline._activate_context` sets a `contextvars.ContextVar` on the
calling thread only. `pool.map` / `loop.run_in_executor` do **not** copy
contextvars into worker threads/processes, so any step that runs off the calling
thread sees `get_context() == {}`:
- `@piped(parallel='thread'|'process')` steps,
- `@piped(timeout=...)` steps (run via the thread pool),
- sync functions executed inside an async pipeline (`run_in_executor`),
- auto-mapped/batched work dispatched to executors.

README presents `get_context()` as the supported way for steps to read context,
so this is a silent correctness gap.

**Location:** `context.py:12`, `pipeline.py` (`_activate_context`, run loops),
`step.py` (`_execute_parallel`, `_execute_parallel_async`, `_execute_sync`
timeout branch, `_execute_auto_map*`, `_execute_batched*`).

**How to fix:** capture the active context at dispatch and run workers inside it.
- Threads: wrap submitted callables with `contextvars.copy_context().run(fn, ...)`,
  or snapshot `get_context()` on the calling thread and re-set it inside the
  worker wrapper.
- Processes: contextvars cannot cross the pickle boundary — pass the context dict
  explicitly to the worker and re-set it there, or document that process pools do
  not see `get_context()`.
- Add tests: `parallel='thread'`, `timeout=`, and async-with-sync-func steps all
  reading `get_context()` inside a `Pipeline(context=...)`.

### CR-2. Half-open probe slot consumed before cancel check → breaker stuck (HIGH)

**What:** In `_preflight_check`, `_check_circuit_breaker()` runs first and
increments `_half_open_calls`; the cancel-event check runs *after*. If the call
is cancelled, the probe slot is consumed but neither `_record_failure` nor
`_reset_circuit_breaker` runs, leaving the breaker `HALF_OPEN` with the slot used
**forever** — every later call raises `CircuitBreakerError` even once healthy.

**Location:** `step.py` `_preflight_check` / `_check_circuit_breaker` (~L155, L387–391).

**How to fix:** check the cancel event **before** consuming a probe slot (reorder
`_preflight_check` so the cancel check precedes `_check_circuit_breaker`), or
release the slot on cancellation (decrement `_half_open_calls` / restore state in
a `finally`/except path when `CancelledError` is raised before the call runs).
Add a regression test: half-open + cancel set → breaker still recovers later.

### CR-3. Breaker reset happens before the schema check (MEDIUM)

**What:** `_finalize_success` calls `_reset_circuit_breaker()` and *then* validates
`schema`. A half-open probe whose function returns a wrong-typed value first
closes the breaker, then raises `TypeError`, which records a failure against a
now-`CLOSED` breaker (count from 0) instead of re-opening immediately.

**Location:** `step.py:169–179` (`_finalize_success`).

**How to fix:** validate the schema **before** calling `_reset_circuit_breaker()`
so a schema failure is treated like any other probe failure. Add a test:
`circuit_breaker` + `schema` where the recovered call returns the wrong type →
breaker re-opens rather than closing.

### CR-4. Graph never activates shared context (MEDIUM)

**What:** Context activation was added to `Pipeline` only. `Graph.run` /
`Graph.async_run` have no `_activate_context`, so a `@piped` node that reads
`get_context()` gets `{}` in a DAG even when run sequentially — inconsistent with
`Pipeline` and with the docs.

**Location:** `graph.py:120` (`run`) and `graph.py` `async_run`.

**How to fix:** give `Graph` an optional `context=` and wrap both run paths in the
same activation, ideally by extracting a shared `_activate_context` helper (e.g.
into `context.py`) reused by `Pipeline` and `Graph`. Note this composes with CR-1
(workers still need propagation). Add a Graph context test.

### CR-5. `on_step` missing on half the public run surface (MEDIUM)

**What:** `on_step` was threaded through `run` / `async_run` / `run_detailed` /
`async_run_detailed` only. `run_async`, `map`, `async_map`, `map_async` ignore it;
`pipeline.run_async(seed, on_step=hook)` raises `TypeError`. README says "Pass
`on_step` to any run method."

**Location:** `pipeline.py` (`run_async`, `map`, `async_map`, `map_async`).

**How to fix:** forward `on_step` from `map`/`async_map` into per-item `run`/
`async_run`, and from `run_async`/`map_async` into the underlying coroutine; OR
narrow the README to list exactly the four methods that accept it. Prefer
forwarding for consistency. Add coverage for at least `map(on_step=...)`.

### CR-6. Spec with both `graph:` and `steps:` builds a silent linear pipeline (LOW)

**What:** `build_pipeline_from_spec` only rejects a graph spec when `steps` is
absent. A spec containing **both** keys silently builds a linear `Pipeline` and
ignores the `graph:` block (wrong execution order / fan-in inputs, no error).

**Location:** `spec.py:144` (`build_pipeline_from_spec`).

**How to fix:** raise `ValueError` when both `graph` and `steps` are present
(ambiguous spec). Mirror the check in `build_graph_from_spec`. Add a test for the
both-keys case.

### CR-7. `_half_open_calls` increment is not atomic across threads (LOW)

**What:** `_check_circuit_breaker` does a non-atomic read-check-increment on
`_half_open_calls`. When one `PipeStep` is invoked concurrently (parallel graph
levels, `parallel='thread'` fan-out), multiple threads can each read `0` and all
proceed, exceeding `half_open_max_calls`. Pre-existing class of issue — breaker
state was never lock-guarded — but the new probe quota relies on it.

**Location:** `step.py:384–391` (`_check_circuit_breaker`, `_record_failure`).

**How to fix:** guard breaker state transitions with a `threading.Lock` on the
step (note: a `frozen`/`object.__setattr__` dataclass needs a non-init lock
field). Lower priority unless shared concurrent steps are a supported pattern;
otherwise document that breaker state is not thread-safe.

### Intentional behavior changes (not bugs — confirm and keep)

- **Single-probe half-open semantics** (`step.py` `_check_circuit_breaker`): the
  breaker now allows only `half_open_max_calls` (default 1) probes and re-opens on
  the first probe failure. This is the requested Phase 2.4 feature, **not** a
  regression — but CR-2/CR-3/CR-7 above are real defects within it.
- **beartype runtime type-checking** (`__init__.py`): every function/method is
  decorated via the import hook; calls that violate annotations now raise
  `BeartypeCallHintParamViolation`. Intended. Caveat to document: it is always-on
  with no opt-out env var, and the numeric tower is enabled (int accepted for
  float). Consider an opt-out hook (e.g. `PIPECRAFT_NO_BEARTYPE`) if consumers
  need to disable runtime checks in production.

---

## Confirmed bugs

### 1. Async + `batch_size` returns unawaited coroutines

When an **async** function is used with `batch_size > 1`, `_invoke_function_async` calls the sync `_execute_batched`, which invokes `self.func(batch)` without awaiting.

```python
@piped(batch_size=2)
async def process_batch(batch):
    return [x * 2 for x in batch]

await process_batch.async_run([1, 2, 3, 4])
# -> [<coroutine ...>, <coroutine ...>]  # RuntimeWarning: never awaited
```

**Location:** `src/stepcraft/step.py` — `_invoke_function_async` → `_execute_batched`

**Impact:** `async_run` silently returns coroutine objects instead of results.

---

### 2. Async + `parallel='thread'` has the same problem

The parallel async path uses `run_in_executor(pool, call, item)`, which always invokes the **sync** wrapper. Async `@piped` functions therefore produce unawaited coroutines.

```python
@piped(parallel='thread')
async def f(x):
    return x * 2

await f.async_run([1, 2, 3])
# -> [<coroutine ...>, <coroutine ...>, <coroutine ...>]
```

**Location:** `src/stepcraft/step.py` — `_execute_parallel_async`

**Impact:** `parallel` + async steps are broken on the async execution path. Sync `run()` is fine (thread pool runs sync callables).

---

### 3. `_async_run_branch_value` may not await `run()` coroutines

If a branch has `.run` but not `.async_run`, and `run()` returns a coroutine, it is returned without awaiting.

```python
# src/stepcraft/utils.py
async def _async_run_branch_value(branch, value):
    ...
    if hasattr(branch, 'run'):
        return branch.run(value)   # coroutine not awaited
    result = branch(value)
    if asyncio.iscoroutine(result):
        return await result
```

**Impact:** Uncommon, but custom steps with only sync `run()` that return coroutines will misbehave in `ConditionalStep` / `SwitchStep` async paths.

---

## Known deferred behavior (documented, still open)

### 4. Whole collections are not auto-mapped

Passing a list into a normal step sends the **entire list** as one argument. Parallel/batch only kick in when explicitly configured.

```python
@piped
def step(x): ...

pipeline.run([1, 2, 3])  # step receives [1,2,3], not three scalars
```

**Status:** Intentionally deferred — see `CODE_REVIEW_CHECKLIST.md` line 10. Surprising if users expect implicit map behavior.

---

## Incomplete features

| Feature | Status |
|--------|--------|
| `Pipeline.from_spec("yaml")` | ✅ Implemented (`spec.py`, `stepcraft[spec]`) |
| `Graph.from_spec("yaml")` | ✅ Implemented — `graph:` block with `nodes`/`edges` |
| `FanOutStep.async_run` + `parallel=` | ✅ Honors `parallel` on the async path |
| `MapReduceStep.async_run` | ✅ Async mappers mapped concurrently per batch |
| `jit` / `vectorize` on `@piped` | ✅ Warns when numba/numpy missing |
| rsloop integration | ✅ Covered by the `test-rsloop` CI job |
| Step hooks (`on_step`) | ✅ `Pipeline` + `Graph` run paths, see `hooks.py` |
| Pipeline shared context | ✅ `Pipeline(steps, context=...)` + `get_context()` |
| Configurable pools | ✅ `configure_pools(thread_workers=, process_workers=)` |
| Runtime type-checking | ✅ beartype decorates every function via import hook |

---

## Design limitations / footguns

**Circuit breaker half-open is single-probe (implemented; has open defects).** After the recovery timeout the breaker moves to `HALF_OPEN` and allows up to `CircuitBreakerConfig.half_open_max_calls` (default 1) trial calls; success closes it, failure re-opens it immediately. See `step.py` `_check_circuit_breaker` / `_record_failure`. ⚠️ Three defects in this logic are open — see **CR-2** (probe slot consumed before cancel check → permanent lock), **CR-3** (reset before schema check), and **CR-7** (non-atomic probe increment) above.

**Sync timeouts leave worker threads running.** On timeout, `fut.result(timeout=...)` raises but the thread-pool worker keeps executing `_invoke_function`. `@piped(cancel_on_timeout=True)` makes a best-effort `fut.cancel()`, which only helps if the worker has not started — a running thread cannot be interrupted. Documented footgun under load.

**Process pools are global and long-lived.** `_POOLS` only shuts down via `cleanup_pools()` (Pipeline context manager or explicit call). Long-running apps can accumulate idle pools.

**`@node` mutates class metadata** via `inst.__class__.__name__ = ...` — unusual pattern; each decorated function gets its own inner class so it works, but fragile.

**`retry` / `circuit_breaker` are immutable (resolved).** Both return a **new** `PipeStep` via `PipeStep.copy()` with fresh breaker counters, so decorating the same step twice or reusing across pipelines no longer shares state.

**`parallel` + `batch_size` together** — parallel wins; batching is skipped. Documented and tested, but easy to misconfigure.

---

## Test coverage gaps

No tests for:

- async `batch_size`
- async `parallel='thread'` / `'process'` with async functions
- `FanOutStep.async_run` with `parallel=`
- `MapReduceStep.async_run` with async mapper/reducer
- half-open circuit breaker probe semantics
- rsloop in CI (only optional local tests)

---

## What looks solid

Previously flagged checklist items appear fixed and covered by tests:

- Circuit breaker records one failure per logical call (not per retry)
- Async timeout honors `timeout=0.0`
- Parallel forwards kwargs (sync and async)
- Batched does not sum booleans
- FanOut process parallel uses picklable helper
- Topo sort cached; deque used instead of `list.pop(0)`
- Shared thread pool for sync timeouts
- Branch helpers deduplicated

**89 tests pass** locally with rsloop installed (3 skipped).

---

## Suggested priority

The async/spec/parallel items from the original audit are now resolved (see
**Incomplete features**). Remaining work is the code-review pass:

1. **CR-1** — propagate shared context into pool workers (or document the limit). Highest user-facing impact.
2. **CR-2** — half-open probe consumed before cancel check → breaker permanently stuck. Correctness.
3. **CR-3** — move schema validation before breaker reset.
4. **CR-4** — activate shared context in `Graph` (extract a shared helper).
5. **CR-5** — forward `on_step` through `map`/`async_map`/`run_async`/`map_async` (or narrow the docs).
6. **CR-6** — reject specs that contain both `graph:` and `steps:`.
7. **CR-7** — lock breaker state (or document non-thread-safety). Lowest priority.
