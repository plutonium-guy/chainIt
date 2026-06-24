# pipecraft Audit — Bugs & Incomplete Features

Audit date: 2026-06-24. Based on code review, runtime reproduction, and test coverage analysis.

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

**Location:** `src/pipecraft/step.py` — `_invoke_function_async` → `_execute_batched`

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

**Location:** `src/pipecraft/step.py` — `_execute_parallel_async`

**Impact:** `parallel` + async steps are broken on the async execution path. Sync `run()` is fine (thread pool runs sync callables).

---

### 3. `_async_run_branch_value` may not await `run()` coroutines

If a branch has `.run` but not `.async_run`, and `run()` returns a coroutine, it is returned without awaiting.

```python
# src/pipecraft/utils.py
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
| `Pipeline.from_spec("yaml")` | Raises `NotImplementedError` — tested, not built |
| `FanOutStep.async_run` + `parallel=` | Sync path uses thread/process pools; async path ignores `parallel` and always uses `asyncio.gather` |
| `MapReduceStep.async_run` | Works for sync mappers; no real parallelism for async mappers (sequential await in comprehension) |
| `jit` / `vectorize` on `@piped` | Silent no-op when numba/numpy missing — no warning |
| rsloop integration | Works when installed, but CI does not install or test `pipecraft[rsloop]` |

---

## Design limitations / footguns

**Circuit breaker half-open is permissive.** After recovery timeout, state moves to `HALF_OPEN` and all traffic is allowed until the next failure — there is no single-probe limit. Works for basic use, not full production breaker semantics.

**Sync timeouts leave worker threads running.** On timeout, `fut.result(timeout=...)` raises but the thread-pool worker keeps executing `_invoke_function`. Documented in code, but can leak work under load.

**Process pools are global and long-lived.** `_POOLS` only shuts down via `cleanup_pools()` (Pipeline context manager or explicit call). Long-running apps can accumulate idle pools.

**`@node` mutates class metadata** via `inst.__class__.__name__ = ...` — unusual pattern; each decorated function gets its own inner class so it works, but fragile.

**`retry` / `circuit_breaker` mutate steps in place** — reusing the same `PipeStep` across pipelines shares breaker/retry state.

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

1. **Fix async batch + async parallel** — highest severity; currently broken for async `@piped` steps.
2. **Fix `_async_run_branch_value`** — small, defensive fix.
3. **Decide on whole-collection auto-map** — product decision, then document or implement.
4. **Implement or remove `from_spec`** — dead API surface today.
5. **Align `FanOutStep.async_run` with `parallel=`** — feature parity gap.
