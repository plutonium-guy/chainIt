# pipecraft — Claude Code Handoff Plan

**Author:** Senior review / planning pass  
**Date:** 2026-06-24  
**Sub-agent:** claude-code (available after quota reset ~12:50 AM Asia/Calcutta)  
**Branch:** `working_barnch`

This file is the **source of truth** for what is done, what was started tonight, and what claude-code should build next. Read this before any feature work.

---

## Already shipped (do not redo)

| Area | Status | Key commits / files |
|------|--------|---------------------|
| Async `batch_size` awaits coroutines | Done | `step._execute_batched_async` |
| Async `parallel` awaits coroutines | Done | `step._execute_parallel_async` |
| `_async_run_branch_value` awaits `run()` coroutines | Done | `utils.py` |
| List auto-map on default `@piped` | Done | `step._should_auto_map`, lists only (not tuples) |
| `Pipeline.from_spec` YAML/JSON | Done | `spec.py`, `pip install pipecraft[spec]` |
| FanOut / MapReduce async parity | Done | `fan.py` |
| jit/vectorize warnings | Done | `decorators.py` |
| rsloop CI job | Done | `.github/workflows/ci.yml` `test-rsloop` |
| Free-threaded 3.14 parallel | Done | `runtime.py`, `parallel='auto'` |

**Tests:** 114+ passed locally. Run `uv run pytest tests/ -v` before every PR.

---

## Started tonight (manager pass — verify / extend)

Claude-code should **read the diff** and finish tests/docs if incomplete:

### A. `@piped(auto_map=False)` opt-out
- **Why:** Steps that intentionally receive a whole list must not be split.
- **Files:** `decorators.py`, `step.py`, `tests/test_pipeline.py`
- **Acceptance:**
  - `@piped(auto_map=False)` passes `[1,2,3]` as one argument
  - Default remains auto-map on multi-item lists
  - `__call__` / partial binding preserves `auto_map`

### B. Immutable `retry` / `circuit_breaker`
- **Why:** Decorating the same `PipeStep` twice or reusing across pipelines must not share breaker counters.
- **Files:** `step.py` (`PipeStep.copy()`), `decorators.py`
- **Acceptance:**
  - `@retry` / `@circuit_breaker` return a **new** `PipeStep` via `copy()`
  - Fresh `_circuit_state`, `_failure_count`, `_last_failure_time`
  - Existing tests still pass; add test that two decorated copies do not share breaker state

### C. `Pipeline.async_run_detailed()`
- **Why:** Parity with `run_detailed()` for async pipelines.
- **Files:** `pipeline.py`, `tests/test_pipeline.py`
- **Acceptance:**
  - Returns `ExecutionResult` with history, `dt`, `n`
  - Works with async `@piped` steps and sync steps (executor fallback in `async_run`)

---

## Phase 1 — Claude-code (high priority, ~1 session)

### 1. Step hooks / observability
**Goal:** Callback after each step without wrapping functions.

```python
# API sketch (finalize in hooks.py)
StepHook = Callable[[str, Any, Any, float], None]  # name, input, output, step_dt

pipeline.run(seed, on_step=hook)
pipeline.run_detailed(seed, on_step=hook)  # hook in addition to history
await pipeline.async_run(seed, on_step=hook)
await pipeline.async_run_detailed(seed, on_step=hook)
```

**Files to add/touch:**
- `src/pipecraft/hooks.py` — `StepHook` type alias, `_call_hook` helper
- `pipeline.py` — optional `on_step` on run paths
- `graph.py` — optional `on_step` on `run` / `async_run` (per node)
- `tests/test_pipeline.py` — hook called N times, receives correct I/O

**Rules:**
- Hook errors must **not** swallow pipeline errors (wrap in try/except log or re-raise based on strict flag — default: log warning, continue)
- `on_step` is optional; no breaking changes to existing signatures (use keyword-only)

### 2. Update `AUDIT.md` + `README.md`
- Mark fixed bugs incomplete features as resolved
- Document: `auto_map`, `parallel='auto'`, `from_spec`, free-threading, `pip install pipecraft[spec]`
- Add short “Parallelism” section: GIL vs 3.14t vs `process`

### 3. `@piped(map=False)` alias
- If manager used `auto_map`, add `map: Optional[bool] = None` where `map=False` ≡ `auto_map=False` for README-friendly naming (single implementation).

---

## Phase 2 — Claude-code (production hardening)

### 4. Circuit breaker single-probe half-open
**Location:** `step.py` `_check_circuit_breaker`, `_record_failure`, `_reset_circuit_breaker`

**Behavior:**
- `HALF_OPEN`: allow **one** trial call; success → `CLOSED`, failure → `OPEN`
- Add `CircuitBreakerConfig.half_open_max_calls: int = 1` (optional, default 1)
- Tests: `test_circuit_breaker_half_open_single_probe`

### 5. Configurable thread/process pools
**Location:** `pools.py`

```python
def configure_pools(*, thread_workers: int | None = None, process_workers: int | None = None) -> None
```

- Apply on next `_get_pool` creation; document that reconfigure requires `cleanup_pools()` first
- Test: pool respects max_workers when set

### 6. Sync timeout cancellation note + optional flag
- Document that timed-out sync work keeps running in thread pool (AUDIT footgun)
- Optional: `cancel_on_timeout=True` on `@piped` — best-effort, document limitations
- **Low priority** if complex; documenting is enough for Phase 2

---

## Phase 3 — Claude-code (larger features)

### 7. Pipeline shared context
```python
Pipeline(steps, context={"request_id": "abc"})
# Steps access via contextvar or optional kw PIPE_CONTEXT
```

- Use `contextvars.ContextVar[dict]` set at pipeline run entry
- `Node.setup()` can read context
- Do **not** break existing step signatures

### 8. `from_spec` graph support
Extend `spec.py`:
```yaml
graph:
  nodes:
    a: { import: "mymodule:fetch" }
    b: { import: "mymodule:parse" }
  edges:
    - [a, b]
  seed_node: a
```

- `Graph.from_spec(path)` or `Pipeline.from_spec` detects `graph:` key
- Tests with fixture YAML

### 9. CI: `python3.14t` job
- Ubuntu job installing free-threaded CPython when available on setup-python
- Run `tests/test_pipeline.py -k "free_threading or parallel_auto"`
- Skip gracefully if 3.14t not on runner

---

## Phase 4 — Defer unless user asks

- Streaming / lazy iterator pipelines
- Checkpoint / resume for graphs
- Pydantic schema validation (optional dep)
- OpenTelemetry exporter (build on hooks)

---

## Conventions for claude-code

1. **Minimal diff** — match existing style in `step.py`, `decorators.py`
2. **Tests required** for every behavior change
3. **No new required dependencies** — optional extras only (`spec`, `rsloop`, etc.)
4. **Do not commit** unless user asks (manager may commit manager pass)
5. **Run before handoff:** `uv run pytest tests/ -q`
6. **Import surface:** export new public APIs from `__init__.py` + `pipeline` shim if tests import from `pipeline`

---

## Suggested claude-code prompt (copy-paste)

```
Read CLAUDE_CODE_HANDOFF.md fully. You are implementing pipecraft on branch working_barnch.

1. Verify items in "Started tonight" (A–C) — add missing tests, fix gaps.
2. Implement Phase 1 items 1–3 (hooks, docs, map= alias).
3. Run full test suite; fix failures.
4. Summarize what changed and what remains for Phase 2.

Do not commit unless I ask. Follow existing code conventions.
```

---

## Manager pass checklist

- [x] Write this handoff file
- [x] `auto_map=False` implementation
- [x] `PipeStep.copy()` + immutable decorators
- [x] `async_run_detailed()`
- [x] Tests for above
- [ ] Commit manager pass (if user approves)
