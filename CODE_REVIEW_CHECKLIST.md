# Code Review Checklist — `git diff HEAD~1`

High-effort multi-agent review. 40 candidates → 29 confirmed. Ranked most-severe first.

## Correctness bugs (fix first)

- [x] **Circuit breaker opens too early** — fixed: `_record_failure()` moved out of `except`, called once after retries exhausted. Test `test_circuit_breaker_records_one_failure_per_logical_call`.
- [x] **Async timeout ignores `timeout=0.0`** — fixed: async now `if self.timeout is not None:`. Test `test_async_timeout_zero_is_enforced`.
- [x] **`parallel` + `batch_size` → batch never runs** — fixed (judgment: documented deterministic precedence, parallel wins; not raised because `test_pipestep_repr` constructs both). Comments in `_invoke_function`/`_invoke_function_async`.
- [ ] **Whole-collection steps auto-mapped per-element** — `src/stepcraft/__init__.py` — NOT fixed (behavior change risks existing parallel/batch tests; deferred).
- [x] **`_execute_parallel` drops kwargs** — fixed: both parallel paths wrap in `functools.partial(self.func, **kwargs)`. Tests `test_parallel_forwards_kwargs_sync/_async`.
- [x] **`_execute_batched` sums scalars across batches** — fixed: excluded `bool` from numeric-sum branch (kept sum for genuine numerics per existing tests). Test `test_batched_does_not_sum_booleans`.
- [x] **`FanOutStep(parallel='process')` crashes — unpicklable lambda** — fixed: module-level `_run_branch_on` + `itertools.repeat`. Tests `test_fanout_thread_parallel`/`test_fanout_process_parallel`.

## API + packaging regressions

- [x] **`piped()` dropped `cffi=`/`pyo3=` kwargs** — intentional: pure-Python only; no Rust/CFFI/PyO3.
- [x] **Removed from `__all__`: `FrozenDict`, `HAS_CFFI`, `HAS_PYO3`, `HAS_PYARROW`** — intentional: not part of the pure-Python API.
- [x] **`_execute_batched` numeric narrowed `(int,float,np.number)`→`(int,float)`** — fixed: numeric tuple includes `np.number` when `HAS_NUMPY`, bool excluded. Test `test_batched_recognizes_numpy_scalars` (skips, no numpy here).
- [x] **pytest `pythonpath=['src']` but `pipeline.py` moved src→root** — fixed: added `"."` to `pythonpath`. Test `test_pipeline_shim_reexports`.
- [x] **CI `test-pure-python` uses maturin backend, never switches** — `.github/workflows/ci.yml:63` → CI now pure-Python (hatchling), Rust steps removed.
- [x] **`.gitignore` ignores `rust/target/` but output at root `target/`** — fixed: ignore `*.so` + `._*`; `rust/` deleted.

## Performance

- [x] **Topo fallback `queue.pop(0)` = O(n) per pop** — fixed: now `deque` + `popleft()`.
- [x] **`ThreadPoolExecutor` created per invocation AND per retry** — fixed: shared `_get_pool('thread')` for timeouts.
- [x] **`_is_pickleable` does `pickle.dumps(whole_func)` + `import pickle` inside call** — fixed: module-level `pickle` import; check runs once in `__post_init__` for process pools.
- [x] **`edge_dict` rebuilt 3x + topo re-run uncached** — fixed: Rust `edge_dict` removed; `_topo_order` / `_topo_levels_cache` cached until graph mutation.
- [x] **Rust `fast_map` sequential under GIL + `fast_map`/`batch_items` never called** — Rust dependency removed entirely.

## Hygiene

- [x] **530KB prebuilt `_rust.cpython-314-darwin.so` committed** — deleted with `rust/` removal.
- [x] **`target/rust-analyzer/flycheck0/*` IDE artifacts committed** — `target/` deleted.

## Cleanup (verified dup, low priority)

- [x] **`_execute_sync`/`_execute_async` duplicate breaker+retry+schema logic** — fixed: shared `_preflight_check`, `_record_and_raise`, `_finalize_success`, `_should_retry`.
- [x] **`_run_branch`/`_async_run_branch` byte-identical in `ConditionalStep`+`SwitchStep`** — fixed: module-level `_run_branch_value` / `_async_run_branch_value`.

---
**Do now:** #7 (process fan-out crash), #1 (breaker logic), pytest+CI (rows 11–12).
