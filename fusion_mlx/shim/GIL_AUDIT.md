# Shim nanobind GIL audit (issue #1193)

Every nanobind bound function in `fusion_mlx/shim/csrc/bindings.cpp` is
audited for GIL hold time. Rule: **no bound function that can run
>ms-scale C++ work may hold the GIL during that work.** New bindings must
follow the rule (add a table row here + `nb::gil_scoped_release` for the
heavy section). The test `tests/unit/test_shim_gil_1193.py` enforces the
table stays complete.

## Bound surface (100% of bindings.cpp)

| Bound member | Heavy work | GIL during heavy work | Evidence |
|---|---|---|---|
| `HardwareProbe.def_ro` | none (field read, x6 members) | held (required) | bindings.cpp HardwareProbe class |
| `hardware_probe` | MTLDevice scan (first call, ms-scale) | released (#1193 fix) | gil_scoped_release in probe lambda |
| `hardware_probe_dict` | MTLDevice scan | released (#1193 fix); dict conversion re-acquires | gil_scoped_release inside envelope lambda |
| `start_memory_sentinel` | none (callback slot assign) | held (required - Python refcount) | trivial binding |
| `stop_memory_sentinel` | none (slot clear) | held (required) | trivial binding |
| `last_memory_pressure` | none | held (trivial) | trivial binding |
| `last_error_code` | none | held (trivial) | trivial binding |
| `last_error_message` | none | held (trivial) | trivial binding |
| `_test_slow_call` | sleep (test hook) | released | gil_scoped_release in lambda |
| `EngineRunner.__init__` | none | held (required) | trivial |
| `EngineRunner.start` | thread spawn + QoS bind | released | gil_scoped_release |
| `EngineRunner.stop` | thread join | released | gil_scoped_release (comment: join may wait on worker needing GIL) |
| `EngineRunner.is_running` | none | held (trivial) | trivial |
| `EngineRunner.submit` | block on decode thread | released during block; decode worker acquires GIL to run the Python callable | gil_scoped_release block + gil_scoped_acquire in worker |
| `EngineRunner.stats` | none | held (trivial) | trivial |

## Fixes made in #1193

- `hardware_probe` / `hardware_probe_dict`: the first-call MTLDevice scan
  is ms-scale; both previously held the GIL across it (H3 fingerprint:
  single Python thread at 100%, all coroutines stopped). Both now release
  the GIL across the scan. `hardware_probe_dict` re-acquires implicitly
  for the dict conversion (`probe_to_dict` touches Python objects).
- Added `_test_slow_call(seconds)` test hook (GIL released) so the audit
  test can deterministically prove a slow shim call does not block a
  concurrent asyncio heartbeat.

## Rule for new bindings

1. Heavy C++ work (>ms): wrap in `nb::gil_scoped_release`.
2. Python-object sections hold the GIL and must stay short.
3. Worker threads touching Python must `gil_scoped_acquire` first.
4. Add a row to this table for every new `m.def` / class-member binding;
   `tests/unit/test_shim_gil_1193.py` checks member names against it.
