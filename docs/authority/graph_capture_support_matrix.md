# Graph Capture Support Matrix

This file is the authoritative stage taxonomy for Phases 3-8. The machine-readable companion is `graph_capture_support_matrix.json`; tooling should key on the JSON `stage_id` values, and stage-level NVTX range names must match those IDs exactly.

## Run Modes

- `sync_debug`: debug-only, not a production acceptance mode.
- `async_no_graph`: required non-graph baseline and fallback target.
- `graph_fixed`: accepted graph mode for stable-address steady-state execution.

## Canonical Stage IDs

| Stage ID | Intended phase | Capture policy | Loop owner | Fallback mode | Notes |
|---|---|---|---|---|---|
| `warmup` | P3 | graph-external boundary | host | `async_no_graph` | One-time warm-up and optional graph upload; outside timed steady-state windows. |
| `pre_solve` | P3 | capture-safe now | host | `async_no_graph` | Project-controlled pre-solve preparation. |
| `outer_iter_body` | P3 | capture-safe now | host outer loop | `async_no_graph` | Composite reusable body for one outer corrector in the Phase 3 baseline. |
| `momentum_predictor` | P5 | grouped inside `outer_iter_body` in Phase 3; explicit later | host outer loop | `async_no_graph` | Becomes an explicit owned stage in Phase 5. |
| `pressure_assembly` | P4/P5 | required graph boundary under project control | host outer loop | `async_no_graph` | Assembly/fix-up work before the selected pressure backend. |
| `pressure_solve_native` | P3+ | graph-external until individually validated | host outer loop | `async_no_graph` | Native pressure solve remains outside capture until explicitly promoted. |
| `pressure_solve_amgx` | P3+ | graph-external until individually validated | host outer loop | `async_no_graph` | AmgX remains graph-external and production-eligible only with `DeviceDirect`. |
| `pressure_post` | P4/P5 | required graph boundary under project control | host outer loop | `async_no_graph` | Post-solve correction / fix-up work. |
| `alpha_pre` | P5 | forward-declared capture-safe stage | host alpha loop | `async_no_graph` | Alpha pre-stage before subcycling. |
| `alpha_subcycle_body` | P5 | forward-declared capture-safe stage | host alpha loop | `async_no_graph` | One host-controlled alpha subcycle body. |
| `mixture_update` | P5 | forward-declared capture-safe stage | host timestep loop | `async_no_graph` | Post-alpha mixture update. |
| `nozzle_bc_update` | P6 | capture-safe only for Phase 6-owned BC rows | host timestep/outer loop | `async_no_graph` | Pressure-backend boundaries remain graph-external. |
| `write_stage` | P3+ | graph-external boundary | host | `async_no_graph` | Explicit output staging; excluded from timed steady-state measurement. |

## Global Capture Rules

- No post-warmup dynamic allocation.
- No hidden CPU patch evaluation in capture-safe stages.
- No silent host reads of device-authoritative fields.
- Any stage that fails capture/update/rebuild downgrades to `async_no_graph` with a logged reason.
