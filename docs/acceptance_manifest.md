# Acceptance Manifest

This file is the authoritative package-level source for case coverage, backend coverage, exact threshold classes, and hard/soft fail policy. Case IDs come from `reference_case_contract.md`; support scope comes from `support_matrix.md`. The machine-readable companion is `acceptance_manifest.json`; automation must consume the JSON companion rather than scrape this Markdown table.

## Accepted Tuple Matrix

Every formal verdict in Phase 5 or Phase 8 must map to one row in this table. If a run does not map to an admitted row, it is not a milestone-1 acceptance run.

The Markdown matrix stays compact by design. The machine-readable companion carries each tuple's exact NVTX contract in `accepted_tuples[*].required_stage_ids` and `nvtx_contract_defaults.required_orchestration_ranges`; `mandatory_nvtx_ranges_present` must be evaluated against those JSON fields rather than reconstructed from phase prose.

| Tuple ID | Phase gate | Case / variant | Backend | Execution mode | Kernel family mode | Admission | Production-eligible | Tolerance class | Restart / reload parity | Execution-parity class | Execution peer tuple | Backend-parity class | Backend peer tuple | Kernel-parity class | Kernel peer tuple |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P5_R2_TRANSPORT_NATIVE_ASYNC_BASELINE` | Phase 5 | `R2` / generic boundedness + interface-transport slice | `native` | `async_no_graph` | `baseline_gpu` | Required | No | `TC_R2_TRANSPORT` | None | None | None | None | None | None | None |
| `P5_R2_SURFACE_NATIVE_ASYNC_BASELINE` | Phase 5 | `R2` / static-droplet + surface-tension slice | `native` | `async_no_graph` | `baseline_gpu` | Required | No | `TC_R2_SURFACE` | None | None | None | None | None | None | None |
| `P5_R1CORE_NATIVE_ASYNC_BASELINE` | Phase 5 | `R1-core` / generic reduced case | `native` | `async_no_graph` | `baseline_gpu` | Required | Yes | `TC_R1CORE_GENERIC` | `RP_STRICT` | None | None | None | None | None | None |
| `P5_R1CORE_AMGX_ASYNC_BASELINE` | Phase 5 | `R1-core` / generic reduced case | `amgx` | `async_no_graph` | `baseline_gpu` | Optional benchmark-only; `DeviceDirect` required | No | `TC_R1CORE_GENERIC` | `RP_STRICT` | None | None | `BP_AMGX_R1CORE` | `P5_R1CORE_NATIVE_ASYNC_BASELINE` | None | None |
| `P8_R1CORE_NATIVE_ASYNC_BASELINE` | Phase 8 | `R1-core` / backend-parity and generic-kernel baseline | `native` | `async_no_graph` | `baseline_gpu` | Required | Yes | `TC_R1CORE_GENERIC` | `RP_STRICT` | None | None | None | None | None | None |
| `P8_R1CORE_NATIVE_ASYNC_CUSTOM` | Phase 8 | `R1-core` / backend-parity and custom-kernel reduced case | `native` | `async_no_graph` | `custom_gpu` | Required | Yes | `TC_R1CORE_GENERIC` | `RP_STRICT` | None | None | None | None | `KP_CUSTOM_VS_BASELINE` | `P8_R1CORE_NATIVE_ASYNC_BASELINE` |
| `P8_R1CORE_NATIVE_GRAPH_BASELINE` | Phase 8 | `R1-core` / backend-parity and generic-kernel baseline | `native` | `graph_fixed` | `baseline_gpu` | Required | Yes | `TC_R1CORE_GENERIC` | `RP_STRICT` | `EP_EXECUTION_MODE` | `P8_R1CORE_NATIVE_ASYNC_BASELINE` | None | None | None | None |
| `P8_R1CORE_NATIVE_GRAPH_CUSTOM` | Phase 8 | `R1-core` / backend-parity and custom-kernel reduced case | `native` | `graph_fixed` | `custom_gpu` | Required | Yes | `TC_R1CORE_GENERIC` | `RP_STRICT` | `EP_EXECUTION_MODE` | `P8_R1CORE_NATIVE_ASYNC_CUSTOM` | None | None | `KP_CUSTOM_VS_BASELINE` | `P8_R1CORE_NATIVE_GRAPH_BASELINE` |
| `P8_R1CORE_AMGX_ASYNC_BASELINE` | Phase 8 | `R1-core` / external-backend benchmark | `amgx` | `async_no_graph` | `baseline_gpu` | Optional benchmark-only; `DeviceDirect` required | No | `TC_R1CORE_GENERIC` | `RP_STRICT` | None | None | `BP_AMGX_R1CORE` | `P8_R1CORE_NATIVE_ASYNC_BASELINE` | None | None |
| `P8_R1_NATIVE_ASYNC_BASELINE` | Phase 8 | `R1` / reduced nozzle baseline | `native` | `async_no_graph` | `baseline_gpu` | Required | Yes | `TC_R1_NOZZLE` | `RP_STRICT` | None | None | None | None | None | None |
| `P8_R1_NATIVE_GRAPH_BASELINE` | Phase 8 | `R1` / reduced nozzle baseline | `native` | `graph_fixed` | `baseline_gpu` | Required | Yes | `TC_R1_NOZZLE` | `RP_STRICT` | `EP_EXECUTION_MODE` | `P8_R1_NATIVE_ASYNC_BASELINE` | None | None | None | None |
| `P8_R1_NATIVE_ASYNC_CUSTOM` | Phase 8 | `R1` / reduced nozzle custom-kernel benchmark | `native` | `async_no_graph` | `custom_gpu` | Optional benchmark-only | No | `TC_R1_NOZZLE` | `RP_STRICT` | None | None | None | None | `KP_CUSTOM_VS_BASELINE` | `P8_R1_NATIVE_ASYNC_BASELINE` |
| `P8_R1_NATIVE_GRAPH_CUSTOM` | Phase 8 | `R1` / reduced nozzle custom-kernel benchmark | `native` | `graph_fixed` | `custom_gpu` | Optional benchmark-only | No | `TC_R1_NOZZLE` | `RP_STRICT` | `EP_EXECUTION_MODE` | `P8_R1_NATIVE_ASYNC_CUSTOM` | None | None | `KP_CUSTOM_VS_BASELINE` | `P8_R1_NATIVE_GRAPH_BASELINE` |
| `P8_R0_NATIVE_GRAPH_BASELINE` | Phase 8 | `R0` / production-shape nozzle acceptance | `native` | `graph_fixed` | `baseline_gpu` | Required | Yes | `TC_R0_NOZZLE` | `RP_STRICT` | None | None | None | None | None | None |
| `P8_R0_NATIVE_GRAPH_CUSTOM` | Phase 8 | `R0` / production-shape nozzle custom-kernel benchmark | `native` | `graph_fixed` | `custom_gpu` | Optional benchmark-only | No | `TC_R0_NOZZLE` | `RP_STRICT` | None | None | None | None | `KP_CUSTOM_VS_BASELINE` | `P8_R0_NATIVE_GRAPH_BASELINE` |

## Coverage Rules

- `async_no_graph` and `graph_fixed` are the only accepted execution-mode labels for formal comparisons.
- `baseline_gpu` means the Phase 5/6 device-resident path without Phase 7 custom hotspot kernels; `custom_gpu` means the Phase 7 hotspot kernels are enabled.
- `R2` is a phase-gate family with exactly two required Phase 5 slices: `generic boundedness + interface-transport` and `static-droplet + surface-tension`.
- `R1-core` is the default reduced case for backend parity, execution-mode parity, and generic-versus-custom-kernel comparisons when Phase 6 nozzle-specific BCs are not required.
- `R1` is the reduced nozzle acceptance case for routine architecture and profiling baselines.
- `R0` is the production-shape acceptance case.
- The required milestone-1 nozzle signoff path is `baseline_gpu` on `R1` / `R0`. Matching `custom_gpu` nozzle rows are profile-driven benchmark rows and are admitted only when the implementation branch intentionally enables Phase 7 custom kernels.
- No milestone-1 acceptance row admits `amgx` on `R1` or `R0`.
- Rows marked `Optional benchmark-only` never block baseline milestone-1 signoff when absent. If they are present, they must satisfy the row's tolerance and parity classes.
- Any accepted `amgx` tuple requires the Phase 4 `DeviceDirect` bridge for that exact case/backend tuple. An AmgX run that still uses `PinnedHost` or any other non-`DeviceDirect` staging mode is diagnostic-only bring-up and does not map to an accepted tuple row.
- `PinnedHost` pressure staging is never production-eligible. Any run that uses it is bring-up/correctness-only regardless of other metrics.
- Write timesteps are excluded from timed steady-state windows unless a named write-cadence variant is explicitly added to this manifest in a later revision.
- Each tuple row above is unique by `(case / variant, backend, execution mode, kernel family mode)`. Automation must use the tuple ID directly and may not infer a row by partially matching only case/backend/mode.

## Tuple-Specific NVTX Contract

- `nvtx_contract_defaults.required_orchestration_ranges` is the required orchestration set for every accepted tuple.
- `accepted_tuples[*].required_stage_ids` freezes the exact stage-level parent ranges for that tuple, including backend-specific pressure solve stage selection.
- `R1` / `R0` nozzle tuples require `nozzle_bc_update`; generic `R2` / `R1-core` tuples do not.
- `write_stage` is excluded from all current accepted tuples because no write-cadence tuple is admitted in this manifest revision.

## Hard Gates

- `unexpected_htod_bytes == 0`
- `unexpected_dtoh_bytes == 0`
- `cpu_um_faults == 0`
- `gpu_um_faults == 0`
- `cudaDeviceSynchronize_calls == 0` inside steady-state inner ranges
- `post_warmup_alloc_calls == 0`
- `mandatory_nvtx_ranges_present == true`
- `cpu_boundary_fallback_events == 0`
- `host_patch_execution_events == 0`
- `pinned_host_pressure_stage_events == 0` in production acceptance runs
- `host_setFields_startup_events == 0` in the accepted startup path
- `unsafe_functionObject_commit_events == 0` inside timed steady-state windows

## Soft Gates

- `graph_launches_per_step <= 4`
- `top_kernel_time_regression_pct <= 10%` versus the locked baseline

## Disposition Rules

- Any hard-gate failure fails formal acceptance for the active tuple and may not support release or baseline-lock claims.
- Any soft-gate failure may be archived diagnostically, but it is not release-eligible or baseline-lock-eligible unless an explicit waiver is recorded against the manifest revision and tuple ID.
- Diagnostic or bring-up runs outside the accepted tuple matrix may report soft-gate results without converting them into a formal acceptance verdict.

## Exact Threshold Classes

### Field and QoI Classes

| Class | Exact thresholds |
|---|---|
| `TC_R2_TRANSPORT` | `max(abs(alpha1 + alpha2 - 1)) <= 1e-12`; `alpha1` bounded within `[-1e-10, 1 + 1e-10]`; integral mass imbalance no worse than CPU baseline by more than `0.1` percentage points; no solver divergence in the accepted transport window |
| `TC_R2_SURFACE` | `max(abs(alpha1 + alpha2 - 1)) <= 1e-12`; `alpha1` bounded within `[-1e-10, 1 + 1e-10]`; static-droplet / Laplace pressure-jump relative error `<= 1.0%`; spurious-current peak no worse than CPU baseline by more than `20%` |
| `TC_R1CORE_GENERIC` | integral mass imbalance no worse than CPU baseline by more than `0.1` percentage points; pressure-drop relative error `<= 0.5%`; mass-flow relative error `<= 0.5%`; discharge-coefficient-proxy relative error `<= 1.0%` after a matched transient window |
| `TC_R1_NOZZLE` | mass-flow relative error `<= 0.5%`; pressure-drop relative error `<= 0.5%`; max alpha overshoot/undershoot magnitude `<= 1e-8`; no solver divergence attributable to the nozzle path within the first `100` timesteps |
| `TC_R0_NOZZLE` | discharge-coefficient relative error `<= 1.0%`; spray half-angle difference `<= 2.0` degrees; air-core onset time difference `<= 5.0%` of the reference startup window |

### Parity and Replay Classes

| Class | Exact thresholds |
|---|---|
| `RP_STRICT` | after an accepted write -> reload cycle on the same mesh, current fields and all required old-time / `prevIter` / device-history fields must match the pre-write state with relative L2 difference `<= 1e-12`; constant / selector fields must match exactly; missing history fields are an automatic fail |
| `EP_EXECUTION_MODE` | exact stage counts; exact outer-loop counts; exact pressure-iteration counts; double-valued residual histories relative difference `< 1e-12`; single-precision scratch-derived diagnostics relative difference `< 1e-6` |
| `BP_AMGX_R1CORE` | small deterministic pressure snapshots: relative L2 difference `< 1e-9`, relative infinity-norm difference `< 1e-8`; real nozzle snapshots: relative L2 difference `< 1e-7`; live `R1-core` QoIs must also satisfy `TC_R1CORE_GENERIC` against the native baseline |
| `KP_CUSTOM_VS_BASELINE` | the `custom_gpu` run must satisfy the row's own tolerance class against CPU/reference and must match its baseline peer with double-valued residual-history relative difference `< 1e-12`, single-precision scratch-derived diagnostics relative difference `< 1e-6`, and QoI deltas no worse than the row's own field/QoI thresholds |

## Production Defaults

- Current default production backend: `native`.
- Current default production execution mode: `graph_fixed`, with `async_no_graph` retained as the required non-graph regression-isolation baseline.
- Contact-angle is not part of milestone-1 acceptance.
