# 1. Frozen global decisions

- **canonical solver family** — Milestone-1 canonical solver family is the algebraic `incompressibleVoF` / explicit MULES / PIMPLE path. `geometric_iso` / `interIsoFoam` may remain **CPU-only shadow references in Phase 0**, but they are not the canonical GPU target or the hard acceptance line.
- **canonical pressure path** — Native SPUMA/OpenFOAM pressure solve is the required baseline. AmgX via `foamExternalSolvers` is the secondary runtime-selectable backend. Phase 4 `PinnedHost` staging is **correctness-only bring-up**; `DeviceDirect` is required before any Phase 5–8 production claim that says “no field-scale host transfer” while using AmgX.
- **GPU-only operational contract**
  - **allowed in production acceptance** — host orchestration, small control-scalar exchange, startup validation/dictionary parsing, explicit write-time output staging, and explicit crash/debug snapshots outside timed steady-state windows.
  - **forbidden in production acceptance** — field-scale host evaluation in hot stages, host patch polymorphism inside the timestep loop, post-warmup dynamic allocation, recurring UVM traffic for registered hot objects, pinned-host pressure staging in accepted production runs, host `setFields` in the accepted startup path, and silent CPU fallback.
- **startup/preconditioning scope** — Phase 0 continues to freeze the full staged CPU reference workflow. Milestone-1 GPU acceptance starts at **transient solve + device startup seeding**. The steady precondition / `potentialFoam` path remains an allowed CPU pre-run utility unless a later explicit milestone promotes it into the GPU acceptance scope.
- **support-matrix ownership** — One centralized support matrix owns BCs, schemes, turbulence scope, surface-tension/contact-angle scope, functionObject policy, mesh constraints, backend availability, and allowed fallbacks. Phase files consume it; they do not restate or reopen it.
- **validation ladder** — `R2` generic VOF -> `R1-core` Phase-5-friendly reduced case using only the frozen generic BC/scheme subset -> `R1` reduced nozzle with Phase 6 BCs/startup -> `R0` full nozzle. Phase gates map to: P0 provenance freeze, P1 toolchain, P2 residency, P3 execution, P4 pressure backend, P5 generic VOF core, P6 nozzle BC/startup, P7 hotspot kernels, P8 final acceptance.
- **source/version/toolchain pin ownership** — One master pin manifest owns SPUMA commit, OpenFOAM flavor/release, `foamExternalSolvers` revision, AmgX revision, CUDA primary lane, experimental lane, driver floor, and Nsight versions. Phase 1 is the initial owner of the proposal; the master roadmap owns the frozen pins. Default until superseded: CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, NVTX3.
- **contact-angle scope** — Contact-angle is **out of milestone-1 scope unless the frozen acceptance cases demonstrably require it**. If required, ownership splits cleanly: Phase 5 interface semantics, Phase 6 patch execution, Phase 7 optimization. Otherwise all contact-angle text in Phases 6–7 is conditional/future-only.
- **runtime configuration normalization** — Normalize under one `gpuRuntime` tree: `execution`, `memory`, `pressure`, `vof`, `boundary`, `startupSeed`, `profiling`, `acceptance`. Existing `gpuMemoryDict`, `gpuVoF`, `gpuNozzleBCDict`, `gpuStartupSeedDict`, `gpuProfilingDict`, and per-phase control structs become compatibility shims or generated subviews, not independent contracts.

# 2. Cross-phase contract matrix

| Phase | Entry criteria | Exit criteria | Produced artifacts | Consumed artifacts | Next-phase handoff | Contract status |
|---|---|---|---|---|---|---|
| **P0** | Clean repo; Baseline A/B environments; builder/extractor/probe/smoke assets present | Frozen A/B reference bundles for `R2`, `R1-core`, `R1`, and `R0`; compare reports; sign-off | manifests; tolerance config; probes; `case_meta.json`; `stage_plan.json`; fingerprints; metrics; compare reports | repo templates/tools; manufacturing sanity data; CPU environments | canonical solver-family freeze; Baseline B ownership; startup/preconditioning scope; validation-ladder case definitions | No contract mismatch. Solver family, Baseline B ownership, startup scope, and `R1-core` are frozen here and consumed downstream. |
| **P1** | Phase 0 freeze available; RTX 5080 workstation; pinned SPUMA checkout | `sm_120` + PTX build; NVTX3; smoke cases; Nsight/Compute Sanitizer working; acceptance report | host/toolchain manifest; CUDA probes; fatbinary report; smoke logs; profiler traces; Phase 1 acceptance report | Phase 0 source/version direction; frozen comparison line | authoritative toolchain proposal and profiler stack | No contract mismatch. The master pin manifest owns the frozen lane; later `12.8` references are feature-floor rationale only. |
| **P2** | Phase 0 references; Phase 1 toolchain/profilers; approved memory mode | persistent/scratch/pinned tiers; residency registry; write/restart staging; UVM-clean reduced case in `productionDevice` | canonical `gpuRuntime.memory` config; allocator modules; residency reports; pool stats; restart/reload parity tests | Phase 0 case inventory; Phase 1 profiler/toolchain; validation ladder; support matrix | stable device residency plus explicit sync/commit APIs for later phases | No contract mismatch. Runtime ownership is `gpuRuntime.memory`, and later phases consume the Phase 2 sync/commit contract rather than redefining it. |
| **P3** | Phase 2 production-device behavior stable; supported GPU solver available | `async_no_graph` and `graph_fixed`; zero hot-path device-wide sync; graph cache/fingerprint; stage taxonomy | execution context; stream policy; graph manager; graph traces; `GraphCaptureSupportMatrix` | Phase 2 residency/pools; Phase 1 profiler stack | graph-safe stage contract and canonical stage IDs | No contract mismatch. Later phases consume `GraphCaptureSupportMatrix` rows and stage IDs from Phase 3. |
| **P4** | Pressure snapshots/native baseline; AmgX toolchain; device allocator stable | persistent LDU->CSR cache; snapshot replay; runtime-selected native + AmgX path; `PinnedHost` correctness path; `DeviceDirect` production bridge definition/completion gate | `PressureMatrixCache`; replay/dump tools; pressure telemetry; `DeviceDirect` bridge contract | Phase 1 toolchain; Phase 2 memory; Phase 3 stage boundaries; Phase 0 snapshots | pressure-backend bridge to Phase 5 | No contract mismatch. `PinnedHost` is correctness-only and `DeviceDirect` is the required bridge before later no-field-scale-host-transfer AmgX claims. |
| **P5** | Device-resident memory/execution substrate; pressure backend bridge; `R1-core` frozen; support matrix frozen | device-authoritative generic VOF core (alpha/MULES, mixture, surface tension subset, momentum, pressure); write-time commit path; restart/reload parity | `gpuRuntime.vof` runtime gate with optional `gpuVoF` shim; `DeviceVoFState`; support scanner; pressure-boundary-state handoff; generic VOF validation artifacts | Phase 4 pressure bridge; Phase 2 memory; Phase 3 execution; Phase 0 references; support matrix | nozzle-specific BC/startup device path | No contract mismatch. `R1-core`, support-matrix ownership, and native-vs-AmgX handoff are frozen for later phases. |
| **P6** | Phase 5 generic core stable; patch names/types frozen; pressure-boundary-state contract available | device nozzle BC/startup subsystem; zero CPU patch evaluation in hot loop; nozzle regression artifacts | boundary support report; boundary manifest; custom swirl BC; startup seeder; nozzle BC execution artifacts | Phase 5 field mirrors and pressure-boundary-state hook; Phase 0 frozen patch/startup data; support matrix | accepted nozzle BC envelope and startup path | No contract mismatch. Authoritative inlet/open BC subset and contact-angle dependency are imported from the support matrix, not reopened locally. |
| **P7** | Stable Phase 5/6 path; hotspot ranking from profiling; source audit complete | runtime-selectable custom kernels for support-matrix-approved hotspot families; graph smoke data; backend comparison artifacts | custom-kernel module; hotspot benchmarks; ranking artifacts | Phase 6 patch manifests; Phase 5 solver semantics; interim profiling artifacts; support matrix | locked hotspot implementations for Phase 8 acceptance | No contract mismatch. Hotspot scope is profile-ranked, support-matrix-bounded, and contact-angle remains conditional. |
| **P8** | Runnable single-GPU nozzle path; validated `R1`/`R0`; profiling build/toolchain pinned | profiling subsystem; acceptance scripts; locked baselines; CI/nightly gates | profile bundles; acceptance summaries; top-kernel reports; sanitizer logs | all earlier phase artifacts; support matrix; acceptance manifest; graph support matrix; master pin manifest | production acceptance baseline / optimization governance | No contract mismatch. Final thresholds, backend scope, and toolchain lane are owned by centralized manifests, not by local defaults. |

# 3. Resolution status

The original continuity conflicts are closed in the patched phase set. Concrete artifact instances may still need to exist in the working branch, but the design contracts below are no longer open decisions.

| Issue ID | Final-state status | Authoritative owner | Notes |
|---|---|---|---|
| **I-01** | Resolved | Phase 0 + this ledger | Algebraic `incompressibleVoF` / explicit MULES / PIMPLE is the milestone-1 canonical GPU family; `geometric_iso` stays CPU-only shadow/reference. |
| **I-02** | Resolved | Phase 0 + this ledger | Baseline B is SPUMA/v2412 CPU when runnable; stock OpenFOAM-v2412 CPU is contingency-only fallback. |
| **I-03** | Resolved | Master pin manifest contract | Phase 1 proposes the lane; later phases consume the manifest instead of reopening toolchain minima. |
| **I-04** | Resolved | Phase 4 handoff contract | `PinnedHost` is correctness-only; `DeviceDirect` is the required AmgX production bridge. |
| **I-05** | Resolved | Support matrix + P5/P6/P7 split | Contact-angle is out of milestone-1 scope unless the frozen accepted cases explicitly require it. |
| **I-06** | Resolved | Phase 0 + this ledger | GPU acceptance starts at transient solve + device startup seeding; CPU preconditioning remains an allowed pre-run utility unless promoted later. |
| **I-07** | Resolved | Validation ladder + acceptance manifest | `R1-core` is the Phase-5-friendly reduced case between `R2` and nozzle-specific `R1`. |
| **I-08** | Resolved | Central support matrix | BCs, schemes, turbulence scope, functionObject policy, and allowed fallbacks are centralized and imported by the phase docs. |
| **I-09** | Resolved | GPU operational contract | Allowed and forbidden host duties for production acceptance are centralized rather than redefined locally. |
| **I-10** | Resolved | Normalized `gpuRuntime` schema | `gpuMemoryDict`, `gpuVoF`, `gpuNozzleBCDict`, `gpuStartupSeedDict`, and `gpuProfilingDict` are compatibility shims or generated subviews only. |
| **I-11** | Resolved | Master roadmap + this ledger | Cross-phase ownership is centralized and the sequencing is Phase 7 before Phase 8 acceptance. |
| **I-12** | Resolved | Phase 5 to Phase 6 handoff | The named pressure-boundary-state contract now bridges `DevicePressureCorrector` and the nozzle boundary executor. |
| **I-13** | Resolved | `GraphCaptureSupportMatrix` | Graph-safety/capture policy is centralized by canonical stage ID and consumed across later phases. |

# 4. Central Package Authorities

The final package now carries the central artifacts that downstream phases import:

1. **`master_pin_manifest.md`**
   - Runtime base, toolkit lanes, driver floor, profiler lane, GPU target, and external bridge defaults.
2. **`reference_case_contract.md`**
   - Concrete `R2` / `R1-core` / `R1` / `R0` case IDs, roles, and phase-gate usage.
3. **`validation_ladder.md`**
   - Navigation alias for the frozen `R2 -> R1-core -> R1 -> R0` ladder defined authoritatively by `reference_case_contract.md`.
4. **`support_matrix.md`**
   - Frozen BCs, schemes, turbulence/contact-angle scope, functionObject policy, backend availability, and fallback rules.
5. **`acceptance_manifest.md`**
   - Hard/soft fail policy, threshold classes, backend coverage, and timing-window rules.
6. **`graph_capture_support_matrix.md`**
   - Canonical stage IDs, graph-external boundaries, loop ownership, and fallback targets.
7. **`semantic_source_map.md`**
   - Exact semantic handoff for SPUMA/v2412, `foamExternalSolvers`, interface-property, pressure, and nozzle-BC touch points.

# 5. Package Consumption Rule

When a phase doc conflicts with one of the authority docs above, the authority doc wins. Phase docs may add local implementation detail, but they may not redefine:

1. toolchain or profiler pins,
2. case-ladder membership or case roles,
3. milestone-1 support scope,
4. acceptance thresholds or hard/soft fail policy,
5. graph stage IDs or fallback modes,
6. semantic patch targets.

# 6. Remaining Post-Freeze Validation Risks

| Risk statement | Required evaluation path | Current package stance | Downstream impact |
|---|---|---|---|
| **Native vs AmgX long-run default may be re-evaluated after `DeviceDirect` benchmarking** | Benchmark native and AmgX on snapshot replay, `R1-core`, `R1`, and `R0` after the `DeviceDirect` bridge exists; measure setup/solve/QoIs separately | Current default remains native; AmgX is secondary until promoted by `acceptance_manifest.md` | May change the preferred backend after implementation evidence exists, but does not block current coding |
| **Graph-capture safety of pressure/backend/library boundaries must still be proven on the exact implementation branch** | Run capture-smoke tests on the exact toolkit + backend versions for native pressure, AmgX wrapper boundaries, and helper-library routines | Current default is graph-external for `pressure_solve_native` / `pressure_solve_amgx` until promoted by `graph_capture_support_matrix.md` | Controls later graph expansion, not current implementation correctness |
