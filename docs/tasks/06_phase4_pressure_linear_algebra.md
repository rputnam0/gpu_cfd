# Phase 4 - Pressure linear algebra

Phase 4 owns `P4-01..P4-09` and delivers the pressure solver bridge: dependency freeze and smoke setup, pressure snapshot/replay assets, CSR topology/value translation, backend runtime selection, persistent cache/staging, telemetry, and the `DeviceDirect` pressure bridge contract. This section consumes Foundation, Phase 0/1, Phase 2, and Phase 3 authority outputs as fixed inputs and does not absorb generic VOF transport or nozzle BC/startup semantics.

## P4-01 Dependency freeze and standalone AmgX smoke

- Objective:
  - Freeze the Phase 4 dependency tuple and validate the AmgX stack with a trivial solve before OpenFOAM integration starts.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/authority/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (canonical pressure path and toolchain pin ownership)
    - `docs/tasks/03_phase1_blackwell_bringup.md` -> `## P1-07 PTX-JIT proof and Phase 1 acceptance bundle`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### Entry criteria`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 1 — Freeze dependency versions and add a standalone AmgX smoke target`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Milestone M4.1 — Dependency freeze and AMgX smoke`
  - Backlog scope:
    - `Pin the SPUMA / foamExternalSolvers / AmgX dependency line for Phase 4 and build the standalone AmgX smoke test on the target workstation.`
  - Backlog done_when:
    - `An identity or trivial system solves correctly on the RTX 5080 before OpenFOAM integration begins.`
- Depends on (backlog IDs):
  - `P1-07`
  - `FND-02`
- Prerequisites:
  - Phase 1 accepted lane artifacts (`host_env.json`, `cuda_probe.json`, `fatbinary_report.json`) are frozen and readable.
  - Foundation pin-manifest ingestion from `FND-02` is active.
- Concrete task slices:
  1. Resolve and lock the exact SPUMA, `foamExternalSolvers`, and AmgX refs from `master_pin_manifest`.
  2. Add a standalone AmgX smoke executable that uploads a tiny identity/trivial system and verifies `AMGX_SOLVE_SUCCESS`.
  3. Emit lane-manifest evidence binding smoke runs to the frozen tuple and probe manifest.
  4. Add fail-fast tuple guardrails so stale dependency tuples abort before any Phase 4 integration run.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - Phase 4 dependency tuple lock artifact
    - Lane-bound smoke artifact with solver status and architecture evidence
  - Consumed:
    - `FND-02` pin-manifest schema and manifest emission contract
    - `P1-07` acceptance tuple evidence bundle
- Validation:
  - Trivial/identity solve returns success using the frozen tuple and frozen profiler/toolchain constraints.
  - Emitted manifests prove this run used the intended lane and dependency refs.
- Done criteria:
  - `P4-01` dependency tuple is locked and standalone AmgX smoke is green.
- Exports to later PRs/phases:
  - Frozen dependency contract consumed by `P4-02..P4-09`.
  - Lane evidence consumed by Phase 8 profiling/provenance tasks.

## P4-02 Pressure snapshot dump utility

- Objective:
  - Add CPU/reference-side pressure snapshot capture at the exact pressure-solve boundary for replay harness use.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (native pressure baseline and staged bring-up)
    - `docs/tasks/02_phase0_reference_problem_freeze.md` -> `## P0-08 Baseline B nozzle freeze and sign-off package`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 4 -> Phase 5`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### Entry criteria`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 2 — Add a pressure-system snapshot dump utility`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Milestone M4.2 — Pressure snapshot dump`
  - Backlog scope:
    - `Add the CPU/reference-side pressure snapshot utility so representative matrices, RHS vectors, and metadata can be captured from frozen cases.`
  - Backlog done_when:
    - `Reduced reference cases can emit reusable pressure snapshots for offline replay and validation.`
- Depends on (backlog IDs):
  - `P4-01`
  - `P0-08`
- Prerequisites:
  - Frozen dependency tuple and smoke validation from `P4-01`.
  - Frozen case bundle and metadata contract from `P0-08`.
- Concrete task slices:
  1. Implement `phase4DumpPressureSystem` capture for `diag`, `lower`, `upper`, `source`, `psi`, `lowerAddr`, `upperAddr`, and metadata.
  2. Capture snapshots at the solve boundary after boundary/reference adjustments are complete.
  3. Add deterministic snapshot hash fields (`solverControlsHash`, case/time identifiers) for replay integrity.
  4. Add reduced-case snapshot fixtures for topology reuse and backend parity tests.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - pressure snapshot schema and dump utility
    - reduced-case snapshot corpus
  - Consumed:
    - `P0-08` case metadata and frozen compare context
    - `P4-01` tuple and environment evidence
- Validation:
  - Snapshot writer and reader reconstruct dimensions/addressing exactly.
  - Malformed snapshots are rejected deterministically with schema diagnostics.
- Done criteria:
  - `P4-02` can produce reusable snapshots for frozen reduced cases.
- Exports to later PRs/phases:
  - Snapshot corpus and schema consumed by `P4-03..P4-06`, `P4-09`, and Phase 5 backend/pressure integration validation.

## P4-03 Topology-only LDU-to-CSR builder

- Objective:
  - Build deterministic topology-only CSR row/column mapping from OpenFOAM/SPUMA LDU addressing and cache by topology signature.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (no post-warmup allocation churn, deterministic contracts)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 4 -> Phase 5`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.2 — Cache topology in the object registry`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.3 — Build a full CSR with diagonal embedded`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.4 — Preserve asymmetric storage semantics`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 3 — Implement the topology-only LDU→CSR builder`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Pattern-build algorithm`
  - Backlog scope:
    - `Implement the topology conversion that builds CSR row offsets and column indices from the OpenFOAM/SPUMA LDU structure and caches it by topology.`
  - Backlog done_when:
    - `Synthetic and snapshot-backed tests prove the structural CSR mapping is correct and reusable.`
- Depends on (backlog IDs):
  - `P4-02`
- Prerequisites:
  - Stable snapshot capture artifacts and case fixtures from `P4-02`.
- Concrete task slices:
  1. Implement `LduCsrPatternBuilder` producing `rowPtr`, `colInd`, and LDU->CSR mapping arrays with topology hash keys.
  2. Add deterministic insertion and duplicate-column checks for debug/replay modes.
  3. Add object-registry keyed cache lookup to reuse structure when topology is unchanged.
  4. Add structural tests for `nnz`, owner/neighbour mapping, and diagonal presence using synthetic and snapshot inputs.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `LduCsrPattern` contract and topology-key schema
    - structural mapping test suite
  - Consumed:
    - snapshot addressing data from `P4-02`
- Validation:
  - Synthetic and snapshot fixtures pass structure equivalence/determinism checks.
  - Repeated same-topology inputs hit cache and do not rebuild topology.
- Done criteria:
  - CSR topology mapping is correct, deterministic, and reusable.
- Exports to later PRs/phases:
  - Stable pattern contract consumed by `P4-04..P4-09`.
  - Downstream handoff artifact for Phase 5 pressure bridge reuse.

## P4-04 CSR value packer and `A*x` validator

- Objective:
  - Pack per-solve coefficients/vectors into CSR arrays and validate mapping by `A*x` equivalence before solver backend attachment.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (correctness-first bring-up and explicit boundary behavior)
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### \`CsrValuePacker\``
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Value-pack algorithm`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### U2 — A*x equivalence test`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 4 — Implement a pure-host CSR value packer`
  - Backlog scope:
    - `Add the value packer that fills CSR coefficients/RHS from live or snapshot data and validate the mapping with \`A*x\` equivalence checks.`
  - Backlog done_when:
    - `Snapshot and synthetic A*x checks pass within tolerance, separating mapping bugs from solver-backend bugs.`
- Depends on (backlog IDs):
  - `P4-03`
- Prerequisites:
  - Topology contract from `P4-03`.
- Concrete task slices:
  1. Implement `CsrValuePacker` for `csrValuesHost`, `rhsHost`, and optional `xHost`.
  2. Bind packing to live/snapshot matrix state at solver boundaries only.
  3. Implement `A*x` equivalence validator against native `lduMatrix::Amul` on synthetic and snapshot fixtures.
  4. Emit mismatch diagnostics isolating row/column/value errors.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - value-packer contract
    - `A*x` parity report schema
  - Consumed:
    - `LduCsrPattern` from `P4-03`
    - snapshot fixtures from `P4-02`
- Validation:
  - `A*x` parity checks pass within declared tolerance across fixtures.
  - Diagnostics identify exact row-level mismatch source.
- Done criteria:
  - `P4-04` provides stable mapping and parity diagnostics independent of backend runtime.
- Exports to later PRs/phases:
  - Verified value packer contract consumed by `P4-05` replay and `P4-07` live integration.

## P4-05 AmgX context wrapper and replay utility

- Objective:
  - Add a safe `AmgXContext` lifecycle wrapper plus standalone replay utility for upload/setup/solve/update decomposition.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (secondary AmgX backend posture)
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### \`AmgXContext\``
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 6 — Implement \`AmgXContext\` RAII wrapper`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 7 — Implement first-call upload path`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 8 — Implement repeated-solve replace/setup/solve path`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 10 — Add debug validators and snapshot replay executable`
  - Backlog scope:
    - `Implement the \`AmgXContext\` wrapper plus a replay tool that loads snapshots and exercises upload/setup/solve/update in isolation.`
  - Backlog done_when:
    - `Snapshot replay solves correctly and emits timing decomposition without involving the live pressure loop.`
- Depends on (backlog IDs):
  - `P4-04`
- Prerequisites:
  - Stable value packer and parity harness from `P4-04`.
- Concrete task slices:
  1. Implement `AmgXContext` RAII wrapper for config/resources/matrix/vector/solver lifecycle and error translation.
  2. Add replay executable that loads snapshots and executes upload, replace, setup, solve, and readback paths.
  3. Emit stable timing decomposition (`upload_all`, `replace`, `setup`, `solve`, `download`) independent of live orchestration.
  4. Enforce `dDDI` mode and fail on unsupported mode or tuple drift.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `AmgXContext` API/contract
    - `phase4ReplayPressureSystem` utility and timing schema
  - Consumed:
    - packed CSR and parity fixtures from `P4-04`
    - dependency tuple evidence from `P4-01`
- Validation:
  - Snapshot replay solves deterministically and emits per-step timing decomposition.
  - Context creation/destruction is leak-free over repeated cycles.
- Done criteria:
  - `P4-05` validates backend behavior without coupling to live pressure loop.
- Exports to later PRs/phases:
  - Reusable backend wrapper consumed by `P4-06..P4-09`.
  - Replay harness reused by Phase 5 backend parity checks.

## P4-06 PressureMatrixCache and persistent staging buffers

- Objective:
  - Persist pressure matrix cache and staging buffers across solves under unchanged topology.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (post-warmup allocation and host-transfer constraints)
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-04 Pinned stage allocator`
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy` (`PinnedHost` correctness-only posture)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### \`PressureMatrixCache\``
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 5 — Implement \`PressureMatrixCache\` with persistent host staging buffers`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Ownership and lifetime model`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Mandatory memory rules`
  - Backlog scope:
    - `Add the persistent pressure-matrix cache object plus reusable host staging buffers that outlive individual solver objects.`
  - Backlog done_when:
    - `Repeated solves with unchanged topology reuse the cached structure and avoid rebuilding row/column data.`
- Depends on (backlog IDs):
  - `P4-05`
  - `P2-04`
- Prerequisites:
  - `AmgXContext` and replay integration from `P4-05`.
  - Pinned-stage allocator contract from `P2-04`.
- Concrete task slices:
  1. Implement `PressureMatrixCache` ownership and lookup in object registry keyed by topology/config.
  2. Allocate and reuse persistent staging buffers for pattern, values, RHS, and solutions.
  3. Implement cache invalidation for topology/hash/config transitions.
  4. Emit cache telemetry (`patternBuildCount`, `patternUploadCount`, allocation counters) for acceptance evidence.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `PressureMatrixCache` contract
    - persistent host staging buffer lifetime policy
  - Consumed:
    - `P2-04` pinned-stage allocator
    - replay and backend path from `P4-05`
- Validation:
  - Same-topology repeated solves do not rebuild row/column structures.
  - Steady-state logs show no repeated topology upload beyond first solve.
- Done criteria:
  - `P4-06` caches topology/values and reuses buffers across steady solves.
- Exports to later PRs/phases:
  - `PressureMatrixCache` contract handed to Phase 5 (`P5-09`, `P5-10`).
  - Cache telemetry consumed by `P4-08` and Phase 8 evidence.

## P4-07 Runtime-selected live solver integration

- Objective:
  - Integrate runtime selectable native/AmgX pressure path into live solve boundary with deterministic fallback.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (native baseline, secondary AmgX backend, no silent fallback)
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.5 — Support only uncoupled scalar systems in Phase 4`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.12 — Native solver fallback is mandatory, not optional`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 9 — Implement runtime-selected \`AmgXSolver\` with fallback`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Unsupported or failed path`
  - Backlog scope:
    - `Wire native and AmgX pressure solver selection into the live solver path with clean fallback on unsupported interface conditions.`
  - Backlog done_when:
    - `A reduced live case can select AmgX at runtime or fall back cleanly to native without corrupting the pressure path.`
- Depends on (backlog IDs):
  - `P4-06`
- Prerequisites:
  - Stable cache and staging contracts from `P4-06`.
- Concrete task slices:
  1. Register runtime pressure selection in solve boundary and integrate `AmgX` in live solver path.
  2. Detect unsupported cases (coupled interfaces, unsupported rows, tuple mismatch) and route explicit native fallback with reason codes.
  3. Guard against recursive fallback and prevent solver-classification drift.
  4. Bind stage accounting to canonical IDs: `pressure_assembly`, `pressure_solve_native`, `pressure_solve_amgx`, `pressure_post`.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - runtime pressure-backend selector contract
    - fallback reason taxonomy/reporting artifact
  - Consumed:
    - `PressureMatrixCache` and replay-proven backend path
    - support-matrix fail-fast policy and graph boundary definitions
- Validation:
  - Reduced live case executes with selected backend and deterministic fallback logs when forced unsupported conditions are injected.
  - Native fallback preserves pressure correctness and solver state integrity.
- Done criteria:
  - Live pressure loop supports runtime backend selection with safe fallback behavior.
- Exports to later PRs/phases:
  - Selection contract consumed by Phase 5 (`P5-09`, `P5-10`).
  - Canonical pressure-stage ownership consumed by Phase 8 instrumentation and Phase 5 stage integration.

## P4-08 Telemetry, profiling hooks, and reduced-case validation

- Objective:
  - Add pressure telemetry and NVTX hooks, then run native-vs-AmgX reduced-case comparison with reviewable evidence.
- Exact citations:
  - Authority:
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/authority/acceptance_manifest.md` -> `## Production Defaults`
    - `docs/authority/acceptance_manifest.json` -> `hard_gates`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/tasks/05_phase3_execution_model.md` -> `## P3-05 Canonical stage scaffolding and parent NVTX ranges`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-03 Solver-stage instrumentation coverage`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 11 — Add NVTX3 ranges and telemetry counters`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 12 — Integrate with reduced live case and benchmark against native`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## NVTX3 ranges to add`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## Telemetry counters to persist`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `## 3. Live reduced-case validation`
  - Backlog scope:
    - `Add pressure telemetry, NVTX boundaries, native-vs-AmgX benchmark scripts, and reduced-case validation for the Phase 4 live path.`
  - Backlog done_when:
    - `Phase 4 can prove no repeated topology upload, can compare native vs AmgX on the reduced case, and emits reviewable telemetry artifacts.`
- Depends on (backlog IDs):
  - `P4-07`
  - `P3-05`
- Prerequisites:
  - Live backend integration from `P4-07`.
  - Stage and NVTX parent scaffolding from `P3-05`.
- Concrete task slices:
  1. Add canonical pressure bridge NVTX hierarchy and subranges aligned to Phase 3 stage ownership.
  2. Persist pressure telemetry counters for topology build/upload/replace/setup/solve/download and fallback events.
  3. Add reduced-case benchmark scripts comparing native and AmgX with explicit backend classification metadata.
  4. Validate hard-gate-relevant counters (`unexpected` transfers/UVM/sync/allocation) without introducing Phase 8 baseline-lock policy.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - pressure telemetry report schema
    - reduced-case native-vs-AmgX benchmark artifact pack
  - Consumed:
    - `P3-05` stage/scaffold contract
    - pass-A instrumentation vocabulary from Phase 8
- Validation:
  - Traces show required pressure stage ranges and bridge sub-ranges in expected order.
  - Telemetry proves topology reuse and no repeated row/column upload after initial build.
  - Reduced-case benchmark artifacts are reproducible and reviewable.
- Done criteria:
  - `P4-08` emits complete pressure telemetry and reduction-case backend comparison evidence.
- Exports to later PRs/phases:
  - Pressure telemetry and benchmark evidence consumed by `P4-09`, Phase 5 parity work, and Phase 8 acceptance pipelines.
  - Canonical pressure-stage instrumentation contract handed to Phase 5 unchanged.

## P4-09 DeviceDirect pressure bridge

- Objective:
  - Implement device-resident pressure staging (`csrValuesDev`, `rhsDev`, `xDev`) and bridge upload/replace semantics required for production-eligible AmgX claims.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (PinnedHost correctness-only; DeviceDirect required for no field-scale host-transfer AmgX claims)
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/authority/acceptance_manifest.json` -> `coverage_rules.amgx_required_pressure_bridge_mode`
    - `docs/authority/acceptance_manifest.json` -> `coverage_rules.amgx_without_required_bridge_classification`
    - `docs/authority/acceptance_manifest.json` -> `coverage_rules.pinned_host_policy`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 4 -> Phase 5`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.8 — Separate bring-up and production paths in the API now`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `### Production path / mandatory bridge (defined now; completed before any later AmgX production claim of no field-scale host transfer)`
    - `docs/specs/phase4_linear_algebra_spec.md` -> `#### Step 13 — Complete the \`DeviceDirect\` pressure bridge`
  - Backlog scope:
    - `Implement the device-resident \`csrValuesDev\` / \`rhsDev\` / \`xDev\` bridge and the device-pointer upload/replace path required for later no-field-scale-host-transfer claims.`
  - Backlog done_when:
    - `Replay and reduced live cases validate the device-pointer path, while \`PinnedHost\` remains correctness-only and clearly non-production.`
- Depends on (backlog IDs):
  - `P4-08`
  - `P2-02`
- Prerequisites:
  - Telemetry and reduced-case baseline from `P4-08`.
  - Device persistent allocation substrate from `P2-02`.
- Concrete task slices:
  1. Implement persistent device-resident `DeviceDirect` buffers for CSR values and pressure vectors.
  2. Add device-pointer upload/replace path in `AmgXContext`, preserving explicit `PinnedHost` bring-up mode.
  3. Validate replay and reduced live cases through `DeviceDirect` mode with UVM/transfer telemetry proving no field-scale host staging in production mode.
  4. Emit bridge-mode status artifact recording tuple eligibility, staging mode, and fallback/diagnostic classification.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceDirect` pressure bridge contract
    - bridge-mode eligibility/classification artifact
  - Consumed:
    - `P2-02` device-persistent memory substrate
    - pressure telemetry and stage contracts from `P4-08`
- Validation:
  - Replay and reduced live paths run with `DeviceDirect` selected and stable bridge-mode reporting.
  - `PinnedHost` remains available for correctness-only bring-up and never classified as production-eligible for accepted AmgX tuples.
- Done criteria:
  - `P4-09` completes production bridge definition and validation.
- Exports to later PRs/phases:
  - Required bridge contract for `P5-10` and later production-eligibility claims.
  - Bridge-mode/eligibility artifacts consumed by Phase 8 acceptance reporting.

## imports_from_prev

- `FND-02` pin-manifest/environment-manifest schema and lane provenance rules.
- `P1-07` Phase 1 acceptance bundle and workstation compatibility evidence.
- `P0-08` frozen reduced/nozzle case metadata and baseline comparison context.
- `P2-04` pinned-stage allocator contract for bring-up staging buffers.
- `P2-02` device-persistent allocation substrate for `DeviceDirect`.
- `P3-05` canonical pressure stage IDs and parent NVTX scaffolding (`pressure_assembly`, `pressure_solve_native`, `pressure_solve_amgx`, `pressure_post`).
- Phase 8 pass-A profiling vocabulary for telemetry alignment (`P8-03`/`P8-05`) without formal baseline-lock policy.

## exports_to_next

- Pressure snapshot and replay tooling for backend parity and offline debugging.
- `PressureMatrixCache` with topology-keyed CSR pattern persistence.
- Runtime pressure-backend selector contract (`native` required baseline, `amgx` secondary, explicit fallback).
- Canonical pressure-stage instrumentation and telemetry artifacts aligned to Phase 3/8 contracts.
- `DeviceDirect` pressure bridge contract and bridge-mode eligibility artifact.
- Explicit seam packet for Phase 5:
  - `pressure_assembly`
  - `pressure_solve_native`
  - `pressure_solve_amgx`
  - `pressure_post`

## shared_terms

- `PressureMatrixCache`: object-registry pressure cache owning CSR topology and staging state across repeated solves.
- `PressureMatrixCache` topology key: stable hash over topology and solver-config inputs used to guarantee deterministic cache reuse.
- `PinnedHost`: correctness-only pressure staging mode; allowed for bring-up and replay, never production-eligible.
- `DeviceDirect`: device-resident pressure staging mode required for production-eligible AmgX no field-scale host transfers.
- `pressure backend`: runtime-selected path (`native` required baseline, `amgx` secondary) with non-optional fallback.
- `fallback reason`: deterministic classification string for unsupported case conditions, consumed by logs and acceptance artifacting.

## open_discontinuities

- `[tracked] coupled-interface-phase4-scope`: Phase 4 supports only uncoupled scalar pressure systems; coupled interface or unsupported patch/interface conditions must trigger native fallback. Impacted PR IDs: `P4-07`, `P5-10`. Citations: `docs/specs/phase4_linear_algebra_spec.md` -> `#### Decision 4.5 — Support only uncoupled scalar systems in Phase 4`, `docs/authority/support_matrix.md` -> `## Global Policy`. Preferred reading: keep coupled-interface support out of Phase 4 and require explicit support-matrix/acceptance change before widening.
- `[tracked] pressure-stage-range-name-drift-risk`: Pressure telemetry and reduced-case comparison must use canonical stage IDs exactly as defined in stage and acceptance artifacts; no local aliases. Impacted PR IDs: `P4-08`, `P4-09`. Citations: `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`, `docs/authority/acceptance_manifest.json` -> `accepted_tuples[*].required_stage_ids`. Preferred reading: fail validation on alias drift.

## validation_checks

- All Phase 4 cards preserve canonical dependency edges exactly:
  - `P4-01 <- P1-07,FND-02`
  - `P4-02 <- P4-01,P0-08`
  - `P4-03 <- P4-02`
  - `P4-04 <- P4-03`
  - `P4-05 <- P4-04`
  - `P4-06 <- P4-05,P2-04`
  - `P4-07 <- P4-06`
  - `P4-08 <- P4-07,P3-05`
  - `P4-09 <- P4-08,P2-02`
- Every card anchors to authority docs + exact Phase 4 spec subsection + backlog `scope` + backlog `done_when`.
- Native remains the required baseline; AmgX remains secondary until/unless acceptance policy elevates it.
- `PinnedHost` remains explicitly correctness-only bring-up; `DeviceDirect` is required for no-field-scale host-transfer production eligibility.
- Phase 4 scope remains bounded to pressure linear algebra bridge work and does not absorb generic VOF transport or nozzle BC/startup semantics.
