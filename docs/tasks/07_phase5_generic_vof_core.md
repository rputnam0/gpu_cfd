# Phase 5 - Generic VOF core

Phase 5 owns `P5-01..P5-11` and delivers the device-authoritative generic VOF core: runtime gating, topology and mirrors, alpha plus MULES and subcycling, mixture and interface updates, momentum and pressure coupling, and the write-time validation freeze package. This section consumes Phase 2 memory and residency contracts, Phase 3 execution and stage contracts, Phase 4 pressure bridge contracts, and Phase 8 pass-A instrumentation prerequisites as fixed inputs. It explicitly excludes nozzle-specific BC kernels and startup seeding behavior owned by Phase 6.

## P5-01 Phase 5 symbol reconciliation note

- Objective:
  - Produce a reviewed semantic-to-local source reconciliation note so all Phase 5 edits target the correct SPUMA/v2412 files before solver changes start.
- Exact citations:
  - Authority:
    - `docs/authority/semantic_source_map.md` -> `## Frozen Mapping`
    - `docs/authority/semantic_source_map.md` -> `## Implementation Rule`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (canonical solver family freeze)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-07 Semantic source-audit helper`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.52 Step 1 — Complete the local semantic source audit and reconcile symbols against the actual SPUMA/v2412 tree`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `## 8.1 Core file map`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `### M5.0 — Symbol reconciliation`
  - Backlog scope:
    - `Generate and review the exact mapping from semantic sources to the local SPUMA/v2412 files touched by the generic VOF port.`
  - Backlog done_when:
    - `The implementation branch has a reviewed patch-target note before any Phase 5 solver edits land.`
- Depends on (backlog IDs):
  - `FND-07`
- Prerequisites:
  - `FND-07` semantic source-audit helper and frozen semantic source map are available.
- Concrete task slices:
  1. Generate `phase5_symbol_reconciliation.md` that maps each contract surface (`alphaPredictor`, `pressureCorrector`, `interfaceProperties`, momentum stage) to local SPUMA/v2412 patch targets.
  2. Record explicit include and ownership boundaries for `DeviceAlphaTransport`, `DeviceMULES`, `DeviceMomentumPredictor`, `DevicePressureCorrector`, and `DeviceSurfaceTension`.
  3. Link each mapping row to the frozen semantic source map entries and mark unresolved symbols as blockers.
  4. Capture reviewer signoff and freeze this note as the only valid patch-target map for `P5-02..P5-11`.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase5_symbol_reconciliation.md`
    - patch-target freeze note for Phase 5
  - Consumed:
    - `FND-07` source-audit helper outputs
    - `semantic_source_map.md` frozen mapping contract
- Validation:
  - Every planned Phase 5 subsystem edit maps to one reviewed local target family.
  - No unresolved symbol remains untagged.
- Done criteria:
  - Reconciliation note is reviewed and accepted before any Phase 5 solver implementation PR proceeds.
- Exports to later PRs/phases:
  - Frozen local target map consumed by `P5-02..P5-11`.
  - Phase 6 import for cross-phase ownership continuity at the generic/nozzle seam.

## P5-02 VOF runtime gate and support scan

- Objective:
  - Implement canonical `gpuRuntime.vof` gating and deterministic support scanning so unsupported Phase 5 tuples fail before the timestep loop.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (runtime normalization under `gpuRuntime`)
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Phase 5 Generic VOF Envelope`
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.md` -> `## FunctionObject Classification`
    - `docs/authority/support_matrix.json` -> `phase5_generic_vof_envelope.runtime_policy`
    - `docs/authority/support_matrix.json` -> `exact_audited_scheme_tuple`
    - `docs/authority/support_matrix.json` -> `function_object_policy`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.53 Step 2 — Add normalized gpuRuntime configuration (gpuVoF compatibility shim) and runtime gate`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.54 Step 3 — Implement CaseSupportReport`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-03 — Feature envelope is strict and validated upfront`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-13 — Strict fail-fast by default, debug fallback only by explicit opt-in`
  - Backlog scope:
    - `Implement the gpuRuntime.vof / compatibility-shim gate plus deterministic support scanning for the allowed Phase 5 envelope.`
  - Backlog done_when:
    - `Unsupported generic VOF tuples fail fast before the timestep loop and accepted tuples are explicitly identified.`
- Depends on (backlog IDs):
  - `FND-04`
  - `P5-01`
- Prerequisites:
  - `P5-01` patch-target reconciliation is frozen.
  - `FND-04` support scanner and fail-fast policy contract exists.
- Concrete task slices:
  1. Add canonical `gpuRuntime.vof` parsing with compatibility shim handling for legacy `gpuVoF` blocks.
  2. Implement startup support scanning against support-matrix phase-5 envelope, exact scheme tuple, and functionObject policy classes.
  3. Enforce `failFast` default with explicit debug-only mode labels for any permitted stage fallback.
  4. Emit `CaseSupportReport` with accepted/rejected tuple details and authority citations.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `gpuRuntime.vof` parser and gate contract
    - `CaseSupportReport` schema
  - Consumed:
    - `FND-04` scanner and failure taxonomy
    - support-matrix machine-readable policy entries
- Validation:
  - Unsupported schemes, BC families, or functionObjects are rejected pre-timestep with deterministic reasons.
  - Accepted phase-5 tuples are explicitly labeled and recorded.
- Done criteria:
  - Runtime gate is canonical and deterministic.
  - Unsupported tuples cannot enter the Phase 5 timestep loop silently.
- Exports to later PRs/phases:
  - `CaseSupportReport` consumed by `P5-03` initialization and `P5-11` validation artifacts.
  - Generic/nozzle seam support classification packet for Phase 6 imports.

## P5-03 Persistent topology and boundary-map substrate

- Objective:
  - Build stable device topology and boundary metadata for generic VOF stages with deterministic memory reporting.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (device-resident hot path and no post-warmup allocation churn)
    - `docs/authority/support_matrix.md` -> `## Phase 5 Generic VOF Envelope`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-08 Mesh mirror and startup registration`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.55 Step 4 — Implement DeviceMeshTopology and DeviceBoundaryMaps`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.19 Boundary representation`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.22 Matrix and index model`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-17 — Index width defaults to 32-bit on device`
  - Backlog scope:
    - `Build the Phase 5 topology, cell/face connectivity, and patch descriptor substrate needed by the generic VOF device path.`
  - Backlog done_when:
    - `The generic VOF path has stable device topology/boundary metadata and emits a memory report for the accepted reduced cases.`
- Depends on (backlog IDs):
  - `P2-08`
  - `P5-02`
- Prerequisites:
  - `P5-02` support gating for accepted generic tuples.
  - `P2-08` mesh mirror/startup registration substrate.
- Concrete task slices:
  1. Implement `DeviceMeshTopology` ownership for cell/face connectivity and patch indexing with stable allocation lifecycle.
  2. Implement `DeviceBoundaryMaps` for generic patch families admitted in Phase 5 support scope.
  3. Add deterministic topology hash and residency accounting suitable for repeated timesteps without rebuild.
  4. Emit `phase5_topology_memory_report` for accepted reduced cases (`R2`, `R1-core`).
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceMeshTopology` contract
    - `DeviceBoundaryMaps` contract
    - `phase5_topology_memory_report`
  - Consumed:
    - `P2-08` startup topology registration outputs
    - `P5-02` accepted tuple classification
- Validation:
  - Topology and boundary maps allocate once and remain stable through repeated iterations.
  - Memory report is deterministic on repeated runs for the same case tuple.
- Done criteria:
  - Generic VOF device topology/boundary substrate is stable and reportable.
- Exports to later PRs/phases:
  - Topology/boundary contracts consumed by `P5-04..P5-11`.
  - Boundary metadata handoff packet consumed by Phase 6 boundary-manifest work.

## P5-04 Field mirrors and old-time state for VOF

- Objective:
  - Mirror the full Phase 5 field set to device authority, including old-time and previous-correction state required by alpha and pressure coupling.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden hot-stage host reads and explicit write-time staging)
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-07 Mirror traits and field mirrors`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-09 Explicit visibility APIs and output stager`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.56 Step 5 — Implement DeviceFieldMirror and DeviceVoFState`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.57 Step 6 — Implement initialization upload and old-time seeding`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.17 Field authority model`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.20 Old-time state`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.20A Restart / checkpoint semantics`
  - Backlog scope:
    - `Mirror the Phase 5 field set, including old-time/previous-correction state needed by alpha transport and pressure coupling.`
  - Backlog done_when:
    - `Required VOF fields are device-visible with explicit host authority boundaries and old-time state is preserved correctly.`
- Depends on (backlog IDs):
  - `P5-03`
  - `P2-07`
- Prerequisites:
  - `P5-03` topology/boundary substrate.
  - `P2-07` field-mirror traits and registration helpers.
- Concrete task slices:
  1. Implement `DeviceFieldMirror` set for phase-fraction, flux, velocity, pressure, mixture, and interface support fields.
  2. Implement `DeviceVoFState` authority markers and unsafe-host-read guards.
  3. Implement old-time and previous-correction persistence update at controlled lifecycle points.
  4. Wire explicit host visibility only through approved commit paths.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceVoFState` contract
    - old-time and previous-correction mirror state contract
  - Consumed:
    - `P2-07` mirror trait infrastructure
    - `P2-09` visibility/commit boundary APIs
- Validation:
  - Required field mirrors remain device-authoritative through timestep execution.
  - Old-time and previous-correction state survives timestep and restart/reload cycles as required.
- Done criteria:
  - Device-visible field set and old-time state are complete and correctness-ready.
- Exports to later PRs/phases:
  - `DeviceVoFState` and old-time state contracts consumed by `P5-05..P5-11`.
  - Restart-sensitive mirror contract consumed by Phase 6 startup semantics.

## P5-05 Alpha skeleton path

- Objective:
  - Stand up the end-to-end alpha stage skeleton (control snapshots, alpha-flux formation, and predictor scaffold) without yet claiming bounded MULES correctness.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (canonical solver family and runtime normalization)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`alpha_pre`, `alpha_subcycle_body`)
    - `docs/tasks/05_phase3_execution_model.md` -> `## P3-03 GpuExecutionContext and execution registry`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.58 Step 7 — Implement alpha Courant reduction and controls snapshots`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.59 Step 8 — Implement alpha flux formation and no-op predictor scaffold`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.24 High-level host orchestration`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.25 Alpha predictor control flow`
  - Backlog scope:
    - `Implement the end-to-end alpha skeleton path with control snapshots, alpha-flux formation, and placeholder/predictor-only logic.`
  - Backlog done_when:
    - `The alpha stage runs end-to-end on the reduced case and establishes the device control-flow skeleton without claiming bounded correctness yet.`
- Depends on (backlog IDs):
  - `P5-04`
  - `P3-03`
- Prerequisites:
  - `P5-04` device field/state ownership.
  - `P3-03` execution context and stage binding contract.
- Concrete task slices:
  1. Implement alpha control snapshot build (including allowed scalar controls such as alpha Courant-derived values).
  2. Implement device alpha-flux formation and stage plumbing through `GpuExecutionContext`.
  3. Implement predictor-only scaffold with explicit placeholder semantics where bounded correction is not yet active.
  4. Emit alpha-stage skeleton trace artifact proving end-to-end control flow.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - alpha skeleton orchestration contract
    - alpha control snapshot artifact
  - Consumed:
    - `DeviceVoFState` from `P5-04`
    - stage execution registry from `P3-03`
- Validation:
  - Reduced case runs alpha stage end-to-end with correct stage ordering and no hidden host fallback.
  - Skeleton mode is explicitly labeled as pre-bounded-correction.
- Done criteria:
  - Device alpha skeleton path is functional and traceable.
- Exports to later PRs/phases:
  - Alpha skeleton control flow consumed by `P5-06`.
  - Stage trace vocabulary reused by `P5-11` and Phase 8 acceptance evidence.

## P5-06 Full alpha + MULES + subcycling

- Objective:
  - Complete bounded alpha transport with explicit MULES semantics, alpha subcycling, and previous-correction flux behavior under the frozen scheme subset.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (canonical algebraic VOF family)
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.json` -> `exact_audited_scheme_tuple`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.61 Step 10 — Port the minimal MULES subset`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.62 Step 11 — Add alpha subcycling and previous-correction persistence`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.28 MULES subset to port`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.29 Device MULES kernel strategy`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-06 — Preserve current numerics before optimization`
  - Backlog scope:
    - `Complete the bounded alpha path, explicit MULES limiter logic, previous-correction flux behavior, and alpha subcycling semantics.`
  - Backlog done_when:
    - `Reduced generic VOF cases keep alpha bounded and numerically consistent with the frozen CPU reference under the accepted scheme subset.`
- Depends on (backlog IDs):
  - `P5-05`
- Prerequisites:
  - `P5-05` alpha skeleton and controls path.
- Concrete task slices:
  1. Implement minimal required MULES correction subset with phase-5 boundedness semantics.
  2. Implement alpha subcycling loop and previous-correction persistence state transitions.
  3. Enforce accepted scheme tuple checks for alpha path at runtime.
  4. Emit alpha boundedness diagnostics aligned with reduced-case validation slices.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceMULES` bounded-correction contract
    - alpha boundedness diagnostics artifact
  - Consumed:
    - alpha skeleton path from `P5-05`
    - phase-5 exact scheme tuple constraints
- Validation:
  - `R2` transport slice and `R1-core` generic case maintain accepted alpha boundedness behavior.
  - Previous-correction state and subcycling semantics match frozen CPU reference intent.
- Done criteria:
  - Full bounded alpha plus MULES and subcycling behavior is complete for accepted generic tuples.
- Exports to later PRs/phases:
  - Bounded alpha subsystem consumed by `P5-07..P5-11`.
  - Generic alpha contract consumed by Phase 6 nozzle-boundary integration.

## P5-07 Mixture update and interface/surface-tension subset

- Objective:
  - Add device mixture-property updates and the constant-sigma interface/surface-tension subset admitted in Phase 5, without reopening contact-angle scope.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Phase 5 Generic VOF Envelope`
    - `docs/authority/support_matrix.json` -> `global_policy.contact_angle_in_scope`
    - `docs/authority/support_matrix.json` -> `global_policy.surface_tension_model_scope`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (contact-angle out of milestone-1 scope)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.63 Step 12 — Implement device mixture-property updates`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.64 Step 13 — Implement interface and surface-tension baseline`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.30 Mixture-property update flow`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.31 Interface and surface-tension flow`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-16 — Surface-tension scope is deliberately narrow`
  - Backlog scope:
    - `Implement rho / rhoPhi updates plus the constant-sigma interface normals/curvature/surface-tension slice admitted in Phase 5 scope.`
  - Backlog done_when:
    - `Mixture properties and the restricted interface path validate on the designated reduced tests without reopening contact-angle scope.`
- Depends on (backlog IDs):
  - `P5-06`
- Prerequisites:
  - Bounded alpha state from `P5-06`.
- Concrete task slices:
  1. Implement device `rho` and `rhoPhi` update flow from alpha state and accepted model inputs.
  2. Implement constant-sigma interface normals, curvature, and surface-tension force path for admitted tuples.
  3. Add explicit rejection paths for contact-angle or unsupported surface-tension model variants.
  4. Emit mixture/interface validation diagnostics for reduced validation slices.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - mixture-property update contract
    - constant-sigma interface/surface-tension contract
  - Consumed:
    - bounded alpha outputs from `P5-06`
    - support-matrix scope gates
- Validation:
  - Reduced surface-tension slice validates without scope widening.
  - Unsupported contact-angle or non-constant sigma paths fail fast.
- Done criteria:
  - Mixture and restricted interface updates are complete and validated for admitted phase-5 tuples.
- Exports to later PRs/phases:
  - Mixture/interface outputs consumed by `P5-08..P5-11`.
  - Contact-angle exclusion evidence consumed by Phase 6 seam review.

## P5-08 Momentum predictor

- Objective:
  - Integrate the device momentum predictor with persistent `rAU` behavior for the accepted laminar reduced-case envelope.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Global Policy` (laminar-only scope)
    - `docs/authority/support_matrix.json` -> `global_policy.turbulence_scope`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (milestone-1 narrow support scope)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.65 Step 14 — Integrate momentum predictor`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.32 Momentum predictor flow`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-05 — Implement alpha path first, then mixture, interface, momentum, pressure`
  - Backlog scope:
    - `Add the device momentum-predictor path, including persistent rAU handling and the accepted laminar reduced-case wiring.`
  - Backlog done_when:
    - `The reduced laminar case exercises the device U equation path successfully and preserves the expected predictor semantics.`
- Depends on (backlog IDs):
  - `P5-07`
- Prerequisites:
  - `P5-07` mixture/interface updates are stable.
- Concrete task slices:
  1. Implement `DeviceMomentumPredictor` integration with persistent intermediate field ownership.
  2. Preserve expected `rAU` lifecycle and sequencing relative to pressure correction.
  3. Enforce laminar-only runtime gate behavior for Phase 5 accepted tuples.
  4. Emit predictor-stage validation artifacts on reduced laminar cases.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceMomentumPredictor` contract
    - momentum predictor validation artifact
  - Consumed:
    - mixture/interface outputs from `P5-07`
    - phase-5 support scope and turbulence constraints
- Validation:
  - Reduced laminar case completes predictor stage with expected semantics and ordering.
  - Unsupported turbulence scope is rejected before timestep execution.
- Done criteria:
  - Device momentum predictor is operational for accepted laminar tuples.
- Exports to later PRs/phases:
  - Momentum predictor outputs consumed by `P5-09` and `P5-10`.
  - Stage evidence consumed by `P5-11` acceptance package.

## P5-09 Native pressure backend integration

- Objective:
  - Integrate the native pressure-corrector backend into the device-authoritative generic core using Phase 4 pressure bridge contracts unchanged.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (native pressure required baseline)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`pressure_assembly`, `pressure_solve_native`, `pressure_post`)
    - `docs/tasks/06_phase4_pressure_linear_algebra.md` -> `## exports_to_next`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 4 -> Phase 5`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.66 Step 15 — Integrate pressure corrector with runtime-selectable backend`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.33 Pressure corrector flow`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.34 Non-orthogonal correction loops`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-09 — Keep native and AmgX pressure backends both alive`
  - Backlog scope:
    - `Integrate the native pressure corrector path, including non-orthogonal loops, into the device-authoritative Phase 5 solver.`
  - Backlog done_when:
    - `A reduced nozzle-friendly case completes the native pressure corrector path correctly with the Phase 5 generic core.`
- Depends on (backlog IDs):
  - `P5-08`
  - `P4-07`
- Prerequisites:
  - `P5-08` momentum predictor integration.
  - `P4-07` runtime-selected pressure backend contract with native baseline semantics.
- Concrete task slices:
  1. Wire native pressure-corrector path to device-authoritative fields and non-orthogonal loop controls.
  2. Consume `PressureMatrixCache` and pressure-stage boundary contracts without redefining backend policy.
  3. Preserve canonical pressure stage IDs and graph-external pressure solve boundaries.
  4. Emit native-pressure integration diagnostics on reduced cases.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - native pressure integration contract for Phase 5
    - pressure-stage sequencing artifact
  - Consumed:
    - `P4-07` backend selector and fallback taxonomy
    - canonical pressure-stage IDs from graph matrix
- Validation:
  - Reduced nozzle-friendly generic case completes native pressure path correctly.
  - Stage IDs and boundaries align with canonical matrix entries.
- Done criteria:
  - Native pressure backend is fully integrated as the required baseline in Phase 5 generic core.
- Exports to later PRs/phases:
  - Native pressure integration contract consumed by Phase 6 (`P6-08`) and Phase 5 freeze (`P5-11`).

## P5-10 AmgX pressure backend integration

- Objective:
  - Add runtime-selectable AmgX backend behind the `DeviceDirect` gate, preserving `PinnedHost` as correctness-only and enforcing acceptance classification rules.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (PinnedHost correctness-only, DeviceDirect required for production-style claims)
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/authority/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/authority/acceptance_manifest.json` -> `coverage_rules.backend_restrictions`
    - `docs/authority/acceptance_manifest.json` -> `coverage_rules.pinned_host_policy`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`pressure_solve_amgx`)
    - `docs/tasks/06_phase4_pressure_linear_algebra.md` -> `## P4-09 DeviceDirect pressure bridge`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.66 Step 15 — Integrate pressure corrector with runtime-selectable backend`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### DD5-09 — Keep native and AmgX pressure backends both alive`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `### Exit criteria` (DeviceDirect requirement for accepted AmgX residency claims)
  - Backlog scope:
    - `Wire the runtime-selectable AmgX backend into the Phase 5 solver behind the DeviceDirect gate and compare it against native behavior.`
  - Backlog done_when:
    - `Accepted reduced tuples can use the AmgX path without violating the no-field-scale-host-transfer contract.`
- Depends on (backlog IDs):
  - `P5-09`
  - `P4-09`
- Prerequisites:
  - `P5-09` native pressure integration is stable.
  - `P4-09` `DeviceDirect` pressure bridge contract is available.
- Concrete task slices:
  1. Wire AmgX backend selection in pressure-corrector flow behind explicit `DeviceDirect` admission checks.
  2. Preserve classification rules: `PinnedHost` runs remain correctness-only and non-accepted for production-style tuples.
  3. Implement native-vs-AmgX parity harness on admitted reduced tuples with explicit tuple ID binding.
  4. Emit AmgX backend eligibility and classification artifact aligned with acceptance coverage rules.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - AmgX pressure backend integration contract for Phase 5
    - backend eligibility/classification artifact
  - Consumed:
    - `P4-09` `DeviceDirect` bridge contract
    - acceptance coverage restrictions for AmgX tuples
- Validation:
  - Admitted reduced tuples can execute AmgX with `DeviceDirect` and compliant transfer classification.
  - Any non-DeviceDirect or `PinnedHost` path is labeled diagnostic/correctness-only and excluded from accepted production-style tuple claims.
- Done criteria:
  - Runtime-selectable AmgX backend is integrated with explicit `DeviceDirect` gate enforcement.
- Exports to later PRs/phases:
  - Backend parity artifacts consumed by `P5-11` baseline freeze.
  - Pressure backend seam contract consumed by Phase 6 pressure-boundary integration.

## P5-11 Write-time commit, validation artifacts, and Phase 5 baseline freeze

- Objective:
  - Finalize explicit write-time commit/restart parity behavior and publish the formal Phase 5 baseline artifact package tied to pass-A stage instrumentation coverage.
- Exact citations:
  - Authority:
    - `docs/authority/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/authority/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/authority/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/authority/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-03 Solver-stage instrumentation coverage`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 8 pass A -> Phase 5 / Phase 7`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.67 Step 16 — Implement write-time commit path, restart/reload parity, and unsafe-read asserts`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.68 Step 17 — Instrument and profile`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.69 Step 18 — Validate and freeze Phase 5 baseline`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.35 Commit and write flow`
    - `docs/specs/phase5_spuma_nozzle_spec.md` -> `### Acceptance checklist`
  - Backlog scope:
    - `Implement the explicit write-time commit path, restart/reload parity checks, validation/profile artifact generation, and the final Phase 5 baseline freeze.`
  - Backlog done_when:
    - `Phase 5 ends with a reviewable reduced-case baseline, restart/reload parity evidence, and artifacts ready for Phase 6 handoff.`
- Depends on (backlog IDs):
  - `P5-10`
  - `P8-03`
- Prerequisites:
  - `P5-10` backend integration paths are stable.
  - `P8-03` stage coverage artifact is available as a prerequisite for baseline freeze evidence.
- Concrete task slices:
  1. Implement write-time commit policy (`writeOnly` default) and unsafe-host-read assertions for device-authoritative fields.
  2. Implement restart/reload parity checks for accepted Phase 5 tuples with strict history-field requirements.
  3. Generate formal validation and profiling artifacts keyed to accepted tuple IDs and canonical stage IDs.
  4. Publish Phase 5 baseline freeze packet and seam handoff bundle for Phase 6.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase5_baseline_freeze_report`
    - restart/reload parity artifact set
    - phase-5-to-phase-6 seam handoff bundle
  - Consumed:
    - `P8-03` stage instrumentation coverage artifact
    - accepted tuple matrix and hard-gate policy from acceptance manifest
    - reference-case ladder roles (`R2`, `R1-core`)
- Validation:
  - Accepted phase-5 tuples pass required hard gates and threshold classes with deterministic disposition mapping.
  - Restart/reload parity for accepted tuples satisfies strict parity class requirements.
  - Artifact package is complete and consumable by Phase 6 planning and validation.
- Done criteria:
  - Phase 5 baseline is frozen with reviewable evidence, restart parity proof, and downstream seam packet.
- Exports to later PRs/phases:
  - `P5-11` baseline freeze and seam handoff consumed by Phase 6 (`P6-01`, `P6-09`) and Phase 8 traceability.
  - Generic-only validation packet preserving `R1-core` boundary for Phase 6 nozzle-specific expansion.

## imports_from_prev

- `FND-07` semantic source-audit helper and frozen mapping contracts.
- `FND-04` support scanner and fail-fast policy taxonomy.
- Phase 2 contracts:
  - `P2-07` mirror traits and field mirrors.
  - `P2-08` mesh mirror and startup registration substrate.
  - explicit visibility/commit boundaries from `P2-09`.
- Phase 3 contracts:
  - `P3-03` `GpuExecutionContext` and execution registry.
  - `P3-05` canonical stage scaffolding and parent ranges.
- Phase 4 pressure bridge contracts:
  - runtime-selected backend selector from `P4-07`.
  - `PressureMatrixCache` and pressure stage boundary rules.
  - `DeviceDirect` admission contract from `P4-09`.
- Phase 8 pass-A prerequisite export:
  - `P8-03` solver-stage instrumentation coverage.

## exports_to_next

- Generic VOF runtime gate and deterministic `CaseSupportReport`.
- Persistent `DeviceMeshTopology` and `DeviceBoundaryMaps` for admitted generic tuples.
- `DeviceVoFState` and old-time/previous-correction mirror contracts.
- Bounded alpha/MULES/subcycling generic core outputs.
- Generic mixture/interface/momentum/pressure stage integration artifacts.
- Runtime pressure backend contracts consumed unchanged from Phase 4:
  - `PressureMatrixCache`
  - native required baseline
  - AmgX via `DeviceDirect` gate
- Write-time commit and strict restart/reload parity artifact package.
- Phase 5 baseline freeze packet for `R2` slices and `R1-core`.
- Phase 5 to Phase 6 seam packet including:
  - generic-only boundary ownership statement
  - pressure boundary-state handoff (`PressureBoundaryStateView`) for nozzle-specific consumption
  - explicit guard that `R1-core` remains generic-only.

## shared_terms

- `DeviceVoFState`: device-authoritative state owner for generic VOF timestep execution and old-time persistence.
- `CaseSupportReport`: deterministic startup scan output binding a run to admitted/denied support-matrix tuples.
- `R1-core`: reduced generic ladder case that excludes nozzle-specific Phase 6 BC/startup semantics.
- `PressureBoundaryStateView`: Phase 5 pressure-boundary-state handoff consumed by Phase 6 nozzle boundary execution.
- `write-time commit`: explicit host visibility boundary executed at accepted write points only.
- `DeviceDirect`: required AmgX bridge mode for accepted no-field-scale-host-transfer claims.

## open_discontinuities

- `[tracked] phase5_phase6_pressure_boundary_state_shape`: Phase 5 exports `PressureBoundaryStateView` semantics, but exact field-membership freeze for Phase 6 consumption is still represented across phase-specific prose rather than one shared schema artifact. Impacted PR IDs: `P5-11`, `P6-08`, `P6-10`. Citations: `docs/specs/phase5_spuma_nozzle_spec.md` -> `#### 5.14A Pressure-boundary-state handoff contract`, `docs/tasks/boundary_matrix.md` -> `### Phase 5 -> Phase 6`. Preferred reading: keep Phase 5 as producer-of-contract only and require Phase 6 to consume without redefining pressure semantics.
- `[tracked] amgx_tuple_admission_guard`: Acceptance manifest admits only `R1-core` async AmgX tuple and classifies non-DeviceDirect/PinnedHost paths as non-accepted; implementation planning must keep this classification explicit in baseline freeze artifacts. Impacted PR IDs: `P5-10`, `P5-11`. Citations: `docs/authority/acceptance_manifest.md` -> `## Accepted Tuple Matrix`, `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`, `docs/authority/acceptance_manifest.json` -> `coverage_rules.backend_restrictions`, `coverage_rules.pinned_host_policy`. Preferred reading: enforce tuple-ID lookup and reject any inferred/ad hoc AmgX admission.

## validation_checks

- All Phase 5 cards preserve canonical dependency edges exactly:
  - `P5-01 <- FND-07`
  - `P5-02 <- FND-04,P5-01`
  - `P5-03 <- P2-08,P5-02`
  - `P5-04 <- P5-03,P2-07`
  - `P5-05 <- P5-04,P3-03`
  - `P5-06 <- P5-05`
  - `P5-07 <- P5-06`
  - `P5-08 <- P5-07`
  - `P5-09 <- P5-08,P4-07`
  - `P5-10 <- P5-09,P4-09`
  - `P5-11 <- P5-10,P8-03`
- Every card anchors to authority docs + exact Phase 5 spec subsection + backlog `scope` + backlog `done_when`.
- Phase 5 preserves milestone-1 canonical solver family: algebraic `incompressibleVoF` with explicit MULES and PIMPLE sequencing.
- Phase 5 keeps scope generic-only:
  - no nozzle-specific BC kernels
  - no startup seeding subsystem ownership
  - no contact-angle expansion
- Phase 5 consumes Phase 4 pressure bridge unchanged:
  - native remains required baseline
  - `PinnedHost` remains correctness-only
  - AmgX accepted claims require `DeviceDirect`
  - canonical pressure stage IDs are preserved.
- `P5-11` baseline-freeze evidence explicitly consumes `P8-03` stage coverage as required by cross-section dependency policy.
