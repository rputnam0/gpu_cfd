# Boundary Matrix

This matrix records the seam rules between section docs. Earlier phases define contracts. Later phases consume them.

## Global Rules

- Authority docs and JSON companions outrank phase prose when they conflict.
- Section docs may refine implementation sequencing but may not widen support scope, fallback policy, solver-family choices, acceptance thresholds, or graph-stage IDs.
- If a seam is resolvable from the docs, update the affected section docs and keep this matrix aligned.
- If a seam is not resolvable from the docs, add a decision note in [decision_notes.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/decision_notes.md) and mark the impacted PRs as blocked until resolved.

## Seam Review Matrix

### Foundation -> Phase 0 / Phase 1

- Review reason: prevent phase-local redefinition of frozen case IDs, ladder roles, toolchain pins, supported tuples, and acceptance vocabulary.
- Handoff anchors:
  - `FND-01` authority loader
  - `FND-02` environment-manifest schema
  - `FND-03` case-role utilities
  - `FND-04` support scanner
  - emitted artifacts: `case_meta.json`, `stage_plan.json`, `host_env.json`, `cuda_probe.json`, `fatbinary_report.json`
- Gate:
  - no Phase 0 or Phase 1 card may create a local copy of ladder roles, supported tuples, fallback policy, or toolchain lanes
  - Phase 0 owns frozen reference bundles
  - Phase 1 consumes `case_meta.json` and the support matrix rather than redefining smoke cases

### Phase 2 -> Phase 3

- Review reason: execution planning must not redefine authority rules, hide CPU touches, or assume capture-safe allocation behavior before residency is frozen.
- Handoff anchors:
  - canonical `gpuRuntime.memory`
  - Residency Registry reports
  - `ensureHostVisible`
  - `ensureDeviceVisible`
  - `OutputStager`
  - `CpuTouchGuard`
  - `ScratchCatalog`
- Gate:
  - memory modes remain orthogonal to execution modes
  - write and restart stay graph-external
  - no post-warmup allocation inside capture-safe stages
  - every fallback path downgrades explicitly to `async_no_graph`

### Phase 4 -> Phase 5

- Review reason: generic VOF planning must consume the pressure bridge unchanged and avoid reopening backend policy.
- Handoff anchors:
  - pressure snapshot and replay tooling
  - `PressureMatrixCache`
  - runtime-selected native and AmgX bridge contract
  - `DeviceDirect`
  - pressure telemetry
  - canonical stage IDs: `pressure_assembly`, `pressure_solve_native`, `pressure_solve_amgx`, `pressure_post`
- Gate:
  - native remains the required baseline
  - `PinnedHost` remains correctness-only
  - AmgX production claims require `DeviceDirect`
  - Phase 5 distinguishes pressure semantics from pressure backend plumbing

### Phase 8 pass A -> Phase 5 / Phase 7

- Review reason: early instrumentation outputs are required upstream of baseline lock and must stay contract-only (not policy-changing).
- Handoff anchors:
  - `P8-03` solver-stage instrumentation coverage
  - `P8-05` Nsight Systems capture scripts and artifact layout
  - canonical stage IDs from the graph stage registry
- Gate:
  - Phase 5 consumes `P8-03` as an instrumentation prerequisite for `P5-11` without changing Phase 5 solver semantics
  - Phase 7 consumes `P8-05` as a hotspot-evidence prerequisite for `P7-01`
  - Phase 8 pass A does not introduce baseline-lock thresholds or CI/nightly lock language reserved for `P8-09`

### Phase 5 -> Phase 6

- Review reason: nozzle planning must not widen the BC envelope or introduce a second boundary-state model.
- Handoff anchors:
  - `support_matrix(.json)` Phase 6 rows
  - `gpuRuntime.startupSeed`
  - `PressureBoundaryStateView`
  - `PressureBoundaryState`
  - device-resident `snGradp`
  - `case_meta.json` patch-role metadata
  - boundary support report
  - boundary manifest
  - flat boundary spans
- Gate:
  - `R1-core` stays generic-only
  - `R1` and `R0` own nozzle semantics
  - `nozzle_bc_update` sequencing is explicit
  - `snGradp` is fresh before pressure assembly each corrector
  - restart reseeding remains forbidden unless `forceReseed yes`

### Phase 6 -> Phase 7

- Review reason: optimization plans must not widen semantics or support scope.
- Handoff anchors:
  - Phase 6 acceptance bundle
  - hotspot-ranking artifact
  - Phase 7 source-audit note
  - boundary manifest
  - flat boundary spans
  - `PressureBoundaryState.*`
  - exact `R1` and `R0` support-matrix subset
- Gate:
  - no new BC kinds
  - no contact-angle logic
  - no scheme widening
  - no graph-semantics changes beyond the canonical stage registry
  - optimization tasks are replaceable backends under frozen Phase 5 and Phase 6 contracts

### Phase 7 -> Phase 8

- Review reason: acceptance work must bind to explicit manifest rows and canonical instrumentation names instead of drifting with implementation.
- Handoff anchors:
  - Phase 7 regression package
  - selected custom-backend decision
  - `acceptance_manifest(.json)`
  - `graph_capture_support_matrix(.json)`
  - `host_env.json`
  - `cuda_probe.json`
  - standard `nsys` artifact layout
  - top-kernel and sanitizer outputs
- Gate:
  - every formal profiling job maps to a manifest row with case ID, backend, execution mode, kernel class, and NVTX stage IDs
  - `async_no_graph` remains the non-graph baseline
  - `graph_fixed` remains the accepted graph mode
  - `uvmAudit`, `syncAudit`, and `graphDebug` stay out of formal timing rows
  - formal baseline locks wait until the Phase 7 path is stable
