# Phase 3 - Execution model

Phase 3 owns `P3-01..P3-08` and defines the execution substrate used by later solver phases: explicit execution modes, stream and event topology, capture boundaries, warmup/upload policy, graph reuse policy, and canonical stage sequencing. This section consumes Phase 2 residency and visibility contracts as fixed upstream authority and does not redefine residency policy, pressure algorithms, or VOF/nozzle numerics.

## P3-01 Synchronization and stream inventory

- Objective:
  - Inventory all synchronization points and stream ownership in supported GPU paths, then freeze an explicit stream policy with no hidden owners.
- Exact citations:
  - Authority:
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `global_capture_rules`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden hot-stage host fallback and post-warmup allocation churn in production acceptance)
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-10 Compute-epoch enforcement`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.3 — Make stream ownership explicit and minimal`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.4 — Remove unconditional \`cudaDeviceSynchronize()\` from normal-mode hot paths`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 1 — Inventory and gate all existing hot-path synchronization`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 2 — Introduce explicit stream ownership`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M0 — Sync and stream inventory`
  - Backlog scope:
    - "Inventory existing sync points, define explicit stream ownership, and lock down the initial stream policy."
  - Backlog done_when:
    - "The branch has a reviewed sync/stream inventory and no hidden stream ownership in supported paths."
- Depends on (backlog IDs):
  - `P2-10`
- Prerequisites:
  - Phase 2 compute-epoch guardrails (`CpuTouchGuard`) and explicit visibility boundaries are active from `P2-09..P2-10`.
  - Phase 2 memory seams are imported as fixed contracts: `gpuRuntime.memory`, `ResidencyRegistry`, `ensureHostVisible`, `ensureDeviceVisible`, `OutputStager`, `ScratchCatalog`.
- Concrete task slices:
  1. Build a synchronization inventory for supported GPU execution paths, classifying each sync call by reason (`debug`, `write_boundary`, `fatal`, `hidden_hot_path`) and by stage context.
  2. Build a stream ownership inventory that traces every kernel launch and event edge to explicit owners (`computeStream`, `stagingStream`) and flags legacy/default stream usage.
  3. Define and land explicit stream policy rules: no implicit stream creation in stage code, no hidden ownership handoff, and no hot-path device-wide synchronization in production lanes.
  4. Emit a reviewed seam artifact (`sync_stream_inventory`) consumed by downstream execution-mode and wrapper PRs.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `sync_stream_inventory.md` (or JSON companion) with stage-tagged sync and stream ownership evidence
    - explicit stream-ownership policy contract for Phase 3 execution layers
  - Consumed:
    - `CpuTouchGuard` and epoch logs from `P2-10`
    - visibility/staging boundaries from `P2-09`
- Validation:
  - Inventory includes every `cudaDeviceSynchronize`, `cudaStreamSynchronize`, and host-blocking event wait in supported paths.
  - No supported hot-path launch remains on legacy/default stream.
  - Hidden stream creation or hidden sync in supported paths fails the Phase 3 gate.
- Done criteria:
  - Reviewed sync/stream inventory exists and is linked to canonical stage contexts.
  - Supported paths have no hidden stream ownership.
- Exports to later PRs/phases:
  - Frozen stream ownership baseline for `P3-02`, `P3-03`, and `P3-04`.
  - Sync audit vocabulary consumed by `P3-08` acceptance checks and Phase 8 instrumentation.

## P3-02 Execution-mode parser and selection policy

- Objective:
  - Implement deterministic execution-mode parsing and selection under normalized runtime config, with explicit downgrade behavior to `async_no_graph`.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/authority/graph_capture_support_matrix.json` -> `run_modes`
    - `docs/authority/graph_capture_support_matrix.json` -> `global_capture_rules.capture_failure_policy`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (runtime normalization under `gpuRuntime`)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3` (fallback must downgrade explicitly to `async_no_graph`)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.1 — Introduce an explicit execution-mode ladder`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`GpuExecutionConfig\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Bring-up path versus production path`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Fallback/rollback behavior`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 3 — Add \`GpuExecutionConfig\` and execution-mode selection`
  - Backlog scope:
    - "Implement runtime parsing and selection for the supported execution modes (including graph-disabled and graph-enabled lanes)."
  - Backlog done_when:
    - "Execution mode is selected deterministically from the normalized runtime tree with clear downgrade behavior."
- Depends on (backlog IDs):
  - `P3-01`
  - `FND-06`
- Prerequisites:
  - `P3-01` stream/sync inventory and policy baseline.
  - Foundation run-mode and stage registry loader from `FND-06`.
- Concrete task slices:
  1. Extend normalized runtime parsing under `gpuRuntime.execution` to resolve supported modes (`sync_debug`, `async_no_graph`, `graph_fixed`) from authority-defined labels.
  2. Implement deterministic mode selection, including explicit downgrade path to `async_no_graph` for capture-unsafe or policy-violating conditions with logged reason codes.
  3. Keep memory-mode and execution-mode concerns orthogonal: execution selection may consume Phase 2 memory assertions but may not redefine `gpuRuntime.memory` semantics.
  4. Emit mode-resolution and downgrade evidence artifact consumed by context and acceptance layers.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - execution-mode selection policy contract
    - `execution_mode_resolution_report` artifact schema with explicit downgrade reasons
  - Consumed:
    - canonical run-mode labels from `graph_capture_support_matrix(.json)`
    - Foundation stage/mode loader interfaces from `FND-06`
- Validation:
  - Same runtime config always resolves to the same mode and same fallback decision.
  - Unsupported mode labels fail fast; no silent aliasing.
  - Every downgrade from `graph_fixed` emits explicit `async_no_graph` reason and evidence.
- Done criteria:
  - Runtime mode is selected deterministically from normalized `gpuRuntime` config.
  - Downgrade behavior is explicit, logged, and reviewable.
- Exports to later PRs/phases:
  - Canonical execution-mode resolution consumed by `P3-03..P3-08`.
  - Stable run-mode labels and fallback semantics for Phase 8 tuple-level acceptance rows.

## P3-03 GpuExecutionContext and execution registry

- Objective:
  - Introduce one long-lived `GpuExecutionContext` owner and stage execution registry so all supported stages resolve execution state from a single authority.
- Exact citations:
  - Authority:
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## exports_to_next` (Phase 2 seam packet artifacts)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.2 — Centralize execution state in a long-lived \`GpuExecutionContext\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`ExecutionStreams\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`DeviceResidencyRegistry\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`GpuExecutionContext\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`HostStageController\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 4 — Create \`GpuExecutionContext\` and residency registry`
  - Backlog scope:
    - "Add the central `GpuExecutionContext` and registry that own stream handles, mode selection, and stage execution state."
  - Backlog done_when:
    - "All supported GPU stages resolve execution context through one owner rather than ad hoc local state."
- Depends on (backlog IDs):
  - `P3-02`
- Prerequisites:
  - Deterministic execution-mode policy from `P3-02`.
  - Imported Phase 2 seam packet (`gpuRuntime.memory`, `ResidencyRegistry`, `ensureHostVisible`, `ensureDeviceVisible`, `OutputStager`, `CpuTouchGuard`, `ScratchCatalog`).
- Concrete task slices:
  1. Implement `GpuExecutionContext` as the single host owner of stream handles, mode, stage registry bindings, graph lifecycle state, and execution counters.
  2. Wire Phase 2 contracts into context-owned services without redefining their policy: residency checks, visibility APIs, output staging, CPU-touch guard, and scratch tracking.
  3. Build a stage execution registry keyed by canonical stage IDs from Foundation/graph matrix and expose a single context lookup path for supported stage launch code.
  4. Refactor supported stage entry points to resolve execution state through `GpuExecutionContext` only.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `GpuExecutionContext` contract
    - stage execution registry contract keyed to canonical stage IDs
  - Consumed:
    - Phase 2 seam artifacts and APIs
    - execution-mode resolution policy from `P3-02`
- Validation:
  - Supported stage launch paths compile and run using only context-resolved execution state.
  - No supported stage path keeps ad hoc local stream ownership or mode state.
  - Context integration does not bypass or shadow Phase 2 residency and visibility APIs.
- Done criteria:
  - One owner (`GpuExecutionContext`) governs stage execution state in supported paths.
  - Stage lookup and execution binding are centralized and canonical-ID-driven.
- Exports to later PRs/phases:
  - `GpuExecutionContext` contract consumed by `P3-04..P3-08`.
  - Phase 4/5 downstream integration point for stage orchestration and explicit execution policy control.

## P3-04 Async launch wrapper layer

- Objective:
  - Add consistent graph-ready async launch wrappers and remove implicit hot-path `cudaDeviceSynchronize()` behavior from wrapped stages.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `global_capture_rules`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates` (`cudaDeviceSynchronize_calls == 0` inside steady-state inner ranges)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.3 — Make stream ownership explicit and minimal`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.4 — Remove unconditional \`cudaDeviceSynchronize()\` from normal-mode hot paths`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.5 — Use stream-capture-built graphs first, with whole-graph update`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Synchronization policy`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 5 — Convert launch wrappers to graph-safe async behavior`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M2 — Async no-graph baseline`
  - Backlog scope:
    - "Introduce graph-ready async launch wrappers and remove implicit hot-path device-wide synchronization from the wrapped stages."
  - Backlog done_when:
    - "Wrapped stages launch through a consistent async interface and no longer rely on hidden `cudaDeviceSynchronize()` behavior."
- Depends on (backlog IDs):
  - `P3-03`
- Prerequisites:
  - Central context/registry ownership from `P3-03`.
- Concrete task slices:
  1. Define context-aware async launch wrapper API that requires explicit stream and stage ID and is capture-safe by default.
  2. Replace wrapped hot-path stage launches to use async wrappers and explicit event-based ordering, eliminating hidden device-wide synchronization.
  3. Keep debug behavior isolated under `sync_debug` while preserving `async_no_graph` as non-graph production fallback.
  4. Add sync-audit hooks that prove steady-state wrappers do not inject device-wide sync.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - async launch wrapper contract
    - sync-audit counters/report used in acceptance packet
  - Consumed:
    - `GpuExecutionContext` ownership from `P3-03`
    - Phase 2 write-boundary exceptions via `OutputStager`
- Validation:
  - Wrapped stages in steady-state use async interface and explicit stream/event ordering.
  - Hot-path `cudaDeviceSynchronize()` usage in wrapped stages drops to zero for production lanes.
  - Debug mode can re-enable diagnostic sync without changing production behavior.
- Done criteria:
  - Wrapped stages launch through one consistent async wrapper layer.
  - Hidden hot-path device-wide synchronization behavior is removed from wrapped stages.
- Exports to later PRs/phases:
  - Graph-ready wrapper substrate used by `P3-06` capture and replay.
  - Sync-audit evidence consumed by `P3-08` acceptance and Phase 8 profiling checks.

## P3-05 Canonical stage scaffolding and parent NVTX ranges

- Objective:
  - Land canonical stage boundaries and parent NVTX scaffolding from the shared graph stage registry before first capture.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages`
    - `docs/authority/graph_capture_support_matrix.json` -> `required_orchestration_ranges`
    - `docs/authority/graph_capture_support_matrix.json` -> `stage_level_range_rule`
    - `docs/authority/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Canonical stage taxonomy and graph-support ownership`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Graph-safe stage composition rules`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### NVTX3 ranges`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 6 — Add NVTX3 domains and coarse solver-stage annotation`
  - Backlog scope:
    - "Add the canonical stage boundaries and parent-stage NVTX scaffolding imported from the graph support matrix."
  - Backlog done_when:
    - "Stage IDs are stable, shared, and visible to instrumentation before the first captured stage lands."
- Depends on (backlog IDs):
  - `P3-03`
  - `FND-06`
- Prerequisites:
  - `GpuExecutionContext` stage registry integration from `P3-03`.
  - Foundation canonical stage registry and matrix loader from `FND-06`.
- Concrete task slices:
  1. Import canonical stage IDs from Foundation registry and bind them to context stage descriptors without introducing local aliases.
  2. Add parent NVTX ranges for required orchestration and stage ranges (`solver/timeStep`, `solver/steady_state`, `solver/pimpleOuter`, and canonical stage IDs).
  3. Scaffold explicit boundaries for graph-external stages (`pressure_solve_native`, `pressure_solve_amgx`, `write_stage`) and capture-safe-now stages (`pre_solve`, `outer_iter_body`).
  4. Add validation checks that fail on non-canonical stage names or missing parent ranges.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - canonical stage scaffolding contract for Phase 3 execution code
    - parent NVTX range coverage report keyed to canonical stage IDs
  - Consumed:
    - Foundation stage registry (`FND-06`)
    - graph capture support matrix stage taxonomy
- Validation:
  - All Stage 3 parent-stage ranges map to canonical stage IDs exactly.
  - No phase-local stage ID aliases are introduced in execution or instrumentation code.
  - Required orchestration ranges are present and visible before graph capture lands.
- Done criteria:
  - Stable shared stage IDs and parent ranges are active before first captured stage.
  - Instrumentation and execution code consume the same canonical stage scaffolding.
- Exports to later PRs/phases:
  - Canonical stage scaffolding and NVTX parent-range contract for Phase 4 pressure boundaries, Phase 5 stage expansion, and Phase 8 instrumentation (`P8-03 <- P3-05`).

## P3-06 First graph-enabled stage

- Objective:
  - Capture and replay the first approved Phase 3 stage in `graph_fixed` mode with stable addresses and strict capture-safety rules.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages` (capture-safe-now rows)
    - `docs/authority/graph_capture_support_matrix.json` -> `global_capture_rules`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.5 — Use stream-capture-built graphs first, with whole-graph update`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.6 — Use stable device-resident launch-parameter blocks`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.9 — Use graph upload in warm-up and keep upload/launch stream consistent`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Graph-safe stage composition rules`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 7 — Build the first capture-enabled stage on a supported solver`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M4 — First graph-enabled stage`
  - Backlog scope:
    - "Capture and replay the first approved graph-enabled stage using the Phase 3 execution contract."
  - Backlog done_when:
    - "At least one allowed stage captures and replays correctly with stable addresses and no illegal capture-time behavior."
- Depends on (backlog IDs):
  - `P3-04`
  - `P3-05`
- Prerequisites:
  - Graph-ready async wrappers from `P3-04`.
  - Canonical stage scaffolding and parent ranges from `P3-05`.
  - Phase 2 no-post-warmup-allocation and visibility contracts are active in supported path.
- Concrete task slices:
  1. Select and freeze first capture target from Phase 3 capture-safe-now stage IDs (`pre_solve` or `outer_iter_body` subset), with explicit declaration in implementation notes.
  2. Implement stream-capture instantiation and replay path using context-owned `computeStream` and stable-address buffers from Phase 2 pools.
  3. Enforce capture rules: no post-warmup allocation, no hidden host access, no graph-internal write/restart behavior, and explicit downgrade to `async_no_graph` on capture failure.
  4. Add warmup/upload path and replay validation traces demonstrating legal capture behavior and correct replay outputs.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - first graph-enabled stage template and replay contract
    - capture/replay legality report with downgrade reason taxonomy
  - Consumed:
    - async wrapper layer from `P3-04`
    - canonical stage boundaries from `P3-05`
    - stable allocation and visibility guarantees from Phase 2
- Validation:
  - Selected stage captures and replays across repeated iterations with stable pointer identity.
  - Capture path performs no disallowed API behavior during capture window.
  - Capture failure path downgrades explicitly to `async_no_graph` with logged reason.
- Done criteria:
  - At least one approved stage captures and replays correctly with stable addresses.
  - No illegal capture-time behavior is present in the supported capture path.
- Exports to later PRs/phases:
  - First validated capture/replay contract consumed by `P3-07`.
  - Capture legality evidence and downgrade taxonomy consumed by Phase 8 graph lifecycle instrumentation.

## P3-07 Graph fingerprint, cache, and rebuild policy

- Objective:
  - Implement deterministic graph fingerprinting, cache reuse, and rebuild/update policy with explicit parameter-mirror handling.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `graph_lifecycle_markers`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates` (`post_warmup_alloc_calls == 0`)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.10 — Add a graph build fingerprint and deterministic rebuild policy`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.6 — Use stable device-resident launch-parameter blocks`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`GraphBuildFingerprint\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### \`GpuGraphManager\``
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Graph build/update algorithm`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Graph rebuild triggers`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 8 — Add graph fingerprinting and cache lookup`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 9 — Add whole-graph update and deterministic rebuild`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M5 — Graph cache, update, rebuild policy`
  - Backlog scope:
    - "Implement graph fingerprinting, cache lookup, update/rebuild policy, and parameter mirror handling for capture reuse."
  - Backlog done_when:
    - "Steady-state runs reuse captured graphs when valid and rebuild only under documented change conditions."
- Depends on (backlog IDs):
  - `P3-06`
- Prerequisites:
  - First captured-stage substrate from `P3-06`.
- Concrete task slices:
  1. Define and implement `GraphBuildFingerprint` key fields (stage set, mode, topology/patch hashes, backend flags, pointer generation) used by graph cache lookup.
  2. Implement graph cache manager with deterministic path split: reuse, whole-graph update, or rebuild with explicit reason codes.
  3. Implement parameter mirror handling via stable host/device launch-parameter blocks so steady-state reuse does not require hot-path allocation.
  4. Emit graph lifecycle and rebuild audit artifacts keyed to canonical stage IDs and lifecycle markers.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - graph fingerprint schema and cache key contract
    - deterministic rebuild/update reason code contract
    - graph lifecycle audit report
  - Consumed:
    - first graph-enabled stage capture and replay contract from `P3-06`
    - Phase 2 stable allocation and scratch discipline
- Validation:
  - Steady-state replay reuses graph cache entries when fingerprint is unchanged.
  - Fingerprint changes trigger only documented rebuild/update reasons.
  - No post-warmup allocation is introduced in capture-safe stage execution.
- Done criteria:
  - Graph reuse and rebuild policy is deterministic, logged, and tied to documented triggers.
  - Capture reuse works in steady-state with explicit parameter mirror handling.
- Exports to later PRs/phases:
  - Graph fingerprint/cache/rebuild policy consumed by Phase 4 and Phase 5 stage integrations.
  - Graph lifecycle markers and rebuild evidence consumed by Phase 8 (`P8-04 <- P3-07`).

## P3-08 Write-boundary staging, production residency assertions, and Phase 3 acceptance

- Objective:
  - Integrate explicit write-boundary staging with execution modes, enforce production residency assertions, and produce the Phase 3 acceptance packet.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`write_stage` as graph-external boundary)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages` (`write_stage`)
    - `docs/authority/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/authority/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/authority/acceptance_manifest.md` -> `## Production Defaults`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (allowed write-time staging and forbidden silent host fallback)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-09 Explicit visibility APIs and output stager`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.11 — Keep write/output staging outside the hot compute graph`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### D3.12 — Fail fast on accidental host access in production mode`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 11 — Add write-boundary staging path`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 12 — Add production residency assertions`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 13 — Add profiling scripts and trace configuration`
    - `docs/specs/phase3_execution_model_spec.md` -> `#### Step 14 — Benchmark stop gate before any Phase 4 or Phase 5 work`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M6 — Write-boundary staging and production residency checks`
    - `docs/specs/phase3_execution_model_spec.md` -> `### Milestone M7 — Acceptance packet`
  - Backlog scope:
    - "Integrate explicit write-boundary staging with execution modes, add production residency assertions, and generate the Phase 3 acceptance packet."
  - Backlog done_when:
    - "Supported execution modes pass acceptance with write paths outside the timed/captured steady-state window and no hot-path global sync regressions."
- Depends on (backlog IDs):
  - `P3-07`
  - `P2-09`
- Prerequisites:
  - Graph cache/rebuild policy from `P3-07`.
  - Explicit visibility/staging interfaces from `P2-09`.
  - Sync-audit and parent-stage range artifacts from `P3-04` and `P3-05`.
- Concrete task slices:
  1. Integrate `OutputStager`, `ensureHostVisible`, and `ensureDeviceVisible` into explicit `write_stage` boundary logic that stays outside captured/timed steady-state execution.
  2. Add production residency assertions for `async_no_graph` and `graph_fixed` lanes using `ResidencyRegistry` and `CpuTouchGuard` evidence; fail fast on forbidden host access.
  3. Run Phase 3 acceptance checks for supported execution modes, including hard-gate evidence for no steady-state `cudaDeviceSynchronize()` regressions and no post-warmup allocation in capture-safe stages.
  4. Publish a Phase 3 acceptance packet and seam handoff bundle for Phase 4, Phase 5, and Phase 8.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - write-boundary staging assertions contract
    - Phase 3 acceptance packet (`phase3_acceptance_report` with sync/residency/graph evidence)
    - Phase 3 seam handoff bundle for downstream phases
  - Consumed:
    - `OutputStager`, `ensureHostVisible`, `ensureDeviceVisible` from `P2-09`
    - `GpuExecutionContext`, stage scaffolding, and graph lifecycle policy from `P3-03..P3-07`
- Validation:
  - Write and restart behavior remains graph-external and excluded from timed steady-state windows.
  - Supported execution modes satisfy acceptance hard-gate checks relevant to Phase 3 (`cudaDeviceSynchronize_calls == 0` in steady-state ranges, `post_warmup_alloc_calls == 0`, mandatory NVTX parent ranges present).
  - Production residency assertions catch forbidden host access with explicit failure or downgrade behavior, never silent migration.
- Done criteria:
  - Supported execution modes pass Phase 3 acceptance with explicit write-boundary staging outside captured/timed steady-state.
  - No hot-path global synchronization regressions remain in accepted Phase 3 paths.
- Exports to later PRs/phases:
  - `GpuExecutionContext` and execution registry handoff for Phase 4/5 integration points.
  - Canonical stage scaffolding and stage-boundary assertions for Phase 4 pressure boundaries and Phase 5 stage expansion.
  - Graph fingerprint/cache/rebuild policy and lifecycle evidence for Phase 8 instrumentation and acceptance pipelines.
  - Write-boundary staging assertions contract consumed by Phase 5 write-time semantics and Phase 8 acceptance evidence.

## imports_from_prev

- `gpuRuntime.memory` canonical parser and policy contract from Phase 2 (`P2-01`).
- `ResidencyRegistry` and deterministic residency reports from Phase 2 (`P2-06`).
- `ensureHostVisible` and `ensureDeviceVisible` explicit visibility APIs from Phase 2 (`P2-09`).
- `OutputStager` write/restart boundary staging contract from Phase 2 (`P2-09`).
- `CpuTouchGuard` compute-epoch enforcement from Phase 2 (`P2-10`).
- `ScratchCatalog` and post-warmup watermark evidence from Phase 2 (`P2-11`).
- Canonical run-mode and stage registry loader from Foundation (`FND-06`).

## exports_to_next

- `GpuExecutionContext` ownership contract for stream/event topology, mode binding, and stage execution state.
- Canonical stage scaffolding with parent NVTX ranges keyed to stage IDs from `graph_capture_support_matrix`.
- First validated graph-enabled stage capture/replay substrate in `graph_fixed` with explicit fallback to `async_no_graph`.
- Graph fingerprint/cache/update/rebuild policy with deterministic reason codes and lifecycle markers.
- Write-boundary staging assertions proving `write_stage` and restart boundaries remain graph-external and outside timed steady-state.
- Phase 3 acceptance packet with sync audits, capture legality evidence, and production residency assertions.
- Downstream seam packet consumed by:
  - Phase 4 pressure bridge planning (`pressure_assembly`, `pressure_solve_native`, `pressure_solve_amgx`, `pressure_post` boundaries)
  - Phase 5 stage expansion and generic VOF orchestration
  - Phase 8 instrumentation and graph lifecycle acceptance checks

## shared_terms

- `execution mode`: authority-defined runtime lane (`sync_debug`, `async_no_graph`, `graph_fixed`) resolved from `gpuRuntime.execution`.
- `capture-safe stage`: canonical stage ID whose current policy allows capture in this phase under global capture rules.
- `graph-external boundary`: canonical stage boundary that must remain outside captured hot compute graphs (for example `write_stage` and pressure solve boundaries until promoted).
- `GpuExecutionContext`: single owner of execution-state services, stream handles, mode, and stage execution bindings.
- `graph fingerprint`: deterministic key describing whether a cached graph execution object is reusable or must rebuild/update.
- `write boundary staging`: explicit output/restart host-visibility sequence through `OutputStager` and visibility APIs, outside timed/captured steady-state.

## open_discontinuities

- `[tracked] first-capture-stage-selection`: Phase 3 authority allows either `pre_solve` or `outer_iter_body` as an initial capture-safe target, but downstream profiling comparability improves if one canonical first-capture target is frozen before Phase 8 pass-A baselines; impacted PR IDs: `P3-06`, `P8-04`; citations: `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`, `docs/specs/phase3_execution_model_spec.md` -> `#### Step 7 — Build the first capture-enabled stage on a supported solver`; preferred reading: freeze `pre_solve` as the minimum canonical first-capture target and treat `outer_iter_body` capture as an additive expansion once cache/rebuild policy is green.

## validation_checks

- All Phase 3 cards preserve canonical backlog dependency edges exactly:
  - `P3-01 <- P2-10`
  - `P3-02 <- P3-01,FND-06`
  - `P3-03 <- P3-02`
  - `P3-04 <- P3-03`
  - `P3-05 <- P3-03,FND-06`
  - `P3-06 <- P3-04,P3-05`
  - `P3-07 <- P3-06`
  - `P3-08 <- P3-07,P2-09`
- Every card anchors to authority docs + exact Phase 3 spec subsections + backlog `scope` + backlog `done_when`.
- Phase 3 consumes Phase 2 seam contracts as fixed imports (`gpuRuntime.memory`, `ResidencyRegistry`, `ensureHostVisible`, `ensureDeviceVisible`, `OutputStager`, `CpuTouchGuard`, `ScratchCatalog`) and does not redefine residency policy.
- Graph governance remains aligned with authority:
  - write/restart remain graph-external boundaries
  - no post-warmup allocation inside capture-safe stages
  - capture failures downgrade explicitly to `async_no_graph`
  - stage IDs come from Foundation/graph matrix and are not redefined locally
- Phase 3 scope remains execution substrate only and does not absorb Phase 4 pressure backend policy or Phase 5/6 solver semantics.
