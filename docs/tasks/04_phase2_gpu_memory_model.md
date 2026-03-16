# Phase 2 — GPU memory model

Phase 2 owns `P2-01..P2-11` and establishes the production memory and residency substrate for later execution and solver phases. This section focuses on allocator tiers, explicit host/device visibility, residency state, and compute-epoch enforcement; it does not own graph orchestration, pressure math, or VOF solver semantics.

## P2-01 Canonical `gpuRuntime.memory` parser

- Objective:
  - Implement one authoritative memory-configuration parser and normalized policy tree for all Phase 2 memory behavior.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (runtime configuration normalization under `gpuRuntime`)
    - `docs/authority/continuity_ledger.md` -> `# 5. Package Consumption Rule`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-01 Authority ingestion scaffold`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 1 — Define five memory tiers`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 10 — Separate bring-up path from production path`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Configuration interface`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 1 — Add gpuRuntime.memory config and compatibility parsing`
  - Backlog scope:
    - "Implement the authoritative memory-configuration parser, enums, and compatibility-shim handling for the Phase 2 contract."
  - Backlog done_when:
    - "The branch has one normalized memory config tree and all later memory code resolves policy from it."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - Foundation authority-loader contract from `FND-01`.
  - Frozen `gpuRuntime` normalization rule from `continuity_ledger.md`.
- Concrete task slices:
  1. Define canonical Phase 2 enums and schema views for `DevicePersistent`, `DeviceScratch`, `PinnedStage`, `ManagedFallback`, and scalar staging rules.
  2. Implement parser and compatibility shims so legacy memory dict inputs resolve into one normalized `gpuRuntime.memory` tree.
  3. Enforce deterministic defaulting and validation for production versus bring-up memory policies.
  4. Export one policy-resolution API consumed by every later allocator and residency component.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - canonical `gpuRuntime.memory` parser contract
    - memory-policy normalization report
  - Consumed:
    - Foundation authority-ingestion interfaces
    - centralized runtime-normalization governance in `continuity_ledger.md`
- Validation:
  - Valid config variants produce one equivalent normalized policy tree.
  - Invalid or conflicting policy keys fail fast with deterministic diagnostics.
  - Downstream components resolve policy only through the canonical parser API.
- Done criteria:
  - One canonical memory config tree exists.
  - Later Phase 2 memory code no longer uses phase-local memory policy parsing.
- Exports to later PRs/phases:
  - Canonical `gpuRuntime.memory` contract consumed by `P2-02..P2-11` and by Phase 3 execution integration.

## P2-02 Device persistent pool

- Objective:
  - Add a deterministic pooled device allocator for long-lived hot objects.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden recurring UVM for registered hot objects)
    - `docs/authority/continuity_ledger.md` -> `# 2. Cross-phase contract matrix` (Phase 2 residency and allocator substrate exit contract)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-01 Authority ingestion scaffold`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 2 — Make DevicePersistent the canonical production location for hot objects`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 5 — Use CUDA async device pools for persistent and scratch tiers`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 3 — Implement DevicePersistentPool`
  - Backlog scope:
    - "Add the persistent device allocator for long-lived hot objects and the unit tests that verify deterministic allocation behavior."
  - Backlog done_when:
    - "Persistent allocations are pooled, leak-free, and validated independently of solver integration."
- Depends on (backlog IDs):
  - `P2-01`
- Prerequisites:
  - Canonical memory-policy parser from `P2-01`.
- Concrete task slices:
  1. Implement `DevicePersistentPool` with deterministic allocation and reuse policy from `gpuRuntime.memory`.
  2. Add ownership and lifetime tracking hooks required by the future residency registry.
  3. Add leak, reuse, and deterministic-allocation unit tests independent of solver runtime behavior.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DevicePersistentPool` allocator contract
    - persistent-pool stats schema
  - Consumed:
    - normalized memory policy from `P2-01`
- Validation:
  - Repeated startup and teardown cycles are leak-free.
  - Persistent allocations reuse as configured and do not drift unexpectedly.
  - Unit tests pass without requiring full solver integration.
- Done criteria:
  - Persistent allocator behavior is deterministic and test-covered.
  - Pool is ready for registry integration in later Phase 2 PRs.
- Exports to later PRs/phases:
  - `DevicePersistentPool` consumed by `P2-06`, `P2-08`, and pressure and VOF phases.

## P2-03 Device scratch pool

- Objective:
  - Implement transient timestep-local scratch allocation with reset and reuse semantics.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden post-warmup dynamic allocation churn in production)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules` (no post-warmup dynamic allocation)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 5 — Use CUDA async device pools for persistent and scratch tiers`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 6 — Use deterministic reuse defaults`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 4 — Implement DeviceScratchPool`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 7. Scratch policy`
  - Backlog scope:
    - "Implement the transient device scratch allocator for timestep-local workspaces with reset and reuse semantics."
  - Backlog done_when:
    - "Scratch allocation and reuse works across repeated iterations without unexpected growth or churn."
- Depends on (backlog IDs):
  - `P2-01`
- Prerequisites:
  - Canonical memory-policy parser from `P2-01`.
- Concrete task slices:
  1. Implement `DeviceScratchPool` with explicit reset points and watermark tracking hooks.
  2. Enforce deterministic growth and reuse rules across repeated iterations.
  3. Add unit and integration tests demonstrating stable reuse across iteration loops.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceScratchPool` allocator contract
    - scratch-watermark counters
  - Consumed:
    - normalized memory policy from `P2-01`
- Validation:
  - Iterative runs stabilize scratch watermarks after warmup.
  - Scratch behavior shows no uncontrolled growth or churn under repeated reuse cycles.
- Done criteria:
  - Scratch pool is stable, deterministic, and ready for cataloging in `P2-11`.
- Exports to later PRs/phases:
  - `DeviceScratchPool` consumed by `P2-06` and `P2-11`, then by Phase 3 and Phase 5 execution paths.

## P2-04 Pinned stage allocator

- Objective:
  - Add one explicit pooled pinned-host staging path for write, restart, and output boundaries.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (field-scale host interaction must be explicit staging, not hot-path reads)
    - `docs/authority/support_matrix.md` -> `## Global Policy` (production and bring-up separation and explicit fail-fast posture)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 7 — Make pinned staging bounded and explicit`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 5 — Implement PinnedStageAllocator`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 7A. Restart / checkpoint policy`
  - Backlog scope:
    - "Add the pooled cudaMallocHost-backed stage allocator for explicit write/restart/output staging."
  - Backlog done_when:
    - "Field-scale host staging goes through one explicit pinned path and unit tests cover allocation and reuse behavior."
- Depends on (backlog IDs):
  - `P2-01`
- Prerequisites:
  - Canonical memory-policy parser from `P2-01`.
- Concrete task slices:
  1. Implement `PinnedStageAllocator` as the only field-scale host staging allocator for supported paths.
  2. Add bounded pool behavior and deterministic reuse semantics for write and restart staging.
  3. Add tests proving staged host transfers route through this allocator only.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `PinnedStageAllocator` contract
    - pinned-stage allocation stats schema
  - Consumed:
    - normalized memory policy from `P2-01`
- Validation:
  - Field-scale host staging in supported paths goes through the pinned-stage allocator.
  - Allocation and reuse behavior is deterministic and covered by tests.
- Done criteria:
  - One explicit pinned staging path exists and is enforceable.
- Exports to later PRs/phases:
  - `PinnedStageAllocator` consumed by `P2-06`, `P2-09`, and Phase 3 write-boundary staging.

## P2-05 Managed fallback allocator

- Objective:
  - Implement managed-memory fallback as a controlled correctness and debug lane, not a production default.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (production forbids recurring UVM for registered hot objects)
    - `docs/authority/support_matrix.md` -> `## Global Policy` (failFast production policy and debug-only fallback separation)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 8 — Keep a managed-fallback tier for incomplete ports`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 10 — Separate bring-up path from production path`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 6 — Wrap the managed fallback path`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 10. Bring-up path versus production path`
  - Backlog scope:
    - "Implement the managed-memory fallback allocator and its policy wiring as a correctness and debug lane rather than a production default."
  - Backlog done_when:
    - "Managed fallback exists for controlled bring-up but is clearly segregated from production-device acceptance."
- Depends on (backlog IDs):
  - `P2-01`
- Prerequisites:
  - Canonical memory-policy parser from `P2-01`.
- Concrete task slices:
  1. Implement managed fallback allocator and explicit mode wiring under `gpuRuntime.memory`.
  2. Enforce mode labeling and guardrails so production mode cannot silently route through managed fallback.
  3. Emit fallback-usage telemetry for acceptance and debugging visibility.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - managed-fallback allocator contract
    - fallback-usage telemetry schema
  - Consumed:
    - normalized memory policy from `P2-01`
    - support-matrix fail-fast and debug-only policy envelope
- Validation:
  - Managed fallback can be enabled for controlled bring-up flows.
  - Production mode rejects silent managed fallback usage.
  - Telemetry clearly distinguishes fallback from production-device runs.
- Done criteria:
  - Managed fallback is available for correctness and debug and clearly segregated from production acceptance.
- Exports to later PRs/phases:
  - Explicit fallback-mode metadata consumed by Phase 3 execution-mode downgrade logic.

## P2-06 Residency registry and reporting

- Objective:
  - Create explicit residency state, transitions, and deterministic reporting for registered hot objects.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (no recurring UVM for registered hot objects)
    - `docs/authority/continuity_ledger.md` -> `# 2. Cross-phase contract matrix` (Phase 2 exit requires explicit residency reports)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Decision 9 — Introduce a residency registry rather than implicit mirroring`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 3. Residency record`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 5. Coherency-state transitions`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 7 — Implement ResidencyRegistry`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 16 — Add pool-stat and residency reports`
  - Backlog scope:
    - "Create the residency state machine, registry types, and reporting layer for all registered hot objects."
  - Backlog done_when:
    - "Registered objects have explicit host and device visibility state and the branch can emit deterministic residency reports."
- Depends on (backlog IDs):
  - `P2-02`
  - `P2-03`
  - `P2-04`
- Prerequisites:
  - `DevicePersistentPool`, `DeviceScratchPool`, and `PinnedStageAllocator` from `P2-02..P2-04`.
- Concrete task slices:
  1. Implement `ResidencyRegistry` records with explicit host and device visibility and coherency transitions.
  2. Connect allocator ownership and residency metadata for registered hot objects.
  3. Emit deterministic residency reports required for the Phase 2 gate and Phase 3 seam review.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `ResidencyRegistry` contract
    - deterministic residency-report schema
  - Consumed:
    - allocator contracts from `P2-02..P2-04`
- Validation:
  - Registered objects always have explicit residency state.
  - State-transition reports are deterministic across repeated runs.
  - Registry data supports hot-object UVM hygiene checks.
- Done criteria:
  - Registry-backed residency state exists for all registered hot objects.
  - Deterministic residency reports are available for handoff.
- Exports to later PRs/phases:
  - `ResidencyRegistry` and residency reports consumed by `P2-07..P2-11` and Phase 3 seam checks.

## P2-07 Mirror traits and field mirrors

- Objective:
  - Add `MirrorTraits` and `FieldMirror` support for core OpenFOAM field/list abstractions with round-trip correctness.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (device-resident hot path with explicit host visibility only)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-01 Authority ingestion scaffold`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Finding C — Field and List cannot be treated the same way`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 2. Object classes and required tier placement`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 4. Binary layout policy`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 8 — Implement MirrorTraits for contiguous OpenFOAM types`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 9 — Implement FieldMirror and registration helpers`
  - Backlog scope:
    - "Implement MirrorTraits and FieldMirror support for OpenFOAM field/list types, including round-trip tests."
  - Backlog done_when:
    - "Core field abstractions can be mirrored between host and device with tested round-trip correctness."
- Depends on (backlog IDs):
  - `P2-06`
- Prerequisites:
  - Residency state and registry contracts from `P2-06`.
- Concrete task slices:
  1. Implement `MirrorTraits` for required contiguous OpenFOAM data classes.
  2. Implement `FieldMirror` registration helpers that bind mirrors into residency records.
  3. Add round-trip tests for host-to-device and device-to-host mirroring and coherence behavior.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `MirrorTraits` contract
    - `FieldMirror` contract
  - Consumed:
    - `ResidencyRegistry` interfaces from `P2-06`
- Validation:
  - Round-trip mirror tests pass for required field and list classes.
  - Registry state transitions align with mirror operations.
- Done criteria:
  - Core field abstractions can be mirrored and validated with deterministic tests.
- Exports to later PRs/phases:
  - Mirror-layer contracts consumed by `P2-08` startup registration and Phase 5 field integration.

## P2-08 Mesh mirror and startup registration

- Objective:
  - Register mesh topology, addressing, and persistent fields into pools and registry at startup using the mirror substrate.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 2. Cross-phase contract matrix` (Phase 2 startup upload of hot mesh and field set)
    - `docs/tasks/02_phase0_reference_problem_freeze.md` -> `## exports_to_next` (canonical case metadata and stage plan for case identity)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 2A. Upstream-supplied hot-object inventory and default registration template`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 1. Initialization sequence`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 10 — Register immutable mesh and addressing objects at startup`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 11 — Register persistent solver fields`
  - Backlog scope:
    - "Add MeshMirror support plus startup registration of mesh topology, addressing, and persistent fields into the registry and pools."
  - Backlog done_when:
    - "A supported GPU solver can complete startup upload of the hot mesh and field set using the Phase 2 substrate."
- Depends on (backlog IDs):
  - `P2-07`
- Prerequisites:
  - `MirrorTraits` and `FieldMirror` from `P2-07`.
  - Registry and pool contracts from `P2-02..P2-06`.
- Concrete task slices:
  1. Implement `MeshMirror` registration path for immutable mesh, topology, and addressing classes.
  2. Register persistent solver fields into canonical tiers with residency ownership at startup.
  3. Emit startup registration report proving the hot-object inventory is uploaded and tracked.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `MeshMirror` contract
    - startup hot-object registration report
  - Consumed:
    - `MirrorTraits`, `FieldMirror`, and `ResidencyRegistry` contracts
    - phase-freeze case identity artifacts for case-consistent startup runs
- Validation:
  - Supported startup path completes registry and pool registration of hot mesh and field sets.
  - Startup reports deterministically list registered classes and assigned tiers.
- Done criteria:
  - Startup upload and registration of hot objects is functional and reportable.
- Exports to later PRs/phases:
  - Startup registration substrate consumed by `P2-09`, Phase 3 execution, and Phase 5 field ownership.

## P2-09 Explicit visibility APIs and output stager

- Objective:
  - Implement explicit host and device visibility boundaries and output staging paths.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbid hot-path field-scale host reads; allow explicit write-time staging)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`write_stage` graph-external boundary)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 5. Host-visibility algorithm`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 7. Output staging algorithm`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 7A. Restart / checkpoint policy`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 12 — Add ensureDeviceVisible() and ensureHostVisible()`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 14 — Add OutputStager`
  - Backlog scope:
    - "Implement ensureHostVisible, ensureDeviceVisible, scalar staging, and the OutputStager so write and restart boundaries are explicit."
  - Backlog done_when:
    - "All field-scale host visibility in supported paths goes through explicit APIs and write-boundary correctness tests pass."
- Depends on (backlog IDs):
  - `P2-08`
- Prerequisites:
  - Startup registration substrate and mirror contracts from `P2-08`.
- Concrete task slices:
  1. Implement `ensureDeviceVisible()` and `ensureHostVisible()` APIs backed by registry state transitions.
  2. Implement `OutputStager` for explicit write and restart boundary commits and scalar staging paths.
  3. Add enforcement so supported field-scale host visibility occurs only through explicit APIs.
  4. Add boundary correctness tests for write and restart staging behavior.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `ensureHostVisible()` and `ensureDeviceVisible()` visibility contracts
    - `OutputStager` contract
  - Consumed:
    - registry, mirror, and startup registration contracts from `P2-06..P2-08`
- Validation:
  - Field-scale host visibility in supported paths routes through explicit APIs only.
  - Write and restart boundary tests pass with deterministic staging behavior.
  - Visibility transitions are auditable from residency reports.
- Done criteria:
  - Explicit visibility and staging boundaries are enforced and test-covered.
- Exports to later PRs/phases:
  - `ensureHostVisible`, `ensureDeviceVisible`, and `OutputStager` consumed by `P2-10`, `P3-08`, and later solver and output phases.

## P2-10 Compute-epoch enforcement

- Objective:
  - Detect and block illegal CPU touches inside compute epochs.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden hot-stage host reads and silent CPU fallback)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3` (no hidden CPU touches)
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 9. Host touch policy`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 6. Compute epoch algorithm`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 13 — Add compute-epoch guards and CPU-touch logging`
  - Backlog scope:
    - "Add CpuTouchGuard, per-epoch logging, and strict mode so illegal CPU touches are detected inside the compute epoch."
  - Backlog done_when:
    - "Strict mode catches unsupported CPU access in the supported GPU solver path before Phase 3 begins."
- Depends on (backlog IDs):
  - `P2-09`
- Prerequisites:
  - Explicit visibility and staging APIs from `P2-09`.
- Concrete task slices:
  1. Implement `CpuTouchGuard` with strict-mode enforcement for compute-epoch host-touch violations.
  2. Add per-epoch CPU-touch logging with violation classification and source attribution.
  3. Integrate guard checks into supported solver-path touch points.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `CpuTouchGuard` contract
    - compute-epoch touch log schema
  - Consumed:
    - explicit visibility APIs from `P2-09`
- Validation:
  - Strict mode catches illegal CPU access in supported GPU path test scenarios.
  - Legal explicit staging and visibility paths do not produce false positives.
- Done criteria:
  - Compute-epoch guardrails are active and enforce unsupported CPU access detection before Phase 3.
- Exports to later PRs/phases:
  - `CpuTouchGuard` and epoch logs consumed by `P2-11` gate bundle and Phase 3 seam validation.

## P2-11 Scratch catalog and Phase 2 gate bundle

- Objective:
  - Add named scratch tracking and produce the Phase 2 acceptance gate evidence package.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (no recurring UVM for registered hot objects; no post-warmup allocation churn)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules` (no post-warmup dynamic allocation)
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates` (`unexpected_htod_bytes`, `unexpected_dtoh_bytes`, `cpu_um_faults`, `gpu_um_faults`, `post_warmup_alloc_calls`)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 2 -> Phase 3`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 7. Scratch policy`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### 9. Warm-up and steady-state transition`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 15 — Add ScratchCatalog`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `#### Step 19 — Run reduced validation-ladder benchmark gate`
    - `docs/specs/phase2_gpu_memory_spec.md` -> `### Acceptance checklist`
  - Backlog scope:
    - "Implement ScratchCatalog, named scratch reset and watermark tracking, and the Phase 2 benchmark and acceptance bundle."
  - Backlog done_when:
    - "Repeated iterations show stable scratch watermarks and the Phase 2 gate proves no uncontrolled UVM traffic for registered hot objects."
- Depends on (backlog IDs):
  - `P2-03`
  - `P2-10`
- Prerequisites:
  - Scratch pool from `P2-03`.
  - `CpuTouchGuard` and explicit visibility and staging from `P2-09..P2-10`.
- Concrete task slices:
  1. Implement `ScratchCatalog` with named scratch groups, reset semantics, and watermark reporting.
  2. Integrate steady-state checks that verify stable post-warmup scratch behavior.
  3. Produce Phase 2 gate bundle: residency reports, touch logs, pool stats, UVM and transfer counters, and watermark evidence.
  4. Emit explicit handoff packet for Phase 3 seam review.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `ScratchCatalog` contract
    - Phase 2 gate bundle schema
  - Consumed:
    - scratch pool and compute-epoch enforcement outputs from `P2-03` and `P2-10`
    - acceptance hard-gate metric vocabulary
- Validation:
  - Iterative runs show stable scratch watermarks after warmup.
  - Gate bundle demonstrates no uncontrolled UVM traffic for registered hot objects.
  - Gate bundle includes explicit evidence for no post-warmup allocation churn.
- Done criteria:
  - Phase 2 gate evidence is complete and reviewable.
  - Phase 3 has the required seam artifacts to start execution-model work safely.
- Exports to later PRs/phases:
  - `ScratchCatalog` and watermark reports.
  - Phase 2 gate bundle with residency, visibility, touch, transfer, and allocation evidence for Phase 3 entry.

## imports_from_prev

- `FND-01` authority-loader and normalized config governance for `gpuRuntime`.
- `FND-02` pin and environment manifests to keep Phase 2 runtime behavior lane-consistent.
- `FND-04` support scanner and fail-fast policy vocabulary for production versus debug behavior.
- `FND-05` acceptance gate terminology and hard-gate metric naming.
- `FND-06` stage and mode vocabulary, especially `async_no_graph` fallback semantics consumed at the Phase 2 -> Phase 3 seam.
- Phase 0 frozen case identity and metadata contracts for consistent reduced-case gate runs.
- Phase 1 environment, build, and probe artifacts, especially `host_env.json`, `cuda_probe.json`, and `fatbinary_report.json`.

## exports_to_next

- Canonical `gpuRuntime.memory` policy tree and parser contract.
- Allocator substrate: `DevicePersistentPool`, `DeviceScratchPool`, `PinnedStageAllocator`, and managed fallback contract.
- `ResidencyRegistry` and deterministic residency reports for registered hot objects.
- Explicit visibility and staging contracts: `ensureHostVisible()`, `ensureDeviceVisible()`, and `OutputStager`.
- Compute-epoch enforcement artifacts: `CpuTouchGuard` and per-epoch touch logs.
- `ScratchCatalog` plus stable post-warmup watermark reports.
- Phase 2 gate bundle proving:
  - no silent CPU fallback
  - no hot-path field-scale host reads
  - no recurring UVM traffic for registered hot objects
  - no post-warmup allocation churn
- Explicit Phase 2 -> Phase 3 seam packet keyed to `async_no_graph` fallback behavior and graph-safe allocation constraints.

## shared_terms

- `DevicePersistent`: long-lived hot-object device residency tier for production.
- `DeviceScratch`: timestep-local transient device workspace tier with reset and reuse semantics.
- `PinnedStage`: bounded host staging tier for explicit write, restart, and output boundaries.
- `ManagedFallback`: correctness and debug tier that is never the production default.
- `ResidencyRegistry`: authoritative state machine for host and device visibility and coherency for registered objects.
- `compute epoch`: interval in which unsupported CPU touches must be detected and rejected.
- `steady-state`: post-warmup run window where recurring UVM traffic and allocation churn are forbidden for registered hot objects.

## open_discontinuities

- `[tracked] hot-object-registry-scope-growth`: Phase 2 gate language proves UVM hygiene for registered hot objects, but the full hot-object universe expands in later solver phases; registry coverage should remain phase-tagged and explicit so later phases cannot claim global coverage from a narrower Phase 2 registration set (`docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions`, `docs/specs/phase2_gpu_memory_spec.md` -> `#### 2A. Upstream-supplied hot-object inventory and default registration template`, impacted PR IDs: `P2-06`, `P2-11`, preferred reading: keep gate wording tied to registered scope and extend registry inventory explicitly in later phases).

## validation_checks

- All `P2-*` cards preserve canonical dependency edges exactly: `P2-01 <- FND-01`, `P2-02 <- P2-01`, `P2-03 <- P2-01`, `P2-04 <- P2-01`, `P2-05 <- P2-01`, `P2-06 <- P2-02,P2-03,P2-04`, `P2-07 <- P2-06`, `P2-08 <- P2-07`, `P2-09 <- P2-08`, `P2-10 <- P2-09`, `P2-11 <- P2-03,P2-10`.
- Every card cites authority docs plus exact Phase 2 spec subsection anchors plus backlog `scope` and `done_when`.
- Phase 2 consumes Foundation and authority contracts instead of redefining `gpuRuntime` normalization, support policy, or acceptance vocabulary locally.
- Phase 2 ownership remains limited to memory, residency, visibility, and epoch enforcement and does not absorb Phase 3 orchestration, Phase 4 pressure bridge, or Phase 5 VOF semantics.
- The Phase 2 -> Phase 3 seam packet explicitly includes `gpuRuntime.memory`, `ResidencyRegistry`, `ensureHostVisible()`, `ensureDeviceVisible()`, `OutputStager`, `CpuTouchGuard`, and `ScratchCatalog`, plus explicit downgrade semantics to `async_no_graph` when seam constraints are not met.
