# Phase 7 — Custom CUDA kernels

Phase 7 owns `P7-01..P7-08` and delivers the custom CUDA hotspot replacement layer as a replaceable backend under frozen Phase 5 and Phase 6 semantics. This section consumes the accepted Phase 6 nozzle baseline and Phase 8 pass-A profiling substrate as fixed inputs, then exports a capture-safe selected custom path and regression package for Phase 8 formal baseline-lock and CI ownership.

## P7-01 Phase 7 source audit and hotspot ranking

- Objective:
  - Freeze a reviewed Phase 7 source-audit note and profiling-backed hotspot ranking so custom-kernel scope is bounded to support-matrix-admitted families only.
- Exact citations:
  - Authority:
    - `docs/authority/semantic_source_map.md` -> `## Frozen Mapping`
    - `docs/authority/semantic_source_map.md` -> `## Implementation Rule`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (support-matrix ownership and contact-angle conditional scope)
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `exact_audited_scheme_tuple`
    - `docs/authority/support_matrix.json` -> `phase6_nozzle_specific_envelope`
    - `docs/authority/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 7 -> Phase 8`
    - `docs/tasks/08_phase6_pressure_swirl_nozzle_bc_startup.md` -> `## P6-10 Solver-stage integration, graph-safety hardening, and Phase 6 acceptance`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-05 Nsight Systems capture scripts and artifact layout`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.1 — Custom-kernel scope is limited to support-matrix-approved hotspot families ranked by interim profiling`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 1 — Audit the exact local v2412 source, consume the frozen support matrix, and freeze the hotspot ranking`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### Entry criteria`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.0 — Source audit and scope freeze`
  - Backlog scope:
    - `Generate the reviewed Phase 7 source-audit note and produce the hotspot ranking artifact that bounds custom-kernel scope.`
  - Backlog done_when:
    - `Custom-kernel work is explicitly limited to support-matrix-approved hotspot families ranked from real profiling data.`
- Depends on (backlog IDs):
  - `P6-10`
  - `P8-05`
- Prerequisites:
  - `P6-10` Phase 6 acceptance bundle and frozen nozzle BC/startup seam packet.
  - `P8-05` standardized `nsys` artifact layout and reproducible profiling command metadata.
- Concrete task slices:
  1. Generate `phase7_source_audit.md` mapping every Phase 7 touch point to exact local SPUMA/v2412 targets and note any semantic drift from public references.
  2. Ingest `P8-05` profiling outputs and produce `phase7_hotspot_ranking.md` with ranked stage/kernel families tied to canonical stage IDs.
  3. Cross-filter ranking rows against support-matrix-admitted scheme/BC scope and freeze an allowed custom-kernel family list.
  4. Record explicit out-of-scope families (pressure linear algebra replacement, scheme widening, contact-angle expansion) as forbidden for `P7-02..P7-08`.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase7_source_audit.md`
    - `phase7_hotspot_ranking.md`
    - `phase7_hotspot_scope_freeze.json`
  - Consumed:
    - `P6-10` Phase 6 acceptance bundle and seam packet
    - `P8-05` standardized profiling artifacts and metadata
- Validation:
  - Every ranked family used by later Phase 7 cards appears in both real profiling data and support-matrix-admitted scope.
  - No non-admitted scheme/BC family is listed as a custom-kernel target.
- Done criteria:
  - Source audit and hotspot ranking are reviewed and explicitly bound Phase 7 scope before implementation cards proceed.
- Exports to later PRs/phases:
  - Frozen hotspot-scope artifact consumed by `P7-02..P7-08`.
  - Source-audit mapping reused by Phase 8 traceability evidence.

## P7-02 Control plane, POD views, and facade skeleton

- Objective:
  - Implement the narrow runtime control plane, POD device views, and host-side facade so custom kernels are selectable without exposing solver code to launch internals.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (runtime normalization under `gpuRuntime`)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/authority/support_matrix.json` -> `backend_operational_policy`
    - `docs/authority/semantic_source_map.md` -> `## Frozen Mapping`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.3 — Use a narrow host-side facade and POD device views`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.4 — Default to persistent device allocations with stable addresses`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 2 — Add runtime controls and fallback switches`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 3 — Introduce POD device views and facade skeleton`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 2. New public host-side API`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 7. Runtime control object`
  - Backlog scope:
    - `Implement the Phase 7 controls parser, POD device views, and the host-side facade that hides kernel-launch details from solver code.`
  - Backlog done_when:
    - `The custom-kernel subsystem has a narrow, typed interface and runtime control surface before any optimized kernel work lands.`
- Depends on (backlog IDs):
  - `P7-01`
- Prerequisites:
  - `P7-01` hotspot scope freeze and reviewed source mapping.
- Concrete task slices:
  1. Add canonical `gpuRuntime.vof.customKernels` controls with deterministic defaults (`disabled` by default, explicit per-family switches).
  2. Implement typed POD views for mesh, patch, field, adjacency placeholders, and launch context with no solver-object leakage into CUDA TUs.
  3. Implement `Phase7KernelFacade` API with backend selector surfaces (`fallback`, `atomic`, `segmented`) and no policy widening.
  4. Add strict scope guards: unsupported kernels or policy combinations fail fast and never silently enable debug-only behavior.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `Phase7KernelFacade` contract
    - `DeviceVoFControls` parser contract for Phase 7 switches
    - POD view schema for Phase 7 kernels
  - Consumed:
    - `P7-01` hotspot scope freeze
    - centralized runtime and fallback policy contracts
- Validation:
  - With all Phase 7 switches off, solver behavior matches pre-Phase-7 baseline.
  - Facade API is typed and stable without direct launch detail leakage into solver orchestration code.
- Done criteria:
  - Runtime control surface and facade skeleton are complete and frozen before backend-specific kernel work.
- Exports to later PRs/phases:
  - Facade and control contracts consumed by `P7-03..P7-08`.
  - Backend-selector vocabulary exported to Phase 8 tuple mapping.

## P7-03 Adjacency preprocessing

- Objective:
  - Build reusable adjacency preprocessing for segmented/gather execution without mixing it with correctness-kernel behavior.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (device-resident hot path and no post-warmup churn)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/support_matrix.md` -> `## Global Policy` (static mesh/no processor patch assumptions)
    - `docs/tasks/05_phase3_execution_model.md` -> `## P3-07 Graph fingerprint, cache, and rebuild policy`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.6 — Implement two limiter backends: atomic baseline and segmented/gather production backend`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 4 — Build adjacency preprocessing`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 2. Adjacency views for segmented/gather backend`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.2 — Adjacency + scratch infrastructure`
  - Backlog scope:
    - `Build the adjacency preprocessing needed by segmented/gather execution, keeping it separate from kernel correctness work.`
  - Backlog done_when:
    - `Adjacency structures are generated once, validated, and reusable by both the atomic and segmented backends.`
- Depends on (backlog IDs):
  - `P7-02`
- Prerequisites:
  - `P7-02` POD views and facade contracts.
- Concrete task slices:
  1. Implement `DeviceAdjacencyBuilder` for owner/neighbour and boundary-owner row views with deterministic index ownership.
  2. Generate adjacency artifacts once per prepared mesh and bind lifecycle to mesh fingerprint validity.
  3. Add dedicated adjacency validation tests (coverage, monotonic offsets, one-entry ownership guarantees).
  4. Wire adjacency view consumption API so both atomic and segmented backends can share the same substrate.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `DeviceAdjacencyView` contract
    - `adjacency_validation_report`
  - Consumed:
    - `P7-02` POD/facade skeleton
    - Phase 3 mesh/graph lifecycle constraints
- Validation:
  - Row offsets are monotonic and complete.
  - Internal and boundary faces satisfy deterministic one-time ownership rules.
- Done criteria:
  - Reusable adjacency structures are generated once and validated independently from kernel correctness work.
- Exports to later PRs/phases:
  - Shared adjacency substrate consumed by `P7-05` and `P7-07`.
  - Adjacency diagnostics consumed by `P7-08` final bundle.

## P7-04 Persistent scratch arena

- Objective:
  - Implement persistent scratch ownership and lifecycle rules for Phase 7 hotspots so hot-loop allocations are eliminated.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden post-warmup dynamic allocation)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules` (no post-warmup allocation)
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## P2-11 Scratch catalog and Phase 2 gate bundle`
    - `docs/tasks/04_phase2_gpu_memory_model.md` -> `## validation_checks` (Phase 2 -> Phase 3 seam includes `ScratchCatalog`)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.4 — Default to persistent device allocations with stable addresses`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 5 — Implement persistent scratch arena`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 6. Scratch arena`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 7. Memory lifetime rules`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 8. Memory budget rule`
  - Backlog scope:
    - `Implement the persistent scratch arena and its ownership/lifetime rules for custom-kernel hot paths.`
  - Backlog done_when:
    - `Custom kernels can acquire named scratch resources without allocation churn, and watermarks remain stable across repeated iterations.`
- Depends on (backlog IDs):
  - `P7-02`
  - `P2-11`
- Prerequisites:
  - `P7-02` control plane and typed resource names.
  - `P2-11` `ScratchCatalog` and watermark reporting baseline.
- Concrete task slices:
  1. Implement `Phase7ScratchArena` named resources mapped onto Phase 2 scratch catalog ownership.
  2. Add deterministic acquire/release/reset semantics keyed to stage lifecycle boundaries.
  3. Ensure stable pointer addresses across warm steady-state iterations and explicit invalidation only on mesh-change/fallback boundaries.
  4. Emit scratch watermark reports compatible with Phase 2 and Phase 8 artifact readers.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `Phase7ScratchArena` contract
    - `phase7_scratch_watermark_report`
  - Consumed:
    - `P2-11` scratch catalog and acceptance gates
    - `P7-02` facade resource binding
- Validation:
  - Repeated iterations show stable post-warmup watermarks.
  - No Phase 7 stage performs dynamic allocations in hot loop execution.
- Done criteria:
  - Persistent scratch arena is live, stable, and allocation-free in steady-state loops.
- Exports to later PRs/phases:
  - Stable scratch handles consumed by `P7-05`, `P7-06`, and `P7-07`.
  - Allocation-hygiene evidence consumed by `P7-08` and Phase 8 evaluators.

## P7-05 Atomic alpha/MULES correctness backend

- Objective:
  - Deliver the atomic baseline backend for alpha/MULES hotspots as the first correctness line before segmented optimization.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (canonical algebraic VOF family and GPU-only contract)
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.json` -> `exact_audited_scheme_tuple`
    - `docs/authority/reference_case_contract.md` -> `## Phase-Gate Mapping` (`R2`, `R1-core`, `R1` usage)
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/authority/acceptance_manifest.json` -> `threshold_classes.field_qoi`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`alpha_pre`, `alpha_subcycle_body`)
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-06 Full alpha + MULES + subcycling`
    - `docs/tasks/07_phase5_generic_vof_core.md` -> `## P5-07 Mixture update and interface/surface-tension subset`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.6 — Implement two limiter backends: atomic baseline and segmented/gather production backend`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 7 — Implement atomic alpha-flux assembly kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 8 — Implement atomic phiBD, phiCorr, SuCorr construction`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 9 — Implement atomic limiter preprocessing and face-lambda update`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 10 — Implement explicit alpha update and fused two-phase update`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 11 — Integrate atomic backend with solver and stop for the first benchmark`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### Validation strategy`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.3 — Atomic alpha/MULES correctness path`
  - Backlog scope:
    - `Implement the atomic baseline for the Phase 7 alpha/MULES hotspot family as the first correctness line.`
  - Backlog done_when:
    - `The atomic backend is numerically correct on the accepted reduced/nozzle cases and benchmark artifacts are captured at the milestone stop.`
- Depends on (backlog IDs):
  - `P7-03`
  - `P7-04`
- Prerequisites:
  - `P7-03` adjacency preprocessing availability.
  - `P7-04` persistent scratch resources.
- Concrete task slices:
  1. Implement atomic alpha-flux and MULES preprocessing/update kernels under frozen scheme tuple assumptions.
  2. Integrate atomic backend into facade selector with explicit `limiterBackend=atomic` runtime route.
  3. Add CPU/GPU snapshot parity tests and boundedness assertions for accepted reduced/nozzle slices.
  4. Capture first benchmark stop artifact (`phase7_atomic_baseline.md`) with launch counts and hotspot timings.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - atomic alpha/MULES backend contract
    - `phase7_atomic_baseline.md`
    - atomic correctness snapshot artifact set
  - Consumed:
    - Phase 5 generic alpha/MULES semantics
    - adjacency and scratch substrates from `P7-03` and `P7-04`
- Validation:
  - Atomic backend satisfies boundedness and threshold-class checks on admitted reduced/nozzle runs.
  - Snapshot parity is deterministic and no hidden host fallback is used.
- Done criteria:
  - Atomic backend is correctness-complete and benchmarked as the baseline for segmented replacement decisions.
- Exports to later PRs/phases:
  - Atomic comparison baseline consumed by `P7-07`.
  - Correctness and benchmark artifacts consumed by `P7-08` and Phase 8.

## P7-06 Interface and patch kernels

- Objective:
  - Implement interface and patch kernel families for the admitted `R1`/`R0` subset without widening contact-angle, scheme, or BC scope.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/authority/support_matrix.json` -> `phase6_nozzle_specific_envelope`
    - `docs/authority/support_matrix.json` -> `accepted_nozzle_case_family_boundary_tuple`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (contact-angle out-of-scope unless authority changes)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs` (`nozzle_bc_update`, `mixture_update`)
    - `docs/tasks/08_phase6_pressure_swirl_nozzle_bc_startup.md` -> `## P6-10 Solver-stage integration, graph-safety hardening, and Phase 6 acceptance`
    - `docs/tasks/08_phase6_pressure_swirl_nozzle_bc_startup.md` -> `## open_discontinuities`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.8 — Support only the audited scheme subset in custom kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.9 — Boundary-condition dispatch happens on host; boundary math happens on device`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 12 — Implement interface gradient/interpolation/normal kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 13 — If required by the frozen acceptance cases, implement contact-angle patch kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 14 — Implement curvature, sigmaK, and surface-tension force kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 15 — Implement patch dispatcher and nozzle-specific patch kernels`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `##### F5. Interface normal and curvature`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `##### F6. Contact-angle correction`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `##### F7. Patch kernels`
  - Backlog scope:
    - `Implement the accepted interface and patch kernel family for the exact patch/scheme subset used by R1 / R0.`
  - Backlog done_when:
    - `The supported interface/patch kernels pass unit tests and integrate without reopening contact-angle scope.`
- Depends on (backlog IDs):
  - `P7-04`
  - `P6-10`
- Prerequisites:
  - `P7-04` persistent scratch infrastructure.
  - `P6-10` frozen nozzle boundary manifests and accepted pressure-boundary-state seam.
- Concrete task slices:
  1. Implement interface pipeline kernels (gradient/interpolation/normal/curvature/surface-force) for admitted scheme tuple only.
  2. Implement patch dispatcher using Phase 6 host-resolved patch kinds and device-boundary manifests; no host patch polymorphism in hot stages.
  3. Keep contact-angle behavior strictly conditional: placeholder or fail-fast unless centralized authorities explicitly admit it.
  4. Integrate kernels behind facade switches and emit interface/patch snapshot parity artifacts.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `InterfaceKernelExecutor` contract
    - `PatchKernelExecutor` contract under admitted nozzle tuple
    - `phase7_interface_patch_snapshot_report`
  - Consumed:
    - `P6-10` boundary-manifest and nozzle acceptance seam contracts
    - support-matrix BC and scheme constraints
- Validation:
  - Interface/patch unit tests and snapshot parity checks pass on admitted `R1`/`R0` slices.
  - No host patch execution events appear in accepted hot-stage windows.
- Done criteria:
  - Interface and patch kernels are integrated for admitted tuples with no scope widening.
- Exports to later PRs/phases:
  - Interface/patch backend artifacts consumed by `P7-08`.
  - Nozzle-kernel traceability bundle consumed by Phase 8 acceptance parsing.

## P7-07 Segmented/gather production backend

- Objective:
  - Implement segmented/gather production backend for repeated face-to-cell accumulation and compare against atomic baseline on the profiled hotspot set.
- Exact citations:
  - Authority:
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (device-resident hot path and no silent fallback)
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/authority/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/authority/acceptance_manifest.json` -> `threshold_classes.parity_replay.KP_CUSTOM_VS_BASELINE`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### DD7.6 — Implement two limiter backends: atomic baseline and segmented/gather production backend`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 17 — Implement segmented/gather backend for repeated face-to-cell accumulation`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 18 — Switch MULES production backend to segmented and keep atomic as fallback`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### 2. Adjacency views for segmented/gather backend`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### Performance expectations`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.5 — Segmented production backend`
  - Backlog scope:
    - `Implement the segmented/gather production backend and compare it against the atomic path on the profiled hotspot set.`
  - Backlog done_when:
    - `A production backend exists that improves or otherwise justifies replacing the atomic baseline for accepted tuples.`
- Depends on (backlog IDs):
  - `P7-05`
  - `P7-03`
- Prerequisites:
  - `P7-05` atomic correctness backend and benchmark baseline.
  - `P7-03` validated adjacency substrate.
- Concrete task slices:
  1. Implement segmented/gather kernels for repeated accumulation paths using shared adjacency structures.
  2. Add runtime selector route `limiterBackend=segmented` while retaining atomic backend as fully supported fallback.
  3. Run atomic-vs-segmented comparison matrix on admitted reduced/nozzle tuples with deterministic artifact outputs.
  4. Record backend-selection decision criteria (performance gain, correctness parity, residency hygiene).
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - segmented/gather backend contract
    - `phase7_segmented_comparison.md`
    - backend-selection decision report
  - Consumed:
    - `P7-05` atomic baseline artifacts
    - `P7-03` adjacency view contract
- Validation:
  - Segmented backend matches atomic correctness within accepted parity tolerances.
  - Comparison evidence shows measurable improvement or explicit retention justification.
- Done criteria:
  - Production segmented backend is implemented with clear keep/replace decision versus atomic baseline.
- Exports to later PRs/phases:
  - Selected backend decision consumed by `P7-08` and Phase 8 tuple evidence mapping.
  - Atomic-vs-segmented comparison artifacts exported to Phase 8 pass-B baseline-lock review.

## P7-08 Graph-safety cleanup, capture validation, and final regression package

- Objective:
  - Remove remaining capture hazards, validate approved-stage graph compatibility, and publish the final Phase 7 regression/comparison package for Phase 8 ownership.
- Exact citations:
  - Authority:
    - `docs/authority/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/authority/graph_capture_support_matrix.json` -> `run_modes`
    - `docs/authority/graph_capture_support_matrix.json` -> `stages`
    - `docs/authority/graph_capture_support_matrix.json` -> `global_capture_rules`
    - `docs/authority/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/authority/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/authority/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/authority/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/authority/acceptance_manifest.json` -> `hard_gates`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (forbidden silent fallback and production GPU-only contract)
    - `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`
    - `docs/tasks/boundary_matrix.md` -> `### Phase 7 -> Phase 8`
    - `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-09 Baseline locks and CI/nightly integration (Deferred to Pass B)`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 19 — Add graph-safety cleanup and capture validation`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 20 — Final profiling and acceptance run`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### Acceptance checklist`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.6 — Graph-safety cleanup and capture validation`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### M7.7 — Final nozzle regression package`
    - `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `### Instrumentation and profiling hooks`
  - Backlog scope:
    - `Remove remaining capture hazards, validate graph compatibility of the chosen custom-kernel path, and generate the final Phase 7 regression/comparison package.`
  - Backlog done_when:
    - `The selected custom-kernel path is capture-safe for the approved stages, UVM-clean in steady state, and backed by final regression artifacts.`
- Depends on (backlog IDs):
  - `P7-06`
  - `P7-07`
- Prerequisites:
  - `P7-06` interface/patch kernels on admitted scope.
  - `P7-07` selected production backend and comparison artifacts.
- Concrete task slices:
  1. Remove capture hazards from selected path: no post-warmup allocation, no hot-loop host polling, no unsupported capture operations.
  2. Run graph-smoke captures on approved Phase-7-owned stage blocks and verify deterministic downgrade to `async_no_graph` with logged reasons on failure.
  3. Execute final regression/comparison matrix for `baseline_gpu` vs selected `custom_gpu` on admitted reduced/nozzle rows as benchmark evidence only.
  4. Publish final Phase 7 bundle in standardized artifact layout and bind every row to manifest tuple/stage IDs for Phase 8 ingestion.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase7_graph_capture_validation.md`
    - `phase7_final_regression_bundle`
    - `phase7_selected_backend_decision.json`
  - Consumed:
    - `P7-06` and `P7-07` backend artifacts
    - centralized acceptance and graph-stage contracts
- Validation:
  - Approved-stage capture smoke tests pass or downgrade deterministically to `async_no_graph` without hidden behavior.
  - Final bundle reports hard-gate metrics (`unexpected_htod_bytes`, `unexpected_dtoh_bytes`, `cpu_um_faults`, `gpu_um_faults`, `post_warmup_alloc_calls`, `host_patch_execution_events`, `cudaDeviceSynchronize_calls`) with stable disposition mapping.
  - Evidence package is explicitly tagged as Phase 7 input to Phase 8 formal baseline locking, not a replacement.
- Done criteria:
  - Selected custom path is capture-safe for approved stages, UVM-clean in steady state, and exported with complete final regression artifacts.
- Exports to later PRs/phases:
  - Final Phase 7 regression package consumed by Phase 8 pass-B `P8-09`.
  - Selected custom-backend decision, tuple binding map, and stage evidence consumed by Phase 8 CI/nightly baseline locks.

## imports_from_prev

- `P6-10` Phase 6 acceptance bundle and frozen nozzle BC/startup seam packet.
- `P2-11` `ScratchCatalog` and stable scratch-watermark contract.
- `P8-05` standardized `nsys` capture artifact layout and command metadata.
- Phase 5 generic-core semantics and contracts:
  - alpha/MULES ordering and boundedness expectations from `P5-06`
  - interface/surface-tension sequencing from `P5-07`
  - baseline-freeze/restart parity discipline from `P5-11`
- Phase 3 stage/run-mode contracts and fallback semantics:
  - canonical stage IDs and capture-policy vocabulary
  - `async_no_graph` as explicit fallback
  - `graph_fixed` as accepted graph mode
- Phase 6 boundary and pressure seam contracts:
  - `DeviceBoundaryManifestView`
  - flat boundary spans
  - `PressureBoundaryState` / `snGradp` as single boundary-state contract

## exports_to_next

- `phase7_source_audit.md` and `phase7_hotspot_ranking.md` scope-freeze artifacts.
- Phase 7 control-plane and facade contracts (`DeviceVoFControls`, `Phase7KernelFacade`, POD view schema).
- `DeviceAdjacencyView` plus validation report for shared backend use.
- `Phase7ScratchArena` and stable watermark evidence.
- Atomic and segmented backend comparison artifacts and selected-backend decision record.
- Interface/patch kernel snapshot parity and nozzle-stage evidence package.
- Final Phase 7 regression/capture bundle mapped to accepted tuple IDs and canonical stage IDs.

## shared_terms

- `baseline_gpu`: Phase 5/6 device-resident baseline path with Phase 7 custom hotspots disabled.
- `custom_gpu`: Phase 7 hotspot kernels enabled through the Phase 7 facade selector.
- `hotspot_scope_freeze`: reviewed ranked family list that bounds all Phase 7 implementation work.
- `atomic backend`: first correctness backend for alpha/MULES hotspots.
- `segmented backend`: production candidate backend using adjacency-aware gather/reduction paths.
- `Phase7ScratchArena`: persistent named scratch resources allocated outside hot-stage loops.
- `selected_backend`: explicit production-choice record exported from `P7-07`/`P7-08`.

## open_discontinuities

- `[tracked] phase6_phase7_hotspot_scope_lock`: Phase 7 may replace hotspot backends but may not widen admitted BC kinds, scheme tuple, contact-angle scope, or graph semantics. Impacted PR IDs: `P7-01`, `P7-02`, `P7-03`, `P7-04`, `P7-06`. Citations: `docs/tasks/boundary_matrix.md` -> `### Phase 6 -> Phase 7`, `docs/tasks/08_phase6_pressure_swirl_nozzle_bc_startup.md` -> `## open_discontinuities`, `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`, `docs/authority/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`. Preferred reading: treat Phase 5/6 behavior as frozen contracts and Phase 7 as replaceable backend implementation only.
- `[tracked] contact_angle_conditionality_guard`: Contact-angle remains out of milestone-1 scope unless centrally promoted; Phase 7 step logic mentions conditional implementation and must not self-admit scope changes. Impacted PR IDs: `P7-06`, `P7-08`. Citations: `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions`, `docs/authority/support_matrix.md` -> `## Global Policy`, `docs/specs/phase7_custom_cuda_kernel_spec.md` -> `#### Step 13 — If required by the frozen acceptance cases, implement contact-angle patch kernels`. Preferred reading: keep contact-angle path disabled/placeholder unless authority docs explicitly change.
- `[tracked] phase7_phase8_formal_acceptance_boundary`: Phase 7 produces benchmark/regression evidence, but formal baseline locking and CI/nightly policy remain Phase 8 pass-B ownership. Impacted PR IDs: `P7-08`, `P8-09`. Citations: `docs/tasks/boundary_matrix.md` -> `### Phase 7 -> Phase 8`, `docs/tasks/10_phase8_profiling_performance_acceptance.md` -> `## P8-09 Baseline locks and CI/nightly integration (Deferred to Pass B)`, `docs/authority/acceptance_manifest.md` -> `## Disposition Rules`. Preferred reading: Phase 7 exports evidence only; Phase 8 issues formal lock verdicts.

## validation_checks

- All Phase 7 cards preserve canonical dependency edges exactly:
  - `P7-01 <- P6-10,P8-05`
  - `P7-02 <- P7-01`
  - `P7-03 <- P7-02`
  - `P7-04 <- P7-02,P2-11`
  - `P7-05 <- P7-03,P7-04`
  - `P7-06 <- P7-04,P6-10`
  - `P7-07 <- P7-05,P7-03`
  - `P7-08 <- P7-06,P7-07`
- Every card anchors to authority docs/json + exact Phase 7 spec subsections + backlog `scope` + backlog `done_when`.
- Phase 7 scope stays bounded to hotspot replacement only:
  - no pressure linear algebra backend redesign
  - no scheme or BC-kind widening
  - no contact-angle policy widening
  - no graph semantics redefinition outside canonical stage registry and run modes
- Flat boundary spans and pressure-boundary-state contracts are consumed unchanged from Phase 6 (`DeviceBoundaryManifestView`, `PressureBoundaryState`, `snGradp`).
- Phase 7 exports evidence for Phase 8 ingestion and does not claim formal baseline lock authority.
