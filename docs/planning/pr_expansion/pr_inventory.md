# PR Inventory

This inventory is copied from [gpu_cfd_pr_backlog.json](/Users/rexputnam/Documents/projects/gpu_cfd/docs/PRs/gpu_cfd_pr_backlog.json). Each PR ID must appear once and only once across the section docs.

## Foundation / authority consumption

- File owner: [01_foundation_authority_consumption.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/01_foundation_authority_consumption.md)
- PR range: `FND-01..FND-07`
- PRs:
  - `FND-01` Authority ingestion scaffold
  - `FND-02` Pin-manifest consumption and environment manifest emission
  - `FND-03` Reference-case and validation-ladder utilities
  - `FND-04` Support-matrix scanner and fail-fast policy
  - `FND-05` Acceptance-manifest evaluator scaffold
  - `FND-06` Graph stage registry and graph-support-matrix loader
  - `FND-07` Semantic source-audit helper

## Phase 0 — Reference problem freeze

- File owner: [02_phase0_reference_problem_freeze.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/02_phase0_reference_problem_freeze.md)
- PR range: `P0-01..P0-08`
- PRs:
  - `P0-01` Environment probe hardening
  - `P0-02` Environment-neutral runner wrappers
  - `P0-03` Case metadata and stage-plan emission
  - `P0-04` I/O normalization overlay
  - `P0-05` Fingerprints, field signatures, and extractor JSON
  - `P0-06` Baseline A reference freeze
  - `P0-07` Baseline B bring-up and R2 smoke
  - `P0-08` Baseline B nozzle freeze and sign-off package

## Phase 1 — Blackwell bring-up

- File owner: [03_phase1_blackwell_bringup.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/03_phase1_blackwell_bringup.md)
- PR range: `P1-01..P1-07`
- PRs:
  - `P1-01` Host and CUDA discovery probes
  - `P1-02` Blackwell build-system enablement
  - `P1-03` Fatbinary inspection and reporting
  - `P1-04` Repo-local smoke-case pack and solver audit
  - `P1-05` Compute Sanitizer memcheck lane
  - `P1-06` Nsight Systems baseline and UVM diagnostic traces
  - `P1-07` PTX-JIT proof and Phase 1 acceptance bundle

## Phase 2 — GPU memory model

- File owner: [04_phase2_gpu_memory_model.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/04_phase2_gpu_memory_model.md)
- PR range: `P2-01..P2-11`
- PRs:
  - `P2-01` Canonical `gpuRuntime.memory` parser
  - `P2-02` Device persistent pool
  - `P2-03` Device scratch pool
  - `P2-04` Pinned stage allocator
  - `P2-05` Managed fallback allocator
  - `P2-06` Residency registry and reporting
  - `P2-07` Mirror traits and field mirrors
  - `P2-08` Mesh mirror and startup registration
  - `P2-09` Explicit visibility APIs and output stager
  - `P2-10` Compute-epoch enforcement
  - `P2-11` Scratch catalog and Phase 2 gate bundle

## Phase 3 — Execution model

- File owner: [05_phase3_execution_model.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/05_phase3_execution_model.md)
- PR range: `P3-01..P3-08`
- PRs:
  - `P3-01` Synchronization and stream inventory
  - `P3-02` Execution-mode parser and selection policy
  - `P3-03` GpuExecutionContext and execution registry
  - `P3-04` Async launch wrapper layer
  - `P3-05` Canonical stage scaffolding and parent NVTX ranges
  - `P3-06` First graph-enabled stage
  - `P3-07` Graph fingerprint, cache, and rebuild policy
  - `P3-08` Write-boundary staging, production residency assertions, and Phase 3 acceptance

## Phase 4 — Pressure linear algebra

- File owner: [06_phase4_pressure_linear_algebra.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/06_phase4_pressure_linear_algebra.md)
- PR range: `P4-01..P4-09`
- PRs:
  - `P4-01` Dependency freeze and standalone AmgX smoke
  - `P4-02` Pressure snapshot dump utility
  - `P4-03` Topology-only LDU-to-CSR builder
  - `P4-04` CSR value packer and `A*x` validator
  - `P4-05` AmgX context wrapper and replay utility
  - `P4-06` PressureMatrixCache and persistent staging buffers
  - `P4-07` Runtime-selected live solver integration
  - `P4-08` Telemetry, profiling hooks, and reduced-case validation
  - `P4-09` DeviceDirect pressure bridge

## Phase 5 — Generic VOF core

- File owner: [07_phase5_generic_vof_core.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/07_phase5_generic_vof_core.md)
- PR range: `P5-01..P5-11`
- PRs:
  - `P5-01` Phase 5 symbol reconciliation note
  - `P5-02` VOF runtime gate and support scan
  - `P5-03` Persistent topology and boundary-map substrate
  - `P5-04` Field mirrors and old-time state for VOF
  - `P5-05` Alpha skeleton path
  - `P5-06` Full alpha + MULES + subcycling
  - `P5-07` Mixture update and interface/surface-tension subset
  - `P5-08` Momentum predictor
  - `P5-09` Native pressure backend integration
  - `P5-10` AmgX pressure backend integration
  - `P5-11` Write-time commit, validation artifacts, and Phase 5 baseline freeze

## Phase 6 — Pressure-swirl nozzle boundary conditions and startup

- File owner: [08_phase6_pressure_swirl_nozzle_bc_startup.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/08_phase6_pressure_swirl_nozzle_bc_startup.md)
- PR range: `P6-01..P6-10`
- PRs:
  - `P6-01` Boundary support report and patch classifier
  - `P6-02` Flat boundary spans and boundary metadata upload
  - `P6-03` Constrained profile grammar and compiler
  - `P6-04` Custom `gpuPressureSwirlInletVelocity` type and CPU snapshot path
  - `P6-05` Alpha boundary kernels
  - `P6-06` Ambient/open velocity boundary kernels
  - `P6-07` Swirl inlet device kernel and invariance tests
  - `P6-08` Pressure boundary integration
  - `P6-09` Startup seeding subsystem
  - `P6-10` Solver-stage integration, graph-safety hardening, and Phase 6 acceptance

## Phase 7 — Custom CUDA kernels

- File owner: [09_phase7_custom_cuda_kernels.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/09_phase7_custom_cuda_kernels.md)
- PR range: `P7-01..P7-08`
- PRs:
  - `P7-01` Phase 7 source audit and hotspot ranking
  - `P7-02` Control plane, POD views, and facade skeleton
  - `P7-03` Adjacency preprocessing
  - `P7-04` Persistent scratch arena
  - `P7-05` Atomic alpha/MULES correctness backend
  - `P7-06` Interface and patch kernels
  - `P7-07` Segmented/gather production backend
  - `P7-08` Graph-safety cleanup, capture validation, and final regression package

## Phase 8 — Profiling and performance acceptance

- File owner: [10_phase8_profiling_performance_acceptance.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/10_phase8_profiling_performance_acceptance.md)
- PR range: `P8-01..P8-09`
- Pass A (`P8-01..P8-08`): instrumentation and profiling substrate needed by downstream phases.
- Pass B (`P8-09`): baseline locks and CI/nightly integration after Phase 7 stability.
- PRs:
  - `P8-01` NVTX wrapper library and build flags
  - `P8-02` Canonical profiling and acceptance config parser
  - `P8-03` Solver-stage instrumentation coverage
  - `P8-04` Graph lifecycle instrumentation
  - `P8-05` Nsight Systems capture scripts and artifact layout
  - `P8-06` Stats export and profile-acceptance parser
  - `P8-07` Diagnostic profiling modes
  - `P8-08` Top-kernel NCU and sanitizer automation
  - `P8-09` Baseline locks and CI/nightly integration

## Cross-Section Dependency Edges (Canonical)

Use this list to verify wave gating and section-level `imports_from_prev`.

- `P0-01 <- FND-02`
- `P0-06 <- FND-03`
- `P1-01 <- FND-02`
- `P2-01 <- FND-01`
- `P3-01 <- P2-10`
- `P3-02 <- FND-06`
- `P3-05 <- FND-06`
- `P3-08 <- P2-09`
- `P4-01 <- P1-07`
- `P4-01 <- FND-02`
- `P4-02 <- P0-08`
- `P4-06 <- P2-04`
- `P4-08 <- P3-05`
- `P4-09 <- P2-02`
- `P5-01 <- FND-07`
- `P5-02 <- FND-04`
- `P5-03 <- P2-08`
- `P5-04 <- P2-07`
- `P5-05 <- P3-03`
- `P5-09 <- P4-07`
- `P5-10 <- P4-09`
- `P5-11 <- P8-03`
- `P6-01 <- FND-04`
- `P6-01 <- P5-11`
- `P6-02 <- P2-08`
- `P6-04 <- P5-01`
- `P6-08 <- P5-09`
- `P6-09 <- P5-11`
- `P7-01 <- P6-10`
- `P7-01 <- P8-05`
- `P7-04 <- P2-11`
- `P7-06 <- P6-10`
- `P8-01 <- P1-02`
- `P8-02 <- FND-05`
- `P8-02 <- FND-06`
- `P8-03 <- P3-05`
- `P8-04 <- P3-07`
- `P8-06 <- FND-05`
- `P8-09 <- P7-08`
