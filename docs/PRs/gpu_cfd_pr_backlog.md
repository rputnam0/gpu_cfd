# GPU CFD Pressure-Swirl Solver — Implementation PR Backlog

This document is the **decomposition layer** between the finalized spec package and coding-agent execution. It is intentionally **not** a set of written PR bodies. It is the recommended PR list, with each PR sliced to keep the body of work atomic, reviewable, and benchmarkable.

## Planning rules used to split the work

- Keep runtime-schema parsing separate from solver semantics.
- Keep state-model changes separate from kernel or numerics changes.
- Use milestone stop points as PR boundaries whenever the phase defines them.
- Do not combine the first correctness implementation with the first optimization implementation.
- Land validation/reporting harnesses before the baseline-freeze PR of a phase.
- Do not let one PR straddle two numerically risky couplings at once, especially alpha+pressure, pressure hook+BC semantics, graph capture+backend replacement, or segmented kernels+wetting/contact-angle scope.

## Backlog summary

Total recommended PRs: **88**

- **Foundation / authority consumption** — 7 PRs
- **Phase 0 — Reference problem freeze** — 8 PRs
- **Phase 1 — Blackwell bring-up** — 7 PRs
- **Phase 2 — GPU memory model** — 11 PRs
- **Phase 3 — Execution model** — 8 PRs
- **Phase 4 — Pressure linear algebra** — 9 PRs
- **Phase 5 — Generic VOF core** — 11 PRs
- **Phase 6 — Pressure-swirl nozzle boundary conditions and startup** — 10 PRs
- **Phase 7 — Custom CUDA kernels** — 8 PRs
- **Phase 8 — Profiling and performance acceptance** — 9 PRs

## Recommended top-level execution order

- Foundation PRs first.
- Phase 0 and Phase 1 next; they can overlap once the authority and pin ingestion exist, but Phase 0 sign-off must complete before any correctness-sensitive GPU comparisons.
- Phase 2 must be green before Phase 3.
- Phase 4 can start after Phase 1/2 are stable enough to support the pressure bridge and snapshot tooling.
- Phase 5 starts only after Phase 2, Phase 3, and the Phase 4 native pressure path are available.
- Phase 6 starts only after the Phase 5 generic VOF core is frozen.
- Phase 7 starts only after Phase 6 nozzle semantics are stable and hotspot data exists.
- Phase 8 instrumentation PRs should start early, but formal baseline locks wait until the Phase 7 path is stable.

## Foundation / authority consumption

Authoritative source(s): `continuity_ledger.md`, `master_pin_manifest.md`, `reference_case_contract.md`, `validation_ladder.md`, `support_matrix.md`, `acceptance_manifest.md`, `graph_capture_support_matrix.md`, `semantic_source_map.md`

### FND-01 — Authority ingestion scaffold
- **Depends on:** None
- **Scope:** Add a shared loader/validator for the package authority docs and JSON companions so later phases consume typed program-wide decisions instead of hard-coded constants.
- **Done when:** One command/library path loads the authority set, validates schema/versioning, and exposes a stable API used by downstream tools.

### FND-02 — Pin-manifest consumption and environment manifest emission
- **Depends on:** FND-01
- **Scope:** Implement ingestion of the master pin manifest and emit machine-readable environment manifests for builds, runs, and profiling artifacts.
- **Done when:** Build/run tooling can resolve the frozen toolchain lane and emit `host_env` / manifest artifacts without duplicating pin logic.

### FND-03 — Reference-case and validation-ladder utilities
- **Depends on:** FND-01
- **Scope:** Create helpers that resolve canonical case IDs, ladder roles (`R2`, `R1-core`, `R1`, `R0`), and phase-gate selection.
- **Done when:** Scripts and tests can select cases by ladder role and reject unknown or out-of-scope case IDs.

### FND-04 — Support-matrix scanner and fail-fast policy
- **Depends on:** FND-01
- **Scope:** Implement centralized support scanning for BCs, schemes, turbulence scope, function objects, backend eligibility, and fallback policy.
- **Done when:** Unsupported tuples are rejected before the run starts, and debug-only fallback stays non-default and explicit.

### FND-05 — Acceptance-manifest evaluator scaffold
- **Depends on:** FND-01
- **Scope:** Build a reusable acceptance evaluator that understands tuple IDs, tolerance classes, parity classes, hard/soft gates, and waiver hooks.
- **Done when:** Later phases can submit artifacts to one common evaluator and get deterministic pass/fail dispositions tied to manifest entries.

### FND-06 — Graph stage registry and graph-support-matrix loader
- **Depends on:** FND-01
- **Scope:** Create the canonical stage-ID registry and graph-policy loader used by execution, instrumentation, and acceptance code.
- **Done when:** All graph-aware code resolves stage IDs from one shared source, and tuple-to-stage validation runs without local restatement.

### FND-07 — Semantic source-audit helper
- **Depends on:** FND-01
- **Scope:** Add tooling and templates that map semantic source references to the exact local SPUMA/v2412 patch targets before any solver modifications.
- **Done when:** Each implementation phase can generate a reviewed source-audit note and avoid patching the wrong upstream analog.


## Phase 0 — Reference problem freeze

Authoritative source(s): `phase0_reference_problem_spec.md`

### P0-01 — Environment probe hardening
- **Depends on:** FND-02
- **Scope:** Extend the existing OpenFOAM environment probe so both Baseline A and Baseline B resolve into machine-readable manifests, with no hard-coded `/opt/openfoam12` assumptions left in tooling.
- **Done when:** The probe captures environment details for both baselines and generated tooling no longer embeds fixed OpenFOAM paths.

### P0-02 — Environment-neutral runner wrappers
- **Depends on:** P0-01
- **Scope:** Add runner wrappers that source the probed environment explicitly and execute the staged workflow in a baseline-neutral way.
- **Done when:** Generated case runners execute against both baselines without manual script edits or hidden environment coupling.

### P0-03 — Case metadata and stage-plan emission
- **Depends on:** P0-02
- **Scope:** Extend the builder/orchestrator to emit `stage_plan.json`, richer `case_meta.json`, resolved numerics, startup-seed metadata, and provenance fields needed by later comparisons.
- **Done when:** Every generated case carries a complete machine-readable execution plan and resolved metadata bundle.

### P0-04 — I/O normalization overlay
- **Depends on:** P0-03
- **Scope:** Implement the non-physics output normalization layer for ASCII/uncompressed/fixed-precision output so comparison artifacts are stable across reruns.
- **Done when:** Same-baseline reruns produce stable normalized outputs and do not fail due to incidental I/O formatting drift.

### P0-05 — Fingerprints, field signatures, and extractor JSON
- **Depends on:** P0-03
- **Scope:** Add mesh/patch fingerprints, selected field-signature extraction, and JSON output from the feature extractor for automated comparison workflows.
- **Done when:** R2/R1-core/R1/R0 cases can emit comparison-ready fingerprints and field signatures non-interactively.

### P0-06 — Baseline A reference freeze
- **Depends on:** P0-04, P0-05, FND-03
- **Scope:** Generate and freeze the Baseline A reference bundles for `R2`, `R1-core`, `R1`, and `R0`, including artifacts, metadata, and initial tolerance checks.
- **Done when:** A complete Baseline A bundle exists for all ladder cases and becomes the initial control truth.

### P0-07 — Baseline B bring-up and R2 smoke
- **Depends on:** P0-06
- **Scope:** Bring up Baseline B in build-only mode first, validate that no legacy OpenFOAM-12 coupling remains, then run the `R2` smoke case with documented backend policy.
- **Done when:** Baseline B builds the frozen cases and `R2` executes cleanly enough to justify proceeding to nozzle cases.

### P0-08 — Baseline B nozzle freeze and sign-off package
- **Depends on:** P0-07
- **Scope:** Run/freeze `R1` and `R0` on Baseline B, compare against Baseline A, and generate the Phase 0 sign-off packet with compare JSON/Markdown and archived artifacts.
- **Done when:** Phase 0 ends with reviewed Baseline A/B compare reports and a frozen reference contract suitable for downstream implementation.


## Phase 1 — Blackwell bring-up

Authoritative source(s): `phase1_blackwell_bringup_spec.md`

### P1-01 — Host and CUDA discovery probes
- **Depends on:** FND-02
- **Scope:** Add the Phase 1 host/CUDA discovery binaries and emit `host_env.json` and `cuda_probe.json` for the target workstation lane.
- **Done when:** The workstation characteristics, driver state, compute capability, and CUDA availability are captured in reproducible machine-readable form.

### P1-02 — Blackwell build-system enablement
- **Depends on:** P1-01
- **Scope:** Patch or wrap the build so the primary lane emits `sm_120` plus PTX, adopts NVTX3, and respects the frozen toolchain lane.
- **Done when:** A clean primary-lane build is produced with the required Blackwell targets and without relying on ad hoc local edits.

### P1-03 — Fatbinary inspection and reporting
- **Depends on:** P1-02
- **Scope:** Add binary inspection tooling that verifies the expected native cubin/PTX composition and writes `fatbinary_report.json`.
- **Done when:** The build artifact proves `sm_120` coverage and PTX retention before smoke cases begin.

### P1-04 — Repo-local smoke-case pack and solver audit
- **Depends on:** P1-02
- **Scope:** Prepare the repo-local smoke cases and the `fvSolution`/runtime sanity checks that gate Phase 1 bring-up.
- **Done when:** All required smoke cases run unprofiled, and unsupported solver settings are caught before deeper tooling passes.

### P1-05 — Compute Sanitizer memcheck lane
- **Depends on:** P1-04
- **Scope:** Add the reduced memcheck workflow for the smallest bring-up case so memory errors are caught before profiling work proceeds.
- **Done when:** The smallest supported case passes memcheck with no actionable errors.

### P1-06 — Nsight Systems baseline and UVM diagnostic traces
- **Depends on:** P1-05
- **Scope:** Create the basic Nsight Systems profiling lane, the UVM-focused diagnostic lane, and the initial NVTX smoke instrumentation used in Phase 1.
- **Done when:** Both baseline and UVM traces show GPU kernels and visible NVTX ranges, and recurring unexplained UVM migration is eliminated or documented.

### P1-07 — PTX-JIT proof and Phase 1 acceptance bundle
- **Depends on:** P1-06
- **Scope:** Run the PTX-JIT compatibility proof and generate the Phase 1 acceptance reports that freeze the bring-up result.
- **Done when:** PTX-JIT succeeds on the workstation lane and `phase1_acceptance_report.json` / `.md` are complete and reviewable.


## Phase 2 — GPU memory model

Authoritative source(s): `phase2_gpu_memory_spec.md`

### P2-01 — Canonical `gpuRuntime.memory` parser
- **Depends on:** FND-01
- **Scope:** Implement the authoritative memory-configuration parser, enums, and compatibility-shim handling for the Phase 2 contract.
- **Done when:** The branch has one normalized memory config tree and all later memory code resolves policy from it.

### P2-02 — Device persistent pool
- **Depends on:** P2-01
- **Scope:** Add the persistent device allocator for long-lived hot objects and the unit tests that verify deterministic allocation behavior.
- **Done when:** Persistent allocations are pooled, leak-free, and validated independently of solver integration.

### P2-03 — Device scratch pool
- **Depends on:** P2-01
- **Scope:** Implement the transient device scratch allocator for timestep-local workspaces with reset/reuse semantics.
- **Done when:** Scratch allocation/reuse works across repeated iterations without unexpected growth or churn.

### P2-04 — Pinned stage allocator
- **Depends on:** P2-01
- **Scope:** Add the pooled `cudaMallocHost`-backed stage allocator for explicit write/restart/output staging.
- **Done when:** Field-scale host staging goes through one explicit pinned path and unit tests cover allocation/reuse behavior.

### P2-05 — Managed fallback allocator
- **Depends on:** P2-01
- **Scope:** Implement the managed-memory fallback allocator and its policy wiring as a correctness/debug lane rather than a production default.
- **Done when:** Managed fallback exists for controlled bring-up but is clearly segregated from production-device acceptance.

### P2-06 — Residency registry and reporting
- **Depends on:** P2-02, P2-03, P2-04
- **Scope:** Create the residency state machine, registry types, and reporting layer for all registered hot objects.
- **Done when:** Registered objects have explicit host/device visibility state and the branch can emit deterministic residency reports.

### P2-07 — Mirror traits and field mirrors
- **Depends on:** P2-06
- **Scope:** Implement `MirrorTraits` and `FieldMirror` support for OpenFOAM field/list types, including round-trip tests.
- **Done when:** Core field abstractions can be mirrored between host and device with tested round-trip correctness.

### P2-08 — Mesh mirror and startup registration
- **Depends on:** P2-07
- **Scope:** Add `MeshMirror` support plus startup registration of mesh topology, addressing, and persistent fields into the registry/pools.
- **Done when:** A supported GPU solver can complete startup upload of the hot mesh/field set using the Phase 2 substrate.

### P2-09 — Explicit visibility APIs and output stager
- **Depends on:** P2-08
- **Scope:** Implement `ensureHostVisible`, `ensureDeviceVisible`, scalar staging, and the `OutputStager` so write/restart boundaries are explicit.
- **Done when:** All field-scale host visibility in supported paths goes through explicit APIs and write-boundary correctness tests pass.

### P2-10 — Compute-epoch enforcement
- **Depends on:** P2-09
- **Scope:** Add `CpuTouchGuard`, per-epoch logging, and strict mode so illegal CPU touches are detected inside the compute epoch.
- **Done when:** Strict mode catches unsupported CPU access in the supported GPU solver path before Phase 3 begins.

### P2-11 — Scratch catalog and Phase 2 gate bundle
- **Depends on:** P2-03, P2-10
- **Scope:** Implement `ScratchCatalog`, named scratch reset/watermark tracking, and the Phase 2 benchmark/acceptance bundle.
- **Done when:** Repeated iterations show stable scratch watermarks and the Phase 2 gate proves no uncontrolled UVM traffic for registered hot objects.


## Phase 3 — Execution model

Authoritative source(s): `phase3_execution_model_spec.md`

### P3-01 — Synchronization and stream inventory
- **Depends on:** P2-10
- **Scope:** Inventory existing sync points, define explicit stream ownership, and lock down the initial stream policy.
- **Done when:** The branch has a reviewed sync/stream inventory and no hidden stream ownership in supported paths.

### P3-02 — Execution-mode parser and selection policy
- **Depends on:** P3-01, FND-06
- **Scope:** Implement runtime parsing and selection for the supported execution modes (including graph-disabled and graph-enabled lanes).
- **Done when:** Execution mode is selected deterministically from the normalized runtime tree with clear downgrade behavior.

### P3-03 — GpuExecutionContext and execution registry
- **Depends on:** P3-02
- **Scope:** Add the central `GpuExecutionContext` and registry that own stream handles, mode selection, and stage execution state.
- **Done when:** All supported GPU stages resolve execution context through one owner rather than ad hoc local state.

### P3-04 — Async launch wrapper layer
- **Depends on:** P3-03
- **Scope:** Introduce graph-ready async launch wrappers and remove implicit hot-path device-wide synchronization from the wrapped stages.
- **Done when:** Wrapped stages launch through a consistent async interface and no longer rely on hidden `cudaDeviceSynchronize()` behavior.

### P3-05 — Canonical stage scaffolding and parent NVTX ranges
- **Depends on:** P3-03, FND-06
- **Scope:** Add the canonical stage boundaries and parent-stage NVTX scaffolding imported from the graph support matrix.
- **Done when:** Stage IDs are stable, shared, and visible to instrumentation before the first captured stage lands.

### P3-06 — First graph-enabled stage
- **Depends on:** P3-04, P3-05
- **Scope:** Capture and replay the first approved graph-enabled stage using the Phase 3 execution contract.
- **Done when:** At least one allowed stage captures and replays correctly with stable addresses and no illegal capture-time behavior.

### P3-07 — Graph fingerprint, cache, and rebuild policy
- **Depends on:** P3-06
- **Scope:** Implement graph fingerprinting, cache lookup, update/rebuild policy, and parameter mirror handling for capture reuse.
- **Done when:** Steady-state runs reuse captured graphs when valid and rebuild only under documented change conditions.

### P3-08 — Write-boundary staging, production residency assertions, and Phase 3 acceptance
- **Depends on:** P3-07, P2-09
- **Scope:** Integrate explicit write-boundary staging with execution modes, add production residency assertions, and generate the Phase 3 acceptance packet.
- **Done when:** Supported execution modes pass acceptance with write paths outside the timed/captured steady-state window and no hot-path global sync regressions.


## Phase 4 — Pressure linear algebra

Authoritative source(s): `phase4_linear_algebra_spec.md`

### P4-01 — Dependency freeze and standalone AmgX smoke
- **Depends on:** P1-07, FND-02
- **Scope:** Pin the SPUMA / foamExternalSolvers / AmgX dependency line for Phase 4 and build the standalone AmgX smoke test on the target workstation.
- **Done when:** An identity or trivial system solves correctly on the RTX 5080 before OpenFOAM integration begins.

### P4-02 — Pressure snapshot dump utility
- **Depends on:** P4-01, P0-08
- **Scope:** Add the CPU/reference-side pressure snapshot utility so representative matrices, RHS vectors, and metadata can be captured from frozen cases.
- **Done when:** Reduced reference cases can emit reusable pressure snapshots for offline replay and validation.

### P4-03 — Topology-only LDU-to-CSR builder
- **Depends on:** P4-02
- **Scope:** Implement the topology conversion that builds CSR row offsets and column indices from the OpenFOAM/SPUMA LDU structure and caches it by topology.
- **Done when:** Synthetic and snapshot-backed tests prove the structural CSR mapping is correct and reusable.

### P4-04 — CSR value packer and `A*x` validator
- **Depends on:** P4-03
- **Scope:** Add the value packer that fills CSR coefficients/RHS from live or snapshot data and validate the mapping with `A*x` equivalence checks.
- **Done when:** Snapshot and synthetic `A*x` checks pass within tolerance, separating mapping bugs from solver-backend bugs.

### P4-05 — AmgX context wrapper and replay utility
- **Depends on:** P4-04
- **Scope:** Implement the `AmgXContext` wrapper plus a replay tool that loads snapshots and exercises upload/setup/solve/update in isolation.
- **Done when:** Snapshot replay solves correctly and emits timing decomposition without involving the live pressure loop.

### P4-06 — PressureMatrixCache and persistent staging buffers
- **Depends on:** P4-05, P2-04
- **Scope:** Add the persistent pressure-matrix cache object plus reusable host staging buffers that outlive individual solver objects.
- **Done when:** Repeated solves with unchanged topology reuse the cached structure and avoid rebuilding row/column data.

### P4-07 — Runtime-selected live solver integration
- **Depends on:** P4-06
- **Scope:** Wire native and AmgX pressure solver selection into the live solver path with clean fallback on unsupported interface conditions.
- **Done when:** A reduced live case can select AmgX at runtime or fall back cleanly to native without corrupting the pressure path.

### P4-08 — Telemetry, profiling hooks, and reduced-case validation
- **Depends on:** P4-07, P3-05
- **Scope:** Add pressure telemetry, NVTX boundaries, native-vs-AmgX benchmark scripts, and reduced-case validation for the Phase 4 live path.
- **Done when:** Phase 4 can prove no repeated topology upload, can compare native vs AmgX on the reduced case, and emits reviewable telemetry artifacts.

### P4-09 — DeviceDirect pressure bridge
- **Depends on:** P4-08, P2-02
- **Scope:** Implement the device-resident `csrValuesDev` / `rhsDev` / `xDev` bridge and the device-pointer upload/replace path required for later no-field-scale-host-transfer claims.
- **Done when:** Replay and reduced live cases validate the device-pointer path, while `PinnedHost` remains correctness-only and clearly non-production.


## Phase 5 — Generic VOF core

Authoritative source(s): `phase5_spuma_nozzle_spec.md`

### P5-01 — Phase 5 symbol reconciliation note
- **Depends on:** FND-07
- **Scope:** Generate and review the exact mapping from semantic sources to the local SPUMA/v2412 files touched by the generic VOF port.
- **Done when:** The implementation branch has a reviewed patch-target note before any Phase 5 solver edits land.

### P5-02 — VOF runtime gate and support scan
- **Depends on:** FND-04, P5-01
- **Scope:** Implement the `gpuRuntime.vof` / compatibility-shim gate plus deterministic support scanning for the allowed Phase 5 envelope.
- **Done when:** Unsupported generic VOF tuples fail fast before the timestep loop and accepted tuples are explicitly identified.

### P5-03 — Persistent topology and boundary-map substrate
- **Depends on:** P2-08, P5-02
- **Scope:** Build the Phase 5 topology, cell/face connectivity, and patch descriptor substrate needed by the generic VOF device path.
- **Done when:** The generic VOF path has stable device topology/boundary metadata and emits a memory report for the accepted reduced cases.

### P5-04 — Field mirrors and old-time state for VOF
- **Depends on:** P5-03, P2-07
- **Scope:** Mirror the Phase 5 field set, including old-time/previous-correction state needed by alpha transport and pressure coupling.
- **Done when:** Required VOF fields are device-visible with explicit host authority boundaries and old-time state is preserved correctly.

### P5-05 — Alpha skeleton path
- **Depends on:** P5-04, P3-03
- **Scope:** Implement the end-to-end alpha skeleton path with control snapshots, alpha-flux formation, and placeholder/predictor-only logic.
- **Done when:** The alpha stage runs end-to-end on the reduced case and establishes the device control-flow skeleton without claiming bounded correctness yet.

### P5-06 — Full alpha + MULES + subcycling
- **Depends on:** P5-05
- **Scope:** Complete the bounded alpha path, explicit MULES limiter logic, previous-correction flux behavior, and alpha subcycling semantics.
- **Done when:** Reduced generic VOF cases keep alpha bounded and numerically consistent with the frozen CPU reference under the accepted scheme subset.

### P5-07 — Mixture update and interface/surface-tension subset
- **Depends on:** P5-06
- **Scope:** Implement `rho` / `rhoPhi` updates plus the constant-sigma interface normals/curvature/surface-tension slice admitted in Phase 5 scope.
- **Done when:** Mixture properties and the restricted interface path validate on the designated reduced tests without reopening contact-angle scope.

### P5-08 — Momentum predictor
- **Depends on:** P5-07
- **Scope:** Add the device momentum-predictor path, including persistent `rAU` handling and the accepted laminar reduced-case wiring.
- **Done when:** The reduced laminar case exercises the device `U` equation path successfully and preserves the expected predictor semantics.

### P5-09 — Native pressure backend integration
- **Depends on:** P5-08, P4-07
- **Scope:** Integrate the native pressure corrector path, including non-orthogonal loops, into the device-authoritative Phase 5 solver.
- **Done when:** A reduced nozzle-friendly case completes the native pressure corrector path correctly with the Phase 5 generic core.

### P5-10 — AmgX pressure backend integration
- **Depends on:** P5-09, P4-09
- **Scope:** Wire the runtime-selectable AmgX backend into the Phase 5 solver behind the DeviceDirect gate and compare it against native behavior.
- **Done when:** Accepted reduced tuples can use the AmgX path without violating the no-field-scale-host-transfer contract.

### P5-11 — Write-time commit, validation artifacts, and Phase 5 baseline freeze
- **Depends on:** P5-10, P8-03
- **Scope:** Implement the explicit write-time commit path, restart/reload parity checks, validation/profile artifact generation, and the final Phase 5 baseline freeze.
- **Done when:** Phase 5 ends with a reviewable reduced-case baseline, restart/reload parity evidence, and artifacts ready for Phase 6 handoff.


## Phase 6 — Pressure-swirl nozzle boundary conditions and startup

Authoritative source(s): `phase6_pressure_swirl_nozzle_bc_spec.md`

### P6-01 — Boundary support report and patch classifier
- **Depends on:** FND-04, P5-11
- **Scope:** Implement the boundary support report, patch classifier, and manifest substrate for the frozen nozzle patch set.
- **Done when:** The branch can classify the R1/R0 boundary set up front and reject unsupported nozzle configurations deterministically.

### P6-02 — Flat boundary spans and boundary metadata upload
- **Depends on:** P6-01, P2-08
- **Scope:** Add the flat boundary-span representation and uploadable metadata views required by device-side boundary execution.
- **Done when:** Supported nozzle cases expose flat boundary spans and patch metadata without falling back to host patch polymorphism.

### P6-03 — Constrained profile grammar and compiler
- **Depends on:** P6-01
- **Scope:** Implement the constrained inlet-profile grammar/parser/compiler used by the frozen nozzle boundary contract.
- **Done when:** Accepted inlet profiles compile into a machine-readable representation that the CPU reference and device path both understand.

### P6-04 — Custom `gpuPressureSwirlInletVelocity` type and CPU snapshot path
- **Depends on:** P6-03, P5-01
- **Scope:** Add the custom nozzle inlet runtime type plumbing together with the CPU snapshot/reference path used to validate inlet semantics before device execution.
- **Done when:** The custom inlet BC can be instantiated, and CPU-side snapshot tests prove the frozen inlet math before GPU kernels land.

### P6-05 — Alpha boundary kernels
- **Depends on:** P6-02
- **Scope:** Implement the accepted alpha BC kernels (`fixedValue`, `zeroGradient`, `inletOutlet`) for the nozzle envelope.
- **Done when:** Frozen-field boundary tests pass for the accepted alpha patch kinds without host patch evaluation in the hot loop.

### P6-06 — Ambient/open velocity boundary kernels
- **Depends on:** P6-02
- **Scope:** Implement the standard ambient/open velocity kernels admitted by the support matrix for nozzle cases.
- **Done when:** The accepted open-boundary velocity subset runs correctly on device and passes frozen snapshot tests.

### P6-07 — Swirl inlet device kernel and invariance tests
- **Depends on:** P6-04, P6-02
- **Scope:** Implement the custom swirl inlet device kernel, including basis regularization and integrated-flux invariance checks.
- **Done when:** Synthetic annular-patch tests and inlet-flux invariance checks pass before full solver integration.

### P6-08 — Pressure boundary integration
- **Depends on:** P6-02, P5-09
- **Scope:** Implement the accepted `prgh*` pressure kernels plus the `fixedFluxPressure` gradient update and the associated pressure-assembly hook as one coupled change.
- **Done when:** `snGradp` matches the CPU reference on frozen tests and a short R1 transient passes the numerically risky pressure/BC coupling point.

### P6-09 — Startup seeding subsystem
- **Depends on:** P6-01, P5-11
- **Scope:** Implement the canonical startup-seed DSL/parser, the seed-region kernel, and the post-seed refresh sequence used by the accepted nozzle path.
- **Done when:** Seed masks and initialized fields match the frozen reference exactly on the accepted startup cases.

### P6-10 — Solver-stage integration, graph-safety hardening, and Phase 6 acceptance
- **Depends on:** P6-05, P6-06, P6-07, P6-08, P6-09
- **Scope:** Wire the nozzle boundary and startup subsystem into alpha/momentum/pressure stages, add runtime-parameter update handling, remove transient allocations, and run the Phase 6 acceptance pass.
- **Done when:** Full R1 runs complete with device-side nozzle BC/startup behavior, graph-safety audits are clean, and the R0 regression package is ready for review.


## Phase 7 — Custom CUDA kernels

Authoritative source(s): `phase7_custom_cuda_kernel_spec.md`

### P7-01 — Phase 7 source audit and hotspot ranking
- **Depends on:** P6-10, P8-05
- **Scope:** Generate the reviewed Phase 7 source-audit note and produce the hotspot ranking artifact that bounds custom-kernel scope.
- **Done when:** Custom-kernel work is explicitly limited to support-matrix-approved hotspot families ranked from real profiling data.

### P7-02 — Control plane, POD views, and façade skeleton
- **Depends on:** P7-01
- **Scope:** Implement the Phase 7 controls parser, POD device views, and the host-side façade that hides kernel-launch details from solver code.
- **Done when:** The custom-kernel subsystem has a narrow, typed interface and runtime control surface before any optimized kernel work lands.

### P7-03 — Adjacency preprocessing
- **Depends on:** P7-02
- **Scope:** Build the adjacency preprocessing needed by segmented/gather execution, keeping it separate from kernel correctness work.
- **Done when:** Adjacency structures are generated once, validated, and reusable by both the atomic and segmented backends.

### P7-04 — Persistent scratch arena
- **Depends on:** P7-02, P2-11
- **Scope:** Implement the persistent scratch arena and its ownership/lifetime rules for custom-kernel hot paths.
- **Done when:** Custom kernels can acquire named scratch resources without allocation churn, and watermarks remain stable across repeated iterations.

### P7-05 — Atomic alpha/MULES correctness backend
- **Depends on:** P7-03, P7-04
- **Scope:** Implement the atomic baseline for the Phase 7 alpha/MULES hotspot family as the first correctness line.
- **Done when:** The atomic backend is numerically correct on the accepted reduced/nozzle cases and benchmark artifacts are captured at the milestone stop.

### P7-06 — Interface and patch kernels
- **Depends on:** P7-04, P6-10
- **Scope:** Implement the accepted interface and patch kernel family for the exact patch/scheme subset used by `R1` / `R0`.
- **Done when:** The supported interface/patch kernels pass unit tests and integrate without reopening contact-angle scope.

### P7-07 — Segmented/gather production backend
- **Depends on:** P7-05, P7-03
- **Scope:** Implement the segmented/gather production backend and compare it against the atomic path on the profiled hotspot set.
- **Done when:** A production backend exists that improves or otherwise justifies replacing the atomic baseline for accepted tuples.

### P7-08 — Graph-safety cleanup, capture validation, and final regression package
- **Depends on:** P7-06, P7-07
- **Scope:** Remove remaining capture hazards, validate graph compatibility of the chosen custom-kernel path, and generate the final Phase 7 regression/comparison package.
- **Done when:** The selected custom-kernel path is capture-safe for the approved stages, UVM-clean in steady state, and backed by final regression artifacts.


## Phase 8 — Profiling and performance acceptance

Authoritative source(s): `phase8_profiling_performance_acceptance_spec.md`

### P8-01 — NVTX wrapper library and build flags
- **Depends on:** P1-02
- **Scope:** Add the permanent NVTX3 wrapper library, compile-time switches, and dependency wiring required by the profiling subsystem.
- **Done when:** Profiling instrumentation can be compiled in/out cleanly and is available to all supported solver modules.

### P8-02 — Canonical profiling and acceptance config parser
- **Depends on:** FND-05, FND-06, P8-01
- **Scope:** Implement the normalized `gpuRuntime.profiling` / `gpuRuntime.acceptance` parser and compatibility shims.
- **Done when:** All profiling and acceptance behavior is runtime-configurable from the canonical tree with no local shadow schema.

### P8-03 — Solver-stage instrumentation coverage
- **Depends on:** P8-02, P3-05
- **Scope:** Instrument the top-level timestep/PIMPLE/alpha/momentum/pressure/surface/boundary/output stages with the canonical NVTX naming hierarchy.
- **Done when:** The required solver-stage ranges appear in traces for accepted tuples and are aligned with the canonical stage registry.

### P8-04 — Graph lifecycle instrumentation
- **Depends on:** P8-02, P3-07
- **Scope:** Instrument graph capture/build/replay/update lifecycle events so graph behavior is visible in traces and parsers.
- **Done when:** Graph-enabled runs emit the required graph lifecycle markers without changing solver semantics.

### P8-05 — Nsight Systems capture scripts and artifact layout
- **Depends on:** P8-03, P8-04
- **Scope:** Create the Phase 8 `nsys` capture scripts, mode selection, environment manifests, and artifact directory layout.
- **Done when:** A baseline `R1` timeline can be captured end-to-end in the standardized artifact layout with reproducible command lines.

### P8-06 — Stats export and profile-acceptance parser
- **Depends on:** P8-05, FND-05
- **Scope:** Implement report export from `nsys` outputs and the acceptance parser that maps metrics back to manifest-driven hard/soft gates.
- **Done when:** The branch can produce a first formal `R1` acceptance summary from captured profiling artifacts.

### P8-07 — Diagnostic profiling modes
- **Depends on:** P8-06
- **Scope:** Add the `uvmAudit`, `syncAudit`, and `graphDebug` modes, keeping them diagnostically useful but outside baseline production timing mode.
- **Done when:** Reduced `R1` runs can separately audit unexpected UVM traffic, inner-loop synchronization, and graph structure with stable artifacts.

### P8-08 — Top-kernel NCU and sanitizer automation
- **Depends on:** P8-06, P8-07
- **Scope:** Add the top-five-kernel Nsight Compute wrapper, section configuration, Compute Sanitizer wrapper, and nightly reduced-case automation.
- **Done when:** The project can produce top-kernel reports and clean reduced-case sanitizer logs without contaminating baseline timing runs.

### P8-09 — Baseline locks and CI/nightly integration
- **Depends on:** P8-08, P7-08
- **Scope:** Run the formal `R1` / `R0` acceptance framework, lock the first approved baselines, and integrate the scheduled profiling/sanitizer jobs.
- **Done when:** The first approved `R0` acceptance bundle exists, `R1` nightly / `R0` scheduled jobs are wired, and retention/lock policy is operational.


## Final sequencing notes

- Treat each PR above as the smallest recommended merge unit. If implementation reality forces a split, split downward; do not merge adjacent PRs upward across risky boundaries.
- For any PR that introduces a new runtime key, loader, or artifact schema, the parser/schema work should land before solver integration work that consumes it.
- For any PR that introduces a new numerically active stage, add its unit/snapshot/reduced-case validation in the same PR or in the immediately following PR before moving to the next coupling point.
- The final PR of each phase should be a freeze/acceptance PR, not the same PR that first lands the underlying capability.