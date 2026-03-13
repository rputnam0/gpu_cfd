# 1. Executive overview

This document preserves the full project context but expands **only Phase 2** to exhaustive implementation depth. Phases 0, 1, and 3-8 are retained only as condensed dependency context so the memory design remains coherent with the overall nozzle-port roadmap; they are not the authoritative owners of cross-phase sequencing, scope, pins, or acceptance. Those project-level contracts live in the master roadmap, master pin manifest, support matrix, validation ladder, GPU operational contract, acceptance manifest, and graph support matrix named in the continuity ledger. The controlling architectural decision for this project is that **SPUMA remains the runtime base**, but **SPUMA's default memory posture does not remain the production memory posture** for the target nozzle workflow. SPUMA's published strategy uses a portable execution abstraction plus memory pools built around unified memory to enable incremental GPU porting with limited invasive change.[R1] That is a strong bring-up strategy. It is **not** the safest production strategy for a transient, subcycled, pressure-swirl nozzle solver on a discrete RTX 5080 where recurring host/device migration, patch-level CPU touches, and temporary allocation churn will amplify across PIMPLE, alpha corrections, and MULES subcycles.[R1][R3][R15][R19]

The project therefore adopts a two-mode interpretation of SPUMA's memory model:

- **Bring-up mode**: preserve SPUMA-compatible managed/unified-memory behavior so incomplete ports continue to run.
- **Production mode**: promote the timestep working set to explicit **device-resident persistent allocations**, stage only selected data to the CPU through **pinned host buffers**, and reduce managed memory to a compatibility/fallback tier.

This is the central point of Phase 2. Every later phase depends on it. Without it, later work on CUDA Graphs, device-resident alpha transport, external pressure solvers, or nozzle-specific boundary kernels will either be masked by managed-memory migrations or repeatedly broken by hidden CPU-side touches.[R1][R3][R15][R19]

The implementation intent is conservative:

- Keep OpenFOAM and SPUMA semantics recognizable.
- Do **not** rewrite numerics in Phase 2.
- Do **not** try to make every OpenFOAM object globally GPU-transparent in one step.
- Introduce a narrow, explicit, testable memory layer that later phases can rely on.

The main Phase 2 deliverables are:

1. a **tiered allocator architecture** with device-persistent, device-scratch, pinned-host-staging, and managed-fallback tiers;
2. a **residency registry** that tracks which OpenFOAM/SPUMA objects have valid device mirrors, valid host mirrors, old-time copies, and write-time snapshots;
3. a **host-touch policy** that forbids implicit CPU access to the hot working set during compute epochs unless a deliberate synchronization call is made;
4. a **staging path** for output, diagnostics, and small host-consumed scalars;
5. a **measurement framework** that treats recurring unified-memory traffic after warm-up as a correctness failure of the porting strategy, not a minor optimization issue.[R3][R15][R19]

The most important design judgment in this specification is the following:

- **Sourced fact**: SPUMA uses memory pools built around unified memory and explicitly describes this as an incremental, minimally invasive porting strategy.[R1]
- **Engineering inference**: that strategy is excellent for initial enablement but insufficient as the end-state for a discrete-GPU, transient multiphase solver where unsupported paths silently trigger host/device copies and page migration can dominate runtime.[R1][R3][R15][R19]
- **Recommendation**: preserve SPUMA's managed-memory path only as a compatibility mode; build a new production memory layer on explicit device allocation plus pinned staging, then require later solver work to use it.

# 2. Global architecture decisions

## 2.1 Keep SPUMA as the code base

- **Sourced fact**: SPUMA is the actively developed OpenFOAM GPU-porting fork in scope here, benchmarked against OpenFOAM-v2412 and exposing current GPU-supported solver lists through its project/wiki material.[R1][R2][R3]
- **Engineering inference**: rebasing the nozzle work onto vanilla OpenFOAM 12 first would add version-migration and GPU-porting risk simultaneously.
- **Recommendation**: implement Phase 2 inside the SPUMA/v2412 code line and keep upstream-diff minimization as a secondary objective, not the primary one.

## 2.2 Make device residency the production invariant

- **Sourced fact**: on discrete GPUs, page migration and host/device copies can dominate runtime in unified-memory-heavy approaches; SPUMA also warns that unsupported features usually slow down through undesired copies instead of failing fast.[R1][R3]
- **Engineering inference**: if a field that participates in every timestep remains in managed memory or is repeatedly touched on the CPU, the future nozzle solver will not show stable performance signatures.
- **Recommendation**: define the canonical production invariant as: **all hot mesh, field, matrix, patch, and scratch data are device-valid and remain device-resident between timesteps unless explicitly staged.**

## 2.3 Preserve OpenFOAM object ownership on the host, but decouple data residence

- **Sourced fact**: OpenFOAM uses `tmp`, `GeometricField`, `Field`, `List`, and patch-field APIs that create many short-lived objects and temporary arrays.[R1][R6][R7][R9][R10]
- **Engineering inference**: replacing the entire ownership and runtime-selection model would be too invasive for this phase.
- **Recommendation**: keep the host-side OpenFOAM object graph as the semantic owner, but let the GPU memory layer own explicit device mirrors and synchronization state.

## 2.4 Use explicit staging instead of transparent CPU reads

- **Sourced fact**: true asynchronous host/device transfers require page-locked host memory; pageable host buffers force less asynchronous behavior and can serialize the path.[R16][R17]
- **Engineering inference**: allowing writes or diagnostics to pull directly from arbitrary host copies will reintroduce hidden transfers.
- **Recommendation**: all write, debug dump, and scalar-reduction paths must route through explicit staging APIs that allocate pinned buffers, issue explicit copies, and record reasons.

## 2.5 Prefer CUDA device pools over repeated raw allocations

- **Sourced fact**: SPUMA measured major gains from pooling because OpenFOAM allocates many temporary arrays; CUDA's stream-ordered allocator supports `cudaMallocAsync`, `cudaMallocFromPoolAsync`, pool attributes, and graph capture semantics.[R1][R14]
- **Engineering inference**: replacing `cudaMallocManaged` churn with `cudaMallocAsync`/pool-backed device allocation is the most direct way to remove allocator overhead and prepare for graphs.
- **Recommendation**: implement explicit device pools as the default production allocator and retain Umpire/managed behavior only behind the fallback tier.

## 2.6 Treat pinned-host pools as optional, not mandatory, in this phase

- **Sourced fact**: current CUDA/CCCL documentation describes host-pinned memory pools, but archived 12.8.1 HTML is not fully consistent with current documentation on host pool creation semantics.[R14][R18]
- **Engineering inference**: defaulting the first production implementation to a feature whose archived documentation is ambiguous is unnecessary risk.
- **Recommendation**: implement the Phase 2 pinned-staging path first as a pooled wrapper around `cudaMallocHost`/`cudaFreeHost`; make stream-ordered host-pinned pools an experimental optimization switch.

## 2.7 Phase 2 is a prerequisite gate, not an optional optimization phase

- **Sourced fact**: SPUMA currently does not support multiphase solvers, and its present synchronization and profiling results already show sensitivity to allocator and synchronization structure.[R1]
- **Engineering inference**: if memory and residency are not stabilized now, later nozzle-specific GPU work will be impossible to debug cleanly.
- **Recommendation**: do not begin Phase 5 productionization until the Phase 2 acceptance gates are passed on the reduced validation-ladder case selected for this stage (normally `R1-core`; `R1` only when nozzle-specific topology or patch-manifest coverage is explicitly under test).

# 3. Global assumptions and constraints

## 3.1 Hardware assumptions

- Single NVIDIA GeForce RTX 5080 workstation for the first production target.
- Blackwell-generation consumer/workstation GPU, compute capability 12.0, 16 GB GDDR7, nominal 960 GB/s memory bandwidth, no NVLink.[R13]
- Linux workstation preferred for development because CUDA 12.8 Linux driver guidance and typical SPUMA deployment patterns are Linux-centric.[R11][R12]

## 3.2 Software assumptions

- Exact toolchain, driver, profiler, and code-generation pins are inherited from the master pin manifest frozen after Phase 1; this file does not reopen them. At the current continuity freeze the default pin set is CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, and NVTX3.
- SPUMA checkout is the branch/SHA frozen in the master pin manifest and aligned with the OpenFOAM-v2412-era code line described in project material.[R1][R2]
- Umpire remains an optional dependency because SPUMA already exposes an Umpire pool path.[R1][R2][R20]

## 3.3 OpenFOAM behavioral assumptions

- `tmp<T>` and `GeometricField` temporaries remain common and semantically important.[R6][R7]
- `List<T>` and `Field<T>` continue to use contiguous storage layouts compatible with explicit mirroring.[R10]
- `oldTime()` and `prevIter()` storage for transient fields must remain semantically correct after mirror introduction.[R4][R8]
- Patch-field matrix coefficient methods continue to produce temporary `Field` objects unless rewritten later.[R9]

## 3.4 Project constraints

- Phase 2 must not change governing equations, limiters, discretization choices, or solver numerics.
- Phase 2 must support both a **bring-up path** and a **production path**.
- Multi-GPU communication buffers are explicitly deferred.
- Device-resident sparse-matrix topology caching for external solvers is not productized in Phase 2, but the memory interfaces must not block it.
- The implementation must remain debuggable by a coding agent with deterministic failure modes and explicit metrics.

# 4. Cross-cutting risks and mitigation

| Risk | Why it matters | Mitigation in this specification |
|---|---|---|
| Silent managed-memory fallback | SPUMA warns unsupported paths usually slow down through unintended copies instead of failing hard.[R3] | Maintain a registry of device-resident objects, expose explicit `ensureHostVisible()` / `ensureDeviceVisible()` calls, and fail the benchmark gate on recurring steady-state UVM events. |
| OpenFOAM temporary churn | `tmp`, `GeometricField`, patch coefficients, and lazy object creation generate many short-lived allocations.[R1][R6][R7][R9] | Separate persistent and scratch tiers; add a scratch catalog; prohibit raw allocation inside hot loops once ported. |
| CPU-side boundary/output code touching hot fields | `primitiveFieldRef()`-style host access increments state and should be avoided in loops.[R8] | Introduce compute epochs with CPU-touch guards, explicit write-time staging, and audit all write/debug code paths. |
| Pool shrinkage or allocator jitter at synchronization points | CUDA pool attributes control release thresholds and reuse behavior.[R14] | Set persistent and scratch release thresholds high during timestep execution; trim explicitly at safe boundaries; default to deterministic reuse settings. |
| Pinned-memory overuse | CUDA warns excessive pinned memory hurts system performance.[R17] | Limit pinned staging to bounded buffers for output and diagnostics; use pool statistics; enforce configurable caps. |
| Documentation ambiguity around host-pinned CUDA mempools | Archived 12.8.1 docs and current docs are not perfectly aligned.[R14][R18] | Do not require host mempools for Phase 2 correctness; use `cudaMallocHost`-backed pooled staging first. |
| Later solver backends needing separate workspaces | AmgX/cuSPARSE/cuSOLVER will allocate their own work areas later.[R1][R12][R22] | Keep device-memory headroom reserve; do not hard-commit all VRAM to Phase 2 pools. |
| Precision and data-layout coupling | OpenFOAM field layouts are semantically entrenched and custom kernels later may want different layouts. | Preserve native binary layout in Phase 2; defer SoA/data-layout experiments until after correctness and residency are stable. |


# 5. Phase-by-phase implementation specification

Only the Phase 2 subsection below is normative in this file. The surrounding Phase 0/1/3-8 subsections remain as dependency context only so local Phase 2 interfaces stay legible. If any condensed cross-phase text here conflicts with the centralized support matrix, validation ladder, master pin manifest, acceptance manifest, GPU operational contract, or graph support matrix, those centralized artifacts win.

## Phase 0 — Freeze the reference problem

### Purpose
Establish CPU-trusted reference cases and a migration-safe SPUMA/v2412 baseline before any production memory surgery.

### Why this phase exists
Phase 2 changes residence, allocation, and object lifetimes. Without fixed reference cases, later numerical deltas cannot be attributed confidently to memory changes versus solver/version drift.

### Entry criteria
- The existing nozzle workflow and reduced development case are runnable on the current CPU environment.
- The case generator/harness can be moved to SPUMA's code line without altering physics choices.

### Exit criteria
- R2, `R1-core`, R1, and R0 are frozen with archived meshes, dictionaries, expected outputs, and acceptance tolerances.
- A SPUMA/v2412 CPU branch reproduces baseline nozzle metrics closely enough that Phase 2 numerical regressions can be isolated.

### Goals
- Separate version-migration risk from GPU-porting risk.
- Produce immutable regression artifacts for later CPU/GPU comparison.

### Non-goals
- No GPU optimization.
- No device-residency work.
- No solver redesign.

### Technical background
SPUMA is the execution base, but its current documented GPU-support list does not include `incompressibleVoF`; therefore the nozzle path remains new solver work even after rebasing.[R2][R3]

### Research findings relevant to this phase
- **Sourced fact**: SPUMA publishes supported GPU solver subsets and warns that unsupported features often fail by slowdown, not by hard error.[R3]
- **Recommendation**: freeze CPU references before enabling GPU-specific memory paths.

### Design decisions
- Freeze four validation-ladder cases: R0 full nozzle, R1 reduced internal nozzle, R1-core reduced generic nozzle core, and R2 generic two-phase verification.
- Archive outputs and tolerances in machine-readable form.

### Alternatives considered
Using ad hoc day-to-day cases was rejected because it would make Phase 2 regression triage ambiguous.

### Interfaces and dependencies
Depends on case-generator portability and SPUMA CPU branch buildability; unblocks Phases 1 and 2.

### Data model / memory model
Reference outputs remain host-side canonical artifacts; no GPU mirrors are introduced here.

### Algorithms and control flow
Port case harness, run baselines, archive metrics, compare against existing production values, and lock files.

### Required source changes
- Add regression harness scripts.
- Add archived baseline metadata.
- Add CPU-only CI target for R1/R2.

### Proposed file layout and module boundaries
- `cases/reference/R0`, `cases/reference/R1`, `cases/reference/R2`
- `tools/regression/reference_manifest.yaml`
- `tools/regression/compare_reference.py`

### Pseudocode
```text
for case in [R0, R1, R2]:
run_cpu_baseline(case)
extract_metrics(case)
write_manifest(case)
compare_to_current_production(case)
if tolerance_fail: stop_and_fix_version_drift()
```

### Step-by-step implementation guide
1. Port the harness to SPUMA/v2412 CPU mode.
2. Freeze meshes and dictionaries.
3. Run multiple repeats for deterministic metrics.
4. Archive field snapshots and scalar metrics.
5. Record tolerances and failure signatures.

### Instrumentation and profiling hooks
- Timestamp runs.
- Record commit hashes.
- Store solver logs and residual histories.

### Validation strategy
- Metric agreement against existing workflow.
- Deterministic reruns on same branch.
- Archived manifests checked into version control.

### Performance expectations
Performance is irrelevant except for ensuring the reference cases are small enough to run repeatedly in development.

### Common failure modes
- Baseline mismatch due to version drift.
- Missing harness dependencies.
- Non-deterministic setup scripts.

### Debugging playbook
- Diff dictionaries first.
- Compare mesh statistics.
- Then compare field norms and integral metrics.

### Acceptance checklist
- Reference manifests committed.
- R2/`R1-core`/R1/R0 rerun successfully on SPUMA CPU branch.
- Tolerances reviewed by a human.

### Future extensions deferred from this phase
- Larger DOE matrix freezing.
- Automated report dashboards.

### Implementation tasks for coding agent
- Build the regression harness.
- Freeze artifacts.
- Add comparison scripts.

### Do not start until
- Do not start until `reference_case_contract.md` and `acceptance_manifest.md` are attached to the work branch and the metric list matches those authorities.

### Safe parallelization opportunities
- R1 and R2 harnessing can proceed in parallel.

### Requires human sign-off on
- Final metric tolerances.
- Which outputs are mandatory for archival.

### Artifacts to produce
- Frozen case directories.
- Reference manifest.
- CPU regression report.


## Phase 1 — Blackwell bring-up and tooling

### Purpose
Validate that the 5080 workstation, CUDA toolchain, SPUMA build, and profiling stack are healthy before changing memory semantics.

### Why this phase exists
A broken toolchain looks similar to a broken memory layer. The project needs a trusted CUDA/driver/profiler baseline first.

### Entry criteria
- Phase 0 references are frozen.
- Target workstation is available.

### Exit criteria
- SPUMA builds on the 5080 system.
- Supported GPU solvers run.
- Nsight Systems / Compute, Compute Sanitizer, and NVTX3 instrumentation are operational.

### Goals
- Confirm Blackwell-native build viability.
- Confirm profiling collection methodology.
- Establish warm-up and driver sanity checks.

### Non-goals
- No nozzle solver port.
- No Phase 2 allocator changes yet.

### Technical background
Blackwell compatibility in CUDA 12.8 depends on including PTX, and CUDA 12.8 release notes add Blackwell compiler support and deprecate NVTX v2.[R11][R12] The RTX 5080 is compute capability 12.0.[R13]

### Research findings relevant to this phase
- **Sourced fact**: CUDA 12.8 adds Blackwell compiler support and recommends PTX/JIT validation; Linux packaged-driver guidance is 570.26+ for CUDA 12.8 GA.[R11][R12]
- **Recommendation**: verify PTX inclusion and profiler functionality before any allocator changes.

### Design decisions
- Consume `master_pin_manifest.md`; current frozen default is CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, NVTX3.
- Include PTX in builds.
- Use NVTX3 only.
- Validate on SPUMA-supported solvers first.[R3]

### Alternatives considered
Deferring profiling setup was rejected because Phase 2 acceptance depends on UVM and allocation traces.

### Interfaces and dependencies
Depends on Phase 0 references; unblocks Phase 2 instrumentation and runtime queries.

### Data model / memory model
No new memory tiers yet; this phase only verifies existing runtime behavior and tool visibility.

### Algorithms and control flow
Build SPUMA, run supported solvers, capture Nsight traces, validate device attributes and PTX/JIT behavior.

### Required source changes
- Build-system flags for Blackwell/PTX.
- NVTX3 migration where required.
- Tooling scripts for `nsys`, `ncu`, and sanitizer.

### Proposed file layout and module boundaries
- `tools/profiling/nsys_gpu_faults.sh`
- `tools/profiling/ncu_topkernels.sh`
- `src/OpenFOAM/device/profiling/nvtx3Wrappers.*`

### Pseudocode
```text
build_spuma(cuda=12.9.1, include_ptx=True)
run_supported_solver(simple_case)
profile_with_nsys(um_faults=True)
profile_top_kernels_with_ncu()
run_compute_sanitizer()
```

### Step-by-step implementation guide
1. Update build flags.
2. Validate driver/toolkit compatibility.
3. Run a supported solver.
4. Capture baseline traces.
5. Confirm NVTX ranges appear.

### Instrumentation and profiling hooks
- `nsys` with UVM page-fault tracing as recommended by SPUMA wiki.[R3]
- `ncu` only on top kernels.
- Compute Sanitizer on reduced cases.

### Validation strategy
- Supported solver completion.
- PTX/JIT test path works.
- Profiler outputs are readable and attributed.

### Performance expectations
No target speedup yet; only health-check execution is required.

### Common failure modes
- Missing PTX.
- Driver/toolkit mismatch.
- NVTX not emitted.
- Profiler permissions issues.

### Debugging playbook
- Check `cudaDriverGetVersion` and toolkit version.
- Force PTX JIT.
- Run a tiny supported case under sanitizer.

### Acceptance checklist
- Build succeeds.
- Profilers work.
- Baseline traces archived.

### Future extensions deferred from this phase
- CUDA Graph tooling presets.
- Automated profiler regression scripts.

### Implementation tasks for coding agent
- Prepare the workstation.
- Fix build flags.
- Archive baseline profiler runs.

### Do not start until
- Do not start until the reference cases from Phase 0 exist.

### Safe parallelization opportunities
- Build fixes and profiling wrappers can proceed in parallel.

### Requires human sign-off on
- Minimum tool versions.
- Whether to support Windows in addition to Linux.

### Artifacts to produce
- Build log.
- Nsight traces.
- Sanitizer log.
- PTX/JIT validation note.


## Phase 2 — Replace SPUMA’s default memory posture for production use

### Purpose
Convert SPUMA's current unified-memory-centric pool strategy into a **tiered production memory system** that keeps the nozzle solver's steady-state working set resident in GPU device memory, stages host-visible data explicitly through pinned buffers, and relegates unified/managed memory to a controlled compatibility tier.

### Why this phase exists
This phase exists because the nozzle solver is not a single sparse solve; it is a transient, repeatedly synchronized workflow whose hot objects will be revisited across timesteps, correctors, limiter passes, and subcycles. A memory model that tolerates accidental host touches or lazy migration will look functional during bring-up but will collapse under real nozzle workloads.

The need is not theoretical:

- **Sourced fact**: SPUMA explicitly uses a memory pool manager that leverages unified memory as part of a minimally invasive, incremental GPU-porting strategy.[R1]
- **Sourced fact**: SPUMA also reports that OpenFOAM uses many temporary arrays and that pooling materially reduces runtime; their semi-automatic policy pools `Field`-derived objects automatically and `List` objects manually.[R1]
- **Sourced fact**: SPUMA's wiki warns that unsupported features usually do not fail fast; they instead produce slowdown from undesired data copies between host and device.[R3]
- **Sourced fact**: CUDA's unified-memory advice API notes that preferred location does not migrate data immediately and can allow indefinite thrashing if CPU and GPU keep touching the same pages while preferred location is set to device memory.[R15]
- **Sourced fact**: Nsight Systems documents that managed-memory HtoD events pause kernels and managed-memory DtoH events pause CPU execution while migration occurs.[R19]

The implication is direct:

- **Engineering inference**: a pressure-swirl nozzle port built on top of unrestricted managed memory will be very difficult to profile, because performance problems will present as migration artifacts rather than genuine kernel bottlenecks.
- **Recommendation**: make device residency explicit now, before Phase 5 solver work.

### Entry criteria
The coding agent may start this phase only when all of the following are true:

1. The centralized validation-ladder cases consumed by Phase 2 exist and have reviewed tolerances. For normal Phase 2 gating this means `R2` plus the reduced case selected for the Phase 2 gate (`R1-core` by default; `R1` only when nozzle-specific topology or patch-manifest coverage is intentionally required). `R0` remains archived for later stages.
2. SPUMA builds and runs on the RTX 5080 workstation in at least one supported GPU solver configuration.
3. Nsight Systems can collect CUDA API, NVTX, and UVM page-fault traces on the workstation.
4. NVTX3 wrappers are available in the code base or the migration plan for them is approved.
5. The project has chosen a default execution mode for this phase:
   - `managedBringup`, or
   - `productionDevice`.
6. A senior reviewer has approved that Phase 2 will be allowed to touch:
   - SPUMA memory-pool interfaces,
   - selected OpenFOAM field/list integration points,
   - solver write/output staging hooks.

### Exit criteria
Phase 2 exits only when all mandatory gates below are green:

#### Correctness gates
1. The new memory subsystem builds in all intended configurations:
   - device-production mode,
   - managed-bringup mode,
   - CPU-only compatibility build.
2. Registration and synchronization round-trips preserve field values for:
   - scalar `Field`,
   - vector `Field`,
   - `List<label>`,
   - `DimensionedField`,
   - `GeometricField` internal field,
   - representative patch data.
3. `oldTime()` and `prevIter()` mirrors are preserved correctly for fields marked as requiring them.
4. Write/output generated from staged host buffers matches the CPU reference for the no-op round-trip path, and a write -> reload -> re-register cycle preserves all restart-visible field values, boundary payloads, and required `oldTime()` / `prevIter()` linkage for the selected support-matrix case.

#### Performance-behavior gates
5. After warm-up, the steady-state reduced validation-ladder compute loop selected for Phase 2 (normally `R1-core`; `R1` only when nozzle-specific topology or patch-manifest coverage is explicitly under test) shows:
   - zero recurring CPU and GPU managed-memory page-fault bursts for all registered hot objects,
   - zero recurring managed-memory HtoD/DtoH migration bursts inside compute epochs,
   - zero raw `cudaMalloc*`, `cudaFree*`, or `cudaMallocManaged` calls from hot loop code paths except where explicitly allowed for external libraries.
6. Device-persistent and device-scratch pool high-water marks stabilize after warm-up and do not grow monotonically across identical timesteps.
7. Output staging uses pinned memory and explicit copies only at write boundaries.

#### Operational gates

7. Output and restart/checkpoint staging use pinned memory and explicit copies only at write/checkpoint boundaries.
8. The code can print a residency report listing every registered hot object, its tier, bytes, coherency state, and synchronization reason history, plus a memory-budget report grouped by persistent fields, mesh/topology, patch manifests, matrix/CSR/pattern storage when present, scratch, pinned staging, output buffers, and reserved headroom.
9. Benchmark scripts can fail the run automatically if steady-state UVM migration thresholds are exceeded.
10. A human review signs off on the default configuration values under `gpuRuntime.memory` (`gpuMemoryDict` only as a compatibility shim if still emitted).

### Goals
This phase has five mandatory goals and four secondary goals.

#### Mandatory goals
1. Introduce a **tiered memory taxonomy**:
   - `DevicePersistent`
   - `DeviceScratch`
   - `HostPinnedStage`
   - `ManagedFallback`
   - `HostOnly`
2. Introduce a **residency registry** that tracks:
   - object identity,
   - canonical location,
   - valid mirrors,
   - host/device dirty state,
   - old-time/prev-iteration linkage,
   - allocation tier and bytes.
3. Make all future hot solver state allocatable into explicit device memory without changing solver numerics.
4. Route all host-visible output and diagnostics through pinned staging buffers.
5. Provide profiler-visible instrumentation and pool statistics sufficient to enforce “no steady-state migration” as a pass/fail condition.

#### Secondary goals
6. Reduce allocator churn from OpenFOAM temporary creation.
7. Prepare for CUDA Graph capture by eliminating allocation and migration variability in the timestep body.
8. Preserve compatibility with SPUMA's existing managed-memory bring-up flow.
9. Keep the implementation modular enough that Phase 4 pressure-solver backends and Phase 5 VOF kernels can adopt it without rework.

### Non-goals
Phase 2 explicitly does **not** do the following:

1. It does not port `incompressibleVoF` numerics to custom device kernels.
2. It does not select the final pressure linear solver backend.
3. It does not introduce CUDA Graph execution.
4. It does not solve multi-GPU communication.
5. It does not change data layout from OpenFOAM's native binary field layout to SoA or any other custom kernel layout.
6. It does not globally rewrite every OpenFOAM allocator.
7. It does not guarantee speedup by itself; it guarantees that future speedups will not be hidden behind memory-path pathologies.

### Technical background
Phase 2 sits at the boundary between OpenFOAM semantics and GPU execution reality. The relevant background is not “memory bandwidth” in the abstract; it is the way OpenFOAM allocates, invalidates, and reuses objects.

#### 1. OpenFOAM object behavior that matters here

OpenFOAM makes heavy use of temporary object wrappers (`tmp<T>`), lazy field creation, free-store `List` allocations, and patch methods that return temporary `Field` objects.[R6][R7][R9][R10] `GeometricField` constructors explicitly document constructors that allocate storage for temporary variables.[R7] This is good CPU-side ergonomics. On a GPU code path, it translates into three practical problems:

1. too many small or medium-lived allocations;
2. implicit host ownership assumptions around field access;
3. no built-in concept of device-valid versus host-valid data.

In addition, representative OpenFOAM API documentation notes that host accessors such as `primitiveFieldRef()` update event counters and should be avoided in loops.[R8] That is a signal that even host-side field access has side effects. A GPU port must therefore treat host touches as semantically meaningful, not as harmless reads.

#### 2. SPUMA memory-pool behavior that matters here

SPUMA's paper describes a memory pool layered over unified memory and shows that pooling is a major win because OpenFOAM allocates many temporary arrays.[R1] SPUMA also distinguishes between `Field` objects, which are pooled automatically, and `List` objects, which are only pooled when specified manually.[R1] That distinction matters directly for nozzle work because topology, addressing, boundary maps, and many matrix structures rely heavily on `List`-like containers.

SPUMA additionally publishes several concrete pool implementations through its project material:
- `dummyMemoryPool` as the default/basic option,
- `fixedSizeMemoryPool`,
- `umpireMemoryPool`.[R2]

That is useful context, but it should not be copied blindly. Those pools solve allocation overhead. They do **not** by themselves guarantee device residency or prevent managed-memory migrations.

#### 3. CUDA semantics that matter here

CUDA's stream-ordered allocator and memory-pool API provide explicit pool creation, pool attributes, async allocation/free semantics, and graph-capture behavior.[R14] Those are the right primitives for device-persistent and device-scratch tiers. CUDA's unified-memory APIs provide `cudaMemAdvise` and `cudaMemPrefetchAsync`, but NVIDIA documents clearly that preferred location is advisory, does not migrate data immediately, and can override page-thrash resolution such that CPU/GPU ping-pong continues indefinitely if pages keep being touched from both sides.[R15] Therefore, preferred-location advice is a useful bring-up hint; it is not a production-residency guarantee.

For CPU-visible staging, CUDA documents that true asynchronous host/device copies require host buffers to be page-locked (pinned).[R16] `cudaMallocHost` creates such buffers, and NVIDIA warns that excessive pinned memory harms general system performance; this means pinned staging must be bounded and purpose-specific.[R17]

#### 4. Why this matters specifically for the nozzle solver

`incompressibleVoF`-class solvers expose a field set that includes `alpha`, `alphaPhi`, `rho`, `rhoPhi`, `U`, `p_rgh`, `rAU`, `Uf`, `divU`, and time-history objects such as old-time and previous-iteration state.[R4][R5] Even before custom kernels exist, these objects define the persistent memory footprint that Phase 2 must be able to host on device. If Phase 2 cannot manage these objects explicitly, later phases will be forced back into managed-memory heuristics.

### Research findings relevant to this phase
The following findings are load-bearing.

#### Finding A — SPUMA's current memory strategy is unified-memory-centric
- **Sourced fact**: SPUMA states that its implementation strategy is based on a portable programming model and the adoption of a memory-pool manager that leverages unified memory.[R1]
- **Engineering inference**: the current pool abstraction is a good integration point, but not the final production policy.
- **Recommendation**: keep the abstraction layer; replace the default production allocation policy under it.

#### Finding B — OpenFOAM temporary churn is real and measured
- **Sourced fact**: SPUMA explicitly states that OpenFOAM uses many temporary arrays requiring allocation/deallocation in a short period of time and that memory pools reduce this overhead.[R1]
- **Engineering inference**: the Phase 2 design must separate persistent objects from scratch objects; otherwise either the persistent pool fragments or the scratch path reallocates constantly.
- **Recommendation**: create distinct `DevicePersistent` and `DeviceScratch` tiers.

#### Finding C — `Field` and `List` cannot be treated the same way
- **Sourced fact**: SPUMA's semi-automatic policy allocates `Field` objects in the pool automatically, but `List` objects must be opted in manually.[R1]
- **Engineering inference**: many topology-heavy or patch-heavy objects that matter on GPU are at risk of being left on the wrong tier unless Phase 2 introduces explicit registration.
- **Recommendation**: implement `MirrorTraits` and explicit registration helpers for `List`-derived and matrix-addressing objects.

#### Finding D — Unified memory can degrade badly on discrete GPUs
- **Sourced fact**: SPUMA cites discrete-GPU page-migration overheads exceeding 65% of runtime in an OpenMP/unified-memory OpenFOAM port.[R1]
- **Sourced fact**: Nsight Systems defines UVM HtoD and DtoH events as pauses in GPU or CPU execution while managed-memory migration occurs.[R19]
- **Engineering inference**: recurring UVM traffic after warm-up is a structural defect in the implementation.
- **Recommendation**: Phase 2 acceptance should treat recurring steady-state UVM traffic as a failure.

#### Finding E — `cudaMemAdviseSetPreferredLocation` is not enough
- **Sourced fact**: CUDA documents that preferred location does not immediately migrate data and can allow indefinite host/device thrashing if preferred location is device and both processors keep touching the pages.[R15]
- **Engineering inference**: “managed memory with preferred location = device” is not a safe production substitute for explicit device residency.
- **Recommendation**: use preferred-location advice only in `ManagedFallback`, never as the main production mode.

#### Finding F — Async host/device staging requires pinned memory
- **Sourced fact**: CUDA documents that asynchronous copies involving CPU memory require pinned/page-locked host buffers to remain asynchronous.[R16]
- **Sourced fact**: `cudaMallocHost` allocates page-locked host memory and accelerates `cudaMemcpy*`, but excessive pinned memory degrades system performance.[R17]
- **Engineering inference**: write/output staging needs its own bounded pool and must not be replaced by ad hoc `std::vector` or pageable OpenFOAM buffers.
- **Recommendation**: implement a capped pinned staging allocator.

#### Finding G — CUDA pools are suitable for device-resident tiers
- **Sourced fact**: CUDA runtime docs describe stream-ordered allocation, pool attributes, reserved/used memory statistics, and graph-capture behavior for memory-pool allocations.[R14]
- **Engineering inference**: device pools can serve both long-lived persistent objects and temporary scratch objects, but these two uses have different reuse and trimming behavior.
- **Recommendation**: use separate logical pools or sub-allocators for persistent and scratch tiers.

#### Finding H — Host-pinned CUDA mempool support is real but version documentation is not perfectly consistent
- **Sourced fact**: current runtime/CCCL documentation describes host-pinned memory pools and stream-ordered host pinned allocations.[R18]
- **Sourced fact**: archived CUDA 12.8.1 HTML for `cudaMemPoolCreate` is not fully consistent with current docs regarding `cudaMemLocationTypeHost` versus `HostNuma` pool creation semantics.[R14][R18]
- **Engineering inference**: relying on stream-ordered host-pinned pools as a required Phase 2 feature would add avoidable integration risk.
- **Recommendation**: implement pinned staging first with a simple pooled `cudaMallocHost` wrapper; treat host-pinned mempools as experimental.

#### Finding I — The 5080 is capacity-constrained enough that headroom must be designed explicitly
- **Sourced fact**: the RTX 5080 provides 16 GB device memory and 960 GB/s bandwidth.[R13]
- **Engineering inference**: a careless “pool everything” strategy can starve later external-solver workspaces, output staging, or graph allocations.
- **Recommendation**: reserve explicit VRAM headroom and track high-water marks as a first-class metric.

### Design decisions
Each major decision is documented here with fact, inference, and recommendation.

#### Decision 1 — Define five memory tiers
- **Sourced fact**: SPUMA already distinguishes pool strategies and relies on a memory-pool interface.[R1][R2]
- **Engineering inference**: one generic pool is insufficient because persistent objects, scratch buffers, output staging, and fallback-compatible objects have fundamentally different lifetimes and synchronization rules.
- **Recommendation**: introduce the following tiers:
  - `HostOnly`
  - `ManagedFallback`
  - `DevicePersistent`
  - `DeviceScratch`
  - `HostPinnedStage`

#### Decision 2 — Make `DevicePersistent` the canonical production location for hot objects
- **Sourced fact**: preferred-location managed memory does not guarantee residency and can thrash indefinitely under CPU/GPU ping-pong.[R15]
- **Engineering inference**: production correctness and profiling clarity require a single canonical compute location.
- **Recommendation**: treat the device mirror as canonical for all hot timestep objects in production mode.

#### Decision 3 — Keep OpenFOAM host objects as semantic owners
- **Sourced fact**: OpenFOAM semantics, objectRegistry behavior, and runtime selection are central to solver composition.[R4][R6][R7]
- **Engineering inference**: replacing ownership entirely would destabilize too much code at once.
- **Recommendation**: the host object remains the semantic owner; the registry manages associated device/pinned/fallback allocations and coherency state.

#### Decision 4 — Preserve native OpenFOAM binary field layout in Phase 2
- **Sourced fact**: OpenFOAM fields and lists are contiguous storage types exposed through `Field` / `List` / `UList` APIs.[R10]
- **Engineering inference**: changing data layout at the same time as changing residency would multiply debugging difficulty.
- **Recommendation**: device mirrors use the same element layout as their host storage in Phase 2. Any SoA/AoSoA transformation is deferred.

#### Decision 5 — Use CUDA async device pools for persistent and scratch tiers
- **Sourced fact**: CUDA pools support stream-ordered allocation/free, pool stats, and graph-capture allocation nodes.[R14]
- **Engineering inference**: future graph-capture and low-overhead steady-state execution require pool-backed device allocations rather than repeated raw allocation.
- **Recommendation**: back `DevicePersistent` and `DeviceScratch` with CUDA memory pools, not plain `cudaMallocManaged` or repeated `cudaMalloc`.

#### Decision 6 — Use deterministic reuse defaults
- **Sourced fact**: CUDA pool attributes include reuse controls and release-threshold behavior.[R14]
- **Engineering inference**: opportunistic reuse and internal dependency insertion may maximize reuse but can complicate reproducibility and latency signatures.
- **Recommendation**: default to:
  - `reuseFollowEventDependencies = true`
  - `reuseAllowOpportunistic = false`
  - `reuseAllowInternalDependencies = false`
  - release threshold = high / effectively unlimited during steady-state compute
  - explicit trimming only at safe phase boundaries
This is an implementation recommendation, not a vendor-documented mandatory setting.

#### Decision 7 — Make pinned staging bounded and explicit
- **Sourced fact**: pinned memory is required for asynchronous host/device copies and excessive pinned memory can hurt system performance.[R16][R17]
- **Engineering inference**: staging must be a managed subsystem with accounting, not an ad hoc helper.
- **Recommendation**: introduce a capped pinned-stage pool and forbid pageable host buffers on critical D2H/H2D paths.

#### Decision 8 — Keep a managed-fallback tier for incomplete ports
- **Sourced fact**: SPUMA's unified-memory strategy enables incremental porting and the project still lacks multiphase support today.[R1][R3]
- **Engineering inference**: removing managed behavior entirely would block incremental bring-up and make early solver development brittle.
- **Recommendation**: keep `ManagedFallback`, but isolate it behind configuration and profiler gating.

#### Decision 9 — Introduce a residency registry rather than implicit mirroring
- **Sourced fact**: OpenFOAM host accessors and temporaries have side effects and allocation churn; SPUMA warns that undesired copies otherwise remain silent.[R1][R3][R8]
- **Engineering inference**: hidden mirror creation or ad hoc copies will become untraceable.
- **Recommendation**: every hot object must be registered explicitly and receive a typed residency record.

#### Decision 10 — Separate bring-up path from production path
- **Sourced fact**: SPUMA's current implementation is a development sandbox and future multiphase work is not yet productized.[R1]
- **Engineering inference**: the coding agent needs a safe path to keep development moving while still converging on production requirements.
- **Recommendation**:
  - `managedBringup` = correctness-first, permissive fallback, heavy tracing;
  - `productionDevice` = fail on unexpected steady-state UVM traffic for registered hot objects.

#### What not to do in this phase
The coding agent must **not** do the following:
- Do not treat `cudaMemAdviseSetPreferredLocation(device)` as “mission accomplished.”
- Do not leave output staging on pageable host memory.
- Do not free and reallocate persistent fields every timestep.
- Do not register only `Field` and forget topology-bearing `List` structures.
- Do not silently synchronize or copy fields to host inside compute epochs just because a legacy helper wants them.
- Do not patch every OpenFOAM allocation site globally in the first implementation pass.

### Alternatives considered
This subsection records alternatives that were considered and either rejected or demoted.

#### Alternative A — Keep SPUMA unified memory plus pool as the production model
Rejected.

- **Why attractive**: minimal code churn; compatible with incremental porting.
- **Why rejected**: does not guarantee residency; unsupported code paths silently degrade through copies; UVM events obscure real bottlenecks on discrete GPUs.[R1][R3][R15][R19]

#### Alternative B — Use managed memory everywhere but add preferred-location and prefetch advice
Rejected as production, retained only as fallback.

- **Why attractive**: smaller code delta than explicit mirrors.
- **Why rejected**: CUDA documents that preferred location is advisory and can leave pages thrashing indefinitely if CPU access continues.[R15]

#### Alternative C — Patch every OpenFOAM allocation globally in one pass
Rejected for Phase 2.

- **Why attractive**: maximum coverage.
- **Why rejected**: too invasive for the first implementation; poor failure isolation; hard to prove correctness. Phase 2 instead patches the memory abstraction layer plus explicit registration for hot types.

#### Alternative D — Use Umpire for all new tiers
Rejected as the default, retained as optional support.

- **Why attractive**: SPUMA already integrates Umpire and Umpire supports pool/advice workflows.[R1][R20]
- **Why rejected**: the project needs direct control of CUDA mempool attributes, capture semantics, and pool statistics for device-resident production tiers. Umpire remains useful for managed fallback or portability experiments, but not as the default Phase 2 production backbone.

#### Alternative E — Use host-mapped zero-copy buffers for output and diagnostics
Rejected.

- **Why attractive**: avoids explicit copies.
- **Why rejected**: output and diagnostics are not the steady-state bottleneck, and mapped host memory would create weaker locality and muddier synchronization semantics than explicit pinned staging on a discrete PCIe GPU.

#### Alternative F — Convert all vector/tensor fields to SoA now
Rejected for Phase 2.

- **Why attractive**: later custom kernels may prefer SoA.
- **Why rejected**: changing data layout at the same time as changing residency would significantly increase debugging complexity and invalidate CPU/GPU parity reasoning.

### Interfaces and dependencies
Phase 2 introduces a narrow set of interfaces. These are the interfaces that later phases must depend on; they must not invent ad hoc memory ownership schemes.

#### External dependencies
1. CUDA runtime API:
   - device allocation and memory pools,
   - events and streams,
   - host-pinned allocation,
   - pointer attributes,
   - mem-advise/prefetch for fallback mode.[R14][R15][R17]
2. Optional Umpire integration for managed fallback and legacy SPUMA continuity.[R1][R20]
3. NVTX3 for instrumentation.[R12]
4. Nsight Systems / Nsight Compute / Compute Sanitizer for validation and profiling.[R3][R19]

#### Internal dependencies
1. Existing SPUMA memory-pool abstraction.
2. Existing SPUMA device executor / backend selection framework.[R1][R21]
3. OpenFOAM `Field`, `List`, `GeometricField`, `DimensionedField`, and patch-field semantics.[R6][R7][R9][R10]
4. Later phases:
   - Phase 3 will require pool-backed, capture-friendly allocations;
   - Phase 4 will require persistent matrix and solver workspace residency;
   - Phase 5 will require device-resident VOF field state.

#### Centralized artifacts consumed by Phase 2

Phase 2 does not define these contracts locally; it consumes them.

1. The master pin manifest owns the exact toolchain, driver, and profiler versions.
2. The centralized support matrix and stage plan supply the registered hot set, allowed fallbacks, and any explicitly cold or waived objects.
3. The validation ladder owns which reduced case is the mandatory Phase 2 gate (`R2`, `R1-core`, `R1`, `R0`) and when nozzle-specific coverage is in scope.
4. The GPU operational contract defines the allowed and forbidden host duties for accepted `productionDevice` runs.
5. The acceptance manifest owns formal tolerance classes, required artifacts, and any written waivers.

Phase 2 memory modes are orthogonal to later execution modes (`sync_debug`, `async_no_graph`, `graph_fixed`) and later solver-stage fallback ladders. `managedBringup` and `productionDevice` inherit the global GPU operational contract; they do not define a separate production exception set.

#### Required public interfaces
The new memory layer shall expose the following public API surface to later phases.

```cpp
namespace Foam::gpuMemory
{
    enum class MemoryTier : uint8_t
    {
        HostOnly,
        ManagedFallback,
        DevicePersistent,
        DeviceScratch,
        HostPinnedStage
    };

    enum class CoherencyState : uint8_t
    {
        HostCanonical,      // host copy authoritative
        DeviceCanonical,    // device copy authoritative
        MirroredClean,      // host and device identical
        SnapshotOnly,       // host staging snapshot valid, not canonical
        Invalid
    };

    enum class SyncReason : uint8_t
    {
        StartupUpload,
        ComputeInput,
        WriteOutput,
        RestartCheckpoint,
        ResidualScalar,
        DebugDump,
        LegacyCpuConsumer,
        FallbackRecovery
    };

    struct AllocationHandle
    {
        void* ptr;
        std::size_t bytes;
        std::size_t alignment;
        MemoryTier tier;
        word tag;
        uint64_t generation;
    };

    struct PoolStats
    {
        std::size_t reservedCurrent;
        std::size_t reservedHigh;
        std::size_t usedCurrent;
        std::size_t usedHigh;
        std::size_t liveAllocationCount;
    };

    struct MirrorOptions
    {
        bool includeBoundary;
        bool includeOldTime;
        bool includePrevIter;
        bool allowCpuTouchInCompute;
        bool createPinnedSnapshot;
    };
}
```

#### Required service classes
The implementation shall provide these service classes or direct equivalents:

1. `GpuMemoryRuntime`
   - singleton-like runtime owner for pools, streams, config, and reporting.
2. `DevicePersistentPool`
   - long-lived pool-backed device allocations.
3. `DeviceScratchPool`
   - scratch allocations or named scratch arena with explicit reset semantics.
4. `PinnedStageAllocator`
   - bounded page-locked host staging allocator.
5. `ManagedFallbackAllocator`
   - compatibility allocator using existing SPUMA/Umpire managed path.
6. `ResidencyRegistry`
   - type-erased catalog of mirrored objects and their state.
7. `MirrorTraits<T>`
   - compile-time bridge from OpenFOAM container types to raw storage spans.
8. `FieldMirror<T>`
   - typed helper for device/host mirror management of contiguous field data.
9. `MeshMirror`
   - helper for owner/neighbour and patch-addressing registration.
10. `OutputStager`
    - explicit device->pinned->writer orchestration.
11. `CpuTouchGuard`
    - compute-epoch guard and logging helper.
12. `ScratchCatalog`
    - named scratch buffer planner or registry.

#### Configuration interface

Canonical runtime configuration is rooted under `gpuRuntime.memory`, typically via `system/gpuRuntimeDict`. A standalone `system/gpuMemoryDict` may remain temporarily as a compatibility shim or generated subview during the Phase 2 rollout, but it is not the authoritative contract.

Recommended baseline structure:

```text
gpuRuntime
{
    memory
    {
        mode                        productionDevice;   // managedBringup | productionDevice
        enableManagedFallback       true;
        reserveHeadroom             2GiB;
        pinnedStageCap              512MiB;
        logPoolStats                true;
        logSyncReasons              true;
        failOnSteadyStateUvm        true;
        steadyStateWarmupSteps      3;
        reuseFollowEvents           true;
        reuseOpportunistic          false;
        reuseInternalDependencies   false;
        trimAtWrite                 false;
        trimAtEndRun                true;
        strictCpuTouchGuard         true;
    }
}
```

#### Compatibility contract for later phases

1. It may request a device span or scratch buffer from the memory layer.
2. It may not allocate unmanaged memory inside the timestep hot path without explicit waiver.
3. It must declare when host visibility is required.
4. It must not assume host fields are current if the registry marks them `DeviceCanonical`.
5. It must tag all synchronization reasons.
6. It must treat Phase 2 memory modes as orthogonal to execution modes. Any later stage-level fallback must be explicit, logged, and support-matrix-authorized; in `productionDevice` it may not reintroduce field-scale host evaluation, pinned-host pressure staging, or silent CPU fallback for registered hot objects.

### Data model / memory model
This is the controlling subsection for implementation.

#### 1. Canonical data-residency model

Phase 2 defines **canonicality** separately from **ownership**.

- **Ownership** answers: which C++ object governs lifetime and semantics?
- **Canonicality** answers: where does the latest valid numerical data live?

For production GPU objects:
- the **host OpenFOAM object** remains the semantic owner;
- the **device mirror** is the canonical numerical storage during compute epochs.

This choice allows OpenFOAM semantics to survive while still preventing accidental CPU reads from being mistaken for free operations.

#### 2. Object classes and required tier placement

The coding agent shall classify objects into the following classes.

##### Class A — Immutable or quasi-immutable mesh/topology data
Examples:
- owner/neighbour addressing,
- face/cell counts,
- patch offsets,
- patch face lists,
- cell volumes,
- face area magnitudes,
- geometric factors used every timestep.

Required placement:
- `DevicePersistent` for the device mirror;
- `HostOnly` or stale host read-only copy retained as semantic source.

Allocation timing:
- allocate once after mesh load;
- reallocate only on mesh topology change.

##### Class B — Persistent solver fields
Examples drawn from the intended nozzle path:
- `alpha1`, `alpha2`,
- `alphaPhi1`, `alphaPhi2`,
- `rho`, `rhoPhi`,
- `U`, `phi`, `p_rgh`,
- `rAU`, `Uf`, `divU`, `trDeltaT`,
- old-time and prev-iteration copies where required.[R4][R5]

Required placement:
- canonical `DevicePersistent` in production mode;
- host copies valid only on explicit sync.

Allocation timing:
- once during solver initialization or at first field registration.

##### Class C — Persistent matrix and addressing data
Examples:
- LDU addressing arrays,
- diag/upper/lower/source arrays,
- row maps or CSR maps for later solver integration,
- smoother/solver work buffers that survive across iterations.

Required placement:
- topology metadata in `DevicePersistent`;
- reusable values/workspaces in `DevicePersistent` unless later external library ownership requires otherwise.

##### Class D — Scratch/intermediate buffers
Examples:
- limiter coefficients,
- face-flux intermediate arrays,
- temporary reduction buffers,
- boundary evaluation work arrays,
- temporary patch coefficient arrays,
- future curvature/normal intermediates.

Required placement:
- `DeviceScratch`.

Lifetime:
- one kernel group, one solver stage, or one timestep depending on size and reuse.

##### Class E — Host-consumed scalars and snapshots
Examples:
- convergence scalars,
- write-time snapshots,
- selected debug samples,
- residual histories written to logs.

Required placement:
- `HostPinnedStage`.

##### Class F — Compatibility/fallback objects
Examples:
- not-yet-ported legacy objects,
- transient bridge buffers,
- debug-only deep-copy mirrors.

Required placement:
- `ManagedFallback` only when explicitly allowed by config.


#### 2A. Upstream-supplied hot-object inventory and default registration template

The registered hot set is supplied by the centralized support matrix, validation ladder, and stage plan; this file does not guess it independently. The table below is the default Phase 2 registration template for the algebraic `incompressibleVoF` / explicit MULES / PIMPLE family once those upstream artifacts declare the active objects. It remains intentionally explicit so later phases cannot silently rely on unmanaged objects after the hot-set manifest is frozen.

| Object / group | Semantic owner | Default tier in `productionDevice` | Allocation timing | Canonical during compute? | Host sync allowed for | Reuse scope | Notes |
|---|---|---|---|---|---|---|---|
| `mesh.owner`, `mesh.neighbour`, face-cell addressing, patch offsets | `fvMesh` / addressing containers | `DevicePersistent` | once after mesh creation | yes | startup diagnostics only | entire run | immutable unless topology changes |
| cell volumes, face areas, geometric weights, `deltaCoeffs` | mesh geometry objects | `DevicePersistent` | once after mesh creation | yes | write/debug only | entire run | quasi-immutable on static mesh |
| patch face lists and compacted patch maps | `fvBoundaryMesh`-derived metadata | `DevicePersistent` | once after mesh creation | yes | debug only | entire run | consumed by later BC kernels |
| `alpha1`, `alpha2` | `incompressibleVoF` fields | `DevicePersistent` | register at solver initialization | yes | write, debug, legacy CPU consumer | entire run | primary interface fields.[R4][R5] |
| `alphaPhi1`, `alphaPhi2` | `incompressibleVoF` fields | `DevicePersistent` | register at solver initialization | yes | write, debug | entire run | reused across alpha corrections.[R4] |
| `rho`, `rhoPhi` | mixture-property objects | `DevicePersistent` | register at solver initialization or first use | yes | write, debug | entire run | updated from alpha every timestep |
| `U`, `phi`, `p_rgh` | solver fields | `DevicePersistent` | register at solver initialization | yes | write, debug, selective monitoring | entire run | core momentum/pressure state |
| `rAU`, `Uf`, `divU` | predictor/corrector support fields | `DevicePersistent` | allocate on first creation; retain after first step | yes | write/debug only | many timesteps | `rAU` specifically must not churn.[R4] |
| `trDeltaT` | local-time-step support | `DevicePersistent` when present | on first use if LTS enabled | yes | write/debug only | many timesteps | absent in fixed-`deltaT` runs |
| `tUEqn` storage and matrix values | transient matrix owner | `DevicePersistent` for values/work buffers | first assembly; topology cached thereafter | yes | residual/debug snapshots only | solver stage to full run depending on implementation | Phase 4 refines matrix backend |
| `talphaPhi1Corr0` or equivalent previous-correction buffer | alpha-correction path | `DevicePersistent` if previous-correction path enabled; otherwise `DeviceScratch` | on first alpha correction | yes | debug only | many alpha iterations | keep persistent when `alphaApplyPrevCorr` is enabled.[R4] |
| reduction buffers, limiter coefficients, temporary patch coeffs | scratch producers | `DeviceScratch` | lazily on first request | yes | never directly | one kernel group / one stage / one timestep | pooled and reused |
| write-time field snapshots | `OutputStager` | `HostPinnedStage` | lazily on first write | no | write only | across write events | never canonical during compute |
| legacy not-yet-ported bridges | explicit compatibility wrappers | `ManagedFallback` | only on explicit opt-in | no | legacy CPU consumer | temporary | must disappear before production sign-off |

Defaults that may be tightened later:
- `tUEqn` may move from host-owned semantic object plus device mirror to a more library-owned matrix object in Phase 4, but Phase 2 must still treat its values and work arrays as pooled device allocations.
- Objects not listed here may remain unmanaged only if the centralized support matrix/stage plan and profiler evidence both classify them cold and the residency report documents the waiver explicitly.

#### 3. Residency record

Every registered object shall have a residency record conceptually equivalent to the following:

```cpp
struct ResidencyRecord
{
    word objectName;                 // unique or registry-qualified
    word typeName;                   // e.g. volScalarField, List<label>, scalarField
    MemoryTier preferredTier;
    CoherencyState coherency;
    bool registeredHot;              // participates in steady-state compute
    bool hasBoundaryMirror;
    bool hasOldTimeMirror;
    bool hasPrevIterMirror;
    bool cpuTouchAllowedInCompute;
    label elementCount;
    std::size_t elementBytes;
    std::size_t payloadBytes;
    AllocationHandle deviceHandle;
    AllocationHandle hostPinnedSnapshotHandle;
    void* hostDataPtr;               // semantic owner storage
    uint64_t versionCounterHost;
    uint64_t versionCounterDevice;
    uint64_t lastUploadEpoch;
    uint64_t lastDownloadEpoch;
    SyncReason lastSyncReason;
};
```

Required behavior:
- a record must exist before an object participates in production GPU compute;
- device and host version counters must make stale-copy bugs diagnosable;
- records must be printable in a human-readable report.

#### 4. Binary layout policy

Phase 2 shall preserve the host binary layout in device mirrors:

- `scalarField` -> contiguous `scalar[]`
- `vectorField` -> contiguous `vector[]`
- `tensorField` -> contiguous `tensor[]`
- `List<label>` -> contiguous `label[]`

No structure-of-arrays transform is permitted in Phase 2.

Reason:
- **Engineering inference**: preserving binary layout keeps pointer-span extraction simple, allows host/device round-trip checks, and avoids invalidating OpenFOAM semantics.

#### 5. Coherency-state transitions

The coding agent shall implement the following logical state transitions.

1. `HostCanonical` -> `DeviceCanonical`
   - triggered by explicit upload.
2. `DeviceCanonical` -> `MirroredClean`
   - triggered by explicit D2H synchronization to the semantic host object.
3. `DeviceCanonical` -> `SnapshotOnly`
   - triggered by D2H copy to pinned output buffer without updating semantic host storage.
4. `MirroredClean` -> `DeviceCanonical`
   - after device writes.
5. `MirroredClean` -> `HostCanonical`
   - after host writes.
6. `ManagedFallback`
   - represented as `HostCanonical` or `DeviceCanonical` plus an allocator/tier tag, but never confused with explicit mirrors.

Rules:
- host-side consumers may only assume fresh data after `ensureHostVisible()` returns.
- device-side kernels may only assume fresh data after `ensureDeviceVisible()` or after they are the last writer in the same compute epoch.

#### 6. Old-time and previous-iteration policy

This is numerically load-bearing for transient PIMPLE/VOF paths.

Required policy:
- if a field is marked `includeOldTime`, the registry must either:
  1. recursively register the `oldTime()` field chain as separate residency records, or
  2. manage equivalent device-side shadow storage with explicit linkage metadata.

Default recommendation:
- register old-time and previous-iteration fields as separate residency records because this preserves OpenFOAM semantics and keeps debugging simple.

#### 7. Scratch policy

A future fully optimized planner may compute overlapping scratch lifetimes, but Phase 2 shall implement the safer initial version:

- named scratch buffers with explicit sizes based on mesh counts,
- allocated from `DeviceScratch`,
- reused across iterations and reset at stage boundaries,
- no ad hoc allocation from inside kernels or limiter loops.

This is the **bring-up path** for scratch management.

Production refinement within Phase 2:
- introduce a simple `ScratchCatalog` that precomputes required sizes and either:
  - allocates distinct named buffers, or
  - assigns non-overlapping offsets in one arena.
The agent may choose either approach, but must start with the safer one if uncertain.

#### 8. Memory-budget policy

The RTX 5080 has 16 GB of device memory.[R13] Phase 2 shall therefore keep explicit headroom.

Default recommendation:
- reserve **2 GiB** as untouchable headroom for:
  - driver/runtime overhead,
  - future graph objects,
  - external solver workspaces,
  - temporary profiler overhead.

Budget rule:
- if `cudaMemGetInfo()` reports free memory below the configured headroom after warm-up, the run shall emit a hard error in `productionDevice` mode.

Required artifact:
- each accepted Phase 2 run shall emit a memory budget report with at least these categories:
  - persistent fields,
  - mesh/topology,
  - patch manifests and boundary metadata,
  - matrix/CSR/pattern storage when present,
  - scratch,
  - pinned staging,
  - write/output buffers,
  - reserved headroom.
- where the runtime can measure both committed and high-water marks, it shall print both.

#### 9. Host touch policy

The following is allowed inside compute epochs:
- explicit synchronization of small scalar reductions,
- reading configuration dictionaries,
- logging static metadata.

The following is forbidden inside compute epochs for registered hot objects unless an explicit sync function is called:
- reading internal field arrays,
- boundary patch walks over hot fields,
- write/output serialization,
- debug dumps of full fields.

This policy is essential. If a legacy CPU helper truly needs data, it must call:

```cpp
registry.ensureHostVisible(fieldName, SyncReason::LegacyCpuConsumer);
```

That call must be visible in traces and logs.

#### 10. Bring-up path versus production path

Bring-up path:
- allow `ManagedFallback`;
- allow more explicit host synchronizations;
- warn on UVM events.

Production path:
- registered hot objects must be explicit-mirror objects in `DevicePersistent` or `DeviceScratch`;
- recurring steady-state UVM events for registered hot objects are fatal.

### Algorithms and control flow

This section defines the runtime choreography the coding agent shall implement.

#### 1. Initialization sequence

Initialization shall execute in this order:

1. parse canonical `system/gpuRuntimeDict` and load `gpuRuntime.memory`; if a legacy `system/gpuMemoryDict` compatibility shim is still accepted in this branch, normalize it into the same `gpuRuntime.memory` view before continuing;
2. select device and query:
   - compute capability,
   - memory-pool support,
   - available/free memory,
   - unified-memory/device attributes required for fallback mode;
3. create or attach pools:
   - `DevicePersistentPool`
   - `DeviceScratchPool`
   - `PinnedStageAllocator`
   - `ManagedFallbackAllocator` if enabled;
4. set pool attributes:
   - release threshold high,
   - deterministic reuse flags,
   - statistics zeroed;
5. create the `ResidencyRegistry`;
6. register mesh/topology objects;
7. register persistent fields;
8. perform initial upload/prefetch as required by mode;
9. run a warm-up synchronization and one lightweight validation round-trip;
10. start steady-state compute epochs.

The initialization sequence must be explicit and must emit one structured report at startup listing:
- total free memory before registration,
- bytes committed to persistent objects,
- pinned staging cap,
- managed fallback enabled/disabled,
- headroom reserve.

#### 2. Registration algorithm

Registration is explicit and type-driven.

##### 2.1 Registration of a contiguous object
For a simple contiguous object such as `scalarField`, `vectorField`, or `List<label>`:

1. extract raw host span using `MirrorTraits<T>::hostSpan(obj)`;
2. allocate mirror in requested tier;
3. copy host -> device if initial canonical state is host;
4. create `ResidencyRecord`;
5. mark state:
   - `MirroredClean` after successful upload,
   - or `HostCanonical` if deferred upload is explicitly configured.

##### 2.2 Registration of `GeometricField`
For `GeometricField<Type,...>`:

1. create parent metadata record for the field;
2. register internal field span as child record;
3. if `includeBoundary == true`, enumerate boundary patches and register each patch payload as a child record:
   - `<field>::patch[patchName]`;
4. if `includeOldTime == true`, recursively register `oldTime()` chain;
5. if `includePrevIter == true`, recursively register `prevIter()` link;
6. aggregate bytes and state into the parent record.

Reason for child records:
- patch storage is not one single flat contiguous buffer in standard OpenFOAM semantics;
- later boundary kernels and write paths must be able to stage or invalidate patch data independently.

##### 2.3 Registration of temporary-producing APIs
For APIs that return `tmp<Field<T>>`:
- never auto-register the `tmp` wrapper itself as a persistent object;
- either:
  1. materialize into a named scratch buffer, or
  2. explicitly persist into a registered field if the algorithm needs reuse beyond the current stage.

This is mandatory. Persistent registration of arbitrary `tmp` objects is forbidden because it would scramble object lifetime semantics.

#### 3. Upload algorithm

Default upload path for a host-owned fresh object:

```cpp
void ensureDeviceVisible(ResidencyRecord& rec, cudaStream_t stream)
{
    switch (rec.coherency)
    {
        case CoherencyState::DeviceCanonical:
        case CoherencyState::MirroredClean:
            return;

        case CoherencyState::HostCanonical:
        {
            cudaMemcpyAsync(
                rec.deviceHandle.ptr,
                rec.hostDataPtr,
                rec.payloadBytes,
                cudaMemcpyHostToDevice,
                stream);

            rec.versionCounterDevice = rec.versionCounterHost;
            rec.lastSyncReason = SyncReason::ComputeInput;
            rec.coherency = CoherencyState::MirroredClean;
            rec.lastUploadEpoch = currentEpoch();
            return;
        }

        default:
            FatalErrorInFunction
                << "Invalid coherency state for upload of " << rec.objectName
                << abort(FatalError);
    }
}
```

Rules:
- uploads must use pageable memory only in `ManagedFallback` mode or CPU-only mode;
- production-mode full-field uploads should be from ordinary host storage only at startup or after deliberate host mutation;
- later phases should avoid full-field uploads during steady-state.

#### 4. Device-write invalidation algorithm

Any kernel or device-side routine that mutates a registered object must call the equivalent of:

```cpp
registry.markDeviceWritten(objectName, epochId);
```

Effect:
- state becomes `DeviceCanonical` unless already `DeviceCanonical`;
- host version stays stale;
- last-writer metadata is recorded for debugging.

This call may be folded into higher-level stage APIs if direct per-kernel calls are too noisy, but the semantics must exist.

#### 5. Host-visibility algorithm

A host consumer requiring fresh data shall call:

```cpp
registry.ensureHostVisible(objectName, SyncReason::WriteOutput, stream);
```

Default behavior:
1. if record is `HostCanonical` or `MirroredClean`, no-op;
2. if record is `DeviceCanonical`:
   - allocate or reuse pinned stage buffer,
   - issue D2H copy into pinned buffer,
   - synchronize on event only at the host consumption point,
   - copy pinned bytes into semantic host storage if the host consumer needs the standard OpenFOAM object,
   - mark `MirroredClean`,
   - log the reason;
3. if record is in `ManagedFallback`, optional no-op or explicit prefetch+fence depending on mode.

The two-step D2H path is intentional:
- device -> pinned stage = transfer-optimized boundary crossing,
- pinned stage -> semantic host object = CPU-side materialization.

#### 6. Compute epoch algorithm

A compute epoch is a time interval during which registered hot objects are assumed to be device-canonical or device-updatable and CPU full-field access is forbidden unless explicitly synchronized.

Required API:

```cpp
class ComputeEpochGuard
{
public:
    ComputeEpochGuard(ResidencyRegistry&, const word& epochName);
    ~ComputeEpochGuard();
};
```

Behavior:
- constructor increments epoch counter, enables CPU-touch guard, emits NVTX range;
- destructor disables guard, emits summary stats.

Typical usage:

```cpp
{
    ComputeEpochGuard epoch(registry, "timeStep::pimpleOuter0");
    registry.ensureDeviceVisible("U");
    registry.ensureDeviceVisible("phi");
    registry.ensureDeviceVisible("p_rgh");
    // later phases launch kernels/solves here
}
```

#### 7. Output staging algorithm

Output staging is the first required real host-consumption path.

At a write boundary:
1. collect the list of registered objects that will be written;
2. for each one, call `ensureHostVisible(..., SyncReason::WriteOutput)`;
3. only after all required objects are host-visible, call OpenFOAM write routines.

Recommendation:
- batch D2H copies on a dedicated I/O stream,
- use one event per field or one batched event group,
- overlap staging of multiple fields where possible.

#### 7A. Restart / checkpoint policy

Write-time visibility and restart equivalence are not the same operation.

At a restart/checkpoint boundary:
1. identify the restart-visible object set required by the support matrix and case manifest;
2. call `ensureHostVisible(..., SyncReason::RestartCheckpoint)` for each such record, not merely `SnapshotOnly` staging;
3. materialize the pinned bytes back into the semantic host objects, including boundary child records and any `oldTime()` / `prevIter()` records required for restart;
4. write restart artifacts only after the committed semantic host state is current;
5. on reload, re-register the same object set and verify that resumed state matches an uninterrupted run at the same timestep within the acceptance-manifest tolerances.

Rule:
- `SnapshotOnly` is sufficient for diagnostics and non-restart dumps;
- restart/checkpoint output requires a committed semantic host state for every restart-visible record.

#### 8. Small-scalar reduction algorithm

Not every host-visible value justifies full-field sync. Residuals and diagnostics shall use a scalar staging path:

1. reduction result lands in device buffer,
2. copy `sizeof(scalar)` or small struct to pinned slot,
3. synchronize only on the scalar event,
4. leave field records `DeviceCanonical`.

This distinction is important. Full-field staging for simple logging is prohibited.

#### 9. Warm-up and steady-state transition

The benchmark harness shall explicitly separate:
- timestep warm-up period,
- steady-state measured period.

Rule:
- UVM events observed during registration/startup are informative but not fatal;
- UVM events observed after the configured warm-up step count are fatal in `productionDevice` mode for registered hot objects.

#### 10. Topology-change invalidation path

Even though the first nozzle target uses a static mesh, the memory layer shall include a topology-change invalidation mechanism:

```cpp
registry.invalidateMeshDependentObjects();
```

This must:
- free or retire mesh-dependent device mirrors,
- clear patch and addressing caches,
- force re-registration before next compute epoch.

This is a correctness feature. It may remain unused in the first production target.

### Required source changes
The following source changes are required. They are split into required-now versus allowed-later.

#### Required now
1. Extend SPUMA/OpenFOAM memory-subsystem interfaces to understand explicit tiers.
2. Add CUDA device-pool wrappers.
3. Add pinned-stage allocator.
4. Add managed-fallback adaptor that can reuse the existing SPUMA/Umpire path.
5. Add residency registry and mirror traits.
6. Add field/list/mesh registration helpers.
7. Add explicit host-visibility and device-visibility calls.
8. Add write/output staging helper.
9. Add startup and runtime reporting.
10. Add unit and integration tests.

#### Strongly recommended now
11. Add compute-epoch guards and CPU-touch logging.
12. Add scratch catalog.
13. Add pool-stat query/reporting wrappers around CUDA mempool attributes.[R14]

#### Explicitly deferred
14. Global rewrite of all OpenFOAM allocation sites.
15. SoA field storage.
16. Multi-GPU buffer specialization.
17. Graph-captured allocator nodes in steady-state paths.

#### Existing files to modify
The exact file set depends on the SPUMA branch, but the following modification classes are mandatory:

1. Existing memory pool interface files under `src/OpenFOAM/memoryPool/`
   - extend interface contracts and runtime selection plumbing.
2. Build-system registration under `src/OpenFOAM/Make/files` and any relevant options files.
3. GPU profiling/NVTX wrappers if not already NVTX3.
4. Solver startup or shared initialization utilities so fields and mesh objects can be registered.
5. Time-loop or write helper code so `OutputStager` runs before `runTime.write()` on GPU-aware solver paths.

### Proposed file layout and module boundaries
The proposed file layout intentionally reuses SPUMA's existing `memoryPool` directory and adds a separate residency layer.

```text
src/OpenFOAM/
├── memoryPool/
│   ├── MemoryPool.H                     # existing base interface; extend
│   ├── MemoryPool.C                     # existing
│   ├── dummyMemoryPool.H                # existing managed/basic path
│   ├── dummyMemoryPool.C
│   ├── fixedSizeMemoryPool.H            # existing legacy path
│   ├── fixedSizeMemoryPool.C
│   ├── umpireMemoryPool.H               # existing optional path
│   ├── umpireMemoryPool.C
│   ├── devicePersistentPool.H           # NEW
│   ├── devicePersistentPool.C           # NEW
│   ├── deviceScratchPool.H              # NEW
│   ├── deviceScratchPool.C              # NEW
│   ├── pinnedStageAllocator.H           # NEW
│   ├── pinnedStageAllocator.C           # NEW
│   ├── managedFallbackAllocator.H       # NEW adaptor over legacy path
│   └── managedFallbackAllocator.C       # NEW

├── device/
│   ├── memory/
│   │   ├── gpuMemoryTypes.H             # NEW enums, handles, config structs
│   │   ├── gpuMemoryRuntime.H           # NEW runtime owner
│   │   ├── gpuMemoryRuntime.C
│   │   ├── residencyRegistry.H          # NEW
│   │   ├── residencyRegistry.C
│   │   ├── mirrorTraits.H               # NEW type bridges
│   │   ├── fieldMirror.H                # NEW typed mirror helpers
│   │   ├── fieldMirror.C
│   │   ├── meshMirror.H                 # NEW owner/neighbour/patch helpers
│   │   ├── meshMirror.C
│   │   ├── scratchCatalog.H             # NEW
│   │   ├── scratchCatalog.C
│   │   ├── outputStager.H               # NEW
│   │   ├── outputStager.C
│   │   ├── cpuTouchGuard.H              # NEW
│   │   └── residencyReport.C            # NEW report generation helper
│   └── profiling/
│       ├── nvtx3Ranges.H                # existing or migrated
│       └── nvtx3Ranges.C

├── include/
│   ├── gpuMemoryInit.H                  # NEW startup include helper
│   ├── gpuMemoryWriteSync.H             # NEW write-boundary helper
│   ├── poolOccupancy.H                  # existing; extend or supersede
│   └── poolMaxOccupancy.H               # existing; extend or supersede

tools/
├── profiling/
│   ├── nsys_gpu_memory.sh               # NEW
│   ├── ncu_allocator_check.sh           # NEW
│   └── compute_sanitizer_gpu_memory.sh  # NEW
└── tests/
    ├── test_device_pool.cpp             # NEW
    ├── test_pinned_stage.cpp            # NEW
    ├── test_residency_registry.cpp      # NEW
    ├── test_geometric_field_mirror.cpp  # NEW
    └── test_write_staging.cpp           # NEW

system/
├── gpuRuntimeDict                       # NEW canonical runtime config
└── gpuMemoryDict                        # OPTIONAL compatibility shim / generated subview
```

#### Module-boundary rules
1. Allocators do not know OpenFOAM field semantics.
2. `MirrorTraits` knows how to expose raw spans from OpenFOAM types.
3. `ResidencyRegistry` owns cross-object coherency state.
4. `OutputStager` is the only module allowed to materialize full-field host data for writes in production mode.
5. Solver code later consumes spans and registry calls; it does not call raw CUDA allocation APIs directly.

### Pseudocode

The pseudocode here is intentionally close to implementable C++.

#### 1. Runtime bootstrap

```cpp
namespace Foam::gpuMemory
{

class GpuMemoryRuntime
{
    dictionary cfg_;
    int deviceId_{0};
    bool productionDevice_{false};
    std::size_t reserveHeadroomBytes_{2ull << 30};

    autoPtr<DevicePersistentPool> persistentPool_;
    autoPtr<DeviceScratchPool> scratchPool_;
    autoPtr<PinnedStageAllocator> pinnedStage_;
    autoPtr<ManagedFallbackAllocator> fallbackPool_;
    autoPtr<ResidencyRegistry> registry_;

    cudaStream_t computeStream_{};
    cudaStream_t ioStream_{};

public:
    void initialise(const Time& runTime, const fvMesh& mesh)
    {
        const IOdictionary runtimeDict
        (
            IOobject
            (
                "gpuRuntimeDict",
                runTime.system(),
                runTime,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );

        cfg_ = runtimeDict.subDict("gpuRuntime").subDict("memory");
        // A legacy gpuMemoryDict may still be accepted earlier as a compatibility shim.

        deviceId_ = readLabel(cfg_.lookupOrDefault("deviceId", 0));
        productionDevice_ = cfg_.lookupOrDefault<word>("mode", "productionDevice")
                            == "productionDevice";
        reserveHeadroomBytes_ = readBytes(cfg_.lookup("reserveHeadroom"));

        cudaSetDevice(deviceId_);

        int poolsSupported = 0;
        cudaDeviceGetAttribute(&poolsSupported, cudaDevAttrMemoryPoolsSupported, deviceId_);
        if (!poolsSupported)
        {
            FatalErrorInFunction << "CUDA memory pools not supported" << abort(FatalError);
        }

        cudaStreamCreate(&computeStream_);
        cudaStreamCreate(&ioStream_);

        persistentPool_.reset(new DevicePersistentPool(deviceId_, cfg_.subDict("persistentPool")));
        scratchPool_.reset(new DeviceScratchPool(deviceId_, cfg_.subDict("scratchPool")));
        pinnedStage_.reset(new PinnedStageAllocator(cfg_.subDict("pinnedStage")));
        fallbackPool_.reset(new ManagedFallbackAllocator(cfg_.subDict("fallbackPool")));
        registry_.reset(new ResidencyRegistry(*persistentPool_, *scratchPool_, *pinnedStage_, *fallbackPool_));

        registerMesh(mesh);
        reportStartup();
    }

    void registerMesh(const fvMesh& mesh)
    {
        registry_->registerList("mesh.owner", mesh.owner(), MemoryTier::DevicePersistent);
        registry_->registerList("mesh.neighbour", mesh.neighbour(), MemoryTier::DevicePersistent);
        registry_->registerPatchAddressing(mesh.boundary(), MemoryTier::DevicePersistent);
        registry_->registerGeometricScalars(mesh, MemoryTier::DevicePersistent); // volumes, magSf, etc.
    }

    template<class GeoField>
    void registerPersistentField(GeoField& fld, const MirrorOptions& opts)
    {
        registry_->registerGeometricField(fld.name(), fld, MemoryTier::DevicePersistent, opts, computeStream_);
    }

    void beginComputeEpoch(const word& tag)
    {
        registry_->beginComputeEpoch(tag);
    }

    void endComputeEpoch(const word& tag)
    {
        registry_->endComputeEpoch(tag);
        enforceHeadroom();
    }

    void enforceHeadroom()
    {
        std::size_t freeBytes = 0, totalBytes = 0;
        cudaMemGetInfo(&freeBytes, &totalBytes);

        if (productionDevice_ && freeBytes < reserveHeadroomBytes_)
        {
            FatalErrorInFunction
                << "Device free memory " << freeBytes
                << " below configured headroom " << reserveHeadroomBytes_
                << abort(FatalError);
        }
    }

    ResidencyRegistry& registry() { return *registry_; }
    cudaStream_t computeStream() const { return computeStream_; }
    cudaStream_t ioStream() const { return ioStream_; }
};

} // namespace Foam::gpuMemory
```

#### 2. Device persistent pool

```cpp
class DevicePersistentPool
{
    cudaMemPool_t pool_{};
    PoolStats stats_{};

public:
    DevicePersistentPool(int deviceId, const dictionary& cfg)
    {
        cudaMemPoolProps props{};
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = deviceId;
        props.allocType = cudaMemAllocationTypePinned; // device-local pool semantics
        props.handleTypes = cudaMemHandleTypeNone;
        cudaMemPoolCreate(&pool_, &props);

        uint64_t releaseThreshold = std::numeric_limits<uint64_t>::max();
        cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, &releaseThreshold);

        int follow = 1;
        int opportunistic = 0;
        int internalDeps = 0;
        cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseFollowEventDependencies, &follow);
        cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowOpportunistic, &opportunistic);
        cudaMemPoolSetAttribute(pool_, cudaMemPoolReuseAllowInternalDependencies, &internalDeps);
    }

    AllocationHandle allocate(std::size_t bytes, std::size_t alignment, const word& tag, cudaStream_t stream)
    {
        void* ptr = nullptr;
        cudaMallocFromPoolAsync(&ptr, bytes, pool_, stream);
        return AllocationHandle{ptr, bytes, alignment, MemoryTier::DevicePersistent, tag, nextGeneration()};
    }

    void trim()
    {
        // Persistent pool is not trimmed during steady-state unless explicitly requested.
    }

    PoolStats sampleStats() const
    {
        PoolStats out{};
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemCurrent, &out.reservedCurrent);
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrReservedMemHigh, &out.reservedHigh);
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemCurrent, &out.usedCurrent);
        cudaMemPoolGetAttribute(pool_, cudaMemPoolAttrUsedMemHigh, &out.usedHigh);
        return out;
    }
};
```

#### 3. Pinned stage allocator

```cpp
class PinnedStageAllocator
{
    std::size_t capBytes_{0};
    std::size_t liveBytes_{0};

    struct Block
    {
        void* ptr{nullptr};
        std::size_t bytes{0};
        bool inUse{false};
        word tag;
    };

    DynamicList<Block> blocks_;

public:
    explicit PinnedStageAllocator(const dictionary& cfg)
    :
        capBytes_(readBytes(cfg.lookup("cap")))
    {}

    AllocationHandle acquire(std::size_t bytes, const word& tag)
    {
        for (auto& blk : blocks_)
        {
            if (!blk.inUse && blk.bytes >= bytes)
            {
                blk.inUse = true;
                blk.tag = tag;
                return AllocationHandle{blk.ptr, blk.bytes, alignof(std::max_align_t), MemoryTier::HostPinnedStage, tag, 0};
            }
        }

        if (liveBytes_ + bytes > capBytes_)
        {
            FatalErrorInFunction << "Pinned staging cap exceeded" << abort(FatalError);
        }

        void* ptr = nullptr;
        cudaMallocHost(&ptr, bytes);
        blocks_.append(Block{ptr, bytes, true, tag});
        liveBytes_ += bytes;
        return AllocationHandle{ptr, bytes, alignof(std::max_align_t), MemoryTier::HostPinnedStage, tag, 0};
    }

    void release(void* ptr)
    {
        for (auto& blk : blocks_)
        {
            if (blk.ptr == ptr)
            {
                blk.inUse = false;
                return;
            }
        }
        FatalErrorInFunction << "Unknown pinned block" << abort(FatalError);
    }
};
```

#### 4. Registration of a `GeometricField`

```cpp
template<class GeoField>
FieldId ResidencyRegistry::registerGeometricField
(
    const word& name,
    GeoField& fld,
    MemoryTier tier,
    const MirrorOptions& opts,
    cudaStream_t stream
)
{
    ParentRecord parent(name, GeoField::typeName, tier);

    // Internal field
    auto internalSpan = MirrorTraits<typename GeoField::Internal>::hostSpan(fld.primitiveFieldRef(false));
    auto internalId = registerSpan(name + "::internal", internalSpan, tier, opts, stream);
    parent.childIds.append(internalId);

    // Boundary patches
    if (opts.includeBoundary)
    {
        forAll(fld.boundaryField(), patchi)
        {
            auto& patchFld = fld.boundaryFieldRef()[patchi];
            auto patchSpan = MirrorTraits<std::remove_reference_t<decltype(patchFld)>>::hostSpan(patchFld);
            auto childId = registerSpan
            (
                name + "::patch[" + fld.boundaryField()[patchi].patch().name() + "]",
                patchSpan,
                tier,
                MirrorOptions{false, false, false, opts.allowCpuTouchInCompute, false},
                stream
            );
            parent.childIds.append(childId);
        }
    }

    // oldTime / prevIter
    if (opts.includeOldTime && fld.nOldTimes())
    {
        auto& oldFld = const_cast<GeoField&>(fld.oldTime());
        auto oldId = registerGeometricField(name + "::oldTime", oldFld, tier, optsWithoutHistory(opts), stream);
        parent.childIds.append(oldId);
    }

    if (opts.includePrevIter && fld.prevIter())
    {
        auto& prevFld = const_cast<GeoField&>(fld.prevIter());
        auto prevId = registerGeometricField(name + "::prevIter", prevFld, tier, optsWithoutHistory(opts), stream);
        parent.childIds.append(prevId);
    }

    return storeParent(std::move(parent));
}
```

#### 5. Host-visible write staging

```cpp
template<class GeoField>
void OutputStager::prepareFieldForWrite(GeoField& fld)
{
    auto ids = registry_.childIdsForParent(fld.name());

    for (const auto id : ids)
    {
        auto& rec = registry_.record(id);
        registry_.ensureHostVisible(rec.objectName, SyncReason::WriteOutput, ioStream_);
    }

    cudaEvent_t ready{};
    cudaEventCreate(&ready);
    cudaEventRecord(ready, ioStream_);
    cudaEventSynchronize(ready);

    // semantic host objects are now current and can be written by existing OpenFOAM I/O
}
```

#### 6. Use from a future solver stage

```cpp
void gpuAwareTimeStep(GpuMemoryRuntime& mem, nozzleSolverState& st)
{
    ComputeEpochGuard epoch(mem.registry(), "timeStep");

    mem.registry().ensureDeviceVisible("U::internal", mem.computeStream());
    mem.registry().ensureDeviceVisible("phi::internal", mem.computeStream());
    mem.registry().ensureDeviceVisible("p_rgh::internal", mem.computeStream());

    auto U = mem.registry().deviceSpan<vector>("U::internal");
    auto phi = mem.registry().deviceSpan<scalar>("phi::internal");
    auto p = mem.registry().deviceSpan<scalar>("p_rgh::internal");

    // Later phases launch kernels or invoke external solvers here

    mem.registry().markDeviceWritten("U::internal");
    mem.registry().markDeviceWritten("phi::internal");
    mem.registry().markDeviceWritten("p_rgh::internal");
}
```

#### Numerically important comments
- Host synchronization of fields is explicit because stale host copies must be a visible, diagnosable state.
- `oldTime`/`prevIter` are mirrored separately to avoid silently aliasing current-time storage.
- `tmp<>` wrappers are not persistent device objects.
- Pinned staging is bounded because pinned memory is not free to the system allocator.[R17]

### Step-by-step implementation guide

The sequence below is the intended build order for Phase 2 itself. Each step includes what to modify, why, expected output, how to verify success, and likely breakages.

#### Step 1 — Add `gpuRuntime.memory` config and compatibility parsing
**Modify**
- add the canonical runtime schema rooted at `gpuRuntime.memory` (typically via `system/gpuRuntimeDict`);
- keep `gpuMemoryDict` only as a compatibility shim or generated subview if existing callers still require it;
- add config parsing in `GpuMemoryRuntime`.

**Why**
- configuration must be explicit and reviewable;
- Phase 2 requires a clean distinction between `managedBringup` and `productionDevice`.

**Expected output**
- runtime can print the selected mode, pinned cap, headroom reserve, and fallback settings.

**Verify success**
- run a supported SPUMA solver and confirm startup prints parsed values.

**Likely breakages**
- dictionary file not found;
- units parsing bugs for `MiB`/`GiB`;
- wrong default mode selected.

#### Step 2 — Introduce core enums and handle types
**Modify**
- create `gpuMemoryTypes.H` with `MemoryTier`, `CoherencyState`, `SyncReason`, `AllocationHandle`, `PoolStats`, `MirrorOptions`.

**Why**
- later modules need a common vocabulary.

**Expected output**
- project compiles with new shared types.

**Verify success**
- unit test that headers compile in CUDA and CPU-only modes.

**Likely breakages**
- circular includes;
- namespace pollution.

#### Step 3 — Implement `DevicePersistentPool`
**Modify**
- add `devicePersistentPool.H/C`.

**Why**
- production hot objects require explicit device allocation with accounting.

**Expected output**
- pool creation, allocation, stat sampling, and teardown work.

**Verify success**
- unit test allocates several blocks, queries reserved/used bytes, then frees or tears down cleanly.

**Likely breakages**
- wrong pool props;
- release-threshold misconfiguration;
- runtime errors on unsupported attributes.

#### Step 4 — Implement `DeviceScratchPool`
**Modify**
- add `deviceScratchPool.H/C`.

**Why**
- scratch and persistent lifetimes differ; mixing them obscures leaks and reuse behavior.

**Expected output**
- scratch allocations or arena offsets are reusable across iterations.

**Verify success**
- allocate/free/reset scratch in a loop; `usedHigh` rises once then stabilizes.

**Likely breakages**
- use-after-free if stream ordering is ignored;
- state leaks across resets.

#### Step 5 — Implement `PinnedStageAllocator`
**Modify**
- add `pinnedStageAllocator.H/C`.

**Why**
- output staging and scalar reductions need explicit pinned memory.

**Expected output**
- bounded pinned blocks can be acquired and released.

**Verify success**
- async D2H copy into pinned block works and profiler attributes it correctly.

**Likely breakages**
- cap enforcement bugs;
- pageable fallbacks sneaking in;
- pinned-memory exhaustion on the host.

#### Step 6 — Wrap the managed fallback path
**Modify**
- add `managedFallbackAllocator.H/C`;
- adapt existing SPUMA legacy/Umpire path behind it.

**Why**
- incremental porting still needs a compatibility mode.

**Expected output**
- one API surface can allocate either explicit device memory or managed fallback based on mode/config.

**Verify success**
- a unit test registers the same object in both modes.

**Likely breakages**
- mismatched lifetime semantics between legacy pool and new registry.

#### Step 7 — Implement `ResidencyRegistry`
**Modify**
- add `residencyRegistry.H/C`.

**Why**
- all later correctness depends on authoritative bookkeeping of coherency and tier placement.

**Expected output**
- registry can create, query, print, and invalidate records.

**Verify success**
- unit tests cover state transitions:
  - host->device upload,
  - device write invalidation,
  - device->host sync,
  - snapshot staging,
  - old-time child record linkage.

**Likely breakages**
- stale version counters;
- wrong parent/child record relationships.

#### Step 8 — Implement `MirrorTraits` for contiguous OpenFOAM types
**Modify**
- add `mirrorTraits.H`.

**Why**
- field/list/span extraction must be centralized.

**Expected output**
- compile-time adapters for:
  - `Field<T>`
  - `List<T>`
  - `DimensionedField<T,...>`
  - patch field payloads
  - `GeometricField` internal field helper

**Verify success**
- type-level tests compile and return correct span lengths.

**Likely breakages**
- accidental copies instead of views;
- calling accessors that mutate event counters more than expected.

#### Step 9 — Implement `FieldMirror` and registration helpers
**Modify**
- add `fieldMirror.H/C`, `meshMirror.H/C`.

**Why**
- reduce boilerplate and make field registration uniform.

**Expected output**
- register mesh owner/neighbour lists and one scalar/vector field successfully.

**Verify success**
- startup report lists registered objects and byte counts.

**Likely breakages**
- patch span extraction mistakes;
- lifetime mismatch when patch fields resize.

#### Step 10 — Register immutable mesh and addressing objects at startup
**Modify**
- solver/mesh initialization path.

**Why**
- topology data is hot and quasi-immutable; it should be the easiest first persistent set.

**Expected output**
- `mesh.owner`, `mesh.neighbour`, selected geometry factors, and patch addressing records appear in the registry.

**Verify success**
- no recurring H2D uploads for these objects after startup.

**Likely breakages**
- missing patch addressing lists;
- incorrect byte counts;
- hidden CPU assumptions in boundary utilities.

#### Step 11 — Register persistent solver fields
**Modify**
- GPU-aware solver initialization utilities.

**Why**
- hot fields must be explicit before later kernels/solvers can consume them.

**Expected output**
- `U`, `phi`, `p`, and representative transport fields register with:
  - internal field child record,
  - patch child records,
  - optional old-time linkage.

**Verify success**
- host->device upload occurs once at startup; later compute epochs do not re-upload unless deliberately modified on host.

**Likely breakages**
- old-time recursion bugs;
- prev-iteration not yet allocated when registration is attempted.

#### Step 12 — Add `ensureDeviceVisible()` and `ensureHostVisible()`
**Modify**
- registry methods and call sites.

**Why**
- explicit synchronization points are the entire point of Phase 2.

**Expected output**
- code can stage or upload data by named reason.

**Verify success**
- round-trip tests show correct state transitions and no stale reads.

**Likely breakages**
- forgetting to update version counters;
- performing device copy into wrong child record.

#### Step 13 — Add compute-epoch guards and CPU-touch logging
**Modify**
- `cpuTouchGuard.H`;
- GPU-aware solver driver / shared loop wrappers.

**Why**
- hidden host access during compute is one of the highest-risk failure modes.

**Expected output**
- entering a compute epoch turns on strict mode and logs explicit host-sync reasons.

**Verify success**
- debug run prints or records host-sync reasons.

**Likely breakages**
- false positives in harmless metadata reads;
- noisy logging.

#### Step 14 — Add `OutputStager`
**Modify**
- `outputStager.H/C`;
- write boundary helper include.

**Why**
- write/output is the first mandatory full-field host consumer.

**Expected output**
- before `runTime.write()`, all registered output objects are materialized explicitly.

**Verify success**
- written fields match CPU reference for no-op synchronization path.

**Likely breakages**
- stale patch values in host object;
- host object partially updated;
- D2H on pageable memory due to wrong buffer path.

#### Step 15 — Add `ScratchCatalog`
**Modify**
- `scratchCatalog.H/C`.

**Why**
- hot temporary allocation churn must stop before later phases.

**Expected output**
- named scratch buffers can be acquired by logical name and reset at safe boundaries.

**Verify success**
- identical repeated iterations show stable scratch high-water marks and no new allocations after warm-up.

**Likely breakages**
- wrong buffer sizes for vector/tensor payloads;
- stage overlap bugs if one buffer is reused too early.

#### Step 16 — Add pool-stat and residency reports
**Modify**
- reporting utilities;
- optional extension of `poolOccupancy.H` and `poolMaxOccupancy.H`.

**Why**
- the coding agent and reviewer need objective evidence of residency behavior.

**Expected output**
- machine-readable report per run and human-readable startup/exit summary.

**Verify success**
- reports include each registered object, tier, bytes, and sync counts.

**Likely breakages**
- expensive reporting in hot paths;
- inconsistent bytes due to parent/child double counting.

#### Step 17 — Add unit tests
**Modify**
- `tools/tests/*` or the project's chosen test location.

**Why**
- Phase 2 is stateful infrastructure; unit tests catch most regressions cheaply.

**Expected output**
- tests for allocators, registry state machine, host/device sync, old-time linkage, and write staging.

**Verify success**
- all unit tests pass under normal run and Compute Sanitizer.

**Likely breakages**
- nondeterministic tests if async operations are not fenced correctly.

#### Step 18 — Add integration run on a supported SPUMA solver
**Modify**
- minimal GPU-aware supported solver path, or a small test harness that uses actual SPUMA/OpenFOAM objects.

**Why**
- Phase 2 must prove it works in the real object model, not only in isolated unit tests.

**Expected output**
- one supported GPU solver runs with registered fields and explicit staging.

**Verify success**
- Nsight trace shows startup uploads only, then stable device residency.

**Likely breakages**
- legacy CPU helper unexpectedly touching a registered field.

#### Step 19 — Run reduced validation-ladder benchmark gate
**Modify**
- benchmark scripts only.

**Why**
- the phase is not complete until the reduced validation-ladder case selected for Phase 2 (normally `R1-core`; `R1` only when nozzle-specific coverage is explicitly under test) proves the steady-state memory behavior.

**Expected output**
- run report with UVM bytes, pool stats, and explicit pass/fail decision.

**Verify success**
- steady-state UVM threshold passes.

**Likely breakages**
- hidden fallback path still active;
- output path accidentally triggering D2H every timestep.

### Instrumentation and profiling hooks
Phase 2 instrumentation is mandatory, not optional.

#### NVTX3 ranges
Add NVTX3 ranges around:
- `GpuMemoryRuntime::initialise`
- mesh registration
- field registration
- startup upload
- compute epoch begin/end
- each `ensureHostVisible()` call
- write staging
- scratch reset
- pool trim (if ever used)
- managed fallback allocation paths

#### Runtime counters
Track the following counters:
- `hostToDeviceBytesExplicit`
- `deviceToHostBytesExplicit`
- `managedFallbackBytes`
- `unexpectedHostSyncCount`
- `unexpectedManagedObjectTouchCount`
- `fullFieldHostSyncCountByReason`
- `scalarSyncCountByReason`
- `scratchResetCount`
- `persistentAllocationCount`
- `scratchAllocationCount`

#### Pool statistics
At minimum, query:
- `cudaMemPoolAttrReservedMemCurrent`
- `cudaMemPoolAttrReservedMemHigh`
- `cudaMemPoolAttrUsedMemCurrent`
- `cudaMemPoolAttrUsedMemHigh`.[R14]

#### Profiler scripts
Provide canned commands:

```bash
nsys profile   --trace=cuda,nvtx,osrt   --cuda-um-cpu-page-faults=true   --cuda-um-gpu-page-faults=true   --output=nsys_phase2   <solver> -case <case>
```

This mirrors SPUMA's published recommendation for tracing CPU/GPU faults, with MPI removed for the single-GPU workstation case.[R3]

```bash
ncu --set full --kernel-name-base demangled --target-processes all <solver> -case <case>
```

Use only after the Nsight Systems timeline identifies top targets.

#### Debug modes
Provide three debug levels:
- `0` = production quiet
- `1` = summary reports
- `2` = per-sync logging
- `3` = strict mode with fatal error on any host sync in compute epoch

### Validation strategy

#### 1. Unit validation
Must include:
1. device pool allocate/free/stat test;
2. pinned stage allocation and async copy test;
3. registry coherency-state test;
4. `GeometricField` internal-field registration test;
5. old-time registration test;
6. write staging test.

#### 2. Numerical validation
Because Phase 2 is not supposed to change numerics:
- any host->device->host round-trip on an unchanged field must reproduce identical values bitwise for plain contiguous fields;
- `GeometricField` internal plus boundary round-trips must match exactly in CPU-only no-op tests;
- write-time staged outputs must match the CPU reference for the same timestep.
- a write -> reload -> re-register test at one checkpoint boundary must reproduce the same restart-visible internal fields, boundary data, and time-history linkage as an uninterrupted run at the same timestep, within the acceptance-manifest tolerances.

If exact equality fails, treat it as a bug unless the field type or packing layer makes a proven, documented copy-format conversion unavoidable.

#### 3. Regression validation
Every Phase 2 merge candidate must run:
- one reduced supported GPU solver case,
- one reduced validation-ladder case (`R1-core` by default; `R1` only when nozzle-specific topology or patch-manifest coverage is explicitly under test),
- unit tests,
- one Compute Sanitizer pass on the smallest case.

#### 4. Profiling validation
For `productionDevice` mode after warm-up:
- target threshold: **0 recurring UVM page-fault bursts** for registered hot objects;
- target threshold: **< 1 MiB total managed-memory HtoD + DtoH per timestep** averaged over the measured steady-state window, excluding write or restart/checkpoint steps;
- target threshold: **0 raw `cudaMallocManaged` calls** in the steady-state timestep body for project-owned code;
- target threshold: **0 raw `cudaMalloc`/`cudaFree` from project-owned hot paths** after warm-up;
- target threshold: persistent/scratch pool reserved-high watermark stable within repeated identical timesteps.

These thresholds are recommendations for pass/fail gates. If a human reviewer decides to relax them temporarily, that waiver must be written down.

#### 5. Acceptance of fallback mode
For `managedBringup` mode:
- correctness matters more than migration absence;
- UVM traces are warnings, not failures;
- every managed fallback allocation must be reported.

### Performance expectations
Phase 2 is infrastructure. Performance expectations are therefore behavioral rather than absolute.

#### Required-now expectations
1. allocator overhead should drop relative to repeated unmanaged allocations;
2. write/output should no longer trigger implicit transfers from arbitrary host objects;
3. steady-state memory behavior should become predictable and profiler-readable.

#### Explicit non-promises
- Phase 2 does not promise nozzle speedup by itself.
- Phase 2 may be neutral or slightly negative in wall time before later kernels are ported, especially if extra validation logging is enabled.
- A temporary regression up to roughly 10-15% on a supported single-phase case is tolerable during bring-up **only if** UVM pathologies and allocation churn are demonstrably reduced.

#### Forward-looking expectation
Once later phases consume explicit device spans and stop synchronizing back to host, the Phase 2 architecture should materially reduce:
- managed-memory migration stalls,
- allocator churn,
- write-path disruptions,
- hidden CPU side tails.

### Common failure modes
1. **Steady-state DtoH bursts every timestep**
   - cause: legacy CPU helper reading a registered field.
2. **Steady-state HtoD bursts every timestep**
   - cause: host code mutating a registered field or a managed fallback object sneaking into a kernel path.
3. **Pool stats grow monotonically**
   - cause: leak, forgotten scratch reset, or ever-growing old-time registrations.
4. **Pinned staging cap exceeded**
   - cause: full-field staging of too many objects at once, or leaked pinned blocks.
5. **Stale output**
   - cause: `ensureHostVisible()` updated only internal field but not boundary children, or did not copy staged data back into semantic host storage.
6. **Crash during write**
   - cause: patch child records missing or wrong byte counts.
7. **Nondeterministic latency spikes**
   - cause: opportunistic reuse enabled, hidden synchronizations, or mixed pageable/pinned buffers.
8. **OOM despite low live bytes**
   - cause: headroom not reserved, external library workspaces, or pools not trimmed at end-run.
9. **Bitwise mismatch on round-trip tests**
   - cause: wrong span extraction, wrong element count, or aliased old-time/prev-iteration storage.
10. **Registry says `MirroredClean` but data is stale**
    - cause: device-write invalidation not called after a mutating path.

### Debugging playbook
When behavior is wrong, debug in this order.

1. **Print the residency registry**
   - confirm the object exists;
   - confirm tier and coherency state;
   - confirm bytes and child records.

2. **Check the Nsight Systems timeline**
   - if UVM events appear in steady-state, identify the time and correlate with host logs.[R19]

3. **Check pool stats**
   - if `reservedHigh` grows indefinitely, suspect leaks or missed resets.

4. **Force strict CPU-touch mode**
   - rerun with `strictCpuTouchGuard true`;
   - any host sync inside compute epoch should now be visible or fatal.

5. **Disable managed fallback temporarily**
   - if the run now fails instead of slowing down, an incomplete port path was hiding behind managed memory.

6. **Use `cudaPointerGetAttributes` on suspect pointers**
   - verify whether the pointer is device, managed, or host-pinned.

7. **Run the smallest reproducer under Compute Sanitizer**
   - catch invalid accesses before chasing performance artifacts.

8. **Check old-time linkage explicitly**
   - dump parent/child record tree for one transient field.

9. **Inspect write staging separately**
   - run with output every timestep and then with output disabled;
   - if the issue disappears when output is disabled, the staging path is wrong.

10. **Reduce to one registered field**
    - prove the registry and staging logic with one scalar field, then add complexity back.

### Acceptance checklist
The phase is accepted only when all of the following are true:

- canonical `gpuRuntime.memory` config exists and is documented (`gpuMemoryDict` only if a compatibility shim is still required).
- `DevicePersistentPool`, `DeviceScratchPool`, `PinnedStageAllocator`, and `ManagedFallbackAllocator` build.
- `ResidencyRegistry` prints records for mesh objects and representative fields.
- startup upload works for representative scalar, vector, and label-list objects.
- `ensureDeviceVisible()` and `ensureHostVisible()` pass unit tests.
- `GeometricField` registration covers internal field and patch children.
- old-time and prev-iteration paths are either implemented or explicitly disabled with a fatal error if requested.
- residency report and memory budget report are emitted with categorized bytes/high-water marks.
- output staging, restart/checkpoint commit, and reload parity work and use pinned buffers for field-scale transfers.
- steady-state `productionDevice` run passes the UVM migration gate on the reduced validation-ladder case selected for Phase 2 (normally `R1-core`; `R1` only when nozzle-specific topology is explicitly under test).
- unit tests and Compute Sanitizer pass.
- a human reviewer approves the default configuration and the final profiler trace.

### Future extensions deferred from this phase
1. Stream-ordered host-pinned CUDA mempools once validated on the exact target toolkit/runtime combination.
2. Arena-style overlap-aware scratch packing with lifetime graph coloring.
3. Direct writer support from pinned snapshot buffers without materializing semantic host fields.
4. Multi-GPU communication buffers in pure device memory.
5. SoA/AoSoA field layouts for selected custom kernels.
6. Cross-run persistent cache of matrix topology and external-solver workspaces.
7. Automatic graph-friendly allocation nodes for captured regions.

### Implementation tasks for coding agent
1. Create the new memory-tier types and config parser.
2. Implement the three new allocator classes plus managed-fallback adaptor.
3. Implement the residency registry and mirror traits.
4. Register mesh and representative field types.
5. Implement explicit host/device sync APIs.
6. Implement pinned write staging.
7. Add reporting, tests, and profiler scripts.
8. Run the benchmark gate and archive results.

### Do not start until
- Phase 0 reference cases exist.
- Phase 1 toolchain and profiling stack are working.
- The reviewer has approved the default memory mode and headroom policy.
- The branch policy allows modifications in `src/OpenFOAM/memoryPool/` and GPU-aware write helpers.

### Safe parallelization opportunities
The following work items can proceed in parallel with low merge risk:
- device pool implementation,
- pinned stage allocator implementation,
- registry data-structure work,
- mirror traits for simple contiguous types,
- profiler script creation,
- unit-test scaffolding.

The following should **not** be parallelized aggressively:
- `GeometricField` boundary/old-time integration,
- write/output staging integration,
- any modification that touches existing SPUMA memory-pool base interfaces.

### Governance guardrails
1. `gpuRuntime.memory` defaults are frozen in this package; any change requires an explicit package revision, not an implementation-time local override.
2. Host-pinned mempool productization is deferred. `cudaMallocHost`-backed staging remains the only required Phase 2 path unless a later measured follow-on is approved.
3. The default Phase 2 registered hot set is imported from `reference_case_contract.md`, `support_matrix.md`, and `acceptance_manifest.md`; the default reduced gate is `R2` plus `R1-core`.
4. The default VRAM headroom reserve on the 16 GB RTX 5080 is 2 GiB.
5. Residual steady-state UVM waivers are not part of the baseline plan. Any temporary exception must be admitted centrally through `acceptance_manifest.md`.

### Artifacts to produce
1. Source files listed in the module map.
2. Unit tests and integration harness.
3. `gpuRuntime` memory schema example plus `gpuMemoryDict` compatibility shim only if needed.
4. Startup residency report example.
5. Memory budget report example.
6. Restart/reload parity test report.
7. Nsight Systems trace showing steady-state residency behavior.
8. Benchmark report with pass/fail summary.
9. Reviewer note documenting approved defaults and any waivers.


## Phase 3 — Execution model overhaul

### Purpose
Replace per-kernel synchronization habits with a stream- and graph-friendly execution structure once Phase 2 guarantees stable residency.

### Why this phase exists
Without explicit residency and bounded allocation behavior from Phase 2, execution-model optimization would mostly optimize noise.

### Entry criteria
- Phase 2 production device mode passes residency gates.
- Hot fields, patch maps, and scratch buffers are explicit objects.

### Exit criteria
- Timestep orchestration uses a small, fixed set of streams.
- Synchronization occurs only at deliberate stage boundaries.
- Candidate graph-capture regions are identified and optionally prototyped.

### Goals
- Remove unnecessary device-wide sync points.
- Make timestep control flow capture-friendly.
- Batch/fuse launches where safe.

### Non-goals
- No major numerical redesign.
- No multi-GPU overlap work.

### Technical background
SPUMA profiling shows very high kernel-launch counts and synchronization-heavy API traces in the profiled path.[R1] CUDA graphs and stream-ordered allocation make more sense only after Phase 2 eliminates uncontrolled migration and allocation churn.[R14]

### Research findings relevant to this phase
- **Sourced fact**: SPUMA reported >53k kernel launches in a small profiled window and that every kernel was followed by `cudaDeviceSynchronize()` in that path.[R1]
- **Recommendation**: Phase 3 should remove global syncs only after Phase 2 has made data residency explicit.

### Design decisions
- Keep one primary compute stream and one I/O stream initially.
- Prototype fixed-iteration graph capture before conditional graphs.

### Alternatives considered
Trying to capture graphs before stabilizing memory/state was rejected.

### Interfaces and dependencies
Depends directly on Phase 2 pools, registry, and explicit sync APIs; later benefits Phase 5 hot paths.

### Data model / memory model
No new object classes; the key change is how existing device-resident objects are scheduled and synchronized.

### Algorithms and control flow
Identify launch clusters, replace device-wide sync with event dependencies, then test graph-capture skeletons.

### Required source changes
- Solver driver orchestration.
- NVTX stage ranges.
- Optional graph wrapper classes.

### Proposed file layout and module boundaries
- `src/OpenFOAM/device/execution/graphExecutor.*`
- `src/OpenFOAM/device/execution/streamPlan.*`

### Pseudocode
```text
for timestep:
begin_epoch()
launch_stage_A on computeStream
event record
launch_stage_B waiting on event
if write: stage on ioStream
end_epoch()
```

### Step-by-step implementation guide
1. Measure current sync points.
2. Replace device-wide syncs with event waits.
3. Stabilize stream ownership.
4. Prototype one captured timestep skeleton.
5. Benchmark before wider rollout.

### Instrumentation and profiling hooks
- Nsight Systems timeline and launch counts.
- NVTX per stage.
- Graph-instantiation timing.

### Validation strategy
- Same numerics as pre-Phase-3 path.
- Lower launch/sync overhead.
- No new stale-data bugs.

### Performance expectations
Kernel count and synchronization overhead should drop materially on repeated subcycled paths.

### Common failure modes
- Hidden host syncs reintroduced.
- Capture-incompatible operations in the loop.
- Stream-order use-after-free.

### Debugging playbook
- First run without graphs.
- Then capture a reduced fixed-iteration loop.
- Compare traces side by side.

### Acceptance checklist
- Global sync count sharply reduced.
- Captured prototype runs correctly on R1.
- No residency regressions.

### Future extensions deferred from this phase
- Conditional graph nodes.
- Fused launch groups by solver stage.

### Implementation tasks for coding agent
- Build stream plan.
- Remove syncs.
- Prototype capture.

### Do not start until
- Do not start until Phase 2 steady-state residency is proven.

### Safe parallelization opportunities
- Stream-plan analysis and graph-wrapper prototyping can proceed in parallel.

### Requires human sign-off on
- Which loops are allowed to be graph-captured first.

### Artifacts to produce
- Before/after Nsight traces.
- Graph-capture prototype report.


## Phase 4 — Linear algebra path

### Purpose
Establish the pressure and sparse-algebra path that consumes persistent matrices and values without rebuilding topology unnecessarily.

### Why this phase exists
Pressure remains the first large linear-algebra consumer, but Phase 2 must already have made matrix/addressing residency possible.

### Entry criteria
- Phase 2 registry and pools can host addressing and matrix storage.
- Phase 3 is optional but helpful.

### Exit criteria
- At least one pressure path runs through the chosen GPU algebra backend while preserving a native baseline for comparison.
- Topology/value separation is explicit.

### Goals
- Keep sparse topology persistent.
- Update values without rebuilding mapping when possible.
- Benchmark external and native paths side by side.

### Non-goals
- No assumption that AmgX automatically wins.
- No full multiphase port yet.

### Technical background
The OGL/Ginkgo approach emphasizes persistent matrix-format mappings and updating only coefficients across solves.[R22] SPUMA's paper notes that AmgX hierarchy rebuild cost can matter when coefficients change between iterations.[R1]

### Research findings relevant to this phase
- **Sourced fact**: SPUMA weak-scaling with AmgX is strong, but changing coefficients can erase hierarchy-caching benefits in some settings.[R1]
- **Recommendation**: keep native pressure path as a benchmark baseline while productizing the external path.

### Design decisions
- CSR first for external path.
- Persistent mapping metadata.
- Native baseline always retained.

### Alternatives considered
Plugin-only linear solve as the whole architecture was already rejected at the project level.

### Interfaces and dependencies
Depends on Phase 2 ability to keep addressing and matrix values resident.

### Data model / memory model
LDU/CSR maps, diag/lower/upper/source arrays, and solver workspaces become `DevicePersistent` objects with values updated in place.

### Algorithms and control flow
Build topology once, update values each solve, invoke backend, compare against native path.

### Required source changes
- Matrix export/mapping code.
- Backend interface adapters.
- Solver selection dictionary entries.

### Proposed file layout and module boundaries
- `src/OpenFOAM/device/linalg/lduCsrMap.*`
- `src/OpenFOAM/device/linalg/gpuLinearSolver.*`

### Pseudocode
```text
if mapping not built:
build_ldu_to_csr_mapping()
update_matrix_values_only()
solve_pressure()
```

### Step-by-step implementation guide
1. Persist addressing.
2. Build mapping.
3. Update values in place.
4. Integrate backend.
5. Benchmark against native path.

### Instrumentation and profiling hooks
- Time mapping separately from solve.
- Track workspace bytes.
- Track hierarchy rebuilds if external AMG is used.

### Validation strategy
- Pressure residual convergence.
- Field equivalence within tolerance.
- Mapping reuse across steps.

### Performance expectations
Matrix rebuild cost should amortize away; pressure path should become stable enough to compare backends fairly.

### Common failure modes
- Rebuilding topology every iteration.
- Workspace allocations in hot path.
- Stale coefficient uploads.

### Debugging playbook
- Dump row counts and nnz.
- Verify only values change across timesteps.
- Compare native versus external residual histories.

### Acceptance checklist
- Persistent mapping proven.
- Native and external baselines benchmarked.
- No hidden residency regressions.

### Future extensions deferred from this phase
- Alternative sparse formats for native kernels.
- Backend-specific workspace tuning.

### Implementation tasks for coding agent
- Build mapping layer.
- Integrate backend.
- Add comparison harness.

### Do not start until
- Do not start until Phase 2 handles matrix/addressing residency.

### Safe parallelization opportunities
- Mapping implementation and backend adaptor can proceed in parallel.

### Requires human sign-off on
- Which pressure backend is the first production candidate.

### Artifacts to produce
- Mapping tests.
- Solver benchmark report.
- Residual comparison report.


## Phase 5 — Port the VOF core

### Purpose
Port the algebraic VOF/MULES/two-phase transient path to explicit device execution on top of the Phase 2 memory layer.

### Why this phase exists
This is the first solver phase that actually makes the nozzle application real rather than preparatory.

### Entry criteria
- Phase 2 residency gates pass.
- Pressure path baseline exists.
- R2 verification case is frozen.

### Exit criteria
- Alpha transport, mixture-property updates, momentum predictor support fields, and pressure-correction data path remain on device for the target solver slice.

### Goals
- Preserve existing numerics first.
- Keep alpha, `alphaPhi`, `rho`, `rhoPhi`, `U`, `p_rgh`, `rAU`, and related fields resident.
- Avoid host round trips across subcycles.

### Non-goals
- No geometric VOF rewrite.
- No algorithmic limiter redesign.

### Technical background
`incompressibleVoF` uses a single mixture momentum equation with PIMPLE coupling and explicit VOF subcycling/MULES behavior.[R4][R5] SPUMA does not currently support multiphase solvers, so this phase is genuine new development.[R1]

### Research findings relevant to this phase
- **Sourced fact**: MULES and alpha subcycling create repeated explicit correction work, not just one sparse solve.[R4][R5]
- **Recommendation**: port existing numerics unchanged before any optimization or kernel fusion.

### Design decisions
- Device namespaces for alpha, MULES, surface tension, and pressure-corrector helpers.
- Reuse Phase 2 scratch and residency APIs.

### Alternatives considered
Jumping directly to geometric VOF or aggressive fused kernels was rejected for the first solver port.

### Interfaces and dependencies
Depends directly on Phases 2-4.

### Data model / memory model
Persistent VOF field set becomes `DevicePersistent`; limiter and correction intermediates use `DeviceScratch`.

### Algorithms and control flow
Mirror fields, run explicit alpha path on device, update mixture properties, assemble momentum/pressure support data, keep host syncs out of the timestep body.

### Required source changes
- New device-side VOF modules.
- Solver integration points.
- Field registration on solver startup.

### Proposed file layout and module boundaries
- `src/OpenFOAM/device/vof/deviceAlphaPredictor.*`
- `src/OpenFOAM/device/vof/deviceMULES.*`
- `src/OpenFOAM/device/vof/deviceSurfaceTension.*`

### Pseudocode
```text
begin_epoch()
alpha_predictor_device()
mixture_update_device()
momentum_support_update_device()
pressure_support_update_device()
end_epoch()
```

### Step-by-step implementation guide
1. Register VOF fields.
2. Port alpha transport unchanged numerically.
3. Port mixture updates.
4. Port reused predictor/corrector support fields.
5. Validate on R2 then R1.

### Instrumentation and profiling hooks
- NVTX per alpha subcycle.
- Scratch usage per limiter stage.
- Host-sync count must remain zero inside subcycles.

### Validation strategy
- Mass conservation.
- Bounded alpha.
- CPU/GPU field comparison by timestep.
- Subcycle-by-subcycle regression checks.

### Performance expectations
Subcycling should no longer amplify host/device pathologies once Phase 2 is in place.

### Common failure modes
- Boundary child records forgotten.
- Scratch under-sizing.
- Host syncs hidden in limiter helpers.

### Debugging playbook
- Validate one alpha correction at a time.
- Compare field snapshots at every subcycle.
- Run reduced two-phase case first.

### Acceptance checklist
- R2 passes boundedness/mass checks.
- R1 runs with device-resident alpha path.
- No steady-state UVM faults in alpha subcycles.

### Future extensions deferred from this phase
- Kernel fusion.
- More aggressive limiter optimization.
- Geometric-VOF experiments.

### Implementation tasks for coding agent
- Port alpha path.
- Port mixture updates.
- Add VOF regression harness.

### Do not start until
- Do not start until Phase 2 and at least baseline Phase 4 pressure support are stable.

### Safe parallelization opportunities
- Mixture-update work and alpha-regression harness can proceed in parallel.

### Requires human sign-off on
- Exact solver slice to port first.
- Acceptance tolerances for VOF fields.

### Artifacts to produce
- R2 validation report.
- R1 alpha-path trace.
- Field comparison snapshots.


## Phase 6 — Nozzle-specific kernels and boundary conditions

### Purpose
Implement pressure-driven inlet, swirl patch groups, startup seeding, and near-field plume boundary logic in a device-resident form.

### Why this phase exists
Generic VOF validation does not exercise the nozzle's boundary-condition and startup complexity.

### Entry criteria
- Phase 5 core VOF path is stable on R2 and at least partially on R1.

### Exit criteria
- Nozzle-specific BCs and startup seeding execute without per-timestep host bounce for registered hot fields.

### Goals
- Compact patch data.
- Make BC evaluation device-friendly.
- Keep startup conditioning in the GPU execution model.

### Non-goals
- No generalized BC framework rewrite for all OpenFOAM patches.

### Technical background
Patch-field APIs commonly return temporaries and rely on host-oriented semantics.[R9] Nozzle-specific swirl and pressure BCs will otherwise become a CPU tail.

### Research findings relevant to this phase
- **Engineering inference**: patch-compacted kernels are the safest way to prevent nozzle BCs from undoing Phase 2 residency gains.
- **Recommendation**: precompute contiguous patch maps and evaluate by patch family, not by scattered host callbacks.

### Design decisions
- Patch-compacted device kernels.
- Startup seeding treated as part of solver initialization, not an external afterthought.

### Alternatives considered
Leaving nozzle BCs on the CPU was rejected because it would reintroduce steady-state D2H/H2D traffic.

### Interfaces and dependencies
Depends on Phase 2 patch child records and Phase 5 field residency.

### Data model / memory model
Patch maps and BC coefficient data become `DevicePersistent`; transient BC work arrays use `DeviceScratch`.

### Algorithms and control flow
Precompute patch groups, upload maps, run device BC evaluators, and stage to host only for write/debug.

### Required source changes
- Nozzle BC modules.
- Startup seeding helpers.
- Patch-map generation utilities.

### Proposed file layout and module boundaries
- `src/OpenFOAM/device/nozzle/deviceNozzleBC.*`
- `src/OpenFOAM/device/nozzle/deviceStartupSeed.*`

### Pseudocode
```text
build_patch_groups_once()
for timestep:
apply_nozzle_bc_device()
if startup_window: apply_seed_updates_device()
```

### Step-by-step implementation guide
1. Compact patch maps.
2. Port inlet/wall/far-field BC families.
3. Integrate startup seeding.
4. Validate against CPU nozzle startup.

### Instrumentation and profiling hooks
- NVTX per patch family.
- Patch-face counts.
- Host-sync reason logging must stay empty inside BC update.

### Validation strategy
- Patch-value parity.
- Startup reproducibility.
- Nozzle integral metrics.

### Performance expectations
Boundary handling should stop showing up as a host-side tail.

### Common failure modes
- Wrong patch grouping.
- Patch-child mirror mismatch.
- Startup seed not mirrored across old-time state.

### Debugging playbook
- Validate one patch family at a time.
- Dump patch maps and compare counts.
- Compare startup field snapshots.

### Acceptance checklist
- BC path device-resident.
- Startup seeding reproducible.
- No steady-state BC-induced UVM traffic.

### Future extensions deferred from this phase
- More generic BC code generation.
- BC fusion with neighboring kernels.

### Implementation tasks for coding agent
- Build patch compaction.
- Port nozzle BCs.
- Add startup validation.

### Do not start until
- Do not start until Phase 5 field residency is working for the relevant nozzle fields.

### Safe parallelization opportunities
- Patch-map generation and startup-validation harness can proceed in parallel.

### Requires human sign-off on
- Which BC families are mandatory for milestone 1.

### Artifacts to produce
- Patch-map dump.
- BC validation report.
- Startup comparison plots.


## Phase 7 — Custom CUDA kernels where actually needed

### Purpose
Write bespoke kernels only for the hotspots that library backends and generic abstractions do not solve well.

### Why this phase exists
The nozzle workflow contains irregular, atomics-heavy, and limiter-heavy work that is not reducible to sparse solves alone.

### Entry criteria
- Profiling identifies stable hotspots after Phases 2-6.
- Generic/device-resident implementations already exist and are correct.

### Exit criteria
- A small, justified set of custom kernels replaces the worst hotspots and improves end-to-end performance without reducing correctness confidence.

### Goals
- Target alpha/MULES, irregular face loops, curvature, patch evaluation, and fused updates only where profiling proves need.
- Preserve device residency and explicit scratch use.

### Non-goals
- No “CUDA everywhere” rewrite.
- No speculative micro-optimization without profiler evidence.

### Technical background
SPUMA profiling already shows gradient/RHS kernels with lower efficiency due in part to atomics, and pressure-GAMG paths with lower efficiency than the best assembly sections.[R1] The sparse-kernel literature also shows format and access-pattern sensitivity for SpMV and irregular kernels.[R22]

### Research findings relevant to this phase
- **Sourced fact**: SPUMA observed weaker sections around atomic-heavy gradient work and pressure solve efficiency.[R1]
- **Recommendation**: only custom-kernel the measured offenders after the memory/execution substrate is stable.

### Design decisions
- Libraries first for standard sparse linear algebra.
- Custom kernels only for irregular CFD-specific hotspots.

### Alternatives considered
Premature custom-kernel work before Phase 2 stability was rejected.

### Interfaces and dependencies
Depends on Phases 2, 3, 5, and 6 plus fresh profiling evidence.

### Data model / memory model
Hot fields remain in `DevicePersistent`; fused-kernel intermediates should shrink `DeviceScratch` usage rather than expand it uncontrollably.

### Algorithms and control flow
Profile, rank hotspots, replace one hotspot at a time with a controlled custom kernel, validate, then keep or rollback.

### Required source changes
- New CUDA kernels and launch wrappers.
- Potential kernel-fused update modules.

### Proposed file layout and module boundaries
- `src/OpenFOAM/device/kernels/alphaLimiterKernels.*`
- `src/OpenFOAM/device/kernels/curvatureKernels.*`

### Pseudocode
```text
hotspots = profile_top_hotspots()
for h in hotspots:
if library_not_sufficient(h):
    implement_custom_kernel(h)
    validate()
    benchmark()
```

### Step-by-step implementation guide
1. Freeze hotspot list.
2. Replace one kernel group at a time.
3. Validate numerics after each replacement.
4. Keep rollback path.

### Instrumentation and profiling hooks
- Nsight Compute on top kernels only.
- Compare atomics, memory transactions, and occupancy before/after.

### Validation strategy
- Per-kernel correctness harness.
- End-to-end regression after each hotspot replacement.

### Performance expectations
Only hotspots with measurable end-to-end impact should be kept.

### Common failure modes
- Faster kernel but slower timestep due to extra sync or scratch use.
- Numerical drift from fused updates.
- Register/shared-memory pressure reducing occupancy excessively.

### Debugging playbook
- Keep old kernel path compile-selectable.
- Compare per-stage timings before and after.
- Use tiny reproducible cases.

### Acceptance checklist
- Each custom kernel improves the full timestep or removes a known bottleneck.
- Rollback path remains available.

### Future extensions deferred from this phase
- Architecture-specific tuning passes.
- Warp-specialized or cluster-aware kernels on Blackwell if justified.

### Implementation tasks for coding agent
- Rank hotspots.
- Implement one custom kernel at a time.
- Benchmark and keep/rollback.

### Do not start until
- Do not start until correctness and residency are stable and profiling identifies genuine hotspots.

### Safe parallelization opportunities
- Separate hotspot prototypes can be explored in parallel if they touch different modules.

### Requires human sign-off on
- Which hotspots justify custom kernels.
- Rollback threshold for keeping a new kernel.

### Artifacts to produce
- Kernel benchmark notes.
- Before/after Nsight Compute reports.
- Rollback/keep decisions.


## Phase 8 — Profiling and performance acceptance

### Purpose
Define the production measurement regime and hard acceptance gates for memory behavior, correctness, and end-to-end runtime.

### Why this phase exists
Without explicit gates, the project will drift back toward hidden fallback paths and anecdotal optimization.

### Entry criteria
- Earlier phases are implemented enough to run R1 and at least one full nozzle-class case.

### Exit criteria
- The project has a reproducible benchmark suite, profiler scripts, acceptance thresholds, and archived baseline traces for future regression detection.

### Goals
- Make performance claims reproducible.
- Separate correctness regressions from performance regressions.
- Preserve visibility into UVM, launch, sync, and allocator behavior.

### Non-goals
- No new solver features.
- No optimization without measurement.

### Technical background
SPUMA already recommends `nsys` plus CPU/GPU unified-memory page-fault tracing.[R3] CUDA/Nsight documentation gives concrete semantics for interpreting UVM events.[R19]

### Research findings relevant to this phase
- **Sourced fact**: Nsight Systems and SPUMA guidance together support a direct UVM-fault tracing methodology.[R3][R19]
- **Recommendation**: make UVM behavior, pool stats, and write-stage copies part of the standard benchmark output.

### Design decisions
- `nsys` for timeline and UVM.
- `ncu` only for top kernels.
- Compute Sanitizer in reduced-case CI.
- NVTX3 ranges everywhere relevant.

### Alternatives considered
A wall-clock-only benchmark was rejected because it cannot distinguish migration, launch overhead, and true kernel inefficiency.

### Interfaces and dependencies
Builds on all earlier phases; especially Phase 2 instrumentation.

### Data model / memory model
Benchmark artifacts include timings, UVM bytes, pool stats, sync counts, and selected numerical metrics.

### Algorithms and control flow
Run warm-up, then steady-state window, collect metrics, compare against thresholds, and archive artifacts.

### Required source changes
- Benchmark harness scripts.
- Report generator.
- CI job definitions if present.

### Proposed file layout and module boundaries
- `tools/benchmark/run_phase_benchmarks.py`
- `tools/benchmark/report_phase_benchmarks.py`

### Pseudocode
```text
warmup(case)
measure(case, window=N)
collect_timeline()
collect_pool_stats()
compare_against_thresholds()
archive_artifacts()
```

### Step-by-step implementation guide
1. Define benchmark windows.
2. Automate profiler collection.
3. Parse UVM and pool metrics.
4. Compare to thresholds.
5. Archive traces and reports.

### Instrumentation and profiling hooks
- `nsys`, `ncu`, NVTX3, sanitizer, runtime counters.

### Validation strategy
- Correctness metrics and performance metrics reported together.
- Threshold breaches fail the run.

### Performance expectations
This phase sets the rules by which performance success is judged.

### Common failure modes
- Unrepeatable measurements.
- Output/write pollution in steady-state window.
- Thresholds too loose to catch regressions.

### Debugging playbook
- Re-run with output disabled.
- Check warm-up window.
- Compare against previous archived traces.

### Acceptance checklist
- Benchmark framework produces stable reports.
- Thresholds are reviewed.
- Baseline traces archived.

### Future extensions deferred from this phase
- Dashboarding.
- Larger DOE automation.
- Power/energy measurements.

### Implementation tasks for coding agent
- Build benchmark harness.
- Parse profiler outputs.
- Archive baselines.

### Do not start until
- Do not start until Phase 2 instrumentation exists and at least one end-to-end run works.

### Safe parallelization opportunities
- Script automation and report rendering can proceed in parallel.

### Requires human sign-off on
- Final pass/fail thresholds and benchmark windows.

### Artifacts to produce
- Benchmark report.
- Archived traces.
- Threshold configuration.


# 6. Validation and benchmarking framework

This section defines the global framework that all phases, especially Phase 2, must use.

## 6.1 Test categories

### A. Unit tests
Must run on every relevant build configuration:
- CPU-only compatibility build,
- managed-bringup GPU build,
- production-device GPU build.

Required unit-test families:
1. allocator creation / destruction;
2. async allocation ordering;
3. pinned stage acquire/release;
4. span extraction for `Field` / `List`;
5. registry state transitions;
6. `GeometricField` internal/boundary child registration;
7. old-time / prev-iteration linkage;
8. write staging.

### B. Integration tests
Required integration levels:
1. one supported SPUMA GPU solver with explicit field registration;
2. one reduced validation-ladder case (`R1-core` by default; `R1` only when nozzle-specific topology or patch-manifest coverage is explicitly under test);
3. one two-phase verification harness (R2) once Phase 5 begins.

### C. Numerical regression tests
For each run, archive:
- selected field norms,
- integral mass/momentum-like quantities,
- pressure drop / discharge proxy metrics,
- alpha boundedness metrics when applicable,
- write-time field snapshots at chosen checkpoints.

### D. Memory-behavior regression tests
Required metrics:
- explicit H2D bytes,
- explicit D2H bytes,
- UVM page-fault counts,
- UVM HtoD / DtoH migration bytes,
- pool reserved/used high-water marks,
- unexpected host-sync counts,
- managed-fallback allocation counts.

## 6.2 Standard benchmark protocol

Every benchmark run shall follow this protocol:

1. build commit hash recorded;
2. config and case manifest recorded;
3. one warm-up window not counted in final metrics;
4. one steady-state measurement window;
5. optional write-disabled repeat to isolate I/O effects;
6. profiler and runtime reports archived together.

Recommended windows:
- R1-core reduced case: warm-up 3 timesteps, measure 10-20 timesteps;
- R0 full case: warm-up 3-5 timesteps, measure at least 5 representative timesteps.

## 6.3 Global pass/fail gates

### Mandatory correctness gates
- No unit-test failures.
- No integration-test failures.
- No stale-data, write-mismatch, or restart/reload parity failures.
- No sanitizer-reported invalid accesses in reduced tests.

### Mandatory Phase 2 memory gates
For `productionDevice` mode after warm-up:
- zero recurring UVM-fault bursts for registered hot objects;
- less than 1 MiB/timestep managed-memory migration averaged over steady-state, excluding write and restart/checkpoint boundaries;
- zero project-owned raw allocation calls in the steady-state hot path;
- stable pool watermarks across repeated equivalent timesteps.

### Conditional waiver policy
Any waiver of the above gates must be:
- written down,
- time-limited,
- tied to a concrete follow-up task.

## 6.4 Benchmark artifacts

Each accepted benchmark drop shall include:
1. solver log;
2. config files;
3. startup residency report;
4. runtime pool-stat report and memory-budget report;
5. Nsight Systems trace;
6. optional Nsight Compute report for top kernels;
7. comparison report versus previous accepted baseline.

# 7. Toolchain / environment specification

## 7.1 Mandatory environment
- Target platform: Linux x86_64 workstation.
- GPU: RTX 5080, compute capability 12.0, 16 GB, 960 GB/s.[R13]
- Toolchain, driver, and profiler versions are inherited from the master pin manifest frozen after Phase 1. At the current continuity freeze the default pins are CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, and NVTX3.
- Profilers: Nsight Systems and Nsight Compute.
- Debugging: Compute Sanitizer.
- Instrumentation: NVTX3 only.

## 7.2 Build requirements
1. Include PTX in the build and validate with `CUDA_FORCE_PTX_JIT=1` at least once per environment.[R11]
2. Build native Blackwell code generation for compute capability 12.0 when the chosen compiler/toolkit combination supports it.[R12][R13]
3. Do not change the SPUMA/OpenFOAM branch's baseline C++ standard in Phase 2; inherit it.
4. Treat Umpire as optional, not mandatory.[R1][R20]

Recommended CUDA code-generation policy:
- native code for Blackwell (`sm_120` when supported by the selected toolkit/compiler),
- plus PTX for forward compatibility / JIT verification.[R11][R12]

## 7.3 Runtime requirements
1. `cudaDevAttrMemoryPoolsSupported` must be true.
2. The runtime must be able to create at least:
   - one compute stream,
   - one I/O stream.
3. Startup must query free/total memory and print it.
4. Managed-fallback mode must verify required unified-memory attributes before enabling preferred-location advice.[R15]

## 7.4 Recommended developer commands

### PTX/JIT check
```bash
CUDA_FORCE_PTX_JIT=1 <solver> -case <case>
```

### Nsight Systems UVM trace
```bash
nsys profile   --trace=cuda,nvtx,osrt   --cuda-um-cpu-page-faults=true   --cuda-um-gpu-page-faults=true   --output=phase2_uvm   <solver> -case <case>
```

### Compute Sanitizer
```bash
compute-sanitizer --tool memcheck <solver> -case <small-case>
```

## 7.5 Environment watch-outs
- Archived and current CUDA docs are not perfectly consistent on host-pinned mempool details; do not make that path mandatory without local validation.[R14][R18]
- Excessive pinned memory can harm workstation usability; keep caps explicit.[R17]
- Blackwell tuning guidance still emphasizes classic best practices: minimize transfers, coalesce accesses, avoid redundant global-memory traffic, and control divergence.[R23]

# 8. Module / file / ownership map

This map is role-based, not people-based. One engineer may own multiple roles. Within this file, the Phase 2 rows are normative; rows for Phases 3-8 are retained only to show where the memory layer hands off to the master roadmap and support matrix.

| Module / file group | Role owner | Responsibility | Depends on | Phase priority |
|---|---|---|---|---|
| `src/OpenFOAM/memoryPool/*` legacy files | GPU memory core owner | Preserve compatibility and expose a common allocator abstraction | Existing SPUMA code | Phase 2 |
| `devicePersistentPool.*` | GPU memory core owner | Long-lived device allocations, stats, trimming policy | CUDA mempools | Phase 2 |
| `deviceScratchPool.*` | GPU memory core owner | Scratch lifetime management | CUDA mempools | Phase 2 |
| `pinnedStageAllocator.*` | GPU memory core owner | Page-locked host staging | CUDA host-pinned APIs | Phase 2 |
| `managedFallbackAllocator.*` | Compatibility owner | Managed-mode bridge over legacy/Umpire path | SPUMA/Umpire | Phase 2 |
| `residencyRegistry.*` | Residency owner | Canonical object state, sync reasons, reporting | All allocators | Phase 2 |
| `mirrorTraits.*` / `fieldMirror.*` / `meshMirror.*` | OpenFOAM integration owner | Span extraction and registration of OpenFOAM objects | OpenFOAM field/list semantics | Phase 2 |
| `outputStager.*` | I/O integration owner | Write-boundary materialization and pinned staging | Registry + pinned stage | Phase 2 |
| `cpuTouchGuard.*` | Debug/profiling owner | Detect or log host access during compute epochs | Registry + NVTX | Phase 2 |
| `scratchCatalog.*` | GPU memory core owner | Named scratch planning | Scratch pool + solver needs | Phase 2 |
| `device/execution/*` | Execution-model owner | Stream plan and graph wrappers | Phase 2 stable memory | Phase 3 |
| `device/linalg/*` | Linear-algebra owner | Matrix mapping and backend integration | Phase 2 residency | Phase 4 |
| `device/vof/*` | Multiphase owner | Alpha/MULES/VOF kernels and helpers | Phases 2-4 | Phase 5 |
| `device/nozzle/*` | Nozzle-physics owner | BCs, startup seeding, patch kernels | Phases 2,5 | Phase 6 |
| `device/kernels/*` | Performance owner | Custom hotspot kernels | Stable profiling data | Phase 7 |
| `tools/profiling/*` | Tooling owner | Reproducible Nsight / sanitizer scripts | NVTX, runtime counters | Phase 1 onward |
| `tools/tests/*` | QA owner | Unit and integration tests | All core modules | Phase 2 onward |
| `tools/benchmark/*` | Benchmark owner | Benchmark automation and report generation | Profiling + reports | Phase 8 |

# 9. Coding-agent execution roadmap

The Phase 2 portion of this roadmap is concrete and dependency-aware. Cross-phase milestone names, sequencing, and stop/go ownership are retained here only as dependency context; the authoritative project roadmap lives in the centralized master roadmap and acceptance artifacts.

## 9.1 Dependency graph

`M0 reference freeze` -> `M1 Blackwell/tooling bring-up` -> `M2 allocator substrate` -> `M3 residency registry` -> `M4 field/mesh integration` -> `M5 write staging and CPU-touch control` -> `M6 scratch stabilization` -> `GATE-A Phase 2 benchmark stop` -> `M7 execution-model cleanup` -> `M8 linear algebra path` -> `M9 VOF core` -> `M10 nozzle BCs` -> `M11 hotspot custom kernels` -> `GATE-B production performance acceptance`

## 9.2 Phase 2-specific milestone order

### M2.1 — Allocator substrate
Build:
- `devicePersistentPool`
- `deviceScratchPool`
- `pinnedStageAllocator`
- `managedFallbackAllocator`

Stop and benchmark:
- allocator unit tests only.

### M2.2 — Residency registry
Build:
- registry types,
- state machine,
- reporting.

Stop and benchmark:
- registry state-transition tests.

### M2.3 — OpenFOAM integration traits
Build:
- `MirrorTraits`,
- `FieldMirror`,
- `MeshMirror`.

Stop and benchmark:
- host/device round-trip tests on `Field`, `List`, `GeometricField`.

### M2.4 — Startup registration
Build:
- mesh registration,
- persistent field registration,
- startup upload.

Stop and benchmark:
- supported GPU solver startup trace.

### M2.5 — Explicit host visibility
Build:
- `ensureHostVisible`,
- `ensureDeviceVisible`,
- `OutputStager`,
- scalar staging path.

Stop and benchmark:
- write-boundary correctness test.

### M2.6 — Compute-epoch control
Build:
- `CpuTouchGuard`,
- per-epoch logging,
- strict mode.

Stop and benchmark:
- one supported GPU solver with strict mode enabled.

### M2.7 — Scratch stabilization
Build:
- `ScratchCatalog`,
- named scratch reset logic.

Stop and benchmark:
- repeated identical iterations; verify stable watermarks.

### GATE-A — Phase 2 acceptance stop
Run:
- unit tests,
- integration test,
- Nsight Systems trace on R1 or closest harness,
- benchmark parser.

Proceed only if:
- residency gates pass,
- no uncontrolled UVM remains for registered hot objects.

## 9.3 What can be parallelized
Safe parallel work:
- allocator implementation,
- registry/reporting scaffolding,
- profiler script automation,
- unit-test scaffolding,
- docs/config schema.

Work that should remain sequential:
- `GeometricField` boundary/old-time integration,
- write staging,
- strict CPU-touch enforcement.

## 9.4 What should be prototyped before productized
Prototype first:
- host-pinned mempool support,
- overlap-aware scratch arena,
- graph-captured allocation nodes.

Productize first:
- explicit device pools,
- pinned staging via `cudaMallocHost`,
- registry state machine,
- explicit write staging,
- strict benchmark gates.

## 9.5 Where to stop and benchmark before proceeding
Mandatory stop points:
1. after allocator unit tests;
2. after first real field registration;
3. after write staging;
4. after compute-epoch guard introduction;
5. before Phase 3 begins;
6. before Phase 5 begins.

If any stop point shows recurring steady-state UVM traffic, do not continue to later phases. Fix Phase 2 first.

# 10. Resolved Local Defaults and Residual Governance Notes

1. **Patch posture is fixed.**
   - Phase 2 uses limited, explicit integration in allocation and field-access code. Whole-framework allocator replacement is out of scope.
2. **Pinned host staging policy is fixed.**
   - `cudaMallocHost`-backed pooled staging is the required path. Host-pinned CUDA mempool productization is deferred until after Phase 2 acceptance.
3. **The Phase 2 gate case set is fixed.**
   - The registered hot set and reduced-case gate are imported from `reference_case_contract.md`, `support_matrix.md`, and `acceptance_manifest.md`; the default reduced gate is `R2` plus `R1-core`.
4. **The default headroom reserve is fixed.**
   - Reserve 2 GiB of VRAM on the RTX 5080 unless a later package revision explicitly changes that policy.
5. **Residual UVM waivers are not part of the baseline contract.**
   - Any temporary exception must be recorded in `acceptance_manifest.md`; Phase 2 itself does not authorize one.
6. **Write-boundary latency policy is fixed.**
   - Blocking at write boundaries is acceptable in milestone 1. Background/asynchronous write productization is deferred.
7. **Dependency posture is fixed.**
   - Umpire remains fallback-only unless a later measured package revision proves a production benefit.

### Human review checklist
- Verify the implementation keeps `gpuRuntime.memory` as the only authoritative memory contract.
- Verify the branch uses the package-fixed 2 GiB headroom reserve unless the package itself is revised.
- Verify boundary patch and history mirroring match the accepted Phase 2 hot set from the central authorities.
- Verify no steady-state UVM waiver was introduced without an `acceptance_manifest.md` revision.
- Verify core OpenFOAM hooks stay within the limited explicit-integration posture.

### Coding agent kickoff checklist
- Pull the SPUMA branch and verify build.
- Ensure the `master_pin_manifest.md` default lane (CUDA 12.9.1 primary, driver `>=595.45.04`) and profiling tools are working.
- Create canonical `gpuRuntime` memory config (and `gpuMemoryDict` shim only if needed).
- Implement allocator substrate first.
- Add registry and tests before solver integration.
- Stop at `GATE-A` and benchmark before touching later phases.

### Highest-risk implementation assumptions
1. That explicit device residency can be introduced with limited core-OpenFOAM surgery.
2. That boundary and old-time mirroring can be integrated without destabilizing write paths.
3. That the selected 16 GB headroom policy leaves enough space for later solver workspaces.
4. That managed fallback remains strictly transitional and does not creep back into production mode.
5. That the coding agent can keep parent/child residency records coherent across transient field history objects.


### References

[R1] Bnà et al., *SPUMA: a minimally invasive approach to the GPU porting of OPENFOAM®*, arXiv HTML version. https://arxiv.org/html/2512.22215v1

[R2] SPUMA project page, benchmarked against OpenFOAM-v2412 and describing current memory-pool options. https://gitlab-hpc.cineca.it/exafoam/spuma

[R3] SPUMA GPU-support wiki (June 10, 2025), including supported solvers and the warning that unsupported features often manifest as host/device-copy slowdowns instead of fatal errors. https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554

[R4] OpenFOAM Foundation API: `Foam::solvers::incompressibleVoF`. https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html

[R5] OpenFOAM Foundation source/API for VOF alpha prediction and subcycling. https://cpp.openfoam.org/v13/incompressibleMultiphaseVoF_2alphaPredictor_8C_source.html

[R6] OpenFOAM API: `Foam::tmp<T>` temporary-object management. https://cpp.openfoam.org/v13/classFoam_1_1tmp.html

[R7] OpenFOAM API: `GeometricField` constructors used to create temporary variables. https://cpp.openfoam.org/v13/classFoam_1_1GeometricField.html

[R8] OpenFOAM API/source note that `primitiveFieldRef()` / internal-field access updates event counters and should be avoided in loops. Representative API pages: https://cpp.openfoam.org/v4/classFoam_1_1GeometricField.html and https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1GeometricField.html

[R9] OpenFOAM API: patch-field matrix coefficient methods return `tmp<Field<Type>>`, contributing to temporary allocation churn. https://cpp.openfoam.org/v13/classFoam_1_1fvPatchField.html

[R10] OpenFOAM API: `List<T>` storage is allocated on the free store; `UList` exposes pointer-based storage/view semantics. https://cpp.openfoam.org/v13/classFoam_1_1List.html and https://www.openfoam.com/documentation/guides/latest/api/classFoam_1_1UList.html

[R11] NVIDIA Blackwell compatibility guide for CUDA 12.8, including PTX/JIT guidance. https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html

[R12] CUDA Toolkit 12.8 release notes, including Blackwell compiler support and NVTX v2 deprecation. https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

[R13] NVIDIA and NVIDIA-maintained sources for RTX 5080 / compute-capability characteristics. https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/ and https://developer.nvidia.com/cuda-gpus

[R14] CUDA 12.8.1 stream-ordered allocator / memory pool runtime API. https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html

[R15] CUDA runtime memory-management docs for `cudaMemAdvise` and `cudaMemPrefetchAsync`. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

[R16] CUDA Programming Guide section on asynchronous execution and the requirement that host buffers be page-locked for true asynchronous host/device copies. https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html

[R17] `cudaMallocHost` runtime documentation: page-locked host memory accelerates `cudaMemcpy*`, but excessive pinned memory hurts system performance. https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html

[R18] Current CUDA runtime / CCCL documentation showing host-pinned and managed memory-pool support; useful because archived 12.8.1 HTML and current docs are not perfectly consistent. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html and https://nvidia.github.io/cccl/libcudacxx/api/classpinned__memory__pool__ref.html

[R19] Nsight Systems unified-memory event descriptions: HtoD events pause kernels to migrate managed memory from host to device, and DtoH events pause CPU execution to migrate managed memory from device to host. https://docs.nvidia.com/nsight-systems/UserGuide/index.html

[R20] Umpire cookbook showing preferred-location advice on unified-memory pools. https://umpire.readthedocs.io/en/bugfix-rtd-docutils/sphinx/cookbook/pool_advice.html

[R21] exaFOAM workshop material on OpenFOAM GPU porting, explicit data management, memory pools, and `devicefvm` / `devicefvc` style operator reimplementation. https://exafoam.eu/wp-content/uploads/2ndWorkshop/exaFOAM_2ndPublicWorkshop_Porting_to_GPUs.pdf

[R22] OGL / Ginkgo paper on persistent OpenFOAM-to-GPU sparse structures and topology/value separation. https://link.springer.com/article/10.1007/s11012-024-01806-1

[R23] NVIDIA Blackwell tuning guide. https://docs.nvidia.com/cuda/blackwell-tuning-guide/
