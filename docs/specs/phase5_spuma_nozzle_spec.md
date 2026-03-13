
# 1. Executive overview

This document expands **only Phase 5** of the nozzle-GPU porting plan into an implementation-ready engineering specification. The surrounding phases are summarized only to the extent required to keep Phase 5 technically coherent, sequence-safe, and reviewable.

The target outcome of Phase 5 is the first **single-GPU, device-resident VOF core** for the SPUMA/OpenFOAM-v2412 code line on an **NVIDIA GeForce RTX 5080 (Blackwell, compute capability 12.0)**. The required solver behavior is the existing algebraic VOF/MULES/PIMPLE family used for incompressible two-phase immiscible flow, not a reformulated solver and not a geometric-VOF branch. The implementation must keep the numerics recognizable, keep the mesh and field topology resident on the GPU, and avoid hidden host/device traffic in the steady-state hot path. [R1] [R2] [R10] [R17] [R18]

The design posture for Phase 5 is driven by three ceilings, not one:

1. **Sparse-algebra memory traffic**, especially in pressure and momentum assembly/solve.
2. **Irregular face-kernel efficiency**, including gradient-like kernels, limiter logic, patch handling, and atomics.
3. **Launch and synchronization overhead**, especially from transient PIMPLE plus repeated alpha correction and alpha subcycling.

That three-ceiling model is not a stylistic preference. It is implied by SPUMA’s own profiling results, which show heavy API/synchronization cost in current paths, weaker performance in irregular kernels than in clean sparse matvec kernels, and nontrivial pressure-path behavior when multigrid and changing coefficients interact. [R2] [R24]

This specification therefore makes the following high-level commitments for Phase 5:

- Preserve the **existing VOF/MULES/PIMPLE numerics first**; optimize only after correctness is proven.
- Treat **host OpenFOAM objects as the control plane** and an explicit **device mirror/state object as the authoritative data plane** during timestepping.
- Keep **all heavy fields, topology, and persistent solver mappings resident on the GPU** between timesteps.
- Allow only **small scalar control transfers** to/from the host during steady-state timestepping.
- Support **single GPU, single rank, static mesh, single region** only in this phase.
- Make all APIs **graph-safe** now, while deferring actual CUDA Graph capture/productization to a later phase.
- Keep both **native SPUMA pressure solve** and **AmgX via foamExternalSolvers** available as runtime-selectable backends. [R2] [R4] [R6] [R19] [R20]

A critical caveat: SPUMA is based on **OpenFOAM-v2412** and the most accessible public source-level VOF documentation is from the OpenFOAM Foundation v11–v13 code line. This document therefore uses the Foundation VOF/module sources as a **semantic proxy for algorithmic flow**, not as an exact file-path prescription for SPUMA’s v2412 tree. The first coding task in Phase 5 is to reconcile those symbols and files against the actual SPUMA/v2412 checkout before patching. That caveat is explicit and intentional; it is safer than pretending exact source identity where it has not been directly verified from the v2412 tree. [R1] [R9] [R10] [R11] [R12]

# 2. Global architecture decisions

The following decisions apply to Phase 5 and are intentionally opinionated.

## 2.1 Base code line

- **Decision:** Use **SPUMA on the OpenFOAM-v2412 line** as the runtime base.
- **Why:** SPUMA is the active GPU-porting branch, its public paper benchmarks against OpenFOAM-v2412, and its current supported GPU solver list does not include multiphase/VoF; therefore Phase 5 is real new solver work on the SPUMA base, not a plugin exercise. [R1] [R2] [R3] [R4]

## 2.2 Scope of solver port

- **Decision:** Port the **current algebraic VOF/MULES/PIMPLE** path first.
- **Why:** `incompressibleVoF` is the relevant solver family for two incompressible, isothermal immiscible fluids with a single mixture momentum equation and PIMPLE coupling. Its alpha path is explicit, bounded, and subcycled through MULES, which is precisely the workload a plugin-only linear-solver offload misses. [R10] [R11] [R15]

## 2.3 Data-plane architecture

- **Decision:** Do **not** make generic OpenFOAM `GeometricField` storage raw device memory in Phase 5. Instead, introduce an explicit **device-resident mirror/state subsystem** whose arrays become authoritative during each timestep.
- **Why:** SPUMA’s current porting guidance emphasizes minimizing invasive changes and isolating GPU kernels. exaFOAM/zeptoFOAM presentations make the same architectural point more explicitly: keep OpenFOAM semantics/runtime-selection, but move heavy execution and data movement into dedicated GPU executors and GPU operator namespaces. [R1] [R2] [R5] [R8]
- **Engineering consequence:** Host fields remain required as metadata owners, runtime-selection anchors, and write/output sources. Device arrays carry the actual timestep state.

## 2.4 Residency policy

- **Decision:** Production Phase 5 runs use **device allocations for persistent state**; unified memory is allowed only in bring-up/debug modes.
- **Why:** SPUMA uses unified memory plus pooling as a migration strategy, but both the SPUMA paper and exaFOAM material show why managed-memory residency is not an acceptable production posture for discrete GPUs. Nsight Systems explicitly documents the pause-and-migrate behavior of managed-memory HtoD/DtoH traffic. [R1] [R2] [R8] [R21] [R22]

## 2.5 Parallel model

- **Decision:** Phase 5 supports **single GPU, single MPI rank, no domain decomposition**.
- **Why:** SPUMA notes that GPU-aware MPI direct transfers require pure device pointers and managed memory can degrade into host/device transfers. Multi-GPU is therefore the wrong first stabilization target. [R2]

## 2.6 Pressure backend strategy

- **Decision:** Keep both **native SPUMA pressure solve** and **AmgX through foamExternalSolvers** runtime-selectable.
- **Why:** SPUMA’s published results show that AmgX can scale well, but can also lose on small GPU counts when changing coefficients force hierarchy rebuilds. The correct engineering posture is “benchmark, do not assume.” [R2] [R6]

## 2.7 Precision policy

- **Decision:** Use **FP64 for baseline correctness** in Phase 5.
- **Why:** The solver is dominated by sparse and irregular CFD kernels, not dense tensor-core GEMMs. On an RTX 5080, FP64 throughput is not a strength, but changing precision before the port is numerically trusted is risk-seeking. [R17] [R24]
- **Deferred:** Mixed precision, FP32 limiter auxiliaries, and reduced-precision surface-tension kernels.

## 2.8 Execution semantics

- **Decision:** Stage APIs must be **stream-ordered and graph-safe**, with **no internal `cudaDeviceSynchronize()`** in hot-path code.
- **Why:** SPUMA’s current profiling shows massive synchronization/API overhead in profiled paths. CUDA 12.8 adds richer graph conditional features that are directly relevant to repeated PIMPLE and subcycling control flow, but Phase 5 should first produce graph-safe stage boundaries and stable object lifetimes. [R2] [R19] [R20]

## 2.9 Feature envelope

- **Decision:** Phase 5 is intentionally **strict**:
  - static mesh only,
  - single region only,
  - no mesh motion,
  - no mesh topology changes,
  - no processor patches,
  - no dynamic contact-angle models,
  - constant surface tension only,
  - no arbitrary `fvModels` / `fvConstraints` unless explicitly proven zero-impact.
- **Why:** This is the smallest envelope that can produce a trustworthy GPU VOF core while preserving the path to later nozzle-specific work. [R10] [R11] [R12] [R13] [R14]

# 3. Global assumptions and constraints

1. The working branch is **SPUMA on OpenFOAM-v2412**, pinned to a reviewed commit or tag. [R1] [R3] [R9]
2. The target development platform is a **Linux workstation** with an **RTX 5080**, **16 GB GDDR7**, **compute capability 12.0**, and **no NVLink**. [R17] [R18]
3. The code path is **single-rank single-GPU** for this phase.
4. The mesh is **static**, the region count is **one**, and topology does not change.
5. The authoritative solver semantics come from the SPUMA/v2412 source tree. Foundation v13 references are used here only as a semantic proxy for the algorithmic structure of:
   - `incompressibleVoF`,
   - `twoPhaseSolver::alphaPredictor()`,
   - `twoPhaseSolver::pressureCorrector()`,
   - `interfaceProperties`. [R10] [R11] [R12] [R13]
6. The current production nozzle workflow can accept a **Phase 5 baseline that is not yet nozzle-BC complete**. Patch-specific swirl/pressure inlet logic is deferred to the later nozzle phase.
7. Phase 5 validation consumes the centralized validation ladder. Hard Phase 5 gates use:
   - one `R2` generic boundedness / interface-transport case,
   - one `R2` generic surface-tension case,
   - one `R1-core` reduced case that uses only the frozen generic BC/scheme subset and no Phase 6-only BCs or startup features.
   `R1` reduced nozzle and `R0` full reference nozzle remain frozen later-ladder cases and are not required for Phase 5 exit.
8. `Euler` is the **required ddt scheme** for Phase 5 bring-up. Other ddt schemes are explicitly unsupported unless the human reviewer signs off on extending scope through the centralized support matrix.
9. Milestone-1 Phase 5 acceptance is **laminar-only** unless the centralized support matrix explicitly includes one previously verified SPUMA GPU-capable turbulence model that is compatible with the device-resident mixture fields. Phase 5 does not reopen turbulence scope locally.
10. Runtime functionObjects are governed by the centralized, machine-readable support matrix and GPU operational contract. In performance acceptance, only entries classified `writeTimeOnly` are allowed. `debugOnly` entries may appear only in explicit debug runs.
11. The allowed production host/device traffic in the hot path is:
    - small scalar control values,
    - linear-solver residual/status scalars,
    - write-time output staging,
    - explicit debug/fallback transfers only when a debug mode is enabled.
12. Peak GPU memory for the reference reduced case should stay comfortably below the 16 GB board limit. The implementation target is:
    - **< 10 GiB** peak for reduced validation cases,
    - **< 13 GiB** peak for the main single-GPU reference nozzle case,
    leaving headroom for context, solver-library buffers, and future Phase 6 additions. This threshold is an engineering recommendation, not a sourced hardware limit.

# 4. Cross-cutting risks and mitigation

| Risk | Why it matters | Early detector | Mitigation | Fallback / rollback |
|---|---|---|---|---|
| SPUMA/v2412 symbol drift versus Foundation docs | The semantic proxy may not match exact class/file boundaries in SPUMA | Initial source reconciliation task | Create a symbol reconciliation note before coding; patch only the real SPUMA/v2412 files | Stop Phase 5 coding until reconciliation note is reviewed |
| Hidden host dereferences of hot fields | Can trigger UVM migrations or crashes if raw device data leaks into host paths | Nsight UM traces; debug asserts on host-shadow freshness | Keep host fields as control plane; use explicit device mirrors; fail-fast if stale host fields are read in GPU mode | Temporary UM bring-up mode for debugging only |
| Managed-memory fallback pathologies | SPUMA and exaFOAM both identify UVM migration as a major discrete-GPU failure mode | Nsight Systems HtoD/DtoH/PtoP UM events and CPU/GPU page-fault traces | Production mode uses device allocations for persistent state; profile with UM tracing during bring-up only | UM mode only as a temporary bring-up tool, never for performance acceptance |
| Atomics in face-based kernels dominate runtime | SPUMA identifies face-based race conditions and uses atomics as the safe baseline | Nsight Compute on gradient, limiter, curvature kernels | Start with atomic baseline, then optimize only after correctness; preserve a reference kernel for A/B comparison | Retain atomic kernel as correctness oracle when trying optimized variants |
| Pressure backend choice is not stable | AmgX can lose when coefficients change every iteration | Pressure backend benchmark matrix | Keep native and AmgX runtime-selectable; benchmark both on reduced nozzle and reference nozzle | Choose native as default if AmgX hierarchy rebuild cost dominates |
| Temporary allocation churn | OpenFOAM-style temporaries create runtime and memory overhead | Nsight memory usage and allocator hot spots | Preallocate scratch arena and persistent matrices; forbid stage-local heap churn in hot path | Increase scratch reservation; revert new temps to preallocated buffers |
| Unsupported BC/model silently reintroduces host path | A single unsupported patch/model can destroy residency | Case support scanner at solver startup | Strict support scanner; fail-fast by default | Debug-only stageFallback mode with loud warning and logs |
| Contact-angle and advanced sigma models | `interfaceProperties` can hide nontrivial wall logic | Dictionary scan of `alpha*` patch types and `sigma` model | Restrict Phase 5 to constant sigma and no contact-angle models | Defer to later phase; reject case in Phase 5 |
| Runtime functionObjects / probes | Can force field commits every timestep | Config scan and runtime logging | Disable or restrict to write-time only during performance runs | Performance mode refuses unsafe functionObjects |
| FP64 weakness on RTX 5080 | Could limit speedup despite correct residency | Benchmark after each major milestone | Optimize memory traffic, kernel count, and atomics before touching precision | Mixed precision deferred, not used as early workaround |
| CUDA Graph capture limitations | Some solver branches and allocations may not be graph-capture-safe | Future graph-capture prototypes | Make APIs graph-safe now: stable lifetimes, no hidden sync, stream-ordered ops | Leave actual graph capture for later phase |
| Phase 5 scope creep | Nozzle-specific BCs and advanced physics can swamp the VOF core port | Review checkpoints after alpha, interface, and pressure milestones | Enforce strict non-goals and human sign-off gates | Stop at validated reduced case if Phase 6 dependencies appear |

# 5. Phase-by-phase implementation specification

## Phase 5 — Port the VOF core

### Purpose

Port the **generic two-phase incompressible VOF core** of the nozzle workflow to the GPU on top of SPUMA’s runtime and memory abstractions, while preserving current algebraic VOF/MULES/PIMPLE solver semantics and establishing a device-resident state model that Phase 6 nozzle-specific kernels can build on.

Phase 5 assumes the algebraic milestone baseline frozen in Phase 0. The canonical GPU target is the algebraic `incompressibleVoF` / explicit MULES / PIMPLE family; any retained `geometric_iso` / `interIsoFoam` material is CPU-only shadow-reference context and not a Phase 5 acceptance target.

The deliverable of this phase is not “some GPU kernels.” It is a functioning, validated, device-resident solver core with explicit module boundaries, explicit ownership, explicit synchronization rules, and explicit failure behavior.

### Why this phase exists

Phase 5 exists because the nozzle workflow is not reducible to “pressure solve acceleration.” The VOF alpha path, MULES corrections, subcycling, mixture-property updates, interface-property work, and pressure assembly/correction all participate materially in time-to-solution and must remain resident to avoid Amdahl collapse. Hybrid/plugin-only solver acceleration has a known ceiling because it offloads only the linear solve, not the rest of the CFD timestep. SPUMA’s own stated roadmap also identifies multiphase as work not yet supported, so this phase is where the project stops being configuration and becomes actual solver engineering. [R1] [R2] [R6] [R8] [R10] [R11] [R12]

### Entry criteria

Phase 5 work shall not start until all of the following are true:

1. The mandatory local semantic source audit is **PR5.1** and must be completed and reviewed before any Phase 5 solver/backend implementation PR.
2. A **CPU reference branch** exists on the SPUMA/v2412 base and reproduces the current workflow within agreed numerical tolerances.
3. SPUMA builds and runs on the RTX 5080 with at least one supported GPU solver.
4. The development environment can run:
   - Nsight Systems,
   - Nsight Compute,
   - Compute Sanitizer,
   - NVTX v3 instrumentation. [R4] [R19] [R20] [R21]
5. A device allocator / memory-pool path exists and has been validated outside the VOF solver.
6. A minimal **native** pressure backend bridge is already buildable. AmgX may participate in Phase 5 only in one of these states:
   - the Phase 4 **DeviceDirect** bridge is available and AmgX is eligible for device-residency / production claims,
   - or the Phase 4 `PinnedHost` path is being used strictly as a correctness-only bring-up mode.
7. The following validation cases are frozen:
   - one `R2` generic boundedness / interface-transport case,
   - one `R2` generic surface-tension case,
   - one `R1-core` reduced case using only the frozen generic BC/scheme subset and no Phase 6-only BCs/startup,
   - later-ladder `R1` and `R0` case definitions for subsequent phases.
8. The human reviewer has signed off on the initial Phase 5 feature envelope imported from the centralized support matrix:
   - single GPU,
   - static mesh,
   - no mesh motion,
   - constant sigma,
   - no contact-angle models,
   - strict BC subset,
   - Euler ddt,
   - laminar-only unless the support matrix explicitly says otherwise.

### Exit criteria

Phase 5 exits only when **all** of the following are satisfied:

1. The alpha transport path (`alphaPhi`, bounded alpha update, MULES correction, alpha subcycling) runs entirely on device-resident field mirrors.
2. Mixture-property updates (`rho`, `rhoPhi`, transport-property update inputs) run on device with no field-sized host round trip.
3. Interface and surface-tension work required by the generic VOF path runs on device for the supported model subset.
4. Momentum predictor and pressure corrector run through the device-resident path, with the native pressure backend as the required baseline. AmgX counts as a Phase 5 device-resident backend only when the Phase 4 **DeviceDirect** bridge is available; otherwise AmgX remains correctness-only bring-up.
5. No hot-stage implementation contains `cudaDeviceSynchronize()` or an equivalent forced host sync.
6. Nsight Systems shows **no recurring field-scale UVM HtoD or DtoH migrations in the steady-state hot path**. For AmgX, this criterion applies only to `DeviceDirect` runs.
7. Matrix topology for persistent linear systems is built once per mesh and values are updated without full remapping each corrector where the backend supports it. [R7]
8. All supported Phase 5 ladder cases (`R2` generic cases plus `R1-core`) pass their numerical acceptance thresholds.
9. `R1-core` benchmarks exist for the native pressure backend and, when `DeviceDirect` is available, for the AmgX backend. Any `PinnedHost` AmgX results are archived as correctness-only bring-up, not production evidence.
10. The module compiles in:
    - CPU-disabled / GPU-disabled mode,
    - GPU enabled with native pressure backend,
    - GPU enabled with AmgX backend.
11. Write/restart parity is validated for supported Phase 5 cases: a case written at an accepted write point can be restarted with current fields and all required old-time state restored into device mirrors without reconstructing missing history from stale host fields.
12. The support scanner enforces the machine-readable functionObject policy imported from the centralized support matrix / GPU operational contract in performance mode.

### Goals

1. Preserve solver semantics of the current algebraic VOF/MULES/PIMPLE path.
2. Introduce a **device-authoritative state object** with explicit host commit semantics.
3. Keep mesh topology, patch maps, fields, old-time state, and persistent matrix mappings on device.
4. Support alpha subcycling and repeated alpha corrections without host staging.
5. Keep the code graph-safe and free from hidden hot-path synchronization.
6. Make unsupported cases fail early and clearly.
7. Create a Phase 5 code structure that Phase 6 can extend for nozzle patch kernels and startup seeding.
8. Produce validation artifacts that separate:
   - correctness,
   - regression stability,
   - pressure backend performance,
   - residency correctness.

### Non-goals

1. Multi-GPU support.
2. MPI domain decomposition.
3. Dynamic mesh / mesh motion / topology change support.
4. Geometric VOF / isoAdvector / reconstructed-interface algorithms.
5. Contact-angle model support.
6. Temperature-dependent or otherwise advanced surface-tension models.
7. Generic support for all OpenFOAM patch fields and `fvModels`.
8. Mixed precision optimization.
9. Final CUDA Graph capture/productization.
10. Generic raw-device replacement of all OpenFOAM field storage.
11. Nozzle-specific pressure/swirl inlet kernels and air-core seeding logic (deferred to the next phase).

### Technical background

#### 5.1 Solver-module semantics

OpenFOAM’s `incompressibleVoF` module is the relevant solver family for **two incompressible, isothermal immiscible fluids** using a VOF phase-fraction representation, a **single mixture momentum equation**, and **PIMPLE** coupling. The accessible Foundation documentation shows the module exposes exactly the objects and methods that matter to this port, including:
- `alphaPredictor()`,
- `surfaceTensionForce()`,
- pressure-correction support through the inherited two-phase solver stack,
- and cached `rAU` handling. [R10]

**Sourced fact:** The Foundation `multiphaseVoFSolver` and `incompressibleVoF` documentation describe a flexible PIMPLE-based solver structure in which phase-fraction transport, mixture-property updates, momentum prediction, and pressure correction are coordinated within the timestep. [R10]

**Engineering inference:** SPUMA’s v2412 implementation is sufficiently semantically close that the same stage decomposition is the correct porting axis, even if exact class or file names differ.

**Recommendation:** Implement Phase 5 as a solver-stage adaptor around that semantic model, not as scattered operator-level patches.

#### 5.2 Alpha transport and MULES

The Foundation `twoPhaseSolver/alphaPredictor.C` shows the key alpha-path behaviors:

- `alphaPhi(phi, alpha)` computes the advective flux of the phase fraction through `fvc::flux`.
- `alphaPredictor()` evaluates the number of alpha subcycles from `nAlphaSubCyclesPtr->value(alphaCoNum)`.
- If `nAlphaSubCycles > 1`, a subcycling loop accumulates the total `alphaPhi1`.
- The implementation supports a `MULESCorr` path and an explicit `MULES::explicitSolve` path.
- If `alphaApplyPrevCorr` is enabled, the previous correction flux is applied and stored across iterations.
- `CrankNicolson` is explicitly incompatible with alpha subcycling in the reference implementation. [R11]

**Sourced fact:** MULES is fundamentally an explicit bounded transport limiter, and the semi-implicit use pattern is “bounded implicit predictor + explicit correction.” [R11] [R15]

**Engineering inference:** The expensive part of the alpha path is not a single linear solve; it is the repeated face-based flux formation, limiter correction, and subcycling logic.

**Recommendation:** Treat alpha transport as a first-class GPU subsystem with its own persistent state, not as a side path around the pressure solve.

#### 5.3 Pressure correction

The Foundation `twoPhaseSolver/pressureCorrector.C` provides the required semantic shape for the incompressible two-phase pressure path:

- cache `rAU` from the momentum matrix,
- interpolate `rAUf`,
- form `HbyA`,
- form `phiHbyA`,
- add surface-tension/buoyancy-like force fluxes (`phig`),
- constrain pressure boundary behavior,
- solve the pressure equation through non-orthogonal correction loops,
- update `phi`, `U`, `p`, `p_rgh`,
- and clear `rAU` only after the correction sequence completes. [R12]

**Sourced fact:** The pressure-corrector path depends on device-resident intermediate fields, not only on the final pressure linear solve. [R12]

**Engineering inference:** A plugin-only pressure-solve offload is architecturally insufficient for this project because assembly and the dependent field updates would still bounce through the CPU.

**Recommendation:** Keep the entire pressure-corrector stage device-resident, with a backend bridge only at the linear-solver boundary.

#### 5.4 Interface properties and surface tension

The Foundation `interfaceProperties` class encapsulates:
- interface curvature,
- surface tension,
- `nHatf`,
- `sigmaK`,
- `surfaceTensionForce()`,
- and alpha contact-angle boundary correction behavior. [R13]

The `surfaceTensionModel` interface is runtime-selectable and supports both constant-value and model-based sigma specification. [R14]

**Sourced fact:** `surfaceTensionForce()` depends on interpolation and face-normal gradients, which are classic irregular finite-volume operations. [R13]

**Engineering inference:** These kernels are likely to behave more like SPUMA’s weaker gradient/atomic kernels than like clean SpMV kernels.

**Recommendation:** Support **constant sigma only** in Phase 5 and explicitly reject contact-angle models and advanced sigma models.

#### 5.5 SPUMA execution and profiling implications

SPUMA’s public paper and arXiv text emphasize:
- a minimally invasive GPU-porting strategy,
- memory-pool usage,
- managed-memory use as a migration aid,
- a current profile containing many kernel launches and synchronizations,
- and performance limitations in irregular kernels and pressure/GAMG behavior. [R1] [R2]

**Sourced fact:** SPUMA’s profiled path reported tens of thousands of kernel launches and `cudaDeviceSynchronize()` calls in a small iteration sample, with synchronization dominating API time. [R2]

**Engineering inference:** A naive one-to-one port of individual VOF operators will work functionally but fail the performance objective.

**Recommendation:** Phase 5 must build stage-level orchestration and persistent workspaces from the start, even before graph capture.

### Research findings relevant to this phase

#### 5.6 Sourced facts

1. **SPUMA is benchmarked against OpenFOAM-v2412 and is intended as a minimally invasive GPU porting path using a portable programming model plus memory pooling.** [R1]
2. **SPUMA does not yet publicly claim support for multiphase/VOF solvers**, making Phase 5 actual new development rather than a feature toggle. [R2] [R3]
3. **Hybrid/plugin-only linear-solver offload is Amdahl-limited** because matrix assembly and other solver work remain on the CPU; exaFOAM’s workshop material states similar limits for OpenFOAM hybrid strategies. [R1] [R8]
4. **SPUMA’s current profiling shows a synchronization-heavy execution pattern** and performance variability across irregular kernels and the pressure path. [R2]
5. **OpenFOAM’s two-phase alpha path includes explicit MULES logic, alpha subcycling, previous-correction flux handling, and scheme-dependent behavior.** [R11]
6. **`interfaceProperties` owns curvature and surface-tension operations and also couples to contact-angle correction behavior.** [R13]
7. **AmgX integration exists through foamExternalSolvers**, but SPUMA results show changing coefficients can reduce its advantage at low GPU counts because multigrid hierarchies cannot always be reused efficiently. [R2] [R6]
8. **OGL/Ginkgo explicitly demonstrates the value of persistent LDU→GPU mapping with value-only updates across solves.** [R7]
9. **Sparse operations such as SpMV are usually memory-bound, but storage format and matrix structure matter materially.** [R24]
10. **CUDA 12.8 adds native Blackwell compiler support including SM_120 and richer graph conditional features.** [R19] [R20]
11. **The RTX 5080 is a Blackwell GeForce part with compute capability 12.0 and 16 GB memory, but no NVLink.** [R17] [R18]
12. **Nsight Systems documents that both HtoD and DtoH unified-memory migrations pause execution and can cause substantial penalties.** [R21]

#### 5.7 Engineering inferences

1. The safest Phase 5 architecture is an **explicit device-state subsystem** rather than attempting to mutate generic OpenFOAM field storage semantics.
2. Phase 5 correctness will fail in subtle ways if `oldTime()` semantics, `alphaApplyPrevCorr`, and `rAU` lifetime are not mirrored explicitly on device.
3. Supported patch fields must be scanned **before** the first timestep; otherwise unsupported patch logic will be discovered too late, usually as hidden host access.
4. Surface-tension/contact-angle support must be narrowed aggressively in Phase 5 because `interfaceProperties` couples geometric operations with boundary-condition behavior.
5. Runtime functionObjects and some diagnostics are likely to cause hidden host commits and must be treated as residency hazards.
6. The cost of rebuilding linear-system topology every corrector would erase a large fraction of expected GPU benefit on a single 16 GB workstation GPU.
7. The initial irregular-kernel baseline should accept atomics rather than prematurely committing to deeper cell-gather refactors.

#### 5.8 Recommendations

1. Build a **`gpuVoF` subsystem** with a `DeviceVoFState`, `DeviceAlphaTransport`, `DeviceMULES`, `DeviceSurfaceTension`, `DeviceMomentumPredictor`, `DevicePressureCorrector`, and `DeviceVoFOrchestrator`.
2. Keep all hot fields device-resident from the end of initialization until write/output.
3. Allow only tiny scalar host/device transfers during steady-state timestepping.
4. Make unsupported configurations fail-fast by default.
5. Treat the atomic face-kernel implementation as the reference baseline and optimize only after validation/profiling.
6. Require a **symbol reconciliation step** before coding against the real SPUMA/v2412 sources.

### Design decisions

The following decisions are normative for implementation.

#### Phase-local assumptions carried into Phase 5

1. The reduced validation cases can be chosen so that they use only the supported Phase 5 patch-field subset.
2. The production nozzle workflow can temporarily accept **Euler-only** alpha and transient discretization during Phase 5 bring-up.
3. Any turbulence usage required in Phase 5 is either laminar or already supported by the SPUMA device path without new turbulence-model porting.
4. The mesh topology is static for the entire run and the owner/neighbour pattern therefore stays reusable.
5. The human reviewer accepts strict fail-fast behavior for unsupported BCs/models rather than silent partial execution.
6. Field-level output during timestepping is not required except at write time.

#### Phase-local imports frozen before implementation

The reduced-case selection, patch targets, backend default, and turbulence/contact-angle scope are imported from the package authorities and are not reopened here.

1. `semantic_source_map.md` identifies the authoritative SPUMA/v2412 classes and files corresponding to `alphaPredictor`, `pressureCorrector`, `momentumPredictor`, and `interfaceProperties`.
2. The device path must match the exact local SPUMA/v2412 `rhoPhi` expression identified during symbol reconciliation; any local refactor is an implementation detail, not a new phase decision.
3. `reference_case_contract.md` freezes `R1-core` as the Phase-5-friendly reduced case and keeps it inside the generic Phase 5 BC/scheme subset.
4. `support_matrix.md` freezes milestone-1 Phase 5 acceptance as laminar-only with no contact-angle in scope.
5. `acceptance_manifest.md` keeps native as the current default backend; AmgX remains optional and benchmark-driven behind `DeviceDirect`.


#### DD5-01 — Separate control plane from data plane

- **Sourced fact:** SPUMA and exaFOAM/zeptoFOAM both emphasize keeping OpenFOAM semantics while isolating GPU execution into dedicated classes/namespaces. [R1] [R2] [R8]
- **Engineering inference:** Replacing generic `GeometricField` storage with raw device pointers in Phase 5 would create broad risk across runtime selection, I/O, and host-side helper code.
- **Recommendation:** Keep host OpenFOAM fields as control-plane objects; introduce explicit device mirrors that become authoritative during timestepping.

**Required behavior**
- Host fields exist at all times.
- Device mirrors are authoritative from `beginTimeStep()` until explicit commit.
- Host fields are considered stale while device state is authoritative.
- Any host read of a stale field in GPU mode must either:
  - trigger a controlled commit, or
  - raise a debug/assert failure if the code path is not approved.

#### DD5-02 — Device state object owns all persistent two-phase fields

- **Sourced fact:** The alpha path, pressure path, and interface-properties path all reuse persistent fields such as `alphaPhi1`, `phi`, `rho`, `rAU`, `HbyA`, `nHatf`, and `p_rgh`. [R11] [R12] [R13]
- **Engineering inference:** These fields must be allocated once and reused to avoid allocation churn and repeated data motion.
- **Recommendation:** Introduce a `DeviceVoFState` object that owns all persistent arrays and their sync metadata.

#### DD5-03 — Feature envelope is strict and validated upfront

- **Sourced fact:** `interfaceProperties` includes contact-angle logic and runtime-selected surface-tension models. `alphaPredictor` is scheme-sensitive. [R11] [R13] [R14]
- **Engineering inference:** Supporting every runtime-selected combination in the first GPU VOF port is unrealistic and unsafe.
- **Recommendation:** The solver startup must build a `CaseSupportReport` from the centralized support matrix and reject unsupported cases before the first timestep.

**Phase 5 supported-by-default set**
This list is the Phase 5 slice of the centralized support matrix:
- static mesh,
- single region,
- no processor patches,
- constant sigma,
- no contact-angle patch fields unless the centralized support matrix is explicitly revised,
- Euler ddt,
- laminar-only unless the centralized support matrix explicitly lists one already-verified SPUMA turbulence model,
- simple generic patch fields required for validation cases,
- functionObjects limited to entries classified `writeTimeOnly` for performance acceptance.

#### DD5-04 — Small scalar host/device control traffic is allowed

- **Sourced fact:** Alpha subcycling count is determined from `alphaCoNum` through a host-side runtime object in the reference solver. [R11]
- **Engineering inference:** Forcing all control logic onto the device in Phase 5 is unnecessary and risk-seeking.
- **Recommendation:** Permit transfer of small scalar control values such as:
  - `alphaCoNum`,
  - solver residual/status scalars,
  - wall-clock/profiling counters,
  - write-time flags.

**Not allowed**
- field-sized host/device transfers in steady-state hot stages.

#### DD5-05 — Implement alpha path first, then mixture, interface, momentum, pressure

- **Sourced fact:** The two-phase solver semantics place `alphaPredictor` ahead of downstream property and pressure updates. [R10] [R11] [R12]
- **Engineering inference:** Pressure debugging is impossible if alpha/mixture state is not already correct.
- **Recommendation:** Implement in this order:
  1. state/mirrors,
  2. alpha transport,
  3. mixture properties,
  4. interface/surface tension,
  5. momentum predictor,
  6. pressure corrector.

#### DD5-06 — Preserve current numerics before optimization

- **Sourced fact:** MULES boundedness and subcycling behavior are numerically delicate. [R11] [R15]
- **Engineering inference:** Algorithmic rewrites and performance tuning must be separated from the initial correctness port.
- **Recommendation:** Port semantics first, optimize second. Do not change limiter math, sequencing, or correction semantics during bring-up.

#### DD5-07 — Use face-atomic kernels as the baseline for irregular operations

- **Sourced fact:** SPUMA’s paper discusses face-based race conditions and atomics as the safe current baseline. [R2]
- **Engineering inference:** A cell-gather/segmented-reduction rewrite may be faster later, but is not the safest correctness baseline for Phase 5.
- **Recommendation:** Start with face-loop kernels using atomics where required, but isolate them so they can be replaced later.

#### DD5-08 — Persist matrix topology, update values only

- **Sourced fact:** OGL/Ginkgo demonstrates persistent LDU→GPU mappings stored in the object registry and value-only updates across solves. [R7]
- **Engineering inference:** The nozzle mesh topology is static, so owner/neighbour sparsity is amortizable.
- **Recommendation:** Build persistent topology mappings once per mesh for pressure (and any other backend that benefits), and update only numeric values in each corrector.

#### DD5-09 — Keep native and AmgX pressure backends both alive

- **Sourced fact:** foamExternalSolvers provides the AmgX interface; SPUMA’s results show AmgX does not automatically dominate GAMG/native paths on all workloads. [R2] [R6]
- **Engineering inference:** The best backend for a transient VOF nozzle case is an empirical result, not an architectural axiom.
- **Recommendation:** Pressure backend selection must be runtime-configurable.

#### DD5-10 — FP64 baseline only

- **Sourced fact:** The target hardware is GeForce-class Blackwell; sparse CFD kernels are not tensor-core-friendly dense GEMMs. [R17] [R18] [R24]
- **Engineering inference:** Mixed precision before validation would confound correctness debugging.
- **Recommendation:** Use double precision throughout Phase 5 for field storage and solver math.

#### DD5-11 — One-stream baseline, no hot-path hard sync

- **Sourced fact:** SPUMA’s current profiling shows hot-path synchronization/API overhead. [R2]
- **Engineering inference:** Multiple streams can help later, but one well-behaved stream is safer for correctness bring-up.
- **Recommendation:** Use one primary compute stream in Phase 5. No stage implementation may call `cudaDeviceSynchronize()`.

#### DD5-12 — Graph-safe API contract now, graph capture later

- **Sourced fact:** CUDA 12.8 adds graph conditional features useful for repeated control-flow patterns. [R19]
- **Engineering inference:** Capturing a badly structured solver into graphs later is harder than writing graph-safe stage boundaries now.
- **Recommendation:** Require stable object lifetimes, stream-ordered work, and no hidden allocation/sync inside stage functions. Capture eligibility by stage is owned by the centralized `GraphCaptureSupportMatrix`; Phase 5 implements graph-safe stage boundaries against that matrix and makes no stronger local capture claim, especially across AmgX solve boundaries.

#### DD5-13 — Strict fail-fast by default, debug fallback only by explicit opt-in

- **Sourced fact:** SPUMA recommends profiling CPU faults because unsupported features can degrade into data motion rather than clean failure. [R4]
- **Engineering inference:** Silent fallback is the fastest path to a misleading “working” implementation that is not actually resident.
- **Recommendation:** Default `fallbackPolicy = failFast`. A `stageFallback` debug mode may exist, but it is forbidden in performance acceptance and forbidden as the default.

#### DD5-14 — Output commits happen only at write time

- **Engineering inference:** Frequent host commits will dominate single-GPU workstation runs and hide whether the port is genuinely device-resident.
- **Recommendation:** Default `commitPolicy = writeOnly`.
- **Allowed exceptions:** human-invoked debugging, stageFallback debug mode, or explicit diagnostic runs.

#### DD5-15 — Phase 5 ddt support is deliberately narrow

- **Sourced fact:** `CrankNicolson` plus alpha subcycling is unsupported in the reference implementation. [R11]
- **Engineering inference:** Supporting multiple ddt schemes expands the correctness matrix significantly.
- **Recommendation:** Phase 5 baseline supports **Euler only**. `localEuler` and `CrankNicolson` are rejected at case-scan time unless a human explicitly broadens scope.

#### DD5-16 — Surface-tension scope is deliberately narrow

- **Sourced fact:** `surfaceTensionModel` is runtime-selectable, and `interfaceProperties` couples curvature with contact-angle-related boundary behavior. [R13] [R14]
- **Engineering inference:** Constant sigma without contact-angle is the smallest stable slice that still tests the GPU interface-property machinery.
- **Recommendation:** Support only constant `sigma` in Phase 5 and reject all contact-angle patch fields.

#### DD5-17 — Index width defaults to 32-bit on device

- **Engineering inference:** The target case sizes are far below 2^31 indices; 32-bit topology indices reduce memory footprint and improve bandwidth efficiency on a 16 GB board.
- **Recommendation:** Device topology uses 32-bit indices by default, with an explicit validation guard that aborts if counts exceed the representable range.

#### DD5-18 — FunctionObjects are residency hazards until proven safe

- **Engineering inference:** Many functionObjects read fields on the host and will force commits or UVM migrations.
- **Recommendation:** The authoritative functionObject policy is machine-readable and imported from the centralized support matrix / GPU operational contract. `inspectCaseSupport` must classify each configured functionObject as `writeTimeOnly`, `debugOnly`, or `unsupported`; performance mode allows only `writeTimeOnly` entries.

### Alternatives considered

#### Alternative A — Put raw device storage directly into generic OpenFOAM fields

- **Rejected because:** too invasive for Phase 5; high risk of hidden host dereferences, runtime-selection regressions, I/O breakage, and widespread undefined behavior.
- **When to revisit:** only after the device-mirror architecture is fully proven and broader refactoring is justified.

#### Alternative B — Rely on unified memory for the full Phase 5 implementation

- **Rejected because:** discrete-GPU UVM page migration is a documented performance hazard; SPUMA and exaFOAM already show why it cannot be the production posture. [R1] [R2] [R8] [R21]
- **Allowed only for:** bring-up/debug mode.

#### Alternative C — Offload only pressure solve

- **Rejected because:** it misses alpha/MULES, interface work, and pressure assembly, and is therefore Amdahl-limited for the nozzle workflow. [R1] [R8]

#### Alternative D — Start with geometric VOF / isoAdvector / interface reconstruction

- **Rejected because:** it expands physics and numerics scope before the GPU substrate is stable.

#### Alternative E — Rewrite irregular kernels immediately as non-atomic cell-gather kernels

- **Rejected because:** safer to validate the atomic baseline first, then optimize with a correctness oracle still available.

#### Alternative F — Mixed precision from day one

- **Rejected because:** would complicate validation, especially for boundedness, pressure correction, and spurious-current behavior.

#### Alternative G — Silent CPU fallback for unsupported BCs or stages

- **Rejected because:** it hides residency failures and makes profiles untrustworthy.
- **Permitted only as:** explicit debug-only `stageFallback` mode with loud runtime logging.

#### Rollback / fallback options

1. **Case-level rollback:** run the CPU reference branch on SPUMA/v2412 if the Phase 5 feature envelope is exceeded.
2. **Pressure-backend fallback:** switch between `native` and `amgx` without changing the rest of the GPU path.
3. **Residency-debug fallback:** use `umBringup` mode only to debug pointer lifetime or host-touch bugs.
4. **Stage-debug fallback:** enable `stageFallback` for one named stage only, commit fields to host, execute the host stage, re-upload, log the event, and then disable once the bug is localized.
5. **Kernel optimization rollback:** always retain the atomic baseline kernel behind a runtime or compile-time switch when testing optimized irregular kernels.

### Interfaces and dependencies

This subsection defines the implementation contract between the Phase 5 modules and the rest of SPUMA/OpenFOAM.

#### 5.9 External dependencies

- **SPUMA runtime and GPU abstractions** for build system integration, kernel launch conventions, memory pool integration, and any existing device operator helpers. [R1] [R2] [R4]
- **Umpire** (or the SPUMA-selected allocator abstraction) for device and pinned allocations, with unified-memory modes retained only for bring-up/debug. [R22]
- **foamExternalSolvers / AmgX** for the optional pressure backend. [R6]
- **CUDA runtime / NVTX v3 / Nsight / Compute Sanitizer** for profiling and debugging. [R19] [R20] [R21]
- **Existing SPUMA linear-algebra path** for native pressure and any reusable momentum/alpha linear solves.

#### 5.10 Host-side binding object

Create a host-only binding structure that centralizes every host object the GPU path is allowed to observe:

```cpp
namespace Foam::gpuVoF
{
    struct HostSolverBindings
    {
        fvMesh& mesh;

        volScalarField& alpha1;
        volScalarField& alpha2;

        surfaceScalarField& phi;
        surfaceScalarField& rhoPhi;

        volScalarField& rho;
        volVectorField& U;

        volScalarField& p_rgh;
        volScalarField& p;

        // Optional / conditional
        autoPtr<surfaceVectorField>* Uf;

        // Model handles / accessors
        const dictionary& transportProperties;
        const dictionary& fvSchemes;
        const dictionary& fvSolution;
        const dictionary& phaseProperties;

        // Solver/runtime services
        pimpleNoLoopControl& pimple;
        IOMRFZoneList* MRF;
        buoyancy* buoyancyModel;
        fvModels* models;
        fvConstraints* constraints;

        // Pressure backend selection
        word pressureBackend;   // "native" | "amgx"
    };
}
```

**Rules**
- `HostSolverBindings` is constructed once during solver setup.
- No stage function may perform ad hoc registry lookups outside this object.
- If a needed host object is missing or unsupported, `CaseSupportReport` must reject the case before runtime.

#### 5.11 Case support scanner

Add a pre-run compatibility scanner:

```cpp
namespace Foam::gpuVoF
{
    struct CaseSupportIssue
    {
        word category;     // scheme, bc, model, mesh, functionObject, etc.
        word objectName;
        word detail;
        bool fatal;
    };

    struct CaseSupportReport
    {
        bool supported;
        List<CaseSupportIssue> issues;
    };

    CaseSupportReport inspectCaseSupport
    (
        const HostSolverBindings& host,
        const dictionary& gpuVoFDict
    );
}
```

**Responsibilities**
- Verify mesh is static and single-region.
- Consume the centralized support matrix / machine-readable compatibility view for BCs, schemes, turbulence scope, functionObject policy, backend availability, and allowed fallbacks.
- Verify patch types for `alpha1`, `U`, `p_rgh`, `p`, `phi` are in the Phase 5 supported set.
- Verify `sigma` is constant.
- Reject contact-angle patch fields unless the centralized support matrix explicitly says otherwise.
- Reject unsupported ddt/div schemes.
- Reject processor patches.
- Classify configured functionObjects as `writeTimeOnly`, `debugOnly`, or `unsupported`, and enforce the global GPU operational contract for the selected mode.
- Verify backend availability for selected pressure backend; if `pressureBackend == amgx` in a production-acceptance mode, require the Phase 4 `DeviceDirect` bridge.
- Verify index ranges fit 32-bit device topology if that mode is requested.

#### 5.12 Controls snapshots

All stage functions consume immutable snapshots. This prevents hidden host lookups and makes graph capture later feasible.

```cpp
namespace Foam::gpuVoF
{
    enum class DdtSchemeKind : uint8_t
    {
        Euler,
        Unsupported
    };

    struct AlphaControlsSnapshot
    {
        int nAlphaCorr;
        int nAlphaSubCycles;
        bool MULESCorr;
        bool alphaApplyPrevCorr;
        DdtSchemeKind ddtScheme;
        double cAlpha;
        double icAlpha;
        double alphaCoNum;
        bool writeAlphaDiagnostics;
    };

    struct MixtureControlsSnapshot
    {
        double rho1;
        double rho2;
        double nu1;
        double nu2;
        bool updateRhoPhi;
    };

    struct SurfaceTensionControlsSnapshot
    {
        double sigma;
        double deltaN;
        bool enabled;
    };

    struct MomentumControlsSnapshot
    {
        bool solveMomentumPredictor;
        bool laminarMode;
        bool allowTurbulenceCoupling;
    };

    struct PressureControlsSnapshot
    {
        int nNonOrthCorr;
        bool finalNonOrthogonalOnlyUpdatesFlux;
        word backend; // native | amgx
        bool needReferenceCell;
    };

    struct TimeStepControlSnapshot
    {
        scalar timeValue;
        scalar deltaT;
        bool writeNow;
    };
}
```

**Allowed host/device scalar transfers**
- `alphaCoNum` reduction to host.
- linear-solver residuals and status scalars to host.
- write-now and timing flags from host to device stage orchestration.
- optional patch-coefficient updates for time-varying uniform BC values.

#### 5.13 Core device subsystem APIs

```cpp
namespace Foam::gpuVoF
{
    class DeviceVoFOrchestrator
    {
    public:
        DeviceVoFOrchestrator(const HostSolverBindings&, const dictionary& gpuVoFDict);
        ~DeviceVoFOrchestrator();

        CaseSupportReport inspectSupport() const;
        void initialize();      // allocate topology, fields, scratch, backends
        void uploadInitialState();
        void beginTimeStep(const TimeStepControlSnapshot&);

        AlphaControlsSnapshot buildAlphaControls();
        void alphaPredictor(const AlphaControlsSnapshot&);

        MixtureControlsSnapshot buildMixtureControls() const;
        void updateMixture(const MixtureControlsSnapshot&);

        SurfaceTensionControlsSnapshot buildSurfaceTensionControls() const;
        void correctInterface(const SurfaceTensionControlsSnapshot&);

        MomentumControlsSnapshot buildMomentumControls() const;
        void momentumPredictor(const MomentumControlsSnapshot&);

        PressureControlsSnapshot buildPressureControls() const;
        void pressureCorrector(const PressureControlsSnapshot&);

        void endTimeStep(const TimeStepControlSnapshot&);
        void commitForWrite();
        void commitForDebug(uint64_t fieldMask);
        void invalidateHostShadows(uint64_t fieldMask);

        const char* backendName() const;
        const struct DeviceSolveStats& lastPressureStats() const;
    };
}
```

#### 5.14 Device state and mirror APIs

```cpp
namespace Foam::gpuVoF
{
    enum class SyncState : uint8_t
    {
        HostFresh,
        DeviceFresh,
        BothFresh,
        Invalid
    };

    template<class T>
    class DeviceFieldMirror
    {
    public:
        void allocateInternal(label n, AllocatorKind alloc);
        void allocateBoundary(label nBoundaryValues, AllocatorKind alloc);
        void uploadFromHost(const UList<T>& internal, const UList<T>& boundary, cudaStream_t);
        void downloadToHost(UList<T>& internal, UList<T>& boundary, cudaStream_t) const;
        T* dInternal();
        T* dBoundary();
        const T* dInternal() const;
        const T* dBoundary() const;
        label sizeInternal() const;
        label sizeBoundary() const;
        SyncState syncState() const;
        void markDeviceFresh();
        void markHostFresh();
        void markBothFresh();
    };
}
```

```cpp
namespace Foam::gpuVoF
{
    struct DeviceVoFState
    {
        // Topology / geometry
        DeviceMeshTopology mesh;
        DeviceBoundaryMaps boundary;

        // Cell fields
        DeviceFieldMirror<double> alpha1;
        DeviceFieldMirror<double> alpha2;
        DeviceFieldMirror<double> alpha1Old;
        DeviceFieldMirror<double> alpha2Old;
        DeviceFieldMirror<double> rho;
        DeviceFieldMirror<double> p_rgh;
        DeviceFieldMirror<double> p;
        DeviceFieldMirror<Vector<double>> U;
        DeviceFieldMirror<Vector<double>> UOld;
        DeviceFieldMirror<double> rAU;
        DeviceFieldMirror<Vector<double>> HbyA;
        DeviceFieldMirror<double> K;       // curvature
        DeviceFieldMirror<double> sigmaK;  // sigma * curvature
        DeviceFieldMirror<double> divU;
        DeviceFieldMirror<double> Su;
        DeviceFieldMirror<double> Sp;

        // Face fields
        DeviceFieldMirror<double> phi;
        DeviceFieldMirror<double> phiOld;
        DeviceFieldMirror<double> alphaPhi1;
        DeviceFieldMirror<double> alphaPhi2;
        DeviceFieldMirror<double> alphaPhi1Corr0;
        DeviceFieldMirror<double> rhoPhi;
        DeviceFieldMirror<double> rAUf;
        DeviceFieldMirror<double> phiHbyA;
        DeviceFieldMirror<double> phig;
        DeviceFieldMirror<Vector<double>> nHatf;
        DeviceFieldMirror<double> snGradAlpha1;
        DeviceFieldMirror<double> surfaceTensionForce;
        DeviceFieldMirror<double> snGradp;   // boundary pressure-state handoff for later nozzle BC work

        // Scratch and matrices
        DeviceScratchArena scratch;
        PersistentLduMapping pressurePattern;
        DeviceLinearSystemBridge pressureBridge;
        DeviceMatrixStorage alphaMatrix;
        DeviceMatrixStorage momentumMatrix;

        // Flags
        bool alphaPhi1Corr0Valid;
        uint64_t hostShadowDirtyMask;
        int lastTimeIndex;
    };
}
```

#### 5.14A Pressure-boundary-state handoff contract

Phase 5 shall export a named pressure-boundary-state contract for later nozzle-boundary work. The contract is owned and refreshed by `DevicePressureCorrector`, and Phase 6 consumes it without host recomputation of pressure boundary data.

```cpp
namespace Foam::gpuVoF
{
    struct PressureBoundaryStateView
    {
        const DeviceBoundaryMaps* boundary;
        const DeviceFieldMirror<double>* phiHbyA;
        const DeviceFieldMirror<double>* phig;
        const DeviceFieldMirror<double>* rAUf;
        const DeviceFieldMirror<double>* rho;
        const DeviceFieldMirror<double>* snGradp;
    };
}
```

**Rules**
- `snGradp` is boundary-addressable and device-resident.
- The view is refreshed before pressure assembly and remains valid through the pressure-correction stage.
- Phase 6 patch executors may consume this view, but they may not rebuild it on the host.

#### 5.15 Class responsibilities

**`DeviceMeshTopology`**
- Owns static topology and geometry arrays:
  - owner,
  - neighbour,
  - face-to-patch map,
  - cell volumes,
  - face area vectors `Sf`,
  - face area magnitudes `magSf`,
  - cell centres / face centres if needed,
  - gravity terms `gh`, `ghf` if part of the solver setup.
- Converts host labels to 32-bit device indices after range validation.
- Never reallocated unless mesh changes; mesh change is unsupported in Phase 5.

**`DeviceBoundaryMaps`**
- Owns patch descriptors and compacted face lists.
- Stores supported BC enums and per-patch coefficient blocks.
- Provides device-side lookup tables for boundary kernels.

**`DeviceScratchArena`**
- One per solver instance.
- Owns all transient scratch buffers required by alpha, interface, momentum, and pressure stages.
- Allocated once using the memory pool.
- Never allocates from inside hot kernels or stage functions.

**`PersistentLduMapping`**
- Owns row pointers, column indices, and value-mapping indirection for the pressure backend.
- Built once from mesh owner/neighbour data and the LDU addressing pattern.
- Exposes `updateValues()` without rebuilding topology.

**`DeviceLinearSystemBridge`**
- Wraps native or AmgX backend.
- Accepts updated coefficient values and RHS.
- Returns solve statistics without forcing field-scale host copies.

**`DeviceAlphaTransport`**
- Forms alpha fluxes.
- Assembles/solves predictor branch.
- Applies MULES corrections.
- Handles subcycling and previous-correction flux persistence.

**`DeviceMULES`**
- Provides only the overload subset actually exercised by the target alpha path.
- Does not attempt to port all generic MULES entry points in Phase 5.

**`DeviceMixtureProperties`**
- Updates `rho`, `rhoPhi`, and any transport-property intermediates required downstream.

**`DeviceSurfaceTension`**
- Implements `correctInterface()`, `nHatf`, curvature, `sigmaK`, and face force flux for the supported model subset.

**`DeviceMomentumPredictor`**
- Assembles the mixture momentum matrix and optionally solves it.
- Maintains `rAU` for the pressure stage.
- Reuses existing SPUMA device operators where possible.

**`DevicePressureCorrector`**
- Assembles and solves the pressure system.
- Updates `phi`, `U`, `p`, and `p_rgh`.
- Maintains non-orthogonal correction loops entirely on device except for residual/status scalars.
- Exports and refreshes `PressureBoundaryStateView`, including boundary-packed `snGradp` or equivalent local pressure-boundary state, for later Phase 6 boundary execution.

### Data model / memory model

#### 5.16 Memory domains

| Domain | Usage in Phase 5 | Allowed contents |
|---|---|---|
| Device memory | Production persistent state and hot scratch | topology, fields, matrices, scratch, patch maps |
| Pinned host memory | Write/output staging and explicit debug downloads | field staging buffers, diagnostics snapshots |
| Unified memory | Bring-up/debug only | incomplete ports, fault tracing, temporary diagnostic modes |

**Normative rule:** Production mode for Phase 5 is **deviceResident**. UM is not an accepted production residency model.

#### 5.17 Field authority model

| Field family | Authoritative location during timestep | When host copy may be refreshed |
|---|---|---|
| `alpha1`, `alpha2`, old-time alpha | Device | write time, debug, stageFallback |
| `phi`, `alphaPhi1`, `alphaPhi2`, `rhoPhi` | Device | write time, debug, stageFallback |
| `rho`, transport-property auxiliaries | Device | write time, debug |
| `nHatf`, curvature, surface-tension force | Device | debug only unless explicitly written |
| `U`, `p_rgh`, `p`, `rAU`, `HbyA` | Device | write time, debug, stageFallback |
| Patch coefficients | Host for scalar control values, device for packed execution data | uploaded at stage entry only if time-dependent |
| Linear-solver residual/status scalars | Host copy allowed | every solve |

**Normative rule:** Host fields are stale once a GPU stage has modified their device mirrors. They shall not be read implicitly in performance mode.

#### 5.18 Persistent device objects

**Allocate once at `DeviceVoFOrchestrator::initialize()`**
- `DeviceMeshTopology`
- `DeviceBoundaryMaps`
- all `DeviceFieldMirror` buffers
- `DeviceScratchArena`
- `PersistentLduMapping`
- backend descriptors/handles for native and/or AmgX pressure solve

**Reused every timestep**
- all field storage,
- all scratch buffers,
- all linear-system mapping and solve descriptors.

**Reallocated only when**
- mesh changes (unsupported),
- user switches case or restarts process,
- selected backend changes between runs.

#### 5.19 Boundary representation

Create a compact device boundary representation rather than relying on host patch object traversal during each stage.

```cpp
namespace Foam::gpuVoF
{
    enum class PatchKind : uint8_t
    {
        Wall,
        GenericPatch,
        SymmetryPlane,
        Empty,
        Wedge,
        Unsupported
    };

    enum class ScalarBcKind : uint8_t
    {
        FixedValue,
        ZeroGradient,
        Calculated,
        Unsupported
    };

    enum class VectorBcKind : uint8_t
    {
        FixedValue,
        NoSlip,
        Slip,
        SymmetryPlane,
        Calculated,
        Unsupported
    };

    struct PatchDescriptor
    {
        PatchKind patchKind;

        ScalarBcKind alphaBc;
        ScalarBcKind pRghBc;
        ScalarBcKind pBc;

        VectorBcKind UBc;

        int faceStartCompact;
        int faceCount;
        int coeffOffsetScalar;
        int coeffOffsetVector;
    };

    struct DeviceBoundaryMaps
    {
        DeviceArray<PatchDescriptor> patches;
        DeviceArray<int32_t> compactFaceIds;
        DeviceArray<double> scalarCoeffs;
        DeviceArray<Vector<double>> vectorCoeffs;
    };
}
```

**Phase 5 supported BC set**
- Scalars: `fixedValue`, `zeroGradient`, `calculated`
- Vectors: `fixedValue`, `noSlip`, `slip`, `symmetryPlane`, `calculated`
- Geometry patches: `wall`, `patch`, `symmetryPlane`, `empty`, `wedge`

**Phase 5 explicitly unsupported**
- contact-angle alpha patch fields,
- `pressureInletOutletVelocity`,
- nozzle-specific swirl inlets,
- `fixedFluxPressure` unless later signed off and implemented,
- processor/coupled patches,
- arbitrary coded BCs.

#### 5.20 Old-time state

Phase 5 must explicitly mirror time-history state on device.

**Required old-time mirrors**
- `alpha1Old`
- `alpha2Old`
- `phiOld`
- `UOld`
- any additional old-time fields required by the chosen ddt schemes

**Rules**
- On initialization from a fresh case, seed old-time mirrors from current fields.
- On restart from disk, upload both current and old-time fields if available.
- At `endTimeStep()`, perform device-to-device copies to old-time mirrors.
- Never rebuild old-time state from host `oldTime()` during the hot path.

#### 5.20A Restart / checkpoint semantics

Phase 5 acceptance includes restart from written time for the supported case set.

**Rules**
- A write/restart cycle must preserve the current fields required for solver restart and the old-time/device-history state required by the accepted ddt/MULES semantics.
- At minimum the restart contract covers current `alpha1`, `alpha2`, `phi`, `U`, `p_rgh`, `p`, `rho`, and the history objects `alpha1Old`, `alpha2Old`, `phiOld`, `UOld`, plus any persisted previous-correction state such as `alphaPhi1Corr0` and its validity flag when `alphaApplyPrevCorr` is enabled.
- `uploadInitialState()` must load these values directly into device mirrors on restart. Reconstructing missing old-time state from current host fields is allowed only for fresh starts, not restart acceptance.
- If the standard write path cannot preserve a required history object, Phase 5 must either emit an explicit auxiliary restart bundle or reject that restart mode in performance acceptance.


#### 5.21 Mixture-property representation

For the incompressible two-phase baseline, the device mixture update shall implement:

- `alpha2 = 1 - alpha1`
- `rho = alpha1*rho1 + alpha2*rho2`
- `rhoPhi = alphaPhi1*(rho1 - rho2) + phi*rho2`

The `rhoPhi` formula above is an engineering inference from standard incompressible two-phase mixture semantics and must be checked against the actual SPUMA/v2412 solver source during symbol reconciliation. If the local source uses an equivalent but differently factored formulation, follow the source exactly.

Transport-property update requirements:

- Required now: constant-phase-property blending needed by the current reduced cases.
- Deferred: temperature-dependent or advanced mixture models.
- Fail-fast: any model whose required property update cannot be reproduced on device in Phase 5.

#### 5.22 Matrix and index model

- Device topology indices default to `int32_t`.
- Field values default to `double`.
- Pressure backend mapping stores:
  - CSR row offsets,
  - CSR column indices,
  - LDU-to-CSR value indirection,
  - optional boundary/reference-cell metadata.

**Normative rule:** topology/mapping build occurs once per mesh. `updateValues()` is called each corrector; `rebuildPattern()` is forbidden in the hot path unless the solver has changed mesh topology, which Phase 5 rejects.

#### 5.23 Memory budget guidance

For planning and code review, use the following back-of-the-envelope device memory estimates:

- scalar cell field: `8 * nCells` bytes
- vector cell field: `24 * nCells` bytes
- scalar face field: `8 * nFaces` bytes
- vector face field: `24 * nFaces` bytes

For a ~2.5M-cell / several-million-face case, a dozen scalar cell fields, several vector cell fields, several scalar face fields, boundary data, and two persistent matrix workspaces will consume multiple gigabytes but should still fit within 16 GB if:
- 32-bit indices are used,
- duplicate fields are minimized,
- scratch buffers are reused,
- topology is not duplicated across backends unnecessarily.

**Recommendation:** Add a `reportMemoryFootprint()` method that prints a per-subsystem estimate after initialization and before the first timestep.

### Algorithms and control flow

#### 5.24 High-level host orchestration

The solver adapter shall preserve host-side control flow but replace hot numerical stages with device calls.

```cpp
void gpuVoFEnabledSolve(HostSolverBindings& host, DeviceVoFOrchestrator& gpu)
{
    gpu.initialize();
    gpu.uploadInitialState();

    while (runTime.run())
    {
        TimeStepControlSnapshot ts = buildTimeStepSnapshot(host);
        gpu.beginTimeStep(ts);

        while (host.pimple.loop())
        {
            AlphaControlsSnapshot alphaCtrl = gpu.buildAlphaControls();
            gpu.alphaPredictor(alphaCtrl);

            MixtureControlsSnapshot mixCtrl = gpu.buildMixtureControls();
            gpu.updateMixture(mixCtrl);

            SurfaceTensionControlsSnapshot stCtrl = gpu.buildSurfaceTensionControls();
            gpu.correctInterface(stCtrl);

            MomentumControlsSnapshot momCtrl = gpu.buildMomentumControls();
            gpu.momentumPredictor(momCtrl);

            PressureControlsSnapshot pCtrl = gpu.buildPressureControls();
            gpu.pressureCorrector(pCtrl);
        }

        gpu.endTimeStep(ts);

        if (ts.writeNow)
        {
            gpu.commitForWrite();
            runTime.write();
        }
    }
}
```

**Normative rule:** the host orchestration above may exchange only control scalars and write-time staging data with the device. It may not pull full fields during each stage.

#### 5.25 Alpha predictor control flow

Phase 5 shall implement the following control sequence for alpha transport.

```cpp
void DeviceAlphaTransport::run
(
    DeviceVoFState& s,
    const AlphaControlsSnapshot& c,
    cudaStream_t stream
)
{
    // 1. Pre-subcycle setup
    //    - s.phi is already authoritative on device
    //    - s.alpha1Old/s.alpha2Old hold previous time state

    if (c.nAlphaSubCycles == 1)
    {
        runSingleAlphaSolve(s, c, stream);
        return;
    }

    zeroField(s.alphaPhi1, stream);   // accumulated end-of-step alpha flux

    for (int sc = 0; sc < c.nAlphaSubCycles; ++sc)
    {
        prepareSubCycleState(s, c, sc, stream);
        runSingleAlphaSolve(s, c, stream);
        accumulateSubCycleAlphaFlux(s, c, sc, stream);
    }

    finalizeSubCycledAlphaFlux(s, c, stream);
}
```

The `runSingleAlphaSolve()` algorithm shall be:

1. Build or alias the advective flux used for alpha transport.
2. Compute any required `divU`, `Su`, and `Sp` terms for the current model subset.
3. Compute unbounded advective phase flux `alphaPhi1Un`.
4. If `MULESCorr`:
   - assemble predictor matrix,
   - solve predictor,
   - obtain predicted alpha flux,
   - compute correction flux,
   - optionally apply previous correction flux if `alphaApplyPrevCorr`,
   - apply current correction flux with bounded limiter,
   - under-relax correction for correction loops after the first, matching source semantics.
5. Else:
   - call the explicit bounded solve path using the ported MULES subset.
6. Update `alpha2 = 1 - alpha1`.
7. Update `alphaPhi2 = phi - alphaPhi1`.
8. Persist `alphaPhi1Corr0` if previous-correction logic is enabled.
9. Mark alpha-related host shadows dirty.

#### 5.26 Alpha control scalar computation

To preserve existing solver semantics while minimizing transfers:

1. Compute `alphaCoNum` on device via reduction.
2. Copy the resulting scalar to host.
3. Evaluate `nAlphaSubCyclesPtr->value(alphaCoNum)` on host and build `AlphaControlsSnapshot`.
4. Upload only the resulting small snapshot/scalars.

This is acceptable Phase 5 control traffic. It avoids field transfers while preserving the host-side runtime control object.

#### 5.27 Predictor matrix assembly

The alpha predictor matrix assembly shall follow the local source semantics exactly after symbol reconciliation. The baseline structure is:

- implicit time term,
- divergence term using the selected alpha flux,
- explicit or implicit source contributions as required by the MULES branch,
- boundedness-preserving handling of `Sp` and `Su`.

**Implementation requirement**
- Reuse existing SPUMA/native matrix assembly helpers if possible.
- If those helpers are insufficient for alpha-specific terms, implement a dedicated alpha matrix assembler using the mesh topology and field mirrors.
- Do not rebuild matrix topology each alpha correction; only update coefficient values and RHS.

#### 5.28 MULES subset to port

Port only the MULES entry points actually exercised by the target alpha path:

Required:
- correction path used by `MULESCorr`
- explicitSolve path used when `MULESCorr == false`

Optional later:
- generic overloads not used by `twoPhaseSolver::alphaPredictor()`

**Normative rule:** Do not attempt a full generic MULES port in Phase 5.

#### 5.29 Device MULES kernel strategy

The initial MULES port shall use a conservative kernel breakdown:

1. compute/pack limiter inputs,
2. perform face-based correction-limiter evaluation,
3. apply limited correction to alpha and fluxes,
4. clamp/validate boundedness,
5. accumulate diagnostics.

**Initial performance strategy**
- face-loop kernels with atomics where required,
- no host sync between these kernels,
- scratch buffers allocated once.

**Do not do now**
- segmented reduction / cell-gather rewrite,
- speculative warp-specialized limiter reformulation.

#### 5.30 Mixture-property update flow

Immediately after alpha update, call:

```cpp
void DeviceMixtureProperties::update
(
    DeviceVoFState& s,
    const MixtureControlsSnapshot& c,
    cudaStream_t stream
);
```

Required work:
1. update `alpha2` if not already done,
2. update `rho`,
3. update `rhoPhi`,
4. update any per-cell transport-property auxiliaries needed by the momentum equation,
5. mark mixture-related host shadows dirty.

**Normative rule:** This stage must not read back `alpha1` to host.

#### 5.31 Interface and surface-tension flow

The supported baseline algorithm is:

1. compute or refresh interface normal-related quantities from `alpha1`,
2. compute `nHatf`,
3. compute curvature `K`,
4. compute `sigmaK = sigma * K`,
5. compute face force flux `surfaceTensionForce`,
6. make the result available to pressure correction.

```cpp
void DeviceSurfaceTension::correct
(
    DeviceVoFState& s,
    const SurfaceTensionControlsSnapshot& c,
    cudaStream_t stream
);
```

**Required restrictions**
- constant sigma only,
- no dynamic contact angle,
- no wall-contact-angle correction.

**Kernel baseline**
- gradient/curvature kernels may use atomics,
- compact patch lists for supported wall/symmetry/patch types,
- no host patch traversal in the hot path.

#### 5.32 Momentum predictor flow

The momentum predictor must:
1. assemble the mixture momentum equation using the authoritative device fields,
2. solve the predictor if enabled in controls,
3. compute and store `rAU`,
4. leave all required intermediates device-resident for pressure correction.

```cpp
void DeviceMomentumPredictor::run
(
    DeviceVoFState& s,
    const MomentumControlsSnapshot& c,
    cudaStream_t stream
);
```

**Implementation rule**
- Reuse existing SPUMA device operators and matrix/solve paths wherever possible.
- Do not create a second independent sparse infrastructure if SPUMA already has the required one.

**Important coupling**
- `rAU` lifetime must extend through the pressure corrector.
- `DevicePressureCorrector` may not assume host `rAU`.

#### 5.33 Pressure corrector flow

The incompressible pressure path shall follow the local solver semantics and, after symbol reconciliation, should match the Foundation reference structure closely:

1. interpolate `rAUf`,
2. compute `HbyA`,
3. compute `phiHbyA`,
4. compute/add `phig` using surface tension and buoyancy terms if enabled,
5. refresh `PressureBoundaryStateView` and compute/store boundary `snGradp` or equivalent local pressure-boundary state required by the supported pressure path,
6. constrain pressure boundary contributions,
7. assemble/update pressure matrix coefficients,
8. solve through native or AmgX backend,
8. update `phi`,
9. reconstruct/correct `U`,
10. update `p` / `p_rgh`,
11. clear `rAU` only when the full pressure stage is complete.

```cpp
void DevicePressureCorrector::run
(
    DeviceVoFState& s,
    const PressureControlsSnapshot& c,
    cudaStream_t stream
);
```

**Backend interface rule**
- `DevicePressureCorrector` may choose backend at runtime via `PressureControlsSnapshot::backend`.
- All non-solver work remains identical between backends.

#### 5.34 Non-orthogonal correction loops

The non-orthogonal loop count remains a host-side control scalar in Phase 5, but all actual loop work stays on the device.

```cpp
for (int corr = 0; corr < c.nNonOrthCorr; ++corr)
{
    updatePressureValuesOnly(...);
    backend.solve(...);

    if (corr == c.nNonOrthCorr - 1)
    {
        updateFluxAndVelocity(...);
    }
}
```

**Normative rule**
- do not rebuild pressure sparsity pattern in each non-orthogonal loop,
- do not commit pressure fields to host between non-orthogonal iterations.

#### 5.35 Commit and write flow

```cpp
void DeviceVoFOrchestrator::commitForWrite()
{
    // Download only fields needed for output / functionObjects at write time.
    // Use pinned host staging.
    // Update host field objects and mark shadows fresh.
}
```

Required write set for baseline:
- `alpha1`, `alpha2`,
- `U`,
- `p_rgh`, `p`,
- `rho`,
- any additional user-requested write fields explicitly enabled in config.

Optional debug write set:
- `alphaPhi1`, `rhoPhi`, `K`, `sigmaK`, `surfaceTensionForce`, `rAU`, `HbyA`.

Required restart bundle when restart validation is enabled:
- `alpha1Old`, `alpha2Old`,
- `phiOld`, `UOld`,
- `alphaPhi1Corr0` plus its validity state when `alphaApplyPrevCorr` is enabled.

If the standard case write set cannot preserve these objects, the implementation must emit an auxiliary restart bundle rather than silently reseeding them from current host fields.

**Normative rule:** no implicit write-side commit of every scratch field.

### Required source changes

The exact file paths must be reconciled to the SPUMA/v2412 tree first. The list below names the required **source-change domains**, not guaranteed final paths.

#### 5.36 Solver/module integration points

1. Add a GPU VOF runtime gate to the relevant solver/module:
   - parse canonical `system/gpuRuntimeDict` / `gpuRuntime.vof`, and normalize any still-accepted legacy `gpuVoF` block into that same runtime view,
   - instantiate `DeviceVoFOrchestrator`,
   - replace host alpha/momentum/pressure stages with device calls when enabled.
2. Add a startup support scan:
   - reject unsupported cases before first timestep.
3. Add write-time commit hooks.
4. Add explicit invalidation of host shadows after device stages.

#### 5.37 New GPU VOF subsystem

Add a new subsystem, tentatively `src/gpu/vof`, containing:
- state,
- alpha,
- MULES,
- mixture,
- interface/surface tension,
- momentum predictor,
- pressure corrector,
- backend bridge,
- instrumentation.

#### 5.38 Existing subsystem extensions

1. Extend allocator/pool integration for persistent VOF fields and scratch.
2. Extend any existing device matrix abstraction to support value-only updates and persistent mapping export for pressure.
3. Extend profiling utilities with VOF-specific NVTX ranges.
4. Extend build system to compile new kernels for Blackwell/SM_120 and include PTX where required. [R19] [R20]

#### 5.39 Dictionary and runtime configuration

Introduce or extend the normalized `gpuRuntime` configuration tree. Phase 5 consumes `gpuRuntime.memory`, `gpuRuntime.pressure`, `gpuRuntime.vof`, `gpuRuntime.profiling`, and `gpuRuntime.acceptance`. A legacy top-level `gpuVoF` block may remain only as a compatibility shim or generated subview during the migration.

```text
gpuRuntime
{
    vof
    {
        enabled                             true;
        mode                                deviceResident;   // off | umBringup | deviceResident
        pressureBackend                     native;           // native | amgx
        requireDeviceDirectForAmgxProduction true;
        fallbackPolicy                      failFast;         // failFast | stageFallback
        commitPolicy                        writeOnly;        // writeOnly | endOfStep | everyStage
        supportedBCPolicy                   strict;           // strict | allowStageFallback
        functionObjectPolicy                strictWriteOnly;  // strictWriteOnly | allowDebugOnly
        profileRanges                       true;
        dumpMemoryReport                    true;
        debugFieldAsserts                   true;
    }
}
```

Compatibility-shim form only, if the branch still accepts it:

```text
gpuVoF
{
    enabled                             true;
    mode                                deviceResident;   // off | umBringup | deviceResident
    pressureBackend                     native;           // native | amgx
    requireDeviceDirectForAmgxProduction true;
    fallbackPolicy                      failFast;         // failFast | stageFallback
    commitPolicy                        writeOnly;        // writeOnly | endOfStep | everyStage
    supportedBCPolicy                   strict;           // strict | allowStageFallback
    functionObjectPolicy                strictWriteOnly;  // strictWriteOnly | allowDebugOnly
    profileRanges                       true;
    dumpMemoryReport                    true;
    debugFieldAsserts                   true;
}
```

**Normative defaults**
- `mode = deviceResident`
- `pressureBackend = native`
- `requireDeviceDirectForAmgxProduction = true`
- `fallbackPolicy = failFast`
- `commitPolicy = writeOnly`
- `supportedBCPolicy = strict`
- `functionObjectPolicy = strictWriteOnly`

Canonical ownership is under `gpuRuntime.*`; the compatibility block above does not create a second independent contract.

### Proposed file layout and module boundaries

The paths below are recommended. Adjust only after the symbol reconciliation step.

```text
src/
  gpu/
    base/
      DeviceFieldMirror.H
      DeviceFieldMirror.C
      DeviceFieldRegistry.H
      DeviceFieldRegistry.C
      DeviceScratchArena.H
      DeviceScratchArena.C
      DeviceProfiling.H
      DeviceProfiling.C

    mesh/
      DeviceMeshTopology.H
      DeviceMeshTopology.C
      DeviceBoundaryMaps.H
      DeviceBoundaryMaps.C

    linalg/
      PersistentLduMapping.H
      PersistentLduMapping.C
      DeviceMatrixStorage.H
      DeviceMatrixStorage.C
      DeviceLinearSystemBridge.H
      DeviceLinearSystemBridge.C
      NativePressureBackend.H
      NativePressureBackend.C
      AmgxPressureBackend.H
      AmgxPressureBackend.C

    vof/
      DeviceVoFControls.H
      DeviceVoFControls.C
      DeviceVoFState.H
      DeviceVoFState.C
      DeviceVoFSupportScan.H
      DeviceVoFSupportScan.C
      DeviceVoFOrchestrator.H
      DeviceVoFOrchestrator.C

      DeviceAlphaTransport.H
      DeviceAlphaTransport.C
      DeviceMULES.H
      DeviceMULES.C
      DeviceMixtureProperties.H
      DeviceMixtureProperties.C
      DeviceSurfaceTension.H
      DeviceSurfaceTension.C
      DeviceMomentumPredictor.H
      DeviceMomentumPredictor.C
      DevicePressureCorrector.H
      DevicePressureCorrector.C

      kernels/
        AlphaKernels.cu
        MulesKernels.cu
        MixtureKernels.cu
        SurfaceTensionKernels.cu
        MomentumKernels.cu
        PressureKernels.cu
        BoundaryKernels.cu
        ReductionKernels.cu

applications/
  modules/
    incompressibleVoF/
      gpuVoFAdapter.H
      gpuVoFAdapter.C
      gpuVoFDict.H
      gpuVoFDict.C

tests/
  gpuVoF/
    unit/
    integration/
    benchmarks/
```

#### 5.40 Ownership boundaries

| Module | Owner role | Responsibility |
|---|---|---|
| `gpu/base` | GPU runtime owner | allocators, mirrors, scratch, profiling helpers |
| `gpu/mesh` | GPU mesh owner | topology extraction, patch compaction |
| `gpu/linalg` | solver-backend owner | persistent mappings, backend bridges |
| `gpu/vof` | two-phase numerics owner | alpha, MULES, mixture, interface, pressure orchestration |
| solver adapter | integration owner | hooks into SPUMA/OpenFOAM solver/module |
| tests | validation owner | unit, regression, benchmark harness |

### Pseudocode

#### 5.41 Initialization and support scan

```cpp
DeviceVoFOrchestrator::DeviceVoFOrchestrator
(
    const HostSolverBindings& host,
    const dictionary& gpuVoFDict
)
:
    host_(host),
    cfg_(parseGpuVoFDict(gpuVoFDict)),
    stream_(createPrimaryStream()),
    state_(nullptr)
{}

CaseSupportReport DeviceVoFOrchestrator::inspectSupport() const
{
    return inspectCaseSupport(host_, cfg_.dict());
}

void DeviceVoFOrchestrator::initialize()
{
    CaseSupportReport report = inspectSupport();
    if (!report.supported)
    {
        throw FatalError(report.toString());
    }

    state_.reset(new DeviceVoFState);

    state_->mesh.buildFrom(host_.mesh);              // validates 32-bit range
    state_->boundary.buildFrom(host_);               // compacts patch maps, coeff blocks
    state_->allocatePersistentFields(host_, cfg_);   // all mirrors and scratch
    state_->pressurePattern.buildOnce(host_, state_->mesh);
    state_->pressureBridge.initialize(cfg_.pressureBackend, state_->pressurePattern);

    if (cfg_.dumpMemoryReport)
    {
        state_->reportMemoryFootprint();
    }
}
```

#### 5.42 Initial upload

```cpp
void DeviceVoFOrchestrator::uploadInitialState()
{
    state_->alpha1.uploadFromHost(host_.alpha1.internalField(), host_.alpha1.boundaryField(), stream_);
    state_->alpha2.uploadFromHost(host_.alpha2.internalField(), host_.alpha2.boundaryField(), stream_);

    state_->phi.uploadFromHost(host_.phi.internalField(), host_.phi.boundaryField(), stream_);
    state_->rho.uploadFromHost(host_.rho.internalField(), host_.rho.boundaryField(), stream_);

    state_->U.uploadFromHost(host_.U.internalField(), host_.U.boundaryField(), stream_);
    state_->p_rgh.uploadFromHost(host_.p_rgh.internalField(), host_.p_rgh.boundaryField(), stream_);
    state_->p.uploadFromHost(host_.p.internalField(), host_.p.boundaryField(), stream_);
    state_->rhoPhi.uploadFromHost(host_.rhoPhi.internalField(), host_.rhoPhi.boundaryField(), stream_);

    // old-time mirrors / restart history
    if (isRestartCase(host_))
    {
        uploadRestartHistory(*state_, host_, stream_);
    }
    else
    {
        seedOldTimeFromCurrent(*state_, stream_);
    }

    state_->markAllDeviceFresh();
}
```

#### 5.43 Alpha control build

```cpp
AlphaControlsSnapshot DeviceVoFOrchestrator::buildAlphaControls()
{
    AlphaControlsSnapshot c{};
    c.ddtScheme = readAndValidateAlphaDdtScheme(host_.fvSchemes);   // Euler only in Phase 5
    c.nAlphaCorr = readInt(host_.fvSolution, "nAlphaCorr");
    c.MULESCorr = readBool(host_.fvSolution, "MULESCorr");
    c.alphaApplyPrevCorr = readBool(host_.fvSolution, "alphaApplyPrevCorr");
    c.cAlpha = readScalar(host_.fvSolution, "cAlpha");
    c.icAlpha = readScalar(host_.fvSolution, "icAlpha");

    // reduction on device, scalar copy to host
    c.alphaCoNum = DeviceAlphaTransport::computeAlphaCoNum(*state_, stream_);
    c.nAlphaSubCycles = evaluateSubCycleControlOnHost(c.alphaCoNum, host_);

    return c;
}
```

#### 5.44 Alpha predictor stage

```cpp
void DeviceVoFOrchestrator::alphaPredictor(const AlphaControlsSnapshot& c)
{
    NVTX_RANGE("alphaPredictor");

    DeviceAlphaTransport alpha;
    alpha.run(*state_, c, stream_);

    state_->hostShadowDirtyMask |= FieldMask::Alpha | FieldMask::Flux;
}
```

#### 5.45 Device alpha solve (baseline semantic version)

```cpp
void DeviceAlphaTransport::runSingleAlphaSolve
(
    DeviceVoFState& s,
    const AlphaControlsSnapshot& c,
    cudaStream_t stream
)
{
    // 1. advective alpha flux
    launchComputeAlphaPhiUnKernel
    (
        s.mesh,
        s.alpha1.dInternal(),
        s.phi.dInternal(),
        s.scratch.alphaPhi1Un,
        c.cAlpha,
        c.icAlpha,
        stream
    );

    // 2. sources required by the target local solver semantics
    launchComputeDivUSuSpKernel
    (
        s.mesh,
        s.phi.dInternal(),
        s.alpha1.dInternal(),
        s.divU.dInternal(),
        s.Su.dInternal(),
        s.Sp.dInternal(),
        stream
    );

    if (c.MULESCorr)
    {
        assembleAlphaPredictorMatrix(s, c, stream);    // values only
        solveAlphaPredictorMatrix(s.alphaMatrix, s.alpha1, stream);

        launchExtractPredictedAlphaFluxKernel
        (
            s.alphaMatrix.faceFlux(),
            s.alphaPhi1.dInternal(),
            stream
        );

        launchComputeAlphaPhiCorrectionKernel
        (
            s.scratch.alphaPhi1Un,
            s.alphaPhi1.dInternal(),
            s.scratch.alphaPhi1Corr,
            s.mesh.nInternalFaces(),
            stream
        );

        if (c.alphaApplyPrevCorr && s.alphaPhi1Corr0Valid)
        {
            DeviceMULES::correct
            (
                s, c,
                s.scratch.alphaPhi1Un,
                s.alphaPhi1Corr0.dInternal(),
                stream
            );
        }

        DeviceMULES::correct
        (
            s, c,
            s.scratch.alphaPhi1Un,
            s.scratch.alphaPhi1Corr,
            stream
        );

        if (c.alphaApplyPrevCorr)
        {
            deviceCopy(s.alphaPhi1Corr0.dInternal(), s.scratch.alphaPhi1Corr, s.mesh.nFaces(), stream);
            s.alphaPhi1Corr0Valid = true;
        }
    }
    else
    {
        deviceCopy(s.alphaPhi1.dInternal(), s.scratch.alphaPhi1Un, s.mesh.nFaces(), stream);
        DeviceMULES::explicitSolve(s, c, stream);
    }

    launchUpdateAlphaComplementKernel
    (
        s.alpha1.dInternal(),
        s.alpha2.dInternal(),
        s.mesh.nCells(),
        stream
    );

    launchComputeAlphaPhi2Kernel
    (
        s.phi.dInternal(),
        s.alphaPhi1.dInternal(),
        s.alphaPhi2.dInternal(),
        s.mesh.nFaces(),
        stream
    );
}
```

#### 5.46 Device MULES correction skeleton

```cpp
void DeviceMULES::correct
(
    DeviceVoFState& s,
    const AlphaControlsSnapshot& c,
    const double* alphaPhi1Un,
    double* alphaPhiCorr,
    cudaStream_t stream
)
{
    // Step 1: compute per-face limiter candidates
    launchMulesFaceLimiterKernel
    (
        s.mesh,
        s.alpha1.dInternal(),
        alphaPhi1Un,
        alphaPhiCorr,
        s.divU.dInternal(),
        s.Su.dInternal(),
        s.Sp.dInternal(),
        s.scratch.faceLimiter,
        stream
    );

    // Step 2: apply limited correction to alpha and alpha flux
    launchApplyLimitedAlphaCorrectionKernel
    (
        s.mesh,
        s.alpha1.dInternal(),
        alphaPhiCorr,
        s.scratch.faceLimiter,
        s.scratch.cellDeltaAlpha,   // atomically accumulated
        stream
    );

    // Step 3: update alpha cell values and enforce boundedness tolerance
    launchFinalizeAlphaKernel
    (
        s.alpha1.dInternal(),
        s.scratch.cellDeltaAlpha,
        s.mesh.nCells(),
        stream
    );

    // Step 4: diagnostics
    launchBoundednessCheckKernel
    (
        s.alpha1.dInternal(),
        s.mesh.nCells(),
        s.scratch.boundStats,
        stream
    );
}
```

#### 5.47 Mixture update

```cpp
void DeviceMixtureProperties::update
(
    DeviceVoFState& s,
    const MixtureControlsSnapshot& c,
    cudaStream_t stream
)
{
    launchUpdateRhoKernel
    (
        s.alpha1.dInternal(),
        s.alpha2.dInternal(),
        c.rho1,
        c.rho2,
        s.rho.dInternal(),
        s.mesh.nCells(),
        stream
    );

    if (c.updateRhoPhi)
    {
        launchUpdateRhoPhiKernel
        (
            s.alphaPhi1.dInternal(),
            s.phi.dInternal(),
            c.rho1,
            c.rho2,
            s.rhoPhi.dInternal(),
            s.mesh.nFaces(),
            stream
        );
    }
}
```

#### 5.48 Surface tension correction

```cpp
void DeviceSurfaceTension::correct
(
    DeviceVoFState& s,
    const SurfaceTensionControlsSnapshot& c,
    cudaStream_t stream
)
{
    if (!c.enabled)
    {
        zeroField(s.surfaceTensionForce, stream);
        return;
    }

    launchComputeSnGradAlphaKernel
    (
        s.mesh,
        s.alpha1.dInternal(),
        s.snGradAlpha1.dInternal(),
        stream
    );

    launchComputeNHatfKernel
    (
        s.mesh,
        s.snGradAlpha1.dInternal(),
        c.deltaN,
        s.nHatf.dInternal(),
        stream
    );

    launchComputeCurvatureKernel
    (
        s.mesh,
        s.nHatf.dInternal(),
        s.K.dInternal(),
        stream
    );

    launchScaleSigmaKKernel
    (
        s.K.dInternal(),
        c.sigma,
        s.sigmaK.dInternal(),
        s.mesh.nCells(),
        stream
    );

    launchSurfaceTensionForceKernel
    (
        s.mesh,
        s.sigmaK.dInternal(),
        s.alpha1.dInternal(),
        s.surfaceTensionForce.dInternal(),
        stream
    );
}
```

#### 5.49 Momentum predictor

```cpp
void DeviceMomentumPredictor::run
(
    DeviceVoFState& s,
    const MomentumControlsSnapshot& c,
    cudaStream_t stream
)
{
    assembleMomentumMatrix(s, c, stream);   // value update only

    if (c.solveMomentumPredictor)
    {
        solveMomentumMatrix(s.momentumMatrix, s.U, stream);
    }

    launchComputeRAUKernel
    (
        s.momentumMatrix.diag(),
        s.rAU.dInternal(),
        s.mesh.nCells(),
        stream
    );
}
```

#### 5.50 Pressure corrector

```cpp
void DevicePressureCorrector::run
(
    DeviceVoFState& s,
    const PressureControlsSnapshot& c,
    cudaStream_t stream
)
{
    launchInterpolateRAUKernel(s.rAU.dInternal(), s.rAUf.dInternal(), s.mesh, stream);

    launchComputeHbyAKernel
    (
        s.momentumMatrix.H(),
        s.rAU.dInternal(),
        s.HbyA.dInternal(),
        s.mesh,
        stream
    );

    launchComputePhiHbyAKernel
    (
        s.HbyA.dInternal(),
        s.rho.dInternal(),
        s.rAU.dInternal(),
        s.phi.dInternal(),
        s.phiHbyA.dInternal(),
        s.mesh,
        stream
    );

    launchComputePhigKernel
    (
        s.surfaceTensionForce.dInternal(),
        s.rAUf.dInternal(),
        s.phig.dInternal(),
        s.mesh,
        stream
    );

    launchAddFaceFluxKernel
    (
        s.phiHbyA.dInternal(),
        s.phig.dInternal(),
        s.mesh.nFaces(),
        stream
    );

    updatePressureBoundaryState(s, c, stream);   // populates PressureBoundaryStateView, including snGradp
    applyPressureBoundaryConstraintsOnDevice(s, c, stream);

    for (int nonOrth = 0; nonOrth < c.nNonOrthCorr; ++nonOrth)
    {
        updatePressureMatrixValuesOnly(s, stream);

        s.pressureBridge.solve
        (
            s.pressurePattern,
            s.p_rgh.dInternal(),
            s.scratch.pressureRhs,
            stream
        );

        if (nonOrth == c.nNonOrthCorr - 1)
        {
            launchUpdatePhiFromPressureFluxKernel
            (
                s.phiHbyA.dInternal(),
                s.pressureBridge.faceFlux(),
                s.phi.dInternal(),
                s.mesh.nFaces(),
                stream
            );

            launchReconstructVelocityKernel
            (
                s.HbyA.dInternal(),
                s.rAU.dInternal(),
                s.rAUf.dInternal(),
                s.phig.dInternal(),
                s.pressureBridge.faceFlux(),
                s.U.dInternal(),
                s.mesh,
                stream
            );
        }
    }

    launchUpdateAbsolutePressureKernel
    (
        s.p_rgh.dInternal(),
        s.rho.dInternal(),
        s.mesh.ghDevice(),
        s.p.dInternal(),
        s.mesh.nCells(),
        stream
    );

    clearRAUAccordingToLocalSolverSemantics(s, stream);
}
```

`updatePressureBoundaryState()` must match the local SPUMA/v2412 pressure-boundary semantics and must not call back into host patch polymorphism. The resulting `PressureBoundaryStateView` is the Phase 5 handoff consumed by Phase 6 nozzle boundary execution.

`clearRAUAccordingToLocalSolverSemantics()` is pseudocode for whatever action the local SPUMA/v2412 solver uses to invalidate or clear the cached reciprocal diagonal after pressure correction. The coding agent must match the local solver semantics exactly rather than unconditionally zeroing `rAU`.

#### 5.51 Write-time commit

```cpp
void DeviceVoFOrchestrator::commitForWrite()
{
    NVTX_RANGE("commitForWrite");

    // Download only write fields using pinned buffers.
    state_->alpha1.downloadToHost(host_.alpha1.internalField(), host_.alpha1.boundaryField(), stream_);
    state_->alpha2.downloadToHost(host_.alpha2.internalField(), host_.alpha2.boundaryField(), stream_);
    state_->U.downloadToHost(host_.U.internalField(), host_.U.boundaryField(), stream_);
    state_->p_rgh.downloadToHost(host_.p_rgh.internalField(), host_.p_rgh.boundaryField(), stream_);
    state_->p.downloadToHost(host_.p.internalField(), host_.p.boundaryField(), stream_);
    state_->rho.downloadToHost(host_.rho.internalField(), host_.rho.boundaryField(), stream_);

    state_->markWriteFieldsHostFresh();
}
```

When restart validation is enabled, `commitForWrite()` must additionally stage the restart bundle defined in 5.20A. The code above shows only the baseline write-field set.

### Step-by-step implementation guide

The sequence below is the required build order for the coding agent.

#### 5.52 Step 1 — Complete the local semantic source audit and reconcile symbols against the actual SPUMA/v2412 tree

**Modify / create**
- Produce a `phase5_symbol_reconciliation.md` artifact from local source inspection, or consume the approved project-level semantic source audit and derive a local Phase 5 patch-target note from it.

**Expected output**
- A mapping table:
  - Foundation semantic object/function
  - actual SPUMA/v2412 file/class/function
  - notes on differences
- The note must also identify the exact SPUMA commit/tag and the exact `foamExternalSolvers` / AmgX revisions used here, or explicitly reference the approved master pin manifest that freezes them.

#### 5.53 Step 2 — Add normalized `gpuRuntime` configuration (`gpuVoF` compatibility shim) and runtime gate

**Modify**
- Solver/module setup path.
- Add the canonical `gpuRuntime.vof` schema and parser, typically under `system/gpuRuntimeDict`.
- Keep `gpuVoF` only as a compatibility shim or generated subview if legacy callers still require it.

**Why**
- Need explicit enable/disable path, backend selection, fail-fast policy, and a single authoritative runtime contract that later phases consume.

**Expected output**
- Canonical `gpuRuntime.vof` config exists and drives the Phase 5 runtime gate.
- `gpuVoF` can be translated or disabled as a compatibility shim without affecting CPU solver behavior.
- Startup prints parsed mode/backend/fallback settings from the normalized tree.

**Verify success**
- CPU runs unchanged with GPU controls disabled through the canonical tree.
- GPU mode refuses unsupported cases before the first timestep.
- If a legacy `gpuVoF` block is still accepted, it resolves to the same normalized runtime state as `gpuRuntime.vof`.

**Likely breakages**
- runtime dictionary lookup failures,
- inconsistent defaults between `gpuRuntime.vof` and the `gpuVoF` shim,
- enabling GPU path too early.

#### 5.54 Step 3 — Implement `CaseSupportReport`

**Modify**
- Add support scanner for mesh, schemes, patch types, functionObjects, models.

**Why**
- Unsupported cases must not discover themselves via hidden host touches.

**Expected output**
- Deterministic support report at startup.

**Verify success**
- Unsupported contact-angle case fails before stepping.
- Supported generic case passes scan.

**Likely breakages**
- incomplete patch-type inspection,
- missing functionObject scan,
- false positives from unused fields.

#### 5.55 Step 4 — Implement `DeviceMeshTopology` and `DeviceBoundaryMaps`

**Modify**
- Add topology extraction and patch compaction.

**Why**
- Every later kernel depends on stable device topology and patch descriptors.

**Expected output**
- One-time upload of owner/neighbour, geometry, patch descriptors.

**Verify success**
- Unit tests compare device-extracted topology/patch maps to host originals.
- Memory report prints sizes and index ranges.

**Likely breakages**
- boundary-face ordering mistakes,
- label-width overflow,
- wrong patch compaction offsets.

#### 5.56 Step 5 — Implement `DeviceFieldMirror` and `DeviceVoFState`

**Modify**
- Add mirror abstraction, sync state tracking, persistent state allocation.

**Why**
- This is the core data-plane architecture.

**Expected output**
- All core fields can be allocated, uploaded, and committed explicitly.

**Verify success**
- Unit tests upload/download random field data with bitwise equality in FP64.
- Host-shadow dirty/fresh transitions behave as specified.

**Likely breakages**
- stale sync metadata,
- missing boundary buffer handling,
- unintended host-side aliasing.

#### 5.57 Step 6 — Implement initialization upload and old-time seeding

**Modify**
- `uploadInitialState()`
- `seedOldTimeFromCurrent()`

**Why**
- Alpha/momentum/pressure paths require consistent old-time state.

**Expected output**
- Device fields and their old-time mirrors populated before the first timestep.

**Verify success**
- First timestep on a trivial case matches CPU for one step.
- Restart case seeds both current and old-time correctly.

**Likely breakages**
- old-time mismatch,
- forgetting `phiOld` or `UOld`,
- stale current/old copies.

#### 5.58 Step 7 — Implement alpha Courant reduction and controls snapshots

**Modify**
- device reduction kernels
- host snapshot builder

**Why**
- Need subcycle count without field transfers.

**Expected output**
- Valid `AlphaControlsSnapshot` built each step from device-computed scalar data.

**Verify success**
- `alphaCoNum` matches CPU reference within tolerance.
- `nAlphaSubCycles` matches CPU decision.

**Likely breakages**
- reduction mismatch on boundary faces,
- incorrect absolute/relative flux usage,
- host control mismatch.

#### 5.59 Step 8 — Implement alpha flux formation and no-op predictor scaffold

**Modify**
- `DeviceAlphaTransport`
- basic kernels for `alphaPhi1Un`, `alpha2`, `alphaPhi2`

**Why**
- Build the transport skeleton before full MULES.

**Expected output**
- Stage runs end-to-end with placeholder or pass-through logic on test cases.

**Verify success**
- Field movement through the stage is consistent.
- No host sync or field download occurs.

**Likely breakages**
- face indexing mistakes,
- wrong boundary contributions,
- accidental host access.

#### 5.60 Step 9 — Implement predictor matrix assembly and solve path

**Modify**
- `DeviceMatrixStorage`
- alpha matrix assembly values/RHS
- native solve invocation for alpha predictor branch

**Why**
- Required for the `MULESCorr` path.

**Expected output**
- Predictor alpha equation solves on device.

**Verify success**
- On a small controlled case, predictor-only results match CPU reference before limiter correction.

**Likely breakages**
- wrong sign conventions,
- bad diagonal/source assembly,
- boundary condition insertion errors.

#### 5.61 Step 10 — Port the minimal MULES subset

**Modify**
- `DeviceMULES`
- face-limiter and alpha-update kernels

**Why**
- This is the boundedness-critical heart of the alpha path.

**Expected output**
- `MULESCorr` and explicit solve path both work for supported cases.

**Verify success**
- `alpha1` remains bounded within tolerance.
- `alpha1 + alpha2` remains approximately 1.
- Generic dam-break style benchmark no longer diverges or develops obvious unboundedness.

**Likely breakages**
- limiter logic sign errors,
- wrong accumulation target,
- under-relaxation mismatch,
- forgetting previous-correction flux behavior.

#### 5.62 Step 11 — Add alpha subcycling and previous-correction persistence

**Modify**
- subcycle loop orchestration
- `alphaPhi1` accumulation
- `alphaPhi1Corr0` validity/lifetime

**Why**
- Production transient VOF runs depend on this.

**Expected output**
- Multi-subcycle alpha path matches CPU decisions and state evolution.

**Verify success**
- `nAlphaSubCycles > 1` cases run and match CPU qualitatively and quantitatively.
- `alphaApplyPrevCorr` behavior matches CPU run flags.

**Likely breakages**
- bad accumulation scaling,
- stale `alphaPhi1Corr0`,
- using wrong `deltaT` in subcycles.

#### 5.63 Step 12 — Implement device mixture-property updates

**Modify**
- `DeviceMixtureProperties`

**Why**
- Pressure and momentum paths depend on updated `rho`/`rhoPhi`.

**Expected output**
- `rho`, `rhoPhi`, and any required transport auxiliaries updated on device every step.

**Verify success**
- CPU vs GPU comparison for `rho` and `rhoPhi` on validation cases.
- No host transfer visible in Nsight during mixture update.

**Likely breakages**
- formula mismatch with actual local solver,
- stale `alpha2`,
- boundary field inconsistency.

#### 5.64 Step 13 — Implement interface and surface-tension baseline

**Modify**
- `DeviceSurfaceTension`
- gradient, normal, curvature, sigmaK, and force kernels

**Why**
- Pressure correction must see device-resident surface-tension forcing.

**Expected output**
- Constant-sigma cases run without host-side interface calculations.

**Verify success**
- Static droplet / Laplace pressure test reproduces CPU/reference trend.
- Uniform-alpha test produces zero or near-zero surface-tension force.
- No contact-angle cases run; contact-angle case is rejected at support scan.

**Likely breakages**
- incorrect face orientation,
- curvature sign errors,
- noisy spurious currents due to gradient mistakes,
- hidden host patch traversal.

#### 5.65 Step 14 — Integrate momentum predictor

**Modify**
- `DeviceMomentumPredictor`
- solver adapter hook for momentum stage

**Why**
- `rAU` and pressure coupling depend on it.

**Expected output**
- Momentum equation assembles/solves on device with `rAU` cached.

**Verify success**
- Laminar validation case matches CPU U-field trend.
- `rAU` nonzero/finite and device-resident before pressure stage.

**Likely breakages**
- wrong transport coefficients,
- turbulence coupling mismatch,
- forgetting to preserve `rAU`.

#### 5.66 Step 15 — Integrate pressure corrector with runtime-selectable backend

**Modify**
- `DevicePressureCorrector`
- `DeviceLinearSystemBridge`
- solver adapter hook for pressure stage

**Why**
- This completes the device-resident VOF/PIMPLE core.

**Expected output**
- Pressure correction runs in native mode without field-scale host transfer.
- AmgX runs without field-scale host transfer only when the Phase 4 `DeviceDirect` bridge is present; otherwise any AmgX result in this step is explicitly labeled correctness-only bring-up.

**Verify success**
- `R1-core` converges with the native backend.
- `R1-core` converges with AmgX only in the correctly labeled mode for the current bridge state (`DeviceDirect` production-resident or correctness-only bring-up).
- `phi`, `U`, `p`, `p_rgh` match CPU within thresholds.
- Topology mapping is built once and values are updated thereafter.

**Likely breakages**
- bad flux reconstruction,
- wrong reference-cell handling,
- non-orthogonal loop mis-sequencing,
- backend value-order mismatch.

#### 5.67 Step 16 — Implement write-time commit path, restart/reload parity, and unsafe-read asserts

**Expected output**
- Writes succeed only after explicit commit.
- Restart from a written time reloads current and required old-time state into the device mirrors without ad hoc host reconstruction.
- Debug builds detect host reads of stale fields.

**Verify success**
- Write-time fields match GPU state.
- A write/restart/write cycle on a supported case preserves current fields and required old-time/device-history state within the accepted tolerances.
- Performance run shows no per-stage host commits.

**Likely breakages**
- missing boundary commits,
- stale field metadata after write,
- unsafe functionObjects,
- missing restart bundle fields or stale old-time metadata on reload.

#### 5.68 Step 17 — Instrument and profile

**Modify**
- NVTX ranges
- optional counters
- memory report hooks

**Why**
- Residency and launch behavior must be demonstrated, not assumed.

**Expected output**
- Nsight timelines clearly show alpha, mixture, surface tension, momentum, pressure, commit stages.

**Verify success**
- No hot-path `cudaDeviceSynchronize`.
- Near-zero steady-state UM traffic in deviceResident mode.

**Likely breakages**
- too many tiny ranges,
- profiling code forcing sync,
- missing backend labeling.

#### 5.69 Step 18 — Validate and freeze Phase 5 baseline

**Modify**
- test harness
- benchmark scripts
- reference comparisons

**Why**
- Need a stable stopping point before Phase 6.

**Expected output**
- frozen validation report,
- `R1-core` benchmark matrix plus conditional AmgX `DeviceDirect` comparison results,
- reviewed support envelope and later-ladder handoff notes for `R1` / `R0`.

**Verify success**
- Acceptance checklist passes.
- Human reviewer signs off baseline.

**Likely breakages**
- chasing performance before numerical closure,
- using unsupported BCs in validation case,
- forgetting backend comparison.

### Instrumentation and profiling hooks

#### 5.70 Required NVTX ranges

Add NVTX v3 ranges around:

- `gpuVoF.initialize`
- `gpuVoF.uploadInitialState`
- `gpuVoF.beginTimeStep`
- `gpuVoF.alphaPredictor`
  - `alpha.computeAlphaCoNum`
  - `alpha.assemblePredictor`
  - `alpha.solvePredictor`
  - `alpha.mulesCorrect`
  - `alpha.subcycle`
- `gpuVoF.updateMixture`
- `gpuVoF.correctInterface`
  - `surface.gradAlpha`
  - `surface.nHatf`
  - `surface.curvature`
  - `surface.force`
- `gpuVoF.momentumPredictor`
- `gpuVoF.pressureCorrector`
  - `pressure.assemble`
  - `pressure.solve.native` or `pressure.solve.amgx`
  - `pressure.reconstructU`
- `gpuVoF.commitForWrite`
- `gpuVoF.endTimeStep`

#### 5.71 Required counters / logs

Log once per run:
- mesh cell/face counts,
- device memory estimate by subsystem,
- selected backend,
- supported BC summary,
- rejected feature summary (if any).

Log per timestep in debug mode:
- `alphaCoNum`,
- `nAlphaSubCycles`,
- boundedness min/max of `alpha1`,
- pressure backend iterations/residual.

#### 5.72 Required Nsight Systems capture

Baseline profile command should include at least the SPUMA-recommended CUDA/NVTX tracing and UM page-fault options when debugging residency. [R4] [R21]

For Phase 5 validation, capture:
- startup initialization,
- 5–10 representative timesteps after transients settle,
- one write event,
- both native and AmgX pressure backend runs.

**Do not** leave UM page-fault tracing enabled in all performance runs; it adds overhead. Use it selectively to diagnose residency.

#### 5.73 Required Nsight Compute capture

Profile only the top few kernels by time:
- alpha limiter/correction kernel,
- gradient/curvature kernel,
- pressure SpMV/solve kernel(s),
- momentum assembly kernel,
- patch update kernel if it appears in top-5.

Metrics of interest:
- achieved memory throughput,
- warp execution efficiency,
- branch efficiency,
- atomic throughput / serialization indicators,
- occupancy,
- L2 hit rate.

### Validation strategy

Validation is split into **correctness**, **regression**, **numerical-invariant**, and **performance/residency** checks.

#### 5.74 Correctness test matrix

**Unit tests**
1. `DeviceMeshTopology` owner/neighbour and patch map extraction.
2. `DeviceFieldMirror` upload/download bitwise equality in FP64.
3. `PersistentLduMapping` stable mapping and value-only update correctness.
4. `alpha2 = 1 - alpha1` complement update.
5. `rho` and `rhoPhi` update formulas.
6. surface-tension zero-force on uniform alpha.
7. stale-host-field assertion behavior.
8. functionObject policy classification from the machine-readable support matrix / GPU operational contract.

**Integration tests**
1. 1D/2D bounded advection of a phase fraction field.
2. small dam-break/bubble transport style VOF benchmark.
3. static droplet / Laplace pressure-jump case.
4. `R1-core` reduced case with the frozen generic BC/scheme subset and no Phase 6-only BCs/startup.
5. restart from written time on one supported generic case and on `R1-core`, including reload of required old-time/device-history state.
6. `R1` / `R0` later-ladder cases only after Phase 6 / Phase 8 readiness.

#### 5.75 Numerical invariants and thresholds

Use the centralized acceptance-manifest classes directly:

- `R2` generic boundedness + interface-transport slice -> `TC_R2_TRANSPORT`
- `R2` static-droplet + surface-tension slice -> `TC_R2_SURFACE`
- `R1-core` generic reduced case -> `TC_R1CORE_GENERIC`
- write/reload parity on accepted Phase 5 tuples -> `RP_STRICT`
- native-vs-AmgX parity on admitted `R1-core` rows -> `BP_AMGX_R1CORE`

Phase 5 does not define local numeric pass/fail thresholds. If a local test harness wants human-readable text, it must mirror the centralized manifest classes verbatim rather than restating or loosening them here.

#### 5.76 Regression checks

A change fails regression if it causes any of the following:
- a new unsupported case without support-scan rejection,
- a functionObject accepted in performance mode without `writeTimeOnly` classification,
- host-field stale-read assertion in performance mode,
- new hot-path UVM traffic,
- worse alpha boundedness,
- worse `R1-core` integral agreement,
- pressure backend mismatch,
- an AmgX production-residency claim without `DeviceDirect`,
- restart/reload parity failure for required current/old-time state,
- increased peak memory > 10% without explicit approval,
- topology rebuilds in the hot path.

#### 5.77 Performance / residency checks

Residency pass criteria in **deviceResident** mode:

1. Nsight Systems shows **no recurring field-scale HtoD or DtoH unified-memory migrations** during steady-state hot stages.
2. No CPU page-fault bursts caused by device-resident field access.
3. No `cudaDeviceSynchronize` in stage implementations.
4. All large field updates are D2D or kernel-generated.
5. Pressure mapping/pattern build occurs only at initialization.

**AmgX note:** production acceptance for AmgX is evaluated only in `DeviceDirect` mode. `PinnedHost` AmgX traces are correctness-only bring-up artifacts and do not satisfy the no-field-scale-host-transfer criterion.

Performance pass criteria:
- `R1-core` GPU timestep time is at least competitive with the CPU baseline,
- pressure backend comparison exists for the native backend and, when `DeviceDirect` is available, for the AmgX backend; otherwise the AmgX lane is archived as correctness-only bring-up,
- top hot kernels identified and documented.

**Important:** Phase 5 does **not** require a specific headline speedup multiple. Correctness plus residency plus a credible performance path is the acceptance bar.

### Performance expectations

1. The pressure path may or may not prefer AmgX on the RTX 5080 reduced nozzle case; the result is empirical.
2. The alpha/MULES and surface-tension kernels are expected to be among the hardest kernels to optimize because they mix indirect addressing, branching, and atomics.
3. Early Phase 5 performance may still show high kernel counts; that is acceptable if:
   - there are no hard syncs,
   - there is no recurring host traffic,
   - and the stage boundaries are graph-safe.
4. Given GeForce-class FP64 limits, Phase 5 success should be defined as:
   - trustworthy residency,
   - validated numerics,
   - persistent mappings,
   - backend comparison,
   - and a clear hotspot map for Phase 7 optimization.

### Common failure modes

1. **Wrong host/device authority assumptions**
   - Symptom: random host-side field values or UVM traffic.
   - Cause: stale host fields read without commit.
2. **Old-time state mismatch**
   - Symptom: timestep-one mismatch or instability under subcycling.
   - Cause: missing `alpha1Old`, `phiOld`, or `UOld` update.
3. **Broken previous-correction flux logic**
   - Symptom: boundedness loss or mismatch with CPU MULESCorr behavior.
   - Cause: stale or wrongly scaled `alphaPhi1Corr0`.
4. **Incorrect patch compaction**
   - Symptom: boundary artifacts, divergence, or patch-only corruption.
   - Cause: wrong compact-face offsets or BC enum mapping.
5. **Pressure matrix value-order mismatch**
   - Symptom: native works but AmgX diverges, or vice versa.
   - Cause: LDU-to-CSR mapping bug.
6. **Incorrect curvature sign / normal orientation**
   - Symptom: wrong pressure jump or amplified spurious currents.
   - Cause: normal direction convention mismatch.
7. **Stealth host fallback**
   - Symptom: run “works” but Nsight shows DtoH/HtoD migrations every step.
   - Cause: unsupported BC/model or functionObject path reading host fields.
8. **Temporary allocation churn**
   - Symptom: allocator hot spots, memory fragmentation, unpredictable slowdowns.
   - Cause: stage-local allocations instead of scratch reuse.
9. **rAU lifetime bug**
   - Symptom: pressure corrector uses invalid or zero `rAU`.
   - Cause: clearing `rAU` too early.
10. **Non-orthogonal correction sequencing bug**
    - Symptom: flux and velocity update mismatch or poor convergence.
    - Cause: updating `phi`/`U` on every non-orth loop instead of the final one, if local semantics say final-only.

#### What not to do

- Do not read host fields inside hot device stages.
- Do not call `cudaDeviceSynchronize()` inside stage functions.
- Do not rebuild matrix topology every corrector.
- Do not silently fall back to host BC logic.
- Do not broaden the supported BC/model set opportunistically while chasing one case.
- Do not optimize kernels before a CPU/GPU correctness baseline exists.
- Do not add mixed precision to “fix” performance before the device-resident path is numerically validated.
- Do not treat Nsight absence of kernel errors as proof of correctness.

### Debugging playbook

#### 5.78 Bring-up order for debugging

1. **Startup support scan only**
   - Confirm case rejection/acceptance is deterministic.
2. **Initialization upload only**
   - Upload then immediately commit and compare host/device equality.
3. **Alpha skeleton without limiter**
   - Verify data flow and flux formation.
4. **Predictor matrix only**
   - Compare predictor alpha solve before limiter correction.
5. **MULES correction**
   - Add boundedness checks and compare alpha min/max versus CPU.
6. **Subcycling**
   - Compare per-subcycle accumulated `alphaPhi1`.
7. **Mixture update**
   - Compare `rho` and `rhoPhi`.
8. **Surface tension**
   - Validate on uniform alpha and static droplet separately.
9. **Momentum**
   - Check `rAU`, `U`, and matrix stats.
10. **Pressure**
    - Validate native first, then AmgX.
11. **Write-time commit**
    - Ensure no per-stage host traffic remains.

#### 5.79 Primary debug tools

- **Compute Sanitizer**
  - first for memory and race bugs in new kernels.
- **Nsight Systems**
  - then for hidden syncs, UVM traffic, and stage timing.
- **Nsight Compute**
  - only after correctness is stable to inspect real hot kernels.
- **Field-diff utility**
  - add a helper to compare host-committed GPU fields to CPU reference fields with max/mean norms.

#### 5.80 Stage isolation techniques

If a stage misbehaves:

1. commit current device state to host,
2. compare against CPU state at the same stage boundary,
3. rerun only that stage under debug instrumentation,
4. if necessary, use debug-only `stageFallback` for a single stage to localize the issue,
5. disable `stageFallback` immediately after diagnosis.

#### 5.81 Specific watch-outs

- `alphaApplyPrevCorr` state must persist across correctors/timesteps as required.
- `Euler` only means any accidental `localEuler` or `CrankNicolson` case must be rejected, not tolerated.
- `interfaceProperties` may hide patch-specific behavior; reject contact-angle patch fields rather than silently ignoring them.
- runtime functionObjects may trigger host commits.
- SPUMA GPU lambdas should follow SPUMA’s own porting guidance: capture by copy, avoid accidental `this` reference capture, and prefer raw pointer access in kernels. [R5]

### Acceptance checklist

A Phase 5 implementation is acceptable only if every item below is checked.

- [ ] Symbol reconciliation against SPUMA/v2412 completed and reviewed.
- [ ] `R1-core` reduced case is frozen and validated as the Phase 5 ladder case.
- [ ] `gpuVoF` runtime gate and dictionary implemented.
- [ ] Startup support scanner rejects unsupported cases before stepping.
- [ ] Support scanner enforces the machine-readable functionObject policy in performance mode.
- [ ] `DeviceMeshTopology` and `DeviceBoundaryMaps` implemented with tests.
- [ ] `DeviceVoFState` owns persistent device-resident fields and old-time mirrors.
- [ ] Alpha predictor runs on device.
- [ ] `MULESCorr` path runs on device.
- [ ] Explicit MULES path runs on device.
- [ ] Alpha subcycling works on device.
- [ ] Previous-correction flux logic implemented and validated.
- [ ] Mixture-property update runs on device.
- [ ] Constant-sigma surface-tension path runs on device.
- [ ] Momentum predictor runs on device and produces valid `rAU`.
- [ ] Pressure corrector runs on device.
- [ ] Pressure backend selectable between native and AmgX.
- [ ] Any AmgX production-residency claim uses the Phase 4 `DeviceDirect` bridge.
- [ ] Topology mapping built once; value-only updates implemented.
- [ ] Write-time commit path implemented with pinned staging.
- [ ] Restart/reload parity validated for supported write points.
- [ ] No hot-path `cudaDeviceSynchronize()`.
- [ ] Nsight shows no recurring hot-path UVM field migrations in deviceResident mode.
- [ ] Validation cases pass numerical thresholds.
- [ ] `R1-core` benchmark exists for the native backend and, when `DeviceDirect` is available, for the AmgX backend.
- [ ] Memory report shows acceptable peak footprint.
- [ ] Debug assertions detect stale host reads in performance mode.

#### Implementation tasks for coding agent

1. Produce `phase5_symbol_reconciliation.md`.
2. Implement `gpuVoF` dictionary parser and runtime gate.
3. Implement `CaseSupportReport`.
4. Implement `DeviceMeshTopology`.
5. Implement `DeviceBoundaryMaps`.
6. Implement `DeviceFieldMirror`.
7. Implement `DeviceVoFState`.
8. Implement initialization upload and old-time seeding.
9. Implement `AlphaControlsSnapshot` build and alpha Courant reduction.
10. Implement alpha flux skeleton.
11. Implement alpha predictor matrix assembly and solve.
12. Port minimal MULES subset.
13. Implement alpha subcycling.
14. Implement previous-correction flux persistence.
15. Implement mixture-property update.
16. Implement constant-sigma interface/surface-tension path.
17. Integrate momentum predictor.
18. Integrate native pressure backend.
19. Integrate AmgX pressure backend.
20. Implement write-time commit.
21. Add NVTX ranges and memory report.
22. Add unit/integration tests.
23. Produce validation and benchmark reports.

#### Do not start until

- the SPUMA/v2412 source reconciliation note is complete,
- the support envelope is approved by the human reviewer,
- the pressure backend bridge from Phase 4 is available,
- the allocator/pool path is stable on the RTX 5080.

#### Safe parallelization opportunities

- `DeviceMeshTopology` and `DeviceBoundaryMaps`
- `DeviceFieldMirror` and `DeviceScratchArena`
- support scanner and dictionary parser
- alpha subsystem and surface-tension subsystem
- native and AmgX pressure backend adapters
- unit-test harness and benchmark scripting

Parallel work is safe only after the symbol reconciliation note defines exact local patch targets.

#### Requires human sign-off on

- broadening supported BCs,
- enabling any ddt scheme other than Euler,
- enabling contact-angle models,
- enabling non-laminar production cases if the turbulence path is not already proven on SPUMA,
- changing fallback policy away from `failFast`,
- introducing mixed precision,
- changing the default pressure backend after benchmark review.

#### Artifacts to produce

1. `phase5_symbol_reconciliation.md`
2. `phase5_case_support_matrix.md`
3. `phase5_memory_report_<case>.md`
4. `phase5_validation_report.md`
5. `phase5_backend_benchmark_native_vs_amgx.md`
6. Nsight Systems timeline captures for:
   - reduced validation case,
   - reduced nozzle case,
   - native and AmgX backends
7. Nsight Compute reports for top kernels
8. Unit/integration test outputs
9. Final Phase 5 support envelope document

### Future extensions deferred from this phase

1. Nozzle-specific pressure/swirl inlet kernels.
2. Air-core startup seeding on device.
3. External near-field plume coupling.
4. Contact-angle models.
5. Temperature-dependent surface tension.
6. Broader BC coverage.
7. Multi-GPU / MPI decomposition.
8. CUDA Graph capture/productization.
9. Non-atomic optimized irregular kernels.
10. Mixed precision.
11. Dynamic mesh support.
12. Generalized functionObject support without write-time commits.

# 6. Validation and benchmarking framework

This section summarizes the validation and benchmarking framework required to support Phase 5.

## 6.1 Validation case set

### V1 — Bounded advection / interface transport
Purpose:
- validate alpha transport,
- boundedness,
- subcycling,
- MULES behavior.

### V2 — Static droplet / Laplace pressure jump
Purpose:
- validate surface tension,
- curvature sign,
- pressure coupling,
- spurious-current level.

### V3 — `R1-core` reduced internal case
Purpose:
- validate production-like field coupling and transient behavior using only the frozen generic BC/scheme subset and no Phase 6-only BCs/startup.

### V4 — `R1` reduced nozzle case
Purpose:
- Phase 6+/8 reduced-nozzle gate once nozzle-specific BCs and startup are device-resident.

### V5 — `R0` full reference nozzle case
Purpose:
- final end-to-end comparison once the later ladder phases are already stable.

## 6.2 Benchmark dimensions

Run each case in:
- CPU reference mode,
- GPU native pressure backend,
- GPU AmgX pressure backend.

Collect:
- timestep wall-clock,
- per-stage wall-clock,
- pressure solver iterations/residual,
- total kernel launches,
- top kernel timings,
- peak device memory,
- hot-path UVM activity,
- integral outputs (mass flow, pressure drop, etc.).

Production AmgX benchmark lines require `DeviceDirect`. If the bridge is not available, any AmgX result is archived as correctness-only bring-up and is excluded from no-field-scale-host-transfer acceptance claims.


## 6.3 Required benchmark stops

Benchmark and review **before proceeding** at these points:

1. after alpha + MULES bring-up,
2. after surface tension integration,
3. after native pressure integration,
4. after AmgX integration,
5. after `R1-core` validation in Phase 5, and again after `R1` reduced-nozzle validation once Phase 6 BC/startup work lands.

## 6.4 Benchmark output format

Every benchmark report shall include:
- case,
- mesh size,
- dt,
- alpha subcycle settings,
- backend,
- device mode (`deviceResident` or debug mode),
- peak memory,
- stage timing table,
- acceptance result,
- reviewer notes.

# 7. Toolchain / environment specification

## 7.1 Required platform

- Linux x86_64 workstation
- NVIDIA GeForce RTX 5080, Blackwell, compute capability 12.0 [R17] [R18]

## 7.2 CUDA baseline

**Sourced fact**
- CUDA 12.8 is the first toolkit release with native compiler support for Blackwell architectures including `SM_120`. [R19] [R20]
- CUDA 12.8 GA requires Linux driver **>= 570.26**; CUDA 12.8 Update 1 requires **>= 570.124.06**. [R20]
- Current CUDA 13.x releases exist and require newer drivers; current 13.2 release notes show Linux driver **>= 595.45.04** for that toolkit release. [R20] [R25]

**Recommendation**
- Phase 5 consumes the master pin manifest rather than defining an independent local minimum.
- Default until superseded: **CUDA 12.9.1** as the primary lane with Linux driver **>= 595.45.04**; **CUDA 13.2** remains the experimental lane. [R20] [R25]
- Do not start Phase 5 on a pre-12.8 toolkit, and do not weaken the master pin manifest locally.

## 7.3 Compiler / binary targets

Compile GPU code with native Blackwell support and embedded PTX for forward compatibility testing:

- `-gencode arch=compute_120,code=sm_120`
- `-gencode arch=compute_120,code=compute_120`

Use `CUDA_FORCE_PTX_JIT=1` in a validation lane to confirm PTX readiness. [R19]

## 7.4 Profiling / debug tools

Required:
- Nsight Systems
- Nsight Compute
- Compute Sanitizer
- NVTX v3 (`nvtx3/nvToolsExt.h` or local equivalent include path) [R19] [R20]
- `cuobjdump` / `nvdisasm` for binary inspection if kernel-target verification is needed. [R20]

## 7.5 Build-system requirements

- One build mode with GPU disabled (control reference).
- One build mode with GPU enabled and native pressure backend.
- One build mode with GPU enabled and AmgX support.
- Unit tests and integration tests compilable in CI or scripted local automation.
- Optional debug build enabling:
  - stale-host-field assertions,
  - stageFallback,
  - heavier field comparison.

## 7.6 Suggested environment pinning

Pin at least:
- SPUMA commit/tag,
- foamExternalSolvers commit/tag,
- CUDA version,
- driver version,
- Umpire version,
- Nsight Systems version,
- compiler version.

**Recommendation:** keep a `phase5_environment_lock.md` artifact checked into the project docs tree, and make it subordinate to the master pin manifest rather than a competing local pin source.

# 8. Module / file / ownership map

## 8.1 Core file map

| File / module | Purpose | Owner role |
|---|---|---|
| `gpuVoFAdapter.*` | solver/module integration and runtime gate | integration owner |
| `DeviceVoFSupportScan.*` | case support scan | integration owner |
| `DeviceVoFState.*` | authoritative persistent device state | two-phase numerics owner |
| `DeviceMeshTopology.*` | static topology extraction/upload | GPU mesh owner |
| `DeviceBoundaryMaps.*` | compact patch descriptors and coeffs | GPU mesh owner |
| `DeviceFieldMirror.*` | host/device mirror abstraction | GPU runtime owner |
| `DeviceScratchArena.*` | pooled temporary buffers | GPU runtime owner |
| `DeviceAlphaTransport.*` | alpha stage orchestration | two-phase numerics owner |
| `DeviceMULES.*` | minimal MULES subset | two-phase numerics owner |
| `DeviceMixtureProperties.*` | rho/rhoPhi and property updates | two-phase numerics owner |
| `DeviceSurfaceTension.*` | interface/surface tension stage | two-phase numerics owner |
| `DeviceMomentumPredictor.*` | momentum assembly/solve and `rAU` | two-phase numerics owner |
| `DevicePressureCorrector.*` | pressure stage and field updates | two-phase numerics owner |
| `PersistentLduMapping.*` | topology mapping / value update indirection | solver-backend owner |
| `DeviceLinearSystemBridge.*` | backend abstraction | solver-backend owner |
| `NativePressureBackend.*` | native solve adapter | solver-backend owner |
| `AmgxPressureBackend.*` | AmgX solve adapter | solver-backend owner |
| `kernels/*.cu` | low-level GPU kernels | corresponding subsystem owner |
| `tests/gpuVoF/*` | unit, integration, benchmark coverage | validation owner |

## 8.2 Ownership rules

1. Only the subsystem owner may change kernel semantics in their module without reviewer approval.
2. Any change to `DeviceVoFState` layout requires:
   - memory report update,
   - write-commit audit,
   - validation rerun.
3. Any change to `PersistentLduMapping` requires backend regression for both native and AmgX.
4. Any new supported BC requires:
   - support scanner update,
   - boundary-kernel tests,
   - validation case proving it.

# 9. Coding-agent execution roadmap

This roadmap is the concrete build order for the coding agent. It is phase-internal and intentionally milestone-driven.

## 9.1 Milestones

### M5.0 — Symbol reconciliation
Deliverable:
- exact mapping from the semantic reference sources to the local SPUMA/v2412 files.

Stop here and review before code changes.

### M5.1 — Runtime gate and support scan
Deliverable:
- `gpuVoF` config,
- deterministic startup accept/reject behavior.

### M5.2 — Topology, boundary maps, and field mirrors
Deliverable:
- persistent device topology,
- patch descriptors,
- upload/download-capable field mirrors,
- memory report.

### M5.3 — Alpha skeleton
Deliverable:
- alpha control snapshots,
- alpha flux formation,
- alpha stage running end-to-end with placeholder or predictor-only logic.

Stop and benchmark residency here.

### M5.4 — Full alpha + MULES + subcycling
Deliverable:
- bounded alpha path,
- previous-correction flux support,
- validation on bounded advection / dam-break type case.

Stop and validate here before interface or pressure work.

### M5.5 — Mixture update + interface/surface tension
Deliverable:
- `rho`/`rhoPhi`,
- constant-sigma interface path,
- droplet validation.

Stop and benchmark kernels here.

### M5.6 — Momentum predictor
Deliverable:
- device `U` equation path,
- persistent `rAU`,
- laminar reduced-case validation.

### M5.7 — Native pressure backend integration
Deliverable:
- device pressure corrector with native backend,
- non-orthogonal loops,
- reduced nozzle run.

Stop and benchmark here.

### M5.8 — AmgX pressure backend integration
Deliverable:
- runtime-selectable AmgX backend,
- pressure backend comparison report.

### M5.9 — Write-time commit and artifact generation
Deliverable:
- explicit write path,
- validation reports,
- benchmark reports,
- profile artifacts.

### M5.10 — Freeze Phase 5 baseline
Deliverable:
- reviewed support envelope,
- acceptance checklist signed off,
- branch ready for Phase 6.

## 9.2 Dependency graph

- `M5.0` → all later milestones
- `M5.1` → `M5.2`
- `M5.2` → `M5.3`, `M5.5` groundwork, and `M5.6` groundwork
- `M5.3` → `M5.4`
- `M5.4` → `M5.5`
- `M5.5` → `M5.6` and `M5.7`
- `M5.6` → `M5.7`
- `M5.7` → `M5.8`
- `M5.8` → `M5.9`
- `M5.9` → `M5.10`

## 9.3 Work that can proceed in parallel

After `M5.0`:
- support scanner,
- topology/boundary map extraction,
- field mirror/scratch infrastructure,
- benchmark harness scaffolding.

After `M5.2`:
- alpha subsystem and interface subsystem can proceed partially in parallel.
- native and AmgX backend wrappers can proceed in parallel once the persistent mapping contract is fixed.

## 9.4 What should be prototyped before productized

Prototype first:
- alpha limiter kernels,
- curvature kernels,
- pressure mapping update path,
- write-time commit set.

Productize only after:
- CPU/GPU equivalence is proven,
- Nsight confirms residency,
- memory footprint is stable.

## 9.5 What should remain experimental

- any non-atomic replacement for irregular kernels,
- multiple streams,
- CUDA Graph capture,
- mixed precision,
- broader BC support.

## 9.6 Where to stop and benchmark before proceeding

Mandatory stop-and-benchmark points:
1. after alpha+MULES correctness,
2. after surface tension correctness,
3. after native pressure correctness,
4. after AmgX integration,
5. before merging Phase 5 to the main development branch.

# 10. Locked Package Imports

## 10.1 Decisions already frozen for implementation

1. **Exact SPUMA/v2412 patch targets**
   - Implementation patches must follow `semantic_source_map.md` and the symbol-reconciliation note produced from it before coding.
2. **Production ddt scheme**
   - Milestone-1 Phase 5 scope is Euler-only.
3. **Turbulence scope import**
   - Milestone-1 Phase 5 acceptance is laminar-only, as frozen in `support_matrix.md`.
4. **Pressure backend default**
   - Native is the current default backend. AmgX remains a supported secondary lane and may be promoted only after `DeviceDirect` evidence is admitted by `acceptance_manifest.md`.
5. **`R1-core` case freeze**
   - `R1-core` is the Phase-5-friendly reduced case frozen in `reference_case_contract.md`.
6. **Fallback policy**
   - Default behavior is `failFast`; any debug fallback remains non-production and non-default.
7. **Memory budget envelope and validation thresholds**
   - Current Phase 5 defaults remain subordinate to `acceptance_manifest.md` and may only change through that central authority.

## 10.2 Human review checklist

- [ ] Does the symbol reconciliation note correctly identify the real SPUMA/v2412 files?
- [ ] Is the Phase 5 support envelope narrow enough to be credible?
- [ ] Is the host/device authority model explicit and safe?
- [ ] Are all hidden host-touch hazards called out?
- [ ] Are alpha subcycling and previous-correction semantics preserved?
- [ ] Is the surface-tension scope appropriately restricted?
- [ ] Are native and AmgX both preserved as options?
- [ ] Is the validation matrix sufficient before Phase 6 begins?
- [ ] Are rollback and debug-only fallback behaviors explicit and limited?
- [ ] Is the coding order realistic for an autonomous agent?

## 10.3 Coding agent kickoff checklist

- [ ] Read and complete `phase5_symbol_reconciliation.md` first.
- [ ] Do not patch any solver source before support-scan targets are identified.
- [ ] Implement `gpuVoF` runtime gate before adding kernels.
- [ ] Build topology, patch maps, mirrors, and old-time state before MULES.
- [ ] Keep host fields as control plane; do not repurpose generic field storage.
- [ ] Treat `failFast` as the default behavior for unsupported cases.
- [ ] Add NVTX ranges as each stage is created.
- [ ] Run unit tests after each infrastructure milestone.
- [ ] Benchmark and review at each mandatory stop point.
- [ ] Do not optimize irregular kernels until correctness is signed off.

## 10.4 Highest risk implementation assumptions

1. The accessible Foundation VOF module semantics are close enough to the SPUMA/v2412 implementation to guide a safe port after reconciliation.
2. The reduced nozzle validation case can avoid unsupported Phase 6 BCs.
3. Existing SPUMA/native device operator infrastructure is sufficient to support momentum and predictor matrix paths without major new generic linalg work.
4. The constant-sigma, no-contact-angle surface-tension slice is enough to validate the interface path before nozzle-specific wall physics.
5. The RTX 5080 has sufficient memory headroom for the Phase 5 persistent working set when 32-bit indices and pooled scratch buffers are used.
6. Pressure backend performance ranking will need empirical selection; no backend is assumed superior a priori.

## 10.5 Reference list

- **[R1]** Carmignani et al., *SPUMA: A minimally invasive approach to the GPU porting of OPENFOAM®*, Computer Physics Communications, Volume 321, April 2026. ScienceDirect. https://www.sciencedirect.com/science/article/pii/S0010465526000674
- **[R2]** SPUMA arXiv full text / HTML preprint. https://arxiv.org/html/2512.22215v1
- **[R3]** SPUMA repository overview. https://gitlab-hpc.cineca.it/exafoam/spuma
- **[R4]** SPUMA GPU-support and porting wiki pages (supported solvers, fvSolution notes, profiling guidance, lambda-capture guidance). https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support and related wiki history pages.
- **[R5]** SPUMA GPU-porting wiki guidance on lambda capture and raw-pointer access in GPU lambdas. https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-porting
- **[R6]** foamExternalSolvers project, interface to AmgX for SPUMA and OpenFOAM. https://gitlab.hpc.cineca.it/exafoam/foamExternalSolvers
- **[R7]** OGL/Ginkgo paper on persistent OpenFOAM linear-system mapping and value-only updates. https://link.springer.com/article/10.1007/s11012-024-01806-1
- **[R8]** exaFOAM public workshop slides, *Porting to GPUs* (FOAM2CSR, AmgX4Foam, zeptoFOAM, memory-pool and explicit-data-management architecture). https://exafoam.eu/wp-content/uploads/2ndWorkshop/exaFOAM_2ndPublicWorkshop_Porting_to_GPUs.pdf
- **[R9]** OpenFOAM-v2412 release page / release timing. https://develop.openfoam.com/Development/openfoam/-/releases/OpenFOAM-v2412 and related OpenFOAM-v2412 news pages.
- **[R10]** OpenFOAM Foundation `incompressibleVoF` class reference. https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html
- **[R11]** OpenFOAM Foundation `twoPhaseSolver/alphaPredictor.C` source/reference. https://cpp.openfoam.org/v13/twoPhaseSolver_2alphaPredictor_8C_source.html
- **[R12]** OpenFOAM Foundation `twoPhaseSolver/pressureCorrector.C` source/reference. https://cpp.openfoam.org/v13/twoPhaseSolver_2pressureCorrector_8C_source.html
- **[R13]** OpenFOAM Foundation `interfaceProperties` class reference. https://cpp.openfoam.org/v13/classFoam_1_1interfaceProperties.html
- **[R14]** OpenFOAM Foundation `surfaceTensionModel` class reference. https://cpp.openfoam.org/v13/classFoam_1_1surfaceTensionModel.html
- **[R15]** OpenFOAM 2.3.0 release note describing MULES as an explicit/semi-implicit bounded transport method and the role of subcycling. https://openfoam.org/release/2-3-0/multiphase/
- **[R16]** OpenFOAM Foundation MULES / CMULES source references (semantic background). https://cpp.openfoam.org
- **[R17]** NVIDIA official GeForce RTX 5080 page. https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
- **[R18]** NVIDIA CUDA GPU compute capability list (RTX 5080 = 12.0). https://developer.nvidia.com/cuda-gpus
- **[R19]** NVIDIA CUDA Blackwell compatibility guide. https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html
- **[R20]** NVIDIA CUDA 12.8/12.8.1 features and release notes (SM_120 support, graph features, driver requirements, NVTX v2 deprecation). https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/ and https://docs.nvidia.com/cuda/archive/12.8.1/cuda-features-archive/index.html
- **[R21]** NVIDIA Nsight Systems User Guide (UM migration semantics, CLI tracing options). https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **[R22]** Umpire documentation (device allocators, pinned memory, preferred location advice). https://umpire.readthedocs.io/
- **[R23]** NVIDIA HPC SDK release notes / current release information. https://developer.nvidia.com/hpc-sdk and https://docs.nvidia.com/hpc-sdk/
- **[R24]** Review article on sparse matrix-vector multiplication on modern architectures and storage-format sensitivity. https://pmc.ncbi.nlm.nih.gov/articles/PMC7295357/
- **[R25]** CUDA 13.2 release notes for current toolkit/driver context. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
