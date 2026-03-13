
# 1. Executive overview

This document expands **only Phase 7** of the broader SPUMA/OpenFOAM GPU-porting plan into an implementation-ready engineering specification. Sections 1–4 and 6–10 preserve the minimum global context required to keep Phase 7 coherent, but the detailed implementation prescription is concentrated in Section 5.

Phase 7 exists because a pressure-swirl nozzle solver based on transient VOF, MULES, and PIMPLE is **not** performance-limited by sparse linear algebra alone. SPUMA’s published profiling shows weaker efficiency in gradient/RHS kernels because of atomic-heavy work, and even lower efficiency in the pressure-solver path with GAMG; the same profiling also shows an extremely high kernel-launch count with a `cudaDeviceSynchronize()` after each kernel invocation in the profiled path. That combination makes the nozzle problem a three-ceiling problem: sparse-memory traffic, irregular face/cell kernel efficiency, and launch/synchronization overhead. [R1]

Phase 7 therefore does **not** attempt to replace the pressure linear solver stack. It implements custom CUDA only where general-purpose linear-algebra libraries do not solve the actual bottlenecks: alpha-flux assembly, MULES limiter/update work, interface curvature and surface-tension source terms, patch/boundary evaluation, and a small number of fused device-resident update kernels.

The implementation target is a **single NVIDIA RTX 5080 workstation** using a **SPUMA v2412-derived codebase**, with device residency as the primary invariant. SPUMA’s project metadata states that SPUMA version `0.1-v2412` is based on the OpenCFD v2412 release, and the SPUMA paper compares against OpenFOAM-v2412 CPU results. [R1][R2]

Two source-of-truth caveats govern this document:

1. The implementation target is **OpenCFD/SPUMA v2412**.
2. The most directly accessible public source-code references for MULES and `interfaceProperties` during research were primarily the OpenFOAM Foundation code guide and raw source pages, which are close enough to illuminate algorithm structure but are **not guaranteed** to be byte-identical to SPUMA’s v2412 tree. In addition, OpenFOAM v13 explicitly changed MULES after v2412, which confirms that branch drift is real. [R8][R9]

Accordingly, the **first mandatory implementation step** in Phase 7 is a local source audit against the exact SPUMA/v2412 checkout used by the project. This is not optional.

The intended deliverable of Phase 7 is:

- a graph-ready, allocation-free, device-resident custom-kernel subsystem for the irregular VOF/MULES/interface/patch hotspots;
- runtime-selectable fallbacks for every new kernel family;
- correctness parity against the CPU/SPUMA reference within explicitly defined tolerances;
- a profiling result that proves the new kernels reduce either hotspot runtime, launch count, or both, without reintroducing unified-memory churn.

# 2. Global architecture decisions

## GA1 — Implement Phase 7 on a SPUMA v2412-derived codebase, not on a separate OpenFOAM branch

- **Sourced fact.** SPUMA version `0.1-v2412` is based on OpenCFD v2412, and SPUMA’s published CPU reference comparisons use OpenFOAM-v2412. [R1][R2]
- **Engineering inference.** Porting Phase 7 against a Foundation v13 tree would introduce branch-drift risk before any CUDA work starts.
- **Recommendation.** Use the exact SPUMA commit already validated in Phases 0–6 as the patch target. Treat Foundation v9/v13/dev sources only as algorithmic references, not as patch targets.

## GA2 — Keep pressure linear algebra library-backed; custom CUDA targets only the non-library hotspots

- **Sourced fact.** `incompressibleVoF` is a two-immiscible-fluid VOF solver with a single mixture momentum equation under PIMPLE; MULES and interface/surface-tension paths sit outside the sparse linear solve itself. [R3][R5][R7]
- **Engineering inference.** Rewriting CSR SpMV or AMG internals in Phase 7 would add risk without addressing the limiter, curvature, and boundary kernels that dominate nozzle-specific overhead.
- **Recommendation.** Keep pressure solve and preconditioning on the existing SPUMA/native or external-solver path. Restrict custom CUDA to alpha/MULES, interface curvature, patch handling, and fused field updates.

## GA3 — Device residency is an invariant, not an optimization hint

- **Sourced fact.** SPUMA uses unified memory for incremental porting but also documents that incomplete ports can degrade into unwanted page migrations; Nsight Systems documents that unified-memory HtoD/DtoH transfers pause execution. [R1][R16]
- **Engineering inference.** A custom kernel subsystem that still causes CPU-side patch evaluation or field inspection during each subcycle will self-sabotage on a discrete RTX 5080.
- **Recommendation.** All heavy fields and all Phase 7 scratch arrays must be persistent device allocations with stable addresses. Host reads are restricted to diagnostics at coarse cadence and write-time output staging.

## GA4 — The default execution contract is graph-ready, allocation-free, and stream-ordered

- **Sourced fact.** CUDA 12.8 adds Blackwell compiler support and extends CUDA Graph conditional control with IF/ELSE and SWITCH nodes. [R12][R19]
- **Engineering inference.** Phase 7 kernels written with dynamic allocation, host callbacks, or address instability will block later graph capture and perpetuate launch overhead.
- **Recommendation.** Every Phase 7 host API must accept preallocated buffers and launch on caller-supplied streams. No kernel family may allocate/free during a timestep or alpha subcycle. Graph-readiness requirements, stage IDs, and graph-unsafe boundaries are imported from the centralized graph support matrix; Phase 7 requires smoke-capture-ready kernels for the Phase-7-owned capture-safe stages, while full graph/performance acceptance remains Phase 8 scope.

## GA5 — Preserve v2412 numerics first; optimize execution second

- **Sourced fact.** OpenFOAM MULES and `interfaceProperties` have specific source-defined formulas, control fields, and boundary handling; OpenFOAM v13 later changed MULES behavior. [R5][R6][R7][R9]
- **Engineering inference.** Any “performance rewrite” that changes limiter semantics, curvature formulae, or patch interpretation during bring-up will make debug impossible.
- **Recommendation.** Phase 7 must begin as a mechanical transliteration of the validated v2412 numerics into CUDA kernels, preserving variable naming and update order wherever possible. Optimization starts only after the transliterated path passes regression.

## GA6 — Single GPU only in this phase

- **Sourced fact.** SPUMA notes that GPU-aware MPI direct GPU-to-GPU transfers require pure GPU pointers; managed-memory pointers can trigger implicit transfers. The RTX 5080 product page does not list NVLink support. [R1][R10]
- **Engineering inference.** Multi-GPU decomposition adds communication, pointer-class, and coupled-patch complexity that is orthogonal to the Phase 7 hotspot work.
- **Recommendation.** Phase 7 supports one rank on one GPU only. Processor patches and decomposed meshes must trigger fallback.

## GA7 — Preserve solver scalar precision; defer mixed precision

- **Sourced fact.** The CUDA Programming Guide’s Blackwell compute-capability description shows far fewer FP64 cores per SM than FP32 cores for compute capability 12.0; the RTX 5080 is compute capability 12.0. [R11][R15]
- **Engineering inference.** FP64-heavy custom kernels will not be especially fast on an RTX 5080, but changing precision during the initial port risks boundedness, capillarity, and nozzle QoI regressions.
- **Recommendation.** Keep storage and arithmetic in `Foam::scalar` for Phase 7 baseline. Mixed precision is explicitly deferred to future work.

# 3. Global assumptions and constraints

1. **Implementation base.** SPUMA `0.1-v2412`-class codebase, pinned to a reviewed commit. [R2]
2. **Target hardware.** Single NVIDIA GeForce RTX 5080: 10,752 CUDA cores, 16 GB GDDR7, 960 GB/s bandwidth, no NVLink, compute capability 12.0. [R10][R11]
3. **Target CUDA stack.** Consume the master pin manifest frozen for the program rather than restating a looser Phase 7-only minimum here. CUDA 12.8+ Blackwell support and the 570.26 Linux driver floor remain compatibility facts, not replacement pins. [R12][R13]
4. **Mesh assumption for Phase 7 production path.** Static mesh topology, no AMR, no mesh motion, no topology changes.
 5. **Patch assumption for Phase 7 production path.** Only boundary-condition classes present in the frozen milestone-1 support matrix for the accepted cases, plus a short list of generic fallbacks, are supported by custom device kernels.
 6. **Scheme assumption for Phase 7 production path.** Only the `fvSchemes` tuple imported from the frozen support matrix is required. Any unsupported scheme must fail fast or fall back cleanly.
7. **Memory assumption.** All persistent Phase 7 fields, topology, and scratch arrays reside in device memory, not managed memory, in production mode.
8. **Execution assumption.** One main compute stream. Optional secondary stream may be used only for asynchronous diagnostics or host-pinned copies after events.
9. **Correctness priority.** Numerical boundedness and reference agreement outrank speed in the first completed version.
10. **Branch-drift constraint.** Publicly accessible Foundation sources are used here for algorithm illumination, but the actual patch must match SPUMA’s v2412 tree. OpenFOAM v13’s MULES changes are explicit evidence that later branches are not safe source-of-truth replacements. [R8][R9]

# 4. Cross-cutting risks and mitigation

## Risk 1 — Branch drift between SPUMA/OpenCFD v2412 and publicly accessible Foundation source references

- **Why it matters.** MULES and VOF internals are sensitive to exact update order and boundary semantics.
- **Mitigation.** Before any CUDA implementation, diff the local SPUMA/v2412 files against the reference files used in this document. If drift exists, port the **local v2412 code**, not the public reference code.

## Risk 2 — Atomics dominate the limiter and interface kernels

- **Why it matters.** SPUMA explicitly reports that one gradient kernel contains several atomic operations and performs worse than pure memory-bound kernels. [R1]
- **Mitigation.** Provide two backends from the start: atomic bring-up path and segmented/gather production path. Keep the atomic path for debug and fallback only.

## Risk 3 — Hidden CPU access reintroduces UVM/page-migration traffic

- **Why it matters.** SPUMA and Nsight both show how incomplete device ports degrade into runtime page migration and execution stalls. [R1][R16]
- **Mitigation.** Ban host-side `correctBoundaryConditions()`, per-subcycle min/max reads, and patch object traversal in the hot loop. Instrument UM traffic in every milestone benchmark.

## Risk 4 — Patch-field runtime selection leaks virtual dispatch into device paths

- **Why it matters.** OpenFOAM boundary behavior is runtime-selected and heavily object-oriented. The GPU hot path cannot evaluate arbitrary patch virtual methods per face.
- **Mitigation.** Resolve runtime selection on the host into POD enums and parameter blocks once per patch. Unsupported patch types must use fallback.

## Risk 5 — Contact-angle and curvature work can generate NaNs or spurious currents

- **Why it matters.** `interfaceProperties` normalization uses `deltaN`, contact-angle correction uses `acos/cos` and a denominator `det = 1 - a12^2`; incorrect handling will produce NaNs or excessive parasitic currents. [R7]
- **Mitigation.** Clamp `a12` to `[-1, 1]`, guard `det`, preserve `deltaN` semantics, and validate against static-interface and capillary regression cases before nozzle runs.

## Risk 6 — Graph capture fails because helper libraries or allocations are not capture-safe

- **Why it matters.** CUDA Graphs are central to the launch-overhead strategy, but not every helper routine is guaranteed capture-safe.
- **Mitigation.** Keep a “capture-safe core path” consisting only of raw kernels plus verified library calls. Treat any CUB/CCCL use inside the steady-state captured path as provisional until explicitly tested. [R18][R20][R21]

## Risk 7 — Shared-memory overuse collapses occupancy on cc12.0

- **Why it matters.** Blackwell cc12.0 allows 99 KB/block, but occupancy remains constrained by shared memory and registers; the tuning guide reports 48 warps/SM and 64K registers/SM. [R14]
- **Mitigation.** Start with low-shared-memory kernels and tune only after measurement. Record achieved occupancy, register count, and eligible warps.

## Risk 8 — Production path silently expands scope into unsupported features

- **Why it matters.** Mesh motion, decomposed meshes, arbitrary patch classes, and unsupported schemes can consume the entire phase.
- **Mitigation.** Fail fast on unsupported combinations. Use explicit capability checks and runtime fallbacks rather than “best-effort” hidden behavior.

## Risk 9 — RTX 5080 memory budget is sufficient but not generous

- **Why it matters.** 16 GB VRAM is enough for the current intended case class but does not leave unlimited scratch headroom. [R10]
- **Mitigation.** Pre-compute scratch upper bounds, reuse buffers aggressively, and measure actual resident bytes before enabling optional compaction or extra mirrors.

# 5. Phase-by-phase implementation specification

## Phase 7 — Custom CUDA kernels for limiter-heavy, irregular, non-library hotspots

### Purpose

Implement a CUDA-native kernel subsystem for the parts of the device-resident nozzle solver that are not adequately addressed by library linear algebra and that are likely to bottleneck on an RTX 5080:

- alpha-flux assembly and explicit alpha update;
- MULES limiter preprocessing and iterative lambda update;
- interface normal, curvature, and surface-tension source calculation;
- patch/boundary evaluation for nozzle-relevant boundary conditions;
- fused mixture-property and face/cell update kernels that remove temporary churn.

The objective is not “maximum theoretical speed.” The objective is a **safe, correct, graph-ready custom-kernel path** that preserves v2412 numerics, keeps data on device, and reduces real hotspot cost on single-GPU workstation runs.

### Why this phase exists

This phase exists because the nozzle solver’s expensive work is not reducible to sparse linear solve time. SPUMA’s own performance analysis shows that irregular kernels, especially gradient/RHS work with atomics, perform worse than simple memory-bandwidth expectations would suggest, and its profiled path shows a very large kernel-launch count with host synchronizations after each launch. [R1]

For the nozzle workflow, the missing GPU value is in:

- repeated explicit VOF/MULES subcycles;
- face-based, indirectly addressed limiter logic;
- curvature and surface-tension kernels around the interface;
- boundary-condition evaluation for swirl inlet/wall/far-field patches;
- temporary-allocation and launch-count reduction.

Those are exactly the parts Phase 7 targets.

### Entry criteria

Phase 7 must not start until all of the following are true:

 1. **Reference cases frozen.** The validation ladder cases `R2`, `R1-core`, `R1`, and any applicable `R0` reference outputs exist and have accepted CPU/SPUMA baselines for the checks that Phase 7 will gate.
 2. **Toolchain validated.** The toolchain pinned in the master pin manifest is proven on the RTX 5080 with supported SPUMA solvers; profiling tools work.
 3. **Persistent device memory exists.** Phase 2 has replaced hot-path managed memory with stable device allocations for the working set.
 4. **Phase 5/6 baseline paths exist.** A device-resident VOF/two-phase path already runs correctly through alpha, mixture-property update, momentum, and pressure on the accepted `R1-core` generic-core path, and the Phase 6 nozzle BC/startup path is already stable on the frozen `R1` nozzle path that Phase 7 will optimize.
 5. **Patch metadata exists.** Phase 6 has already defined compacted device patch maps and nozzle BC parameter blocks.
 6. **Support matrix frozen and consumed.** The exact `fvSchemes`, patch classes, and any milestone-1 contact-angle scope used by the frozen reference cases are imported from the centralized support matrix; Phase 7 may not reopen them locally.
 7. **Hotspot ranking frozen.** Interim profiling artifacts from the accepted Phase 5/6 path identify the hotspot families to optimize, and every planned custom kernel family is both present in that ranking and present in the frozen support matrix.
 8. **Source audit complete.** The local SPUMA/v2412 implementations of:
   - `twoPhaseSolver` / `incompressibleVoF` alpha predictor logic,
   - MULES files,
   - `interfaceProperties`,
   - patch classes present in the nozzle cases,
   have been diffed against the public reference sources used in this document, and all semantic drift has been recorded.
9. **Fallbacks available.** For every planned custom kernel family, there is an existing non-custom path that can still run end-to-end for regression isolation.
 10. **Graph support matrix available.** The centralized graph support matrix already identifies which Phase 7 stages are capture-safe, graph-external, or smoke-test-only.

### Exit criteria

Phase 7 exits only when all of the following are satisfied:

1. Custom CUDA implementations exist, are integrated, and are runtime-selectable for every Phase 7 hotspot family admitted by the frozen support matrix and interim hotspot ranking, within the following candidate set:
   - alpha-flux assembly;
   - MULES limit + explicit update;
   - interface normal/curvature/surface tension;
   - patch update kernels for all in-scope patch classes in the frozen support matrix for the milestone-1 nozzle cases;
   - fused mixture update.
2. Every kernel family is allocation-free during a timestep/subcycle.
3. Every kernel family has a clean fallback switch.
4. The production path runs with device-resident fields and shows **near-zero steady-state UVM traffic** in Nsight Systems.
5. Kernel APIs are graph-capture-safe for the Phase-7-owned stages marked capture-safe in the centralized graph support matrix: stable addresses, no host callbacks, no unsupported stream-capture operations in the default path. Phase 7 requires successful smoke capture/replay of those stages; formal graph-performance acceptance remains Phase 8 scope.
6. Correctness gates pass:
   - boundedness,
   - conservation,
   - no NaN/Inf,
   - CPU/SPUMA regression thresholds,
   - nozzle QoI thresholds.
7. Performance gates pass:
   - launch count for the targeted alpha/interface/BC portion is reduced materially;
   - at least one targeted hotspot shows a measurable stage speedup over the pre-Phase-7 path;
   - no custom-kernel family is retained if it is both slower and more complex without providing a correctness or residency benefit.
8. A profiling package exists: Nsight Systems timeline, top-kernel Nsight Compute captures, and a benchmark summary comparing fallback vs custom backends.

### Goals

1. Preserve v2412 numerics while moving limiter/interface/patch hot paths to CUDA.
2. Eliminate host-side patch evaluation and host-driven field inspection in the alpha/interface subcycle path.
3. Provide a **correctness-first** custom MULES backend.
4. Provide a **production** MULES backend that reduces atomics through adjacency-aware gather/segmented reductions.
5. Keep all topology, fields, and scratch arrays resident in device memory.
6. Make the kernel subsystem graph-ready by design.
7. Reduce temporary-allocation churn and kernel-launch explosion.
8. Support the exact scheme and patch set used by the frozen reference nozzle cases.
9. Expose instrumentation detailed enough to debug boundedness failures and performance regressions.
10. Keep solver-module integration narrow and reversible.

### Non-goals

1. Writing a custom pressure linear solver, custom AMG, or custom generic SpMV.
2. Replacing all of SPUMA’s `fvc` / `fvm` operators globally.
3. Supporting mesh motion, topology change, AMR, or decomposed multi-GPU meshes.
4. Introducing geometric VOF / isoAdvector in this phase.
5. Changing turbulence or physical models.
6. Introducing mixed precision, Tensor Core paths, block clusters, distributed shared memory, or device graph launch as baseline requirements.
7. Generalizing to arbitrary boundary-condition classes not present in the frozen nozzle cases.
8. Generalizing to arbitrary `fvSchemes` beyond the audited supported subset.
9. Making phase 7 backend-agnostic across vendors. The public API remains narrow, but this phase deliberately implements CUDA for NVIDIA Blackwell.

### Technical background

The target solver family is `incompressibleVoF`, which OpenFOAM describes as a solver module for two incompressible, isothermal, immiscible fluids using a VOF phase-fraction interface-capturing approach, with a single mixture momentum equation and PIMPLE coupling. [R3]

The VOF hot path is not just a sparse solve. The relevant algorithmic pieces include:

- alpha predictor with explicit subcycling;
- MULES bounded flux limiting;
- mixture-property updates from alpha;
- interface normal and curvature computation;
- surface-tension source computation;
- pressure-correction coupling through `rhoPhi` and related fields.

OpenFOAM’s MULES implementation is an explicit, bounded transport limiter. The public MULES description and source code show a sequence that, in simplified terms, does the following:

1. build a bounded upwind flux `phiBD`;
2. compute correction flux `phiCorr = psiPhi - phiBD`;
3. build a correction-equation source `SuCorr`;
4. compute or update a per-face limiter `lambda`;
5. form limited flux `psiPhi = phiBD + lambda*phiCorr`;
6. explicitly update the transported field through a surface integration of `psiPhi`. [R5][R6]

The limiter path is heavily face/cell coupled. It uses owner/neighbour addressing, boundary patch loops, per-cell extrema and sum accumulators, and an iterative update of cell-side and face-side limiter coefficients. This is precisely the kind of irregular work that a generic sparse library does not solve. [R5][R6]

For interface curvature and surface tension, `interfaceProperties::calculateK()` in the OpenFOAM source computes a cell gradient of alpha, interpolates that gradient to faces, forms a normalized face interface normal, applies contact-angle correction on the boundary, computes the face-normal flux `nHatf`, and then computes curvature as `K = -div(nHatf)`. `surfaceTensionForce()` then returns `interpolate(sigmaK()) * snGrad(alpha1_)`. [R7]

This is important for implementation planning because it means the “surface-tension kernel” is not one kernel. It is a pipeline of gradients, interpolations, boundary corrections, and divergence-like reductions, with numerically sensitive normalization and patch handling. A correct port must preserve that structure before trying to fuse it.

Finally, the target architecture is NVIDIA Blackwell on a GeForce RTX 5080. NVIDIA lists the RTX 5080 as a compute capability 12.0 device with 16 GB GDDR7 and 960 GB/s memory bandwidth. CUDA 12.8 adds native Blackwell compiler support and additional CUDA Graph conditional control. The Blackwell tuning guide reports, for compute capability 12.0, 48 resident warps/SM maximum, 64K registers/SM, and 99 KB maximum shared memory per block; the tuning guide also reports 128 KB shared-memory capacity per SM. NVIDIA’s generic compute-capabilities appendix presently reports a lower “maximum shared memory per SM” number for 12.x, so the exact interpretation of aggregate per-SM shared-memory accounting should not be treated as a tuning invariant. The one limit that is consistently documented is the **99 KB per-block ceiling**, which is the safe implementation constraint. [R10][R11][R12][R14][R15]

### Research findings relevant to this phase

1. SPUMA’s performance study shows that GPU efficiency is not uniform across kernels; gradient/RHS work is weaker because the gradient kernel includes several atomic operations, and the pressure solver path with GAMG is also weaker. [R1]
2. SPUMA’s profiled path shows 53,259 kernel launches and `cudaDeviceSynchronize()` after each kernel invocation in the measured case, with synchronization dominating API time. [R1]
3. SPUMA’s project metadata states that SPUMA `0.1-v2412` is based on OpenCFD v2412. [R2]
4. OpenFOAM’s `incompressibleVoF` solves two incompressible immiscible fluids using VOF with a single mixture momentum equation under PIMPLE. [R3]
5. OpenFOAM MULES is explicitly documented as a multidimensional universal limiter for explicit solution. [R6]
6. The raw MULES source shows that `limit()` builds `phiBD`, forms `phiCorr`, constructs `SuCorr`, allocates/updates `lambda`, and then either returns the limited correction or reconstructs `psiPhi = phiBD + lambda*phiCorr`. [R5][R6]
7. The raw `interfaceProperties` source shows the curvature pipeline: `grad(alpha)` → `interpolate(grad)` → normalize to `nHatfv` → `correctContactAngle` → `nHatf = nHatfv & Sf` → `K = -div(nHatf)`; `surfaceTensionForce()` uses `interpolate(sigmaK()) * snGrad(alpha1_)`. [R7]
8. OpenFOAM v13 later changed MULES and explicitly advertises improvements to guarantee boundedness and accelerate solutions, so using v13 MULES behavior as a silent proxy for v2412 is unsafe. [R8][R9]
9. CUDA 12.8 adds Blackwell compiler support (`sm_120`) and extends CUDA Graph control with IF/ELSE and SWITCH conditional nodes. [R12][R19]
10. NVIDIA lists the RTX 5080 as compute capability 12.0, 16 GB, 960 GB/s, with no NVLink. [R10][R11]
11. The Blackwell tuning guide reports 48 warps/SM, 64K registers/SM, 32 thread blocks/SM max, 128 KB shared-memory capacity/SM, and 99 KB shared memory per thread block for compute capability 12.0. [R14]
12. NVIDIA’s Blackwell compatibility guidance recommends including PTX and suggests `CUDA_FORCE_PTX_JIT=1` to test PTX readiness; it also gives the minimum driver versions for CUDA 12.8 compatibility. [R13]
13. Nsight Systems documents that managed-memory HtoD/DtoH transfers pause execution and can incur significant penalties. [R16]
14. NVIDIA’s warp-aggregated atomics guidance shows that aggregating updates can reduce the number of atomic operations substantially in suitable patterns. [R17]
15. CUB provides segmented reduction primitives, but graph-capture support for all routines has not historically been uniformly documented; CCCL issue history shows this area requires explicit validation rather than assumption. [R18][R20][R21]

### Design decisions

#### DD7.1 — Custom-kernel scope is limited to support-matrix-approved hotspot families ranked by interim profiling

- **Sourced fact.** Library linear algebra does not cover MULES limiter logic or `interfaceProperties` pipelines. [R5][R6][R7]
- **Engineering inference.** Custom pressure SpMV would consume effort without addressing nozzle-specific irregular hotspots.
- **Recommendation.** Build custom CUDA only for hotspot families that are both present in the frozen milestone-1 support matrix and ranked as material by interim profiling artifacts from the accepted Phase 5/6 path:
  - alpha-flux assembly;
  - MULES limit + explicit solve;
  - interface normal/curvature/surface tension;
  - patch kernels;
  - fused field updates.
  Pressure solve stays on existing SPUMA/native or external-solver infrastructure.

#### DD7.2 — Phase 7 supports the **two-phase single-transported-alpha** production path first

- **Sourced fact.** `incompressibleVoF` is a two-immiscible-fluid solver. [R3]
- **Engineering inference.** General n-phase MULES support adds complexity (`limitSum`, multiple phase correction fluxes, broader data layout) that the nozzle use case does not need immediately.
- **Recommendation.** Implement production kernels around transported `alpha1`, derived `alpha2 = 1 - alpha1`, and derived `rho`, `rhoPhi`, and `alphaPhi2` where needed. Keep APIs extensible enough to add n-phase later, but do not make n-phase support a Phase 7 gate.

#### DD7.3 — Use a narrow host-side façade and POD device views

- **Sourced fact.** OpenFOAM boundary conditions and fields are runtime-selected, object-heavy constructs.
- **Engineering inference.** Passing full OpenFOAM objects into device code would couple CUDA to host runtime semantics and block graph capture.
- **Recommendation.** Introduce a host façade that translates OpenFOAM/SPUMA objects into POD device views:
  - mesh topology view,
  - geometry view,
  - alpha/phi/U/rho field views,
  - patch metadata view,
  - scratch view,
  - launch/config view.
  CUDA translation units see only POD views and raw device pointers.

#### DD7.4 — Default to persistent device allocations with stable addresses

- **Sourced fact.** CUDA Graph replay benefits from stable graph structure; Phase 7 must be graph-ready. CUDA 12.8 expands graph conditional control. [R12][R19]
- **Engineering inference.** Reallocating scratch per subcycle will destabilize pointer arguments and reintroduce allocator overhead.
- **Recommendation.** All Phase 7 scratch arrays are allocated once from the project’s device allocator/pool at solver initialization or mesh-prepare time and are never resized during a run unless the mesh changes, in which case the Phase 7 path falls back.

#### DD7.5 — Keep vector-valued geometry in AoS layout initially; use SoA for indices and scalar work arrays

- **Sourced fact.** OpenFOAM naturally stores vectors as packed vector objects; the main irregularity in MULES/interface work comes from owner/neighbour and patch addressing, not dense vector math.
- **Engineering inference.** Mirroring all vector fields into SoA would add memory overhead and conversion complexity before the real bottlenecks are understood.
- **Recommendation.** Keep `vector` geometry/field buffers in the existing packed layout initially. Store indices, offsets, flags, scalar fields, and scalar scratch in separate contiguous arrays. Revisit vector SoA mirrors only if profiling proves a component-access bottleneck.

#### DD7.6 — Implement two limiter backends: atomic baseline and segmented/gather production backend

- **Sourced fact.** SPUMA shows atomics in gradient kernels are a real performance issue. NVIDIA documents warp aggregation as a mitigation and CUB provides segmented reduction primitives. [R1][R17][R18]
- **Engineering inference.** A face-parallel atomic scatter is the fastest path to correctness but probably not the best end-state for nozzle limiter kernels.
- **Recommendation.**
  - **Bring-up backend:** face-parallel scatter with atomics for `SuCorr`, flux sums, and/or extrema support where needed.
  - **Production backend:** owner/neighbour-sorted adjacency lists and cell-parallel gather/segmented reductions for repeated face-to-cell accumulation.
  Both backends must share identical public APIs and validation tests.

#### DD7.7 — Preserve scalar precision and disable `--use_fast_math` by default

- **Sourced fact.** The RTX 5080 is compute capability 12.0, and Blackwell cc12.0 exposes much less FP64 throughput than FP32 throughput. [R11][R15]
- **Engineering inference.** Precision-reducing optimizations may help speed but are too risky during the initial port, especially around boundedness and capillary terms.
- **Recommendation.** Keep `Foam::scalar` precision end-to-end for Phase 7 baseline. Default build uses standard IEEE math, not `--use_fast_math`. Optional fast-math experiments may be added only behind explicit experimental flags after validation.

#### DD7.8 — Support only the audited scheme subset in custom kernels

- **Sourced fact.** `interfaceProperties::calculateK()` and alpha transport use specific gradient, interpolation, and flux semantics from the solver’s numerics stack. [R5][R7]
- **Engineering inference.** A “generic all-schemes” CUDA operator layer would balloon scope and likely re-implement half of `fvc`.
- **Recommendation.** At solver initialization, parse and validate the scheme tuple imported from the centralized support matrix for the frozen reference cases. Phase 7 consumes that tuple; it does not determine it locally. The custom path must support only that tuple plus a small explicitly enumerated superset that is also added to the same support matrix. Unsupported schemes must fail fast or fall back to the existing path.

#### DD7.9 — Boundary-condition dispatch happens on host; boundary math happens on device

- **Sourced fact.** `interfaceProperties` and MULES both require patch-specific boundary handling. [R5][R6][R7]
- **Engineering inference.** Per-face virtual dispatch on the GPU is both slow and fragile.
- **Recommendation.** Resolve each patch once into:
  - a `PatchKernelKind` enum,
  - a POD parameter block,
  - a compacted face-range descriptor,
  - optional mate/transform data for coupled patches.
  Here “boundary-condition dispatch happens on host” means only prevalidated enum selection, parameter-block selection, and kernel-launch choice performed outside the hot loop. It does **not** permit per-face coefficient evaluation, host patch-object traversal, or runtime dictionary lookups during stage execution. Device kernels operate only on those resolved descriptors.

#### DD7.10 — Fixed-iteration limiter loops are the baseline; tolerance-based early exit is optional and device-side

- **Sourced fact.** MULES has iterative limiter controls (`nIter`, `tol`, and related settings). [R6]
- **Engineering inference.** Host-side tolerance checks inside every limiter iteration would break graph capture and add synchronization.
- **Recommendation.** Baseline production path uses fixed `nIter` iterations from the validated case settings. An optional device-side convergence flag/residual path may be added later, but host-side early-exit in the steady-state path is not allowed.

#### DD7.11 — Shared memory is a tactical optimization, not a baseline requirement

- **Sourced fact.** Blackwell cc12.0 allows at most 99 KB shared memory per block, with occupancy limits driven by registers and shared memory. [R14][R15]
- **Engineering inference.** Over-aggressive tiling can easily reduce occupancy below the point where irregular-memory latency is hidden.
- **Recommendation.** Start with register/global-memory implementations and small shared-memory staging where obviously beneficial. Only escalate to large shared-memory kernels after Nsight Compute confirms a win.

#### DD7.12 — Do not baseline Tensor Cores, block clusters, DSM, or device graph launch

- **Sourced fact.** Blackwell exposes cluster and advanced graph features, but the dominant workload here is irregular face-based limiter logic, not dense MMA. [R14][R19]
- **Engineering inference.** These features add complexity and are unlikely to be first-order wins for this phase.
- **Recommendation.** Mark them explicitly as deferred experiments, not implementation requirements.

### Alternatives considered

#### Alternative A — Library-only path

- **Description.** Keep only external/native linear solver acceleration and rely on generic SPUMA operators elsewhere.
- **Why rejected.** This does not address the irregular VOF/MULES/interface/boundary work that motivates Phase 7.

#### Alternative B — Atomics-only face-parallel custom kernels

- **Description.** One thread per face, atomically scatter all face contributions into cells forever.
- **Why rejected as final architecture.** It is useful for bring-up but likely leaves performance on the table on the exact kernels SPUMA already identifies as atomic-sensitive. Keep only as fallback/debug backend.

#### Alternative C — Segmented/gather-only implementation from day one

- **Description.** Build only the optimized adjacency-aware production backend.
- **Why rejected.** It front-loads the hardest correctness work and removes a simple debug fallback. The safe path is atomic correctness first, segmented optimization second.

#### Alternative D — Global replacement of `fvc::grad`, `fvc::div`, `fvc::interpolate`

- **Description.** Create a general-purpose GPU operator library that replaces all hot finite-volume operators everywhere.
- **Why rejected.** Scope explosion. Phase 7 needs a solver-targeted implementation for a narrow set of schemes and fields, not a global operator rewrite.

#### Alternative E — Geometric VOF / isoAdvector during Phase 7

- **Description.** Replace algebraic VOF/MULES with a geometric scheme while porting to CUDA.
- **Why rejected.** This changes numerics and validation scope simultaneously. It is a future extension, not a safe first production target.

#### Alternative F — Managed-memory hot path with advice

- **Description.** Keep Phase 7 fields in managed memory with preferred-location hints.
- **Why rejected.** Useful for incomplete ports and bring-up, but inconsistent with the production residency goals for a discrete RTX 5080. [R1][R16]

#### Alternative G — Immediate mixed precision

- **Description.** Store/compute limiter scratch or curvature in FP32 while leaving some state in FP64.
- **Why rejected.** Too much numerical risk before the exact v2412 semantics are re-established on the device.

### Interfaces and dependencies

#### 1. External dependencies

Phase 7 depends on the following existing subsystems from earlier phases:

- **SPUMA/OpenFOAM solver integration layer**
  - owns `fvMesh`, field objects, time loop, `PIMPLE`, and solver controls;
- **device memory/pool layer**
  - provides persistent device allocations;
- **device mesh cache**
  - provides device-resident topology/geometry arrays;
- **device field registry**
  - provides stable device addresses for alpha, phi, U, rho, etc.;
- **pressure-solver path**
  - unchanged by Phase 7 except for consuming updated `rhoPhi`/source fields;
- **profiling layer**
  - NVTX3 ranges, Nsight integration, diagnostics.

#### 2. New public host-side API

The Phase 7 subsystem must introduce a narrow façade. Recommended namespace and API shape:

```cpp
namespace Foam::spuma::deviceVoF
{

struct DeviceVoFControls;
struct DeviceMeshView;
struct DevicePatchView;
struct DeviceAdjacencyView;
struct TwoPhaseFieldView;
struct InterfaceFieldView;
struct PatchStateView;
struct Phase7ScratchView;
struct LaunchContext;
struct Phase7Diagnostics;

class Phase7KernelFacade
{
public:
    Phase7KernelFacade() = default;

    void prepare
    (
        const DeviceVoFControls& controls,
        const DeviceMeshView& mesh,
        const DevicePatchView& patches,
        const DeviceAdjacencyView& adjacency,
        Phase7ScratchView& scratch,
        LaunchContext& ctx
    );

    void runAlphaSubCycle
    (
        TwoPhaseFieldView& fields,
        const DeviceVoFControls& controls,
        LaunchContext& ctx
    );

    void runInterfaceCorrection
    (
        InterfaceFieldView& fields,
        const DeviceVoFControls& controls,
        LaunchContext& ctx
    );

    void runPatchUpdates
    (
        PatchStateView& patches,
        const DeviceVoFControls& controls,
        LaunchContext& ctx
    );

    void runFusedMixtureUpdate
    (
        TwoPhaseFieldView& fields,
        const DeviceVoFControls& controls,
        LaunchContext& ctx
    );

    void collectDiagnostics
    (
        Phase7Diagnostics& diag,
        LaunchContext& ctx
    ) const;
};

}
```

#### 3. New internal executors

`Phase7KernelFacade` delegates to specialized executors:

- `AlphaFluxExecutor`
- `MulesLimiterExecutor`
- `InterfaceExecutor`
- `PatchExecutor`
- `FusedUpdateExecutor`
- `Phase7DiagnosticsExecutor`

Each executor owns only launch parameters and helper metadata. It does **not** own OpenFOAM fields.

#### 4. Required enums and POD descriptors

```cpp
enum class AlphaConvScheme : uint8_t
{
    Unsupported = 0,
    Upwind,
    VanLeer
};

enum class InterpScheme : uint8_t
{
    Unsupported = 0,
    Linear
};

enum class GradScheme : uint8_t
{
    Unsupported = 0,
    GaussLinear
};

enum class LimiterBackend : uint8_t
{
    Atomic = 0,
    Segmented = 1
};

enum class LimiterIterationMode : uint8_t
{
    FixedIter = 0,
    DeviceTol = 1,
    HostTolDebug = 2
};

enum class PatchKernelKind : uint8_t
{
    Unsupported = 0,
    FixedValueScalar,
    FixedValueVector,
    ZeroGradientScalar,
    ZeroGradientVector,
    NoSlipWall,
    InletOutletScalar,
    PressureDrivenSwirlInlet,
    AlphaContactAngleConst,
    AlphaContactAngleDynamic,
    Calculated
};
```

#### 5. Host-launch context

```cpp
struct LaunchContext
{
    cudaStream_t stream;
    bool captureMode;              // true when inside graph capture
    bool diagnosticsEnabled;
    bool debugSynchronous;         // allowed only in debug builds/small tests
    cudaEvent_t* optionalEvent;    // may be null
};
```

#### 6. Ownership model

- `fvMesh` and OpenFOAM fields are owned by solver modules outside Phase 7.
- Device-resident field buffers are owned by the project’s field registry from earlier phases.
- `DeviceMeshView`, `DevicePatchView`, and `DeviceAdjacencyView` are non-owning POD views.
- `Phase7ScratchView` owns or references persistent scratch buffers from the device allocator.
- `Phase7KernelFacade` owns no heavy memory; it owns only cached launch metadata and capability checks.

#### 7. Runtime control object

Recommended dictionary-backed control object (materialized as the Phase 7 subview of `gpuRuntime.vof` plus compatibility shims, not as a new independent top-level contract):

```cpp
struct DeviceVoFControls
{
    bool enableCustomAlphaFlux;
    bool enableCustomMules;
    bool enableCustomInterface;
    bool enableCustomPatchKernels;
    bool enableCustomFusedUpdates;

    AlphaConvScheme alphaConvScheme;
    InterpScheme interfaceInterpScheme;
    GradScheme interfaceGradScheme;

    LimiterBackend limiterBackend;
    LimiterIterationMode limiterIterationMode;

    int limiterIterations;         // fixed-iter baseline
    double limiterTolerance;       // used only in DeviceTol/HostTolDebug
    bool globalBounds;
    double smoothingCoeff;
    double extremaCoeff;
    double boundaryExtremaCoeff;

    int threadsPerBlockFaces;
    int threadsPerBlockCells;
    int threadsPerBlockPatches;

    bool allowCaptureUnsafeHelpers; // default false
    bool allowExperimentalFastMath; // default false
};
```

### Data model / memory model

#### 1. Core mesh topology view

The production kernels require a **stable, device-resident, topology-centric** view of the mesh:

```cpp
struct DeviceMeshView
{
    int nCells;
    int nInternalFaces;
    int nBoundaryFaces;
    int nFacesTotal;               // internal + boundary

    const int* owner;              // [nInternalFaces]
    const int* neighbour;          // [nInternalFaces]

    const int* boundaryFaceCell;   // [nBoundaryFaces]
    const int* boundaryGlobalFace; // [nBoundaryFaces] -> global face index

    const FoamVec3* C;             // [nCells]
    const FoamVec3* Cf;            // [nFacesTotal]
    const FoamVec3* Sf;            // [nFacesTotal]
    const double* magSf;           // [nFacesTotal]
    const double* V;               // [nCells]
    const double* deltaCoeffs;     // [nFacesTotal], if already available
};
```

Notes:

- `FoamVec3` is a POD-compatible representation of `Foam::vector`.
- Geometry remains in AoS form initially.
- `deltaCoeffs` is optional if existing SPUMA device infrastructure already owns it; if not, it must be added in earlier phases or in Phase 7 prepare.

#### 2. Adjacency views for segmented/gather backend

To eliminate repeated atomics in the production path, precompute CSR-like adjacency lists **once per mesh**:

```cpp
struct DeviceAdjacencyView
{
    // owner-sorted internal faces
    const int* ownerRowOffsets;      // [nCells + 1]
    const int* ownerSortedFaceIdx;   // [nInternalFaces]

    // neighbour-sorted internal faces
    const int* neighRowOffsets;      // [nCells + 1]
    const int* neighSortedFaceIdx;   // [nInternalFaces]

    // boundary faces grouped by owner cell
    const int* bndRowOffsets;        // [nCells + 1]
    const int* bndSortedFaceIdx;     // [nBoundaryFaces]
};
```

These arrays are built once during `prepare()`. They are mandatory for the production limiter backend and for cell-gather divergence kernels.

#### 3. Patch metadata view

```cpp
struct PatchMeta
{
    PatchKernelKind kind;
    uint32_t flags;            // coupled, wall, needsTransform, etc.

    int faceStart;             // offset into compacted boundary arrays
    int nFaces;

    int paramIndex;            // index into patch-parameter table
    int mateStart;             // offset into coupled mate map, -1 if none
    int transformIndex;        // offset into transform data, -1 if none
};

struct DevicePatchView
{
    int nPatches;
    const PatchMeta* patchMeta;

    const int* compactBoundaryFaceIdx;    // boundary faces sorted by patch/kind
    const int* coupledMateFaceLocalIdx;   // for cyclic/coupled on single GPU
    const FoamMat3* coupledTransforms;    // optional transform matrices
};
```

Patch compaction rule:

- compact by `PatchKernelKind`,
- then by patch id,
- then preserve original face order within a patch unless a stronger locality ordering is proven safe.

#### 4. Two-phase field view

Production path for Phase 7 is **single transported alpha** with derived secondary phase.

```cpp
struct TwoPhaseFieldView
{
    // Cell fields
    double* alpha1;              // [nCells]
    const double* alpha1Old;     // [nCells]
    double* alpha2;              // [nCells], may be derived/write-through
    double* rho;                 // [nCells]
    double* mu;                  // [nCells], if needed by later phases
    const FoamVec3* U;           // [nCells]
    const FoamVec3* UOld;        // optional
    const double* Sp;            // [nCells], may alias zero field
    const double* Su;            // [nCells], may alias zero field

    // Face fields
    const double* phi;           // [nFacesTotal]
    double* alphaPhi1;           // [nFacesTotal]
    double* alphaPhi2;           // [nFacesTotal]
    double* rhoPhi;              // [nFacesTotal]
    double* phic;                // [nFacesTotal], if compression coeff stored

    // Physical constants / scalar controls
    double rho1;
    double rho2;
    double dt;
    double rDeltaT;
    bool movingMesh;             // production path requires false
};
```

#### 5. Interface field view

```cpp
struct InterfaceFieldView
{
    double* alpha1;              // [nCells]
    const FoamVec3* U;           // [nCells]

    FoamVec3* gradAlpha;         // [nCells]
    FoamVec3* gradAlphaf;        // [nFacesTotal]
    FoamVec3* nHatfv;            // [nFacesTotal]
    double* nHatf;               // [nFacesTotal]
    double* K;                   // [nCells]
    double* sigmaK;              // [nCells]
    double* surfaceTensionForce; // [nFacesTotal]

    double sigma;
    double deltaN;
};
```

#### 6. Scratch arena

Scratch buffers must be persistent, not ephemeral. Minimum required buffers:

```cpp
struct Phase7ScratchView
{
    // limiter scratch
    double* phiBD;               // [nFacesTotal]
    double* phiCorr;             // [nFacesTotal]
    double* lambdaFace;          // [nFacesTotal]
    double* SuCorr;              // [nCells]
    double* sumPhip;             // [nCells]
    double* mSumPhim;            // [nCells]
    double* psiMaxn;             // [nCells]
    double* psiMinn;             // [nCells]
    double* lambdap;             // [nCells]
    double* lambdam;             // [nCells]
    double* divPsiPhi;           // [nCells]

    // diagnostics / reductions
    double* blockReduceScratch;
    int* violationCount;         // [1]
    int* nanCount;               // [1]
    double* residualScalar;      // [1]
    double* minAlphaScalar;      // [1]
    double* maxAlphaScalar;      // [1]

    size_t bytesReserved;
};
```

#### 7. Memory lifetime rules

- **Allocated once**:
  - mesh topology,
  - adjacency lists,
  - patch descriptors,
  - all scratch buffers.
- **Updated each timestep/subcycle**:
  - alpha/rho/rhoPhi,
  - interface fields,
  - patch state,
  - diagnostics scalars.
- **Never allocated/freed in hot loop**:
  - no temporary `surfaceScalarField` mirrors,
  - no per-iteration lambda allocation,
  - no dynamic compaction buffers created on demand.
- **Host-pinned only**:
  - diagnostics snapshots,
  - write-time staging,
  - optional debug dumps.

#### 8. Memory budget rule

Before enabling production kernels on the full nozzle case, compute and log:

- persistent mesh bytes;
- persistent field bytes;
- persistent scratch bytes;
- high-water mark during one timestep.

Phase 7 is rejected if it cannot fit with a safety margin of at least **10% free VRAM** on the validated reduced nozzle case and a documented margin on the full reference case. This threshold is a recommendation, not a hardware fact; it is chosen to reduce allocator fragmentation and profiling distortion.

### Algorithms and control flow

#### A. Initialization / prepare path

`prepare()` is a one-time (or rare) host function. It must:

1. validate mesh assumptions:
   - no decomposition,
   - no mesh motion,
   - no topology change,
   - supported patch set only;
2. validate scheme assumptions:
   - supported alpha convection scheme,
   - supported interface grad/interp scheme,
   - supported boundary models;
3. build or verify adjacency lists:
   - owner-sorted internal faces,
   - neighbour-sorted internal faces,
   - boundary faces grouped by owner cell;
4. allocate persistent scratch from device pool;
5. compile or cache launch configurations;
6. precompute patch parameter blocks;
7. record capability flags into diagnostics.

If any validation fails, `prepare()` must disable the relevant custom path and leave the fallback path active.

#### B. Alpha subcycle control flow

The default Phase 7 alpha subcycle is:

1. apply required patch updates for alpha/phi/U-dependent patch state;
2. assemble `alphaPhi1` on device using the supported scheme path;
3. build limiter inputs `phiBD`, `phiCorr`, `SuCorr`;
4. run `nIter` limiter iterations:
   - preprocess cell extrema and correction sums;
   - compute cell-side limiter coefficients;
   - update face `lambda`;
   - optionally compute device-side convergence residual;
5. finalize limited `alphaPhi1`;
6. explicit alpha update by surface integration;
7. enforce/refresh alpha boundary state;
8. derive `alpha2 = 1 - alpha1`;
9. fused update of `rho`, `alphaPhi2`, and `rhoPhi`;
10. optional diagnostics reduction.

#### C. Interface correction control flow

The default Phase 7 interface path is:

1. compute `gradAlpha` on cells;
2. interpolate `gradAlpha` to faces;
3. normalize face gradients to `nHatfv`;
4. apply contact-angle correction on relevant patches;
5. compute `nHatf = dot(nHatfv, Sf)`;
6. compute `K = -div(nHatf)`;
7. compute `sigmaK = sigma * K`;
8. compute face `surfaceTensionForce = interpolate(sigmaK) * snGrad(alpha1)`;
9. write all outputs into persistent field buffers for later momentum/pressure use.

#### D. Patch update control flow

Boundary updates are executed as explicit kernel groups, not hidden side effects. At minimum, Phase 7 must support:

- fixed value scalar/vector;
- zero gradient scalar/vector;
- no-slip wall for `U`;
- inlet/outlet scalar;
- pressure-driven swirl inlet;
- alpha contact-angle patch model(s) used in the frozen cases, **if** the centralized support matrix marks them as required for milestone-1 acceptance;
- calculated/pass-through fields.

The host façade schedules patch kernels by patch kind and field family. No per-face virtual dispatch is permitted.

#### E. Synchronization rules

Allowed:

- stream ordering inside one compute stream;
- optional event after a major stage for diagnostics;
- optional one host sync at the end of a debug-only small test.

Not allowed in the production path:

- `cudaDeviceSynchronize()` between every kernel;
- host polling per limiter iteration;
- synchronous min/max extraction per subcycle;
- device allocation in any timed stage.

#### F. Algorithmic details by kernel family

##### F1. Alpha-flux assembly

Bring-up implementation requirements:

- one internal-face kernel;
- one boundary-face kernel;
- exact sign conventions preserved from the v2412 source;
- boundary face fluxes resolved using patch descriptors, not host patch objects.

Production implementation requirements:

- fuse convective and compressive flux assembly where dependencies are local and scheme support is fixed;
- keep `phi`, `phic`, and `nHatf` device-resident;
- support the exact audited alpha scheme tuple only.

Recommended function split:

```cpp
void assembleAlphaPhi
(
    const DeviceMeshView& mesh,
    const DevicePatchView& patches,
    TwoPhaseFieldView& f,
    Phase7ScratchView& s,
    const DeviceVoFControls& c,
    LaunchContext& ctx
);
```

##### F2. MULES limiter preprocessing

Required scalar variables and naming should mirror source terminology as closely as possible:

- `phiBD`
- `phiCorr`
- `SuCorr`
- `sumPhip`
- `mSumPhim`
- `psiMaxn`
- `psiMinn`
- `lambdap`
- `lambdam`
- `lambdaFace`

Mechanical transliteration rule:

- port the local SPUMA/v2412 loops in the same conceptual stages and keep variable names intact in the CUDA version;
- do **not** algebraically simplify formulas during the first implementation.

##### F3. MULES iterative lambda update

Two implementations are required.

**Atomic backend**
- one face thread computes contribution and atomically updates cell accumulators;
- acceptable only for bring-up, debug, and correctness reference on small cases.

**Segmented/gather backend**
- face contributions are precomputed into face arrays;
- cell accumulations are gathered using precomputed owner/neighbour/boundary adjacency lists;
- face `lambda` update remains face-parallel;
- this is the required production backend.

Face limiter selection rule for the two-phase path:

- if correction flux on an internal face is positive in owner-to-neighbour orientation, the owner is constrained by the lower-bound coefficient and the neighbour by the upper-bound coefficient;
- if correction flux is negative, the owner is constrained by the upper-bound coefficient and the neighbour by the lower-bound coefficient.

This sign logic is an engineering reconstruction consistent with the variable naming and standard MULES semantics; it **must** be checked against the exact local v2412 source during implementation. [R5][R6]

##### F4. Explicit alpha update

The custom explicit-update kernel must implement the same formula as the source-of-truth v2412 MULES explicit solve, including the handling of source terms and `rDeltaT`. The Phase 7 production path assumes a static mesh, so the moving-mesh branch is not implemented here; if `movingMesh == true`, fallback is mandatory. [R5]

##### F5. Interface normal and curvature

For the baseline supported scheme tuple, implement the source-equivalent sequence:

- `gradAlpha = grad(alpha1)`
- `gradAlphaf = interpolate(gradAlpha)`
- `nHatfv = gradAlphaf / (|gradAlphaf| + deltaN)`
- `correctContactAngle(...)`
- `nHatf = dot(nHatfv, Sf)`
- `K = -div(nHatf)`
- `sigmaK = sigma * K`
- `surfaceTensionForce = interpolate(sigmaK) * snGrad(alpha1)` [R7]

The baseline implementation may use separate kernels for these stages. Fusion is permitted only after correctness is established.

##### F6. Contact-angle correction

This kernel family is conditional. Implement it only if the frozen milestone-1 support matrix and acceptance cases require contact-angle handling; otherwise keep the interface, capability checks, and validation hooks only, and route any contact-angle patch kind to explicit unsupported/fallback behavior.

 The contact-angle correction kernel must support at least the wall wetting models used by the frozen nozzle cases when it is enabled. For the constant/dynamic contact-angle logic shown in the reference source, the patch-face update follows the `a12/b1/b2/det/a/b` formulation in `interfaceProperties`, including normalization by `(mag(nHatp) + deltaN)` and gradient update. [R7]

Numerical safeguards:

- clamp `a12` into `[-1, 1]` before `acos`;
- guard `det` with a small floor;
- detect NaN/Inf and trip diagnostics;
- fallback on unsupported contact-angle model kinds.

##### F7. Patch kernels

Patch kernels must operate on compacted patch ranges. Each patch kind gets either:

- a specialized kernel, or
- a small generic kernel for simple algebraic cases.

`PressureDrivenSwirlInlet` must be a specialized kernel. It must not call host patch logic in the hot path.

##### F8. Fused mixture update

At minimum, the fused update kernel must compute:

- `alpha2 = 1 - alpha1`
- `rho = alpha1*rho1 + alpha2*rho2`
- `alphaPhi2 = phi - alphaPhi1` or the exact v2412 equivalent used by the local solver path
- `rhoPhi = alphaPhi1*rho1 + alphaPhi2*rho2`

If viscosity or any additional mixture property is alpha-derived and used immediately afterward, include it in the same kernel if the data dependency is local.

### Required source changes

The following source changes are required. File names are proposed and may be adapted to local SPUMA conventions, but module boundaries should remain the same.

1. Add a new Phase 7 custom-kernel library under a dedicated device-VOF module.
2. Add a controls parser for `deviceVoFControls`.
3. Add a mesh-preparation step that builds adjacency lists and patch descriptors.
4. Add a host façade that converts OpenFOAM fields into device views.
5. Add CUDA translation units for:
   - alpha flux assembly,
   - MULES atomic backend,
   - MULES segmented backend,
   - interface kernels,
   - patch kernels,
   - fused updates,
   - diagnostics helpers.
6. Add solver integration hooks in the existing VOF/two-phase module so the solver can select custom or fallback kernels per stage.
7. Add unit tests and regression harness support.
8. Add profiling labels (NVTX3) around every Phase 7 stage.
9. Add capability checks and clean fallback logging.

### Proposed file layout and module boundaries

Recommended layout:

```text
src/deviceVoF/
  DeviceVoFTypes.H
  DeviceVoFControls.H
  DeviceVoFControls.C
  DeviceMeshView.H
  DevicePatchView.H
  DeviceAdjacencyBuilder.H
  DeviceAdjacencyBuilder.C
  Phase7ScratchArena.H
  Phase7ScratchArena.C
  Phase7KernelFacade.H
  Phase7KernelFacade.C
  AlphaFluxExecutor.H
  AlphaFluxExecutor.C
  MulesLimiterExecutor.H
  MulesLimiterExecutor.C
  InterfaceExecutor.H
  InterfaceExecutor.C
  PatchExecutor.H
  PatchExecutor.C
  FusedUpdateExecutor.H
  FusedUpdateExecutor.C
  Phase7Diagnostics.H
  Phase7Diagnostics.C

src/deviceVoF/cuda/
  CudaLaunchUtils.cuh
  DeviceCommonTypes.cuh
  AlphaFluxKernels.cu
  MulesAtomicKernels.cu
  MulesSegmentedKernels.cu
  InterfaceKernels.cu
  PatchKernels.cu
  FusedUpdateKernels.cu
  DiagnosticsKernels.cu

applications/modules/incompressibleVoF/
  DevicePhase7Hooks.H
  DevicePhase7Hooks.C

tests/deviceVoF/
  testSchemeAudit.C
  testAdjacencyBuilder.C
  testMulesAtomic.C
  testMulesSegmented.C
  testInterfaceCurvature.C
  testPatchKernels.C
  testFusedMixtureUpdate.C

benchmarks/deviceVoF/
  R1_reducedNozzle/
  R2_vofVerification/
  scripts/
```

Boundary rules:

- `src/deviceVoF/` headers expose only POD views and façade APIs.
- `src/deviceVoF/cuda/` contains CUDA-specific code and must not include OpenFOAM runtime-selection classes directly.
- solver modules call only `Phase7KernelFacade`.
- tests may include OpenFOAM objects on the host side, but CUDA kernels see only POD descriptors.

### Pseudocode

#### 1. Prepare path

```cpp
void Phase7KernelFacade::prepare(
    const DeviceVoFControls& controls,
    const DeviceMeshView& mesh,
    const DevicePatchView& patches,
    const DeviceAdjacencyView& adjacency,
    Phase7ScratchView& scratch,
    LaunchContext& ctx)
{
    auditSchemesOrDisableCustomPath(controls);          // fail fast / fallback
    auditPatchKindsOrDisableCustomPath(patches);        // fail fast / fallback
    requireStaticMeshOrFallback(mesh);                  // no moving/topo-changing mesh
    ensureAdjacencyBuilt(mesh, adjacency, ctx.stream);  // owner/neigh/bnd CSR-like lists
    ensureScratchAllocated(mesh, scratch, ctx.stream);  // persistent allocations only
    cacheLaunchConfigs(mesh, patches, controls);
    zeroDiagnostics(scratch, ctx.stream);
}
```

#### 2. Host-side alpha subcycle orchestration

```cpp
void Phase7KernelFacade::runAlphaSubCycle(
    TwoPhaseFieldView& f,
    const DeviceVoFControls& c,
    LaunchContext& ctx)
{
    NVTX_RANGE("Phase7::AlphaSubCycle");

    patchExecutor_.updateAlphaDependentPatches(f, c, ctx);

    alphaFluxExecutor_.assembleAlphaPhi(mesh_, patchView_, f, scratch_, c, ctx);

    mulesExecutor_.limitAndExplicitSolve(
        mesh_,
        patchView_,
        adjacency_,
        f,
        scratch_,
        c,
        ctx);

    fusedUpdateExecutor_.updateTwoPhaseDerivedFields(
        mesh_,
        f,
        c,
        ctx);

    if (ctx.diagnosticsEnabled)
    {
        diagExecutor_.reduceAlphaDiagnostics(mesh_, f, scratch_, ctx);
    }
}
```

#### 3. Host-side interface correction orchestration

```cpp
void Phase7KernelFacade::runInterfaceCorrection(
    InterfaceFieldView& g,
    const DeviceVoFControls& c,
    LaunchContext& ctx)
{
    NVTX_RANGE("Phase7::InterfaceCorrection");

    interfaceExecutor_.computeGradAlpha(mesh_, g, c, ctx);
    interfaceExecutor_.interpolateGradAlpha(mesh_, g, c, ctx);
    patchExecutor_.correctContactAngle(mesh_, patchView_, g, c, ctx);
    interfaceExecutor_.computeNHatf(mesh_, g, c, ctx);
    interfaceExecutor_.computeCurvature(mesh_, adjacency_, g, c, ctx);
    interfaceExecutor_.computeSurfaceTensionForce(mesh_, g, c, ctx);

    if (ctx.diagnosticsEnabled)
    {
        diagExecutor_.reduceInterfaceDiagnostics(mesh_, g, scratch_, ctx);
    }
}
```

#### 4. Alpha flux internal-face kernel (scheme-specialized baseline)

```cpp
__global__ void kAssembleAlphaPhiInternal_UpwindLinearCompression(
    int nInternalFaces,
    const int* __restrict__ owner,
    const int* __restrict__ neigh,
    const double* __restrict__ phi,
    const double* __restrict__ phic,
    const double* __restrict__ nHatf,
    const double* __restrict__ alpha1,
    double* __restrict__ alphaPhi1)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nInternalFaces) return;

    int o = owner[face];
    int n = neigh[face];

    double alpha_o = alpha1[o];
    double alpha_n = alpha1[n];

    // Baseline convective upwind contribution.
    double phi_f = phi[face];
    double alpha_up = (phi_f >= 0.0) ? alpha_o : alpha_n;
    double conv = phi_f * alpha_up;

    // Compression contribution.
    // Exact local v2412 formula must be verified during source audit.
    double phir = phic[face] * nHatf[face];
    double alpha2_o = 1.0 - alpha_o;
    double alpha2_n = 1.0 - alpha_n;
    double alpha2_up = (-phir >= 0.0) ? alpha2_o : alpha2_n;
    double tmpFlux = (-phir) * alpha2_up;
    double alpha1_comp_up = (tmpFlux >= 0.0) ? alpha_o : alpha_n;
    double comp = tmpFlux * alpha1_comp_up;

    alphaPhi1[face] = conv + comp;
}
```

#### 5. MULES limiter: high-level orchestrator

```cpp
void MulesLimiterExecutor::limitAndExplicitSolve(
    const DeviceMeshView& mesh,
    const DevicePatchView& patches,
    const DeviceAdjacencyView& adj,
    TwoPhaseFieldView& f,
    Phase7ScratchView& s,
    const DeviceVoFControls& c,
    LaunchContext& ctx)
{
    NVTX_RANGE("Phase7::MULES");

    // Stage 0: build phiBD/phiCorr and correction source
    launchBuildPhiBD(mesh, patches, f, s, c, ctx);
    launchBuildPhiCorr(mesh, f, s, c, ctx);
    launchBuildSuCorr(mesh, patches, adj, f, s, c, ctx);

    // Stage 1: initialize face lambda = 1
    launchInitLambda(mesh, s, ctx);

    // Stage 2: iterative limiter
    for (int iter = 0; iter < c.limiterIterations; ++iter)
    {
        launchPrepareLimiterCellState(mesh, patches, adj, f, s, c, ctx);
        launchComputeCellLambdaBounds(mesh, f, s, c, ctx);
        launchUpdateFaceLambda(mesh, patches, f, s, c, ctx);

        if (c.limiterIterationMode == LimiterIterationMode::DeviceTol)
        {
            launchUpdateLimiterResidual(mesh, adj, s, c, ctx);
            if (graphSafeDeviceResidualSatisfied(s, c, ctx))
            {
                break; // only valid in non-captured path or later conditional-graph path
            }
        }
    }

    // Stage 3: finalize limited flux and explicit solve
    launchFinalizeLimitedAlphaPhi(mesh, f, s, ctx);
    launchExplicitAlphaUpdate(mesh, patches, f, s, c, ctx);
}
```

#### 6. MULES face-lambda update kernel

```cpp
__global__ void kUpdateFaceLambdaInternal(
    int nInternalFaces,
    const int* __restrict__ owner,
    const int* __restrict__ neigh,
    const double* __restrict__ phiCorr,
    const double* __restrict__ lambdap,
    const double* __restrict__ lambdam,
    double* __restrict__ lambdaFace)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nInternalFaces) return;

    int o = owner[face];
    int n = neigh[face];
    double corr = phiCorr[face];

    double lambda;
    if (corr >= 0.0)
    {
        // owner loses alpha, neighbour gains alpha
        lambda = min(lambdam[o], lambdap[n]);
    }
    else
    {
        // owner gains alpha, neighbour loses alpha
        lambda = min(lambdap[o], lambdam[n]);
    }

    lambdaFace[face] = min(lambdaFace[face], lambda);
}
```

#### 7. Segmented gather kernel skeleton

```cpp
__global__ void kGatherFaceContribToCells(
    int nCells,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ sortedFaceIdx,
    const double* __restrict__ faceContrib,
    double sign,
    double* __restrict__ out)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    double sum = 0.0;
    for (int k = rowOffsets[cell]; k < rowOffsets[cell + 1]; ++k)
    {
        int face = sortedFaceIdx[k];
        sum += faceContrib[face];
    }
    out[cell] += sign * sum;
}
```

#### 8. Explicit alpha update kernel (static mesh path)

```cpp
__global__ void kExplicitAlphaUpdate_StaticMesh(
    int nCells,
    const double* __restrict__ rho,
    const double* __restrict__ rhoOld,
    const double* __restrict__ alphaOld,
    const double* __restrict__ Sp,
    const double* __restrict__ Su,
    const double* __restrict__ divPsiPhi,
    double rDeltaT,
    double* __restrict__ alpha1)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    double numer = rhoOld[cell] * alphaOld[cell] * rDeltaT
                 + Su[cell]
                 - divPsiPhi[cell];

    double denom = rho[cell] * rDeltaT - Sp[cell];

    double a = numer / denom;
    alpha1[cell] = a;
}
```

#### 9. Contact-angle correction kernel skeleton

```cpp
__global__ void kCorrectContactAngle(
    int nFaces,
    const PatchFaceMap patch,
    const FoamVec3* __restrict__ patchNormals,
    const FoamVec3* __restrict__ UBoundary,
    FoamVec3* __restrict__ nHatPatch,
    const FoamVec3* __restrict__ gradAlphafPatch,
    double deltaN,
    ContactAngleParams params,
    double* __restrict__ alphaGradientOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nFaces) return;

    FoamVec3 nf = patchNormals[i];
    FoamVec3 nHatp = nHatPatch[i];

    double theta = evaluateContactAngle(params, UBoundary[i], nHatp); // model-specific
    double a12 = clamp(dot(nHatp, nf), -1.0, 1.0);
    double b1  = cos(theta);
    double b2  = cos(acos(a12) - theta);
    double det = max(1.0 - a12 * a12, 1e-20);

    double a = (b1 - a12 * b2) / det;
    double b = (b2 - a12 * b1) / det;

    FoamVec3 corrected = a * nf + b * nHatp;
    corrected = corrected / (norm(corrected) + deltaN);

    nHatPatch[i] = corrected;
    alphaGradientOut[i] = dot(nf, corrected) * norm(gradAlphafPatch[i]);
}
```

#### 10. Curvature pipeline kernels

```cpp
__global__ void kComputeNHatf(
    int nFaces,
    const FoamVec3* __restrict__ nHatfv,
    const FoamVec3* __restrict__ Sf,
    double* __restrict__ nHatf)
{
    int face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nFaces) return;
    nHatf[face] = dot(nHatfv[face], Sf[face]);
}

__global__ void kComputeSigmaK(
    int nCells,
    const double* __restrict__ K,
    double sigma,
    double* __restrict__ sigmaK)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;
    sigmaK[cell] = sigma * K[cell];
}
```

#### 11. Fused two-phase update kernel

```cpp
__global__ void kFusedUpdateTwoPhaseFields(
    int nCells,
    int nFaces,
    double rho1,
    double rho2,
    const double* __restrict__ alpha1,
    const double* __restrict__ phi,
    const double* __restrict__ alphaPhi1,
    double* __restrict__ alpha2,
    double* __restrict__ rho,
    double* __restrict__ alphaPhi2,
    double* __restrict__ rhoPhi)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nCells)
    {
        double a1 = alpha1[tid];
        double a2 = 1.0 - a1;
        alpha2[tid] = a2;
        rho[tid] = a1 * rho1 + a2 * rho2;
    }

    if (tid < nFaces)
    {
        double aPhi1 = alphaPhi1[tid];
        double aPhi2 = phi[tid] - aPhi1;
        alphaPhi2[tid] = aPhi2;
        rhoPhi[tid] = aPhi1 * rho1 + aPhi2 * rho2;
    }
}
```

### Step-by-step implementation guide

Each step below includes: what to modify, why, expected output, verification, and likely breakages.

#### Step 1 — Audit the exact local v2412 source, consume the frozen support matrix, and freeze the hotspot ranking

- **Modify.** Local engineering notes and a new source-audit artifact. Inspect:
  - local SPUMA/v2412 `alphaPredictor` path,
  - local MULES source files,
  - local `interfaceProperties`,
  - exact boundary classes used by frozen nozzle/reference cases,
  - exact `fvSchemes` entries used for alpha and interface terms.
- **Why.** Public references are not guaranteed identical to the target branch; v13 changed MULES after v2412. [R8][R9]
 - **Expected output.** Checked-in artifacts `docs/phase7_source_audit.md` and `docs/phase7_hotspot_ranking.md` (or one equivalent artifact) recording:
   - local file paths,
   - semantic differences from public references,
   - supported scheme tuple as imported from the frozen support matrix,
   - supported patch classes as imported from the frozen support matrix,
   - ranked hotspot families from interim profiling of the accepted Phase 5/6 path.
- **Verify success.** Senior reviewer signs off that the source audit is complete and that no hidden branch drift remains unresolved.
- **Likely breakages.** Underestimating branch drift; missing a patch subclass that appears only in the nozzle case.

#### Step 2 — Add runtime controls and fallback switches

- **Modify.** New `DeviceVoFControls` parser and dictionary entries.
- **Why.** Every kernel family needs an independent on/off switch for debug isolation.
- **Expected output.** Solver starts with custom kernels disabled by default unless explicitly enabled.
- **Verify success.** Run the existing solver with all custom flags off; results are identical to pre-Step-2 behavior.
- **Likely breakages.** Incorrect default values; controls not propagated into solver hooks.

#### Step 3 — Introduce POD device views and façade skeleton

- **Modify.** Add `DeviceMeshView`, `DevicePatchView`, `DeviceAdjacencyView`, `TwoPhaseFieldView`, `InterfaceFieldView`, `Phase7ScratchView`, and `Phase7KernelFacade`.
- **Why.** CUDA translation units need stable, minimal interfaces independent of OpenFOAM object graphs.
- **Expected output.** Code compiles with empty façade methods and no behavior change.
- **Verify success.** Build passes; no runtime change with custom kernels disabled.
- **Likely breakages.** Incorrect pointer lifetimes; accidental inclusion of heavyweight host headers in CUDA TUs.

#### Step 4 — Build adjacency preprocessing

- **Modify.** Add `DeviceAdjacencyBuilder`.
- **Why.** Segmented/gather backend depends on stable owner/neighbour/boundary row offsets.
- **Expected output.** Per-mesh adjacency arrays created once and cached.
- **Verify success.** Unit test `testAdjacencyBuilder` on a small known mesh:
  - row offsets monotonic,
  - every internal face appears once in owner-sorted and once in neighbour-sorted list,
  - every boundary face appears exactly once in boundary-owner list.
- **Likely breakages.** Off-by-one row offsets; wrong global/local face indexing; coupled patch face mishandling.

#### Step 5 — Implement persistent scratch arena

- **Modify.** `Phase7ScratchArena` / allocation code.
- **Why.** All hot-path temporaries must become stable persistent buffers.
- **Expected output.** Scratch bytes reserved once and logged.
- **Verify success.** Repeated timesteps show no additional device allocations in the Phase 7 path.
- **Likely breakages.** Under-sized scratch; stale pointer reuse after mesh reload.

#### Step 6 — Add CPU mirror/reference kernels for transliteration targets

- **Modify.** Add small CPU reference helpers for the exact local formulas of:
  - `phiBD/phiCorr/SuCorr`,
  - limiter cell-state preparation,
  - face lambda update,
  - explicit alpha update,
  - contact-angle correction, if contact-angle is in the frozen milestone-1 scope.
- **Why.** They allow micro-regression independent of full solver behavior.
- **Expected output.** Test harness can compare CPU scalar reference vs GPU kernels on frozen snapshots.
- **Verify success.** Small-mesh scalar tests pass before any optimized GPU path is enabled.
- **Likely breakages.** Accidentally reusing fallback solver logic rather than the exact formula being ported.

#### Step 7 — Implement atomic alpha-flux assembly kernels

- **Modify.** `AlphaFluxKernels.cu`.
- **Why.** This is the minimal GPU implementation of alpha flux construction.
- **Expected output.** `alphaPhi1` produced fully on device for supported schemes/patches.
- **Verify success.**
  - compare `alphaPhi1` face values against CPU reference on a small snapshot;
  - check internal and boundary face sign conventions;
  - check no host readbacks needed.
- **Likely breakages.** Owner/neighbour orientation mistakes; unsupported scheme accidentally slipping through.

#### Step 8 — Implement atomic `phiBD`, `phiCorr`, `SuCorr` construction

- **Modify.** `MulesAtomicKernels.cu`.
- **Why.** This is the first limiter bring-up step.
- **Expected output.** Device arrays `phiBD`, `phiCorr`, `SuCorr` match CPU reference.
- **Verify success.**
  - small-mesh exact regression vs CPU helper;
  - global sum of `SuCorr` behaves as expected for closed test problems.
- **Likely breakages.** Missing boundary contributions; wrong source-term signs; stale `alphaOld` / `rhoOld`.

#### Step 9 — Implement atomic limiter preprocessing and face-lambda update

- **Modify.** `MulesAtomicKernels.cu`.
- **Why.** Establish correctness before segmentation optimization.
- **Expected output.** `lambdaFace` values are produced and bounded in `[0,1]`.
- **Verify success.**
  - assert `0 <= lambdaFace <= 1 + eps`;
  - compare limited flux against CPU helper on small cases;
  - boundedness check on one subcycle.
- **Likely breakages.** Misinterpreted owner/neighbour limiter logic; race conditions; incorrect extrema handling.

#### Step 10 — Implement explicit alpha update and fused two-phase update

- **Modify.** `MulesAtomicKernels.cu`, `FusedUpdateKernels.cu`.
- **Why.** Close the loop from limited face flux to updated alpha/rho/rhoPhi.
- **Expected output.** Entire alpha subcycle can run on device via the atomic bring-up backend.
- **Verify success.**
  - `alpha1` stays bounded after one subcycle on R2;
  - `alpha2 = 1 - alpha1`;
  - `rho` and `rhoPhi` match CPU reference within tolerance.
- **Likely breakages.** Wrong `divPsiPhi`; forgetting boundary refresh after explicit update; mismatch in `alphaPhi2` convention.

#### Step 11 — Integrate atomic backend with solver and stop for the first benchmark

- **Modify.** `Phase7KernelFacade.C`, solver hooks.
- **Why.** Need a complete though unoptimized path to validate correctness and measure baseline.
- **Expected output.** Solver can run R1/R2 with `limiterBackend=Atomic`.
- **Verify success.**
  - end-to-end run completes for the reduced verification cases;
  - Nsight shows custom alpha path active;
  - no steady-state UVM transfers introduced by the new path.
- **Likely breakages.** Hidden host-side boundary calls; diagnostics readback too frequent; patch state not updated in correct order.

#### Step 12 — Implement interface gradient/interpolation/normal kernels

- **Modify.** `InterfaceKernels.cu`.
- **Why.** Surface tension and curvature remain on the critical path and must stay device-resident.
- **Expected output.** `gradAlpha`, `gradAlphaf`, `nHatfv`, `nHatf` produced on device for supported schemes.
- **Verify success.**
  - compare against CPU reference on a static-interface snapshot;
  - `nHatfv` finite and normalized within tolerance where interface exists.
- **Likely breakages.** Scheme mismatch; incorrect boundary face indexing; normalization NaNs.

#### Step 13 — If required by the frozen acceptance cases, implement contact-angle patch kernels

- **Modify.** `PatchKernels.cu` and/or `InterfaceKernels.cu`.
- **Why.** If contact-angle is in scope, wall wetting is a common source of capillary instability and host fallback; if it is out of scope, this step is limited to explicit placeholders and validation hooks.
- **Expected output.** If contact-angle is in the frozen milestone-1 scope, device-side contact-angle correction for the exact wall models present in the accepted cases; otherwise explicit placeholder hooks and fallback/unsupported-path coverage only.
- **Verify success.**
  - unit test for constant-angle wall;
  - compare patch gradients and normals against CPU reference on a wall-interface snapshot.
- **Likely breakages.** `acos` domain errors; unsupported dynamic model coefficients; transform handling on coupled patches.

#### Step 14 — Implement curvature, sigmaK, and surface-tension force kernels

- **Modify.** `InterfaceKernels.cu`.
- **Why.** Complete the device-resident surface-tension path.
- **Expected output.** `K`, `sigmaK`, and face `surfaceTensionForce` available without host round-trips.
- **Verify success.**
  - static droplet/parasitic-current benchmark;
  - no NaN/Inf;
  - acceptable agreement with CPU reference.
- **Likely breakages.** Divergence sign mistakes; poor handling far from the interface; unsupported non-orthogonal correction assumptions.

#### Step 15 — Implement patch dispatcher and nozzle-specific patch kernels

- **Modify.** `PatchExecutor.*`, `PatchKernels.cu`.
- **Why.** Nozzle BC logic must not remain a CPU tail.
- **Expected output.** Patch kernels exist for all patch classes in R1/R0.
- **Verify success.**
  - patch-only unit tests;
  - reduced nozzle startup run reaches stable first window of timesteps.
- **Likely breakages.** Incorrect mapping of runtime-selected BC to `PatchKernelKind`; stale parameter blocks after restart.

#### Step 16 — Stop and benchmark the full atomic/custom path before optimizing further

- **Modify.** Benchmark scripts only.
- **Why.** Need a measured baseline before segmented optimization and fusion.
- **Expected output.** Benchmark report `benchmarks/deviceVoF/phase7_atomic_baseline.md`.
- **Verify success.** Contains:
  - correctness status,
  - launch counts,
  - top kernels,
  - UVM traffic,
  - time breakdown.
- **Likely breakages.** Skipping this step and losing the ability to attribute later wins.

#### Step 17 — Implement segmented/gather backend for repeated face-to-cell accumulation

- **Modify.** `MulesSegmentedKernels.cu`.
- **Why.** Reduce atomic bottlenecks in the production path.
- **Expected output.** Production backend using adjacency lists for:
  - `SuCorr` accumulation,
  - limiter sum accumulation,
  - divergence/surface integration where beneficial.
- **Verify success.**
  - exact small-case agreement with atomic backend;
  - profiler shows reduced atomic pressure and improved hotspot time.
- **Likely breakages.** Row-offset bugs; duplicate/missing contributions; unexpected floating-point reordering differences.

#### Step 18 — Switch MULES production backend to segmented and keep atomic as fallback

- **Modify.** `MulesLimiterExecutor.C`.
- **Why.** Establish the optimized production path while retaining debug/reference backend.
- **Expected output.** Runtime option `limiterBackend=Segmented` becomes default for production benchmarks.
- **Verify success.**
  - segmented backend passes all atomic-backend tests;
  - no boundedness regression;
  - measurable speedup in limiter stages.
- **Likely breakages.** Hidden dependence on atomic update order; convergence residual mismatch.

#### Step 19 — Add graph-safety cleanup and capture validation

- **Modify.** façade/executors/build flags as needed.
- **Why.** Phase 7 kernels must be ready for Phase 3/8 graph packaging.
- **Expected output.** Capture-safe path with:
  - stable addresses,
  - no allocations,
  - no unsupported helper routines,
  - no host polling in the default fixed-iteration mode.
- **Verify success.**
  - stream-capture smoke test around one alpha subcycle and one interface-correction block;
  - successful graph replay on R1.
- **Likely breakages.** CUB helper inside capture; debug logging on capture stream; implicit host sync.

#### Step 20 — Final profiling and acceptance run

- **Modify.** Benchmark scripts and documentation only.
- **Why.** Need final evidence package.
- **Expected output.** Final report with:
  - correctness tables,
  - performance deltas,
  - Nsight captures,
  - runtime-control matrix,
  - fallback behavior summary.
- **Verify success.** Human reviewer can decide go/no-go without additional reverse engineering.
- **Likely breakages.** Incomplete documentation; no isolation of custom-kernel contribution.

### Instrumentation and profiling hooks

#### Mandatory NVTX3 ranges

Add NVTX3 ranges for at least the following:

- `Phase7::Prepare`
- `Phase7::PatchUpdate`
- `Phase7::AlphaFluxAssembly`
- `Phase7::MULES::BuildPhiBD`
- `Phase7::MULES::BuildSuCorr`
- `Phase7::MULES::Iter`
- `Phase7::MULES::FinalizeFlux`
- `Phase7::MULES::ExplicitSolve`
- `Phase7::FusedUpdate`
- `Phase7::Interface::GradAlpha`
- `Phase7::Interface::InterpolateGrad`
- `Phase7::Interface::ContactAngle`
- `Phase7::Interface::Curvature`
- `Phase7::Interface::SurfaceTensionForce`
- `Phase7::Diagnostics`

#### Mandatory device diagnostics

For debug and validation runs, maintain device-side counters or reductions for:

- count of `alpha < -eps`;
- count of `alpha > 1 + eps`;
- count of NaN/Inf in alpha;
- count of NaN/Inf in `K`;
- min/max `alpha1`;
- max `|K|`;
- max limiter residual per iteration;
- total scratch bytes reserved.

Read these back only at coarse cadence, not every kernel.

#### Nsight Systems requirements

Collect:

- CUDA API timeline;
- kernel launches;
- stream usage;
- graph capture/replay activity if enabled;
- unified-memory traffic and page faults;
- CPU thread activity around patch and diagnostics code.

#### Nsight Compute requirements

Profile only the top 5 kernels by time on R1. Capture at minimum:

- achieved occupancy;
- registers/thread;
- shared-memory usage;
- DRAM throughput;
- L2 hit rate;
- atomic throughput/stalls where relevant;
- warp execution efficiency;
- branch efficiency.

#### Compute Sanitizer requirements

Run `memcheck`, `racecheck`, and `synccheck` on the smallest reproducible mesh/unit tests before scaling up.

### Validation strategy

Validation is staged. Do not jump directly to the full nozzle case.

#### 1. Unit-level kernel validation

Small synthetic meshes:

- 2-cell 1-face line mesh;
- 4-cell diamond / T-junction mesh;
- boundary-only patch strip;
- cyclic/coupled mini-case if such patches are in-scope.

Checks:

- exact face flux values vs CPU helpers;
- exact `SuCorr` accumulation;
- exact `lambdaFace` on simple cases;
- bounded explicit update;
- contact-angle math on one wall face, only if contact-angle is in the frozen milestone-1 scope;
- adjacency offsets and coverage.

#### 2. Snapshot operator regression

Extract one or more frozen field snapshots from the CPU/SPUMA baseline and compare:

- `alphaPhi1`
- `phiBD`
- `phiCorr`
- `lambdaFace`
- updated `alpha1`
- `rho`
- `rhoPhi`
- `gradAlpha`
- `K`
- `surfaceTensionForce`

Norms to record:

- `L_inf`
- relative `L2`
- integrated phase volume error

Suggested thresholds for initial pass/fail:

- `alpha1` boundedness: `-1e-10 <= alpha1 <= 1 + 1e-10`
- no NaN/Inf anywhere
- `lambdaFace` in `[0, 1 + 1e-12]`
- relative `L2` difference of face/cell operator outputs vs CPU helper: `<= 1e-9` on small tests where ordering is identical, `<= 1e-7` where segmented reductions alter summation order
- phase-volume relative error after one subcycle: `<= 1e-9`

These thresholds are recommendations for double-precision baseline runs and may need slight relaxation if the local v2412 implementation or segmented order differs.

#### 3. Verification-case regression (R2)

Required cases:

- interface advection / dam-break style case already defined in Phase 0;
- static-interface or droplet case for curvature/surface tension;
- optional parasitic-current case if capillary effects dominate the nozzle physics;
- one wall-wetting snapshot/component case only if the frozen support matrix requires contact-angle handling.

Checks:

- boundedness over many timesteps;
- phase conservation drift;
- interface position / shape metrics;
- capillary stability / parasitic current magnitude;
- no fallback silently triggered unless configured.

Suggested pass/fail:

- integrated phase-volume drift after 100 timesteps `<= 1e-6` relative;
- no unbounded alpha excursions;
- capillary benchmark macroscopic metrics within 5% of CPU/SPUMA reference.

#### 4. Reduced nozzle regression (R1)

Checks:

- mass flow;
- pressure drop;
- nozzle startup stability;
- swirl-inlet BC correctness;
- air-core proxy / internal interface behavior;
- spray-angle proxy if already part of the reduced case.

Suggested pass/fail:

- key nozzle QoIs remain within the Phase 0 accepted CPU/SPUMA tolerance band;
- no custom-kernel-only instabilities appear in the startup window.

#### 5. Fallback matrix validation

For each kernel family, verify that these combinations run:

- all fallback,
- custom alpha only,
- custom MULES only,
- custom interface only,
- custom patch only,
- all custom.

This is required for debug isolation.

### Performance expectations

Performance expectations are deliberately hierarchical. They are Phase 7 engineering stop/go gates for keeping or dropping custom hotspot families, not the program’s formal performance-acceptance contract; Phase 8 remains the owner of final performance acceptance.

#### Correctness-first expectation

The first fully working custom path may not be faster than the fallback path in every stage, especially with the atomic backend. That is acceptable temporarily.

#### Production expectation

By the end of Phase 7, the **segmented production backend** should satisfy all of the following on R1:

1. no steady-state unified-memory migration attributable to the custom path;
2. no per-kernel `cudaDeviceSynchronize()` in the hot loop;
3. reduced kernel count for the targeted alpha/interface/patch portion relative to the pre-Phase-7 device path;
4. at least one of:
   - MULES stage time improved by **>= 1.3x**,
   - interface-curvature stage improved by **>= 1.2x**,
   - total alpha+interface+patch block improved by **>= 1.15x**.

These thresholds are recommendations, not sourced hardware facts. They are intended to separate meaningful improvement from noise. If a kernel family fails to deliver any measurable performance or residency benefit, keep the fallback path and reconsider whether that family should remain custom.

#### Blackwell-specific expectation

Do not expect Tensor Core-style leaps. The practical wins should come from:

- fewer launches,
- fewer atomics,
- better face/cell locality,
- fewer host/device transitions,
- less temporary churn.

### Common failure modes

1. **Alpha leaves bounds immediately**
   - Likely causes: wrong sign convention, missing boundary face contribution, wrong explicit-update denominator, stale `alphaOld`.
2. **Mass drift grows steadily**
   - Likely causes: divergence kernel missing a face class, incorrect `rhoPhi` derivation, failure to refresh boundary values.
3. **NaNs in curvature or contact-angle patches**
   - Likely causes: `acos` argument outside `[-1,1]`, zero `det`, zero-norm normal vector, missing `deltaN`.
4. **Atomic backend passes, segmented backend fails**
   - Likely causes: adjacency bug, duplicate/missing face in CSR lists, incorrect boundary grouping.
5. **Graph capture fails**
   - Likely causes: hidden allocation, unsupported helper call, debug logging, host synchronization during capture.
6. **Performance is worse even though kernels are correct**
   - Likely causes: too much shared memory, too many registers, uncoalesced patch ordering, diagnostics too frequent, segmented path doing more work than atomic baseline.
7. **Nozzle BCs misbehave but generic tests pass**
   - Likely causes: incorrect translation of runtime patch fields into parameter blocks, time-dependent inlet table not mirrored on device.
8. **UVM traffic reappears**
   - Likely causes: host-side field inspection or patch method still touching managed/host-backed objects.

### Debugging playbook

#### Symptom: boundedness failure on first subcycle

1. Run atomic backend only.
2. Disable custom interface and patch kernels except the minimum needed.
3. Dump `phi`, `alphaPhi1`, `phiBD`, `phiCorr`, `lambdaFace`, `divPsiPhi`, `alpha1`.
4. Compare against CPU mirror on smallest failing mesh.
5. Check owner/neighbour orientation on the first failing face.
6. Verify boundary face cell mapping.

#### Symptom: segmented backend diverges but atomic backend is stable

1. Validate adjacency counts and row offsets.
2. Compare per-cell accumulated sums between atomic and segmented implementations.
3. Ensure each internal face contributes once to owner and once to neighbour, with correct sign.
4. Re-run under `compute-sanitizer`.
5. If needed, temporarily replace one segmented stage at a time with the atomic equivalent.

#### Symptom: curvature kernel produces NaNs or spurious spikes

1. Check `gradAlpha` near the interface.
2. Verify `deltaN` magnitude and nonzero.
3. Clamp `a12` and guard `det`.
4. Disable contact-angle patches and compare interior curvature only.
5. Compare `nHatf` and `K` against CPU reference on a static droplet.

#### Symptom: unexpected UVM traffic

1. Inspect Nsight Systems for HtoD/DtoH/UVM events.
2. Search for host reads of field min/max or patch data in the timestep loop.
3. Check whether any fallback path secretly aliases managed memory.
4. Disable diagnostics and retest.

#### Symptom: graph capture fails

1. Run the same stage outside capture.
2. Remove optional diagnostics and any helper-library path.
3. Verify no allocations/free in the captured region.
4. Verify all pointers remain stable between captures/replays.
5. Reduce to one captured alpha subcycle block before expanding scope.

### What not to do

1. Do not re-derive or simplify MULES equations during bring-up. Transliterate local v2412 first.
2. Do not start by optimizing with mixed precision.
3. Do not introduce custom generic SpMV or AMG in this phase.
4. Do not call host-side patch-field methods per face in the hot path.
5. Do not allow silent fallback on unsupported schemes or patch kinds; log it explicitly.
6. Do not allocate scratch in the timestep loop.
7. Do not enable `--use_fast_math` by default.
8. Do not graph-capture code that still performs host polling per limiter iteration.
9. Do not tune shared memory or block size blindly from A100 guidance; measure on the RTX 5080.
10. Do not remove the atomic backend until the segmented backend is fully validated.

### Rollback / fallback options

Every family of custom kernels must have an independent rollback switch:

- `enableCustomAlphaFlux`
- `enableCustomMules`
- `enableCustomInterface`
- `enableCustomPatchKernels`
- `enableCustomFusedUpdates`
- `limiterBackend = Atomic | Segmented`

Rollback rules:

1. If the segmented backend fails validation, keep the atomic backend for debug and restore fallback path for production.
2. If contact-angle kernels fail and the nozzle cases do not use those models, disable only that patch kind, not the whole interface subsystem.
3. If graph capture fails, continue with stream-ordered launches; graph capture is a readiness requirement, not a reason to ship incorrect kernels.
4. If a custom kernel family is slower and adds no residency benefit, retain the fallback path and mark that family “experimental.”

### Acceptance checklist

A Phase 7 implementation is not accepted until every item below is checked.

#### Correctness

- [ ] Source audit against local SPUMA/v2412 completed and reviewed.
- [ ] Supported schemes and patch kinds explicitly enumerated.
- [ ] Atomic backend passes all unit tests.
- [ ] Segmented backend passes all atomic-backend comparison tests.
- [ ] `alpha1` remains bounded on R2 and R1.
- [ ] `lambdaFace` remains in valid bounds.
- [ ] No NaN/Inf in alpha, curvature, or surface-tension fields.
- [ ] R1/R2 regression metrics pass thresholds.
- [ ] No unsupported feature is silently accepted.

#### Residency / execution

- [ ] No hot-path allocations after prepare.
- [ ] No steady-state UVM traffic from the custom path.
- [ ] No per-kernel `cudaDeviceSynchronize()` in production mode.
- [ ] All heavy fields remain device-resident between timesteps.
- [ ] Graph-capture smoke test passes for at least one alpha subcycle block and one interface-correction block.

#### Performance

- [ ] Benchmarks recorded for all-fallback, atomic, and segmented paths.
- [ ] Launch count reduced for the targeted hot block.
- [ ] At least one targeted hotspot shows measurable speedup.
- [ ] Diagnostics overhead measured and shown acceptable.

#### Maintainability

- [ ] Runtime fallback matrix works.
- [ ] New source files follow module boundaries in this spec.
- [ ] Documentation and benchmark reports are checked in.
- [ ] Coding agent leaves no hidden TODOs in hot-path code.

### Future extensions deferred from this phase

1. Multi-GPU / decomposed meshes and processor patches.
2. Mesh motion / topology changes / AMR.
3. Generalized n-phase MULES `limitSum` production path.
4. Geometric VOF / isoAdvector.
5. Mixed precision experiments.
6. Interface-active-cell compaction to skip bulk cells.
7. Large-shared-memory kernels tuned specifically for cc12.0.
8. Block clusters / distributed shared memory.
9. Device-side conditional graph loops for limiter early exit.
10. Additional patch classes not present in frozen cases.

### Implementation tasks for coding agent

1. Create the Phase 7 device-VOF module, façade, and control parser.
2. Audit and document the exact local v2412 alpha/MULES/interface/patch semantics.
3. Implement adjacency builders and persistent scratch arena.
4. Implement atomic alpha/MULES path and corresponding unit tests.
5. Integrate atomic path into solver and run first correctness benchmark.
6. Implement interface and patch kernels needed by frozen cases.
7. Implement segmented MULES backend and regression tests vs atomic backend.
8. Add fused updates, diagnostics, and NVTX3 ranges.
9. Validate capture safety and produce benchmark/report artifacts.
10. Leave all custom stages individually switchable.

### Do not start until

- the local source audit is complete;
- supported `fvSchemes` are frozen;
- supported patch classes are frozen;
- persistent device field ownership from earlier phases is already stable;
- the fallback path runs cleanly end-to-end on R1 and R2.

### Safe parallelization opportunities

1. `DeviceAdjacencyBuilder` and `Phase7ScratchArena` can be implemented in parallel after POD view definitions exist.
2. Atomic alpha/MULES kernels and interface kernels can be developed in parallel after the source audit is frozen.
3. Patch dispatcher implementation can proceed in parallel with interface kernels once patch kinds and parameter blocks are defined.
4. Diagnostics and benchmark automation can proceed in parallel with kernel development.
5. Segmented backend can begin after atomic correctness path stabilizes; it does not need to wait for all patch kernels.

### Governance guardrails

1. Record the exact SPUMA commit/branch used for the run in `manifest_refs.json`; Phase 7 does not reopen the frozen SPUMA/v2412 family.
2. The supported `fvSchemes` tuple for the frozen nozzle cases is imported from the centralized support envelope and may not be broadened locally.
3. Contact-angle / wall wetting remains out of milestone-1 scope per `support_matrix.md`.
4. QoI tolerance bands for reduced and full nozzle cases are imported from `acceptance_manifest.md`.
5. Moving mesh / topology change remains out of scope for the first production milestone.
6. Unsupported patch classes must follow the package `failFast` default in production mode; any debug fallback remains explicitly bring-up-only.

### Artifacts to produce

 1. `docs/phase7_source_audit.md`
 2. `docs/phase7_hotspot_ranking.md` (or equivalent section merged into the source-audit artifact)
 3. `docs/phase7_controls.md`
 4. New source files under `src/deviceVoF/` and `src/deviceVoF/cuda/`
 5. Unit tests under `tests/deviceVoF/`
 6. Benchmark reports:
    - `phase7_atomic_baseline.md`
    - `phase7_segmented_final.md`
 7. Nsight Systems timeline capture for R1
 8. Nsight Compute reports for top 5 kernels
 9. Validation summary comparing fallback vs custom paths

# 6. Validation and benchmarking framework

This section applies globally to the Phase 7 work.

## Test tiers

### Tier 0 — Build and static checks

- compile with warnings enabled;
- build with debug symbols in non-release profile;
- ensure no CUDA translation unit includes unintended OpenFOAM runtime-selection internals.

### Tier 1 — Unit tests

- adjacency builder;
- small-mesh flux assembly;
- small-mesh MULES atomic path;
- small-mesh MULES segmented path;
- contact-angle patch math, only if contact-angle is in the frozen milestone-1 scope;
- curvature and surface-tension mini-cases;
- fused two-phase update.

### Tier 2 — Snapshot regression

- one-step operator comparisons against CPU mirror functions on frozen field dumps.

### Tier 3 — R2 verification cases

- interface transport;
- static interface / droplet / capillary case;
- optional parasitic-current case.

### Tier 4 — R1 reduced nozzle

- startup window;
- stable operating window;
- QoI comparison.

## Benchmark matrix

Record at least these configurations:

1. all fallback
2. custom alpha only
3. custom alpha + MULES atomic
4. custom alpha + MULES segmented
5. custom interface only
6. custom patch only
7. all custom without graph capture
8. all custom with graph capture smoke test

## Required metrics

- wall time per timestep;
- wall time per alpha subcycle block;
- launch count;
- stream sync count;
- steady-state UVM bytes transferred;
- top-kernel timings;
- boundedness violations;
- QoI deltas vs reference.

## Benchmark stop points

Mandatory stop-and-measure points:

1. after the first full atomic custom path works;
2. after the interface + patch path works;
3. after the segmented backend is integrated;
4. after graph-capture validation.

Do not continue optimization without a benchmark at each stop point.

# 7. Toolchain / environment specification

## Recommended baseline environment

- **OS:** Ubuntu 24.04 LTS on the workstation.
- **Compiler toolchain:** the same host compiler family already validated by the SPUMA build for v2412 on the project workstation.
 - **Toolchain / driver / profiler pins:** consume the master pin manifest frozen for the program rather than restating a looser Phase 7-only minimum here.
 - **Compatibility-floor reference:** CUDA 12.8 introduced Blackwell support and at least 570.26 on Linux for CUDA 12.8 GA compatibility; the active project pin remains the master-pin-manifest lane. [R13]
 - **Primary profiling tools:** Nsight Systems, Nsight Compute, Compute Sanitizer, at the versions pinned in the master pin manifest.
- **Instrumentation:** NVTX3, not deprecated NVTX2-era integration. CUDA 12.8 deprecates NVTX v2. [R12]

## Required CUDA code generation flags

Native Blackwell build must include both:

- `sm_120` cubin
- `compute_120` PTX

Recommended `nvcc` gencode pair:

```text
-gencode arch=compute_120,code=sm_120
-gencode arch=compute_120,code=compute_120
```

Reason:

- **Sourced fact.** CUDA 12.8 adds Blackwell architecture support and NVIDIA’s compatibility guide recommends including PTX for forward compatibility and JIT validation. [R12][R13]
- **Recommendation.** Always ship PTX with the native cubin.

## PTX readiness smoke test

Run at least one smoke test with:

```text
CUDA_FORCE_PTX_JIT=1
```

This validates that the build can JIT from PTX on Blackwell per NVIDIA guidance. [R13]

## Build profiles

### Debug profile

Use for unit tests and sanitizer runs:

- `-O0` or low optimization;
- debug symbols;
- line info;
- diagnostics enabled;
- no fast math.

### Release profile

Use for benchmarks:

- `-O3`
- line info retained for profiling
- no `--use_fast_math` by default
- high warning level
- stable codegen flags for `sm_120` + PTX

## Shared-memory policy

- do not request dynamic shared memory above legacy 48 KB unless the kernel explicitly opts in and profiling justifies it;
- treat **99 KB/block** as the hard practical ceiling on cc12.0. [R14][R15]

## Blackwell feature policy

Allowed in Phase 7 baseline:

- native `sm_120` compilation;
- standard CUDA kernels;
- CUDA Graph capture readiness;
- warp-level primitives where useful.

Deferred:

- Tensor Core/MMA specialization;
- thread-block clusters;
- distributed shared memory;
- device graph launch.

# 8. Module / file / ownership map

## Ownership model

### Solver module owner

Owns:

- `fvMesh`
- OpenFOAM field objects
- timestep sequencing
- runtime dictionaries
- fallback path selection

Must not own:

- CUDA kernel internals
- adjacency builders
- scratch management logic

### Phase 7 device-VOF module owner

Owns:

- POD device views
- adjacency building
- scratch arena
- executors
- CUDA kernels
- diagnostics reductions
- runtime capability checks

Must not own:

- pressure linear-solver selection
- global OpenFOAM runtime selection
- I/O and write-time staging beyond diagnostics

### Validation/benchmark owner

Owns:

- unit tests
- regression scripts
- Nsight capture scripts
- benchmark reports

## File map

- `DeviceVoFTypes.H` — common POD types, enums, lightweight vector wrappers
- `DeviceVoFControls.*` — parse and validate controls dictionary
- `DeviceAdjacencyBuilder.*` — build owner/neigh/boundary CSR-like lists
- `Phase7ScratchArena.*` — persistent scratch allocation and size accounting
- `Phase7KernelFacade.*` — narrow solver-facing orchestration API
- `AlphaFluxExecutor.*` — alpha flux host launch logic
- `MulesLimiterExecutor.*` — limiter backend selection and orchestration
- `InterfaceExecutor.*` — interface/curvature host launch logic
- `PatchExecutor.*` — patch dispatch and kernel launch grouping
- `FusedUpdateExecutor.*` — derived field update launch logic
- `Phase7Diagnostics.*` — reductions and debug extraction
- `cuda/*.cu` — actual CUDA kernels only
- `applications/modules/incompressibleVoF/DevicePhase7Hooks.*` — minimal integration glue
- `tests/deviceVoF/*` — unit and regression tests

# 9. Coding-agent execution roadmap

## Milestone order

### M7.0 — Source audit and scope freeze
Dependency: none beyond earlier completed phases.
Stop here until reviewed.

### M7.1 — Control plane and POD views
Dependency: M7.0.
Parallelizable:
- controls parser,
- POD views,
- façade skeleton.

### M7.2 — Adjacency + scratch infrastructure
Dependency: M7.1.
Parallelizable:
- adjacency builder,
- scratch arena.

### M7.3 — Atomic alpha/MULES correctness path
Dependency: M7.2 and CPU mirror helpers.
Stop and benchmark before continuing.

### M7.4 — Interface and patch kernels
Dependency: M7.2 and source-audited patch/scheme matrix.
Parallelizable:
- interface kernels,
- patch dispatcher,
- patch kernel unit tests.
Stop and benchmark before continuing.

### M7.5 — Segmented production backend
Dependency: M7.3.
Stop and benchmark before continuing.

### M7.6 — Graph-safety cleanup and capture validation
Dependency: M7.4 + M7.5.
Stop and benchmark before continuing.

### M7.7 — Final nozzle regression package
Dependency: all above.

## Dependency graph

- `M7.0 -> M7.1 -> M7.2`
- `M7.2 -> M7.3`
- `M7.2 -> M7.4`
- `M7.3 + M7.4 -> M7.5`
- `M7.4 + M7.5 -> M7.6`
- `M7.6 -> M7.7`

## What should be prototyped before being productized

Prototype first:

- atomic limiter backend;
- fixed-iteration limiter loop;
- separate interface kernels before fusion;
- patch kernels for only the patch kinds present in R1/R0.

Productize later in the same phase:

- segmented/gather limiter backend;
- fused mixture updates;
- capture-safe path.

## What should remain experimental

- mixed precision;
- active interface compaction;
- large-shared-memory kernels;
- CUB inside captured steady-state path until verified;
- cluster/DSM/device graph launch.

## Where to stop and benchmark

Mandatory benchmark stops:

1. after M7.3
2. after M7.4
3. after M7.5
4. after M7.6

Do not continue to the next milestone without capturing:
- correctness status,
- launch counts,
- top kernels,
- UVM behavior.

# 10. Imported Authorities and Residual Governance Notes

1. **Source-line ownership is fixed.**
   - Use the canonical SPUMA/v2412 patch target from `master_pin_manifest.md` and record the exact commit/branch in `manifest_refs.json`.
2. **Scheme support scope is fixed.**
   - Use the exact `fvSchemes` entries already frozen in the centralized support matrix for `R1-core` / `R1` / `R0`; Phase 7 does not reopen them locally.
3. **Wall wetting/contact-angle scope is fixed.**
   - Contact-angle is out of milestone-1 scope per `support_matrix.md`; Phase 7 does not promote related kernels into accepted production scope.
4. **Nozzle BC contract import is fixed.**
   - Preserve the Phase 6 `gpuPressureSwirlInletVelocity` algorithm exactly as frozen in the package; Phase 7 may optimize it, but may not reinterpret the inlet math.
5. **Static-topology scope is fixed.**
   - Mesh motion and topology change are out of scope for the first production milestone; Phase 7 assumes a static mesh throughout.
6. **QoI tolerance ownership is fixed.**
   - Accepted tolerance bands for mass flow, pressure drop, spray angle, and air-core proxy on `R1` / `R0` come from `acceptance_manifest.md`.
7. **Fallback policy is fixed.**
   - Unsupported patch/scheme combinations must follow the package `failFast` default in production mode. Any warning-only fallback remains debug-only.

### Human review checklist

- [ ] SPUMA branch/commit pinned
- [ ] v2412 source audit reviewed
- [ ] supported schemes approved
- [ ] supported patch classes approved
- [ ] nozzle BC math approved
- [ ] validation thresholds approved
- [ ] fallback policy approved

### Coding agent kickoff checklist

- [ ] pull the reviewed SPUMA/v2412 tree
- [ ] generate `phase7_source_audit.md`
- [ ] add `deviceVoFControls`
- [ ] add POD views and façade skeleton
- [ ] build adjacency + scratch infrastructure
- [ ] implement atomic alpha/MULES path first
- [ ] add unit tests before segmented optimization
- [ ] stop and benchmark after each milestone

### Highest-risk implementation assumptions

1. The public Foundation source structure is close enough to local SPUMA/v2412 for a mechanical transliteration after source audit.
2. The frozen nozzle cases use a narrow enough scheme/patch subset to avoid a generic operator rewrite.
3. The segmented/gather backend will materially outperform the atomic backend on the RTX 5080.
4. The RTX 5080’s 16 GB VRAM is sufficient for persistent fields plus Phase 7 scratch with acceptable margin.
5. Graph capture will be viable once allocations and host polling are removed.

### References

- **[R1]** Bnà et al., “SPUMA: a minimally invasive approach to the GPU porting of OPENFOAM,” arXiv HTML, 2025. https://arxiv.org/html/2512.22215v1
- **[R2]** SPUMA GitLab project page, noting version `0.1-v2412` is based on OpenCFD v2412. https://gitlab-hpc.cineca.it/exafoam/spuma
- **[R3]** OpenFOAM Foundation `incompressibleVoF` class reference. https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html
- **[R4]** OpenFOAM Foundation raw/source references for `incompressibleVoF` / two-phase solver internals used during audit (representative public references). https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-dev/master/applications/modules/incompressibleVoF/incompressibleVoF.C
- **[R5]** OpenFOAM Foundation raw `MULESTemplates.C`. https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-dev/master/src/finiteVolume/fvMatrices/solvers/MULES/MULESTemplates.C
- **[R6]** OpenFOAM Foundation raw `MULES.C`, `MULESlimiter.C`, and `MULES.H`/source-guide references. https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-dev/master/src/finiteVolume/fvMatrices/solvers/MULES/MULES.C ; https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-dev/master/src/finiteVolume/fvMatrices/solvers/MULES/MULESlimiter.C ; https://cpp.openfoam.org/v13/MULES_8H.html
- **[R7]** OpenFOAM Foundation `interfaceProperties` source guide reference (algorithmic structure for curvature/contact-angle/surface tension). https://cpp.openfoam.org/v9/interfaceProperties_8C_source.html
- **[R8]** OpenCFD OpenFOAM v2412 release page. https://www.openfoam.com/news/main-news/openfoam-v2412
- **[R9]** OpenFOAM Foundation v13 release notes noting MULES improvements. https://openfoam.org/version/13/
- **[R10]** NVIDIA GeForce RTX 5080 specifications. https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
- **[R11]** NVIDIA CUDA GPU compute capability table. https://developer.nvidia.com/cuda-gpus
- **[R12]** CUDA 12.8 toolkit release notes and features archive. https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/ ; https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html
- **[R13]** NVIDIA Blackwell compatibility guide for CUDA 12.8. https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html
- **[R14]** NVIDIA Blackwell Tuning Guide. https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- **[R15]** CUDA C++ Programming Guide and compute-capabilities appendix. https://docs.nvidia.com/cuda/cuda-c-programming-guide/ ; https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- **[R16]** NVIDIA Nsight Systems User Guide. https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **[R17]** NVIDIA Technical Blog, warp-aggregated atomics. https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
- **[R18]** CUB segmented reduction reference. https://github.com/NVIDIA/cub/blob/main/cub/device/device_segmented_reduce.cuh
- **[R19]** NVIDIA CUDA Graph conditional nodes documentation/blog. https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/ ; https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- **[R20]** CCCL issue discussing graph-capture support/documentation for CUB routines. https://github.com/NVIDIA/cccl/issues/321
- **[R21]** CCCL issue discussing NVTX in CUB and noting current graph-capture considerations. https://github.com/NVIDIA/cccl/issues/1674
