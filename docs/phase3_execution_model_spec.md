
# Phase 3 Execution Model Overhaul — Implementation-Ready Engineering Specification

This document expands **only Phase 3** of the provided plan in full implementation detail. It preserves enough context from the surrounding phases to keep the design coherent, but all other phases remain contextual dependencies, not fully expanded work packages.

---

## 1. Executive overview

Phase 3 exists to replace SPUMA’s current **kernel-by-kernel, synchronization-heavy execution posture** with a **persistent, graph-ready, device-resident runtime model** suitable for the later VOF/MULES/nozzle solver port. This is not a cosmetic runtime refactor. It is the execution substrate that later phases depend on to make transient pressure-swirl nozzle CFD run efficiently on a single NVIDIA RTX 5080 workstation. The immediate target is not “peak kernel efficiency”; it is **eliminating hot-path device-wide synchronization, reducing launch overhead, formalizing stream ownership, and preserving device residency across repeated solver stages**. SPUMA’s published profile for a supported solver shows 53,259 kernel launches and the same number of `cudaDeviceSynchronize()` calls over five iterations, with synchronization dominating CUDA API time. That is precisely the behavior this phase must remove from the nozzle path.

The safest implementation path is **not** to start with a fully dynamic single CUDA graph containing all future PIMPLE, alpha-subcycle, and pressure-corrector control flow. CUDA graphs support that direction in principle through conditional nodes (`IF`/`ELSE`, `WHILE`, `SWITCH`), graph update, and graph upload, but the restrictions are real: capture cannot tolerate synchronizing a captured stream/device/context, graph update depends on a stable topology, and graph memory nodes/device graph launch add extra lifecycle and remapping constraints. The recommended bring-up path is therefore: **(1)** explicit non-legacy stream ownership, **(2)** removal of unconditional device-wide sync in the hot path, **(3)** graph capture for fixed stage sequences on a supported SPUMA solver, **(4)** whole-graph update and graph upload, **(5)** reuse of the same execution substrate for the future VOF/nozzle stages.

The deliverable of Phase 3 is a production-capable **execution substrate**, not the full nozzle numerics. Specifically, the coding agent must build a long-lived `GpuExecutionContext`, explicit stream policy, graph cache/update/rebuild machinery, graph-safe launch wrappers, device-resident launch parameter blocks, write-boundary staging rules, and profiling/instrumentation support. The output must be usable first with a currently supported SPUMA solver such as `pimpleFoam`, because that is the lowest-risk way to validate the execution model before Phase 4 and Phase 5 introduce pressure-backend and VOF complexity. SPUMA’s current public GPU-support notes list `pimpleFoam` among the supported GPU solvers, while multiphase support is explicitly future work in the SPUMA paper. 

Phase 3 also owns the first publication of the program-wide graph stage contract. Canonical stage IDs, graph-safe boundaries, graph-external boundaries, host-controlled loops, and fallback targets must be emitted in a centralized `GraphCaptureSupportMatrix` artifact so Phases 4–7 consume the same taxonomy instead of redefining stage names locally.

The acceptance standard for this phase is strict:

1. No steady-state `cudaDeviceSynchronize()` in the timestep hot path in normal mode.
2. No legacy/default stream use in captured regions.
3. Graph launch count per timestep should scale with **major solver stages**, not with **individual kernels**.
4. Device residency must be preserved between stages and timesteps, with near-zero steady-state UVM page-fault activity in production mode.
5. All fallback paths must remain available: `sync_debug`, `async_no_graph`, and `graph_fixed`.


## 2. Global architecture decisions

### Decision G1 — Use SPUMA/OpenFOAM-v2412 as the runtime base

**Sourced fact.** SPUMA is the actively developed GPU-porting fork in this problem context; its paper benchmarks against OpenFOAM-v2412, and its public GPU support documentation lists supported GPU solvers including `pimpleFoam`.

**Engineering inference.** Re-basing the nozzle work on that code line minimizes uncertainty in the runtime and GPU abstraction layers compared with backporting SPUMA concepts into a different OpenFOAM baseline.

**Recommendation.** Phase 3 shall target the checked-out SPUMA branch selected for the project, with the assumption that OpenFOAM-v2412 semantics are the baseline reference. Do not implement Phase 3 against OpenFOAM 12 and hope to forward-port later.

---

### Decision G2 — Single GPU first, no multi-GPU behavior in Phase 3

**Sourced fact.** SPUMA documents that GPU-aware MPI transfers work best with pure GPU pointers, while managed-memory pointers can trigger implicit host/device transfers and hurt scalability. The RTX 5080 product page also confirms there is no NVLink on this GPU.

**Engineering inference.** Multi-GPU adds communication semantics, graph partitioning complexity, and more residency hazards before the single-GPU execution model is stable.

**Recommendation.** Phase 3 shall assume **one rank, one GPU, no domain decomposition**, and no MPI communication inside captured execution regions.

---

### Decision G3 — Device memory is the production working-set location; managed memory is transitional only

**Sourced fact.** SPUMA’s paper uses unified memory to enable incremental porting, but explicitly states that incomplete ports show up as slowdowns caused by page migrations. Earlier OpenMP/unified-memory work on discrete GPUs spent more than 65% of runtime on migrations. Nsight Systems documents that both CPU and GPU unified-memory page-fault tracing correspond to CPU/GPU access of pages resident on the other side.

**Engineering inference.** A graph-based execution model is not enough if hot fields remain migratory. Launch overhead reduction does not compensate for sustained UVM thrashing.

**Recommendation.** Phase 3 assumes Phase 2 has already moved the persistent working set to device allocations. Managed memory may exist in fallback/debug mode, but the normal Phase 3 acceptance target is **near-zero steady-state UVM page-fault activity**.

---

### Decision G4 — Use host-launched CUDA graphs first; do not use device graph launch in Phase 3

**Sourced fact.** CUDA graphs can reduce repeated workflow overhead, but device graph launch imposes additional constraints: graphs must be instantiated explicitly for device launch; graph structure is fixed at instantiation; only certain node types are allowed; launching the same device graph simultaneously from host and device is undefined; and device graphs must be re-uploaded after update.

**Engineering inference.** Device graph launch adds complexity without solving the first-order Phase 3 problem, which is excessive host-side launch/sync overhead in repeated fixed-stage workflows.

**Recommendation.** Phase 3 shall use **host-launched graphs only**. Device graph launch is explicitly deferred.

---

### Decision G5 — Start with fixed-topology stage graphs and host-controlled loop counts

**Sourced fact.** OpenFOAM’s top-level solver loop is highly repetitive (`preSolve`, repeated PIMPLE outer loop, `prePredictor`, `momentumPredictor`, `thermophysicalPredictor`, `pressureCorrector`, `postCorrector`, write). CUDA graph update works when topology remains identical, and stream capture is convenient when the caller does not own node handles. Conditional nodes can represent loops and branches, but they have body-graph restrictions and should not be treated as free-form control flow.

**Engineering inference.** Stable topology plus repeated stage launch is the highest-probability path to a usable first graph-based implementation. The nozzle solver’s varying counts for PIMPLE correctors, pressure correctors, and alpha subcycles should initially stay under host control.

**Recommendation.** Phase 3 shall implement **host-controlled repeated graph launches** for stage bodies. Conditional graphs are deferred until after fixed-topology graph reuse is working.

---

### Decision G6 — One mandatory compute stream and one optional staging stream

**Sourced fact.** Legacy/default stream use is invalid during capture when other streams in the same context are being captured and are not legacy-safe; cross-stream capture dependencies are supported via events recorded and waited within the capture graph. Nsight Systems can correlate stream/event behavior, but event tracing in graphs has limitations and overhead.

**Engineering inference.** Multiple compute streams complicate ordering, graph composition, and debugging before the solver is graph-stable.

**Recommendation.** Phase 3 shall define:
- `computeStream` — required, non-blocking, all hot-path work and graph launches.
- `stagingStream` — optional, non-blocking, write/output staging only.

No other stream may be introduced in production code without explicit rationale and benchmark evidence.

---

### Decision G7 — Graphs will use preallocated stable-address buffers; graph memory nodes are not part of the baseline

**Sourced fact.** CUDA graph memory nodes can reduce allocation overhead but introduce remapping behavior; changing the launch stream can trigger remap; `cudaGraphUpload()` should be done on the same stream as later launches to avoid additional remapping; graph memory management has its own pool and trimming semantics.

**Engineering inference.** For a CFD solver with large persistent fields and already-planned Phase 2 allocator work, graph-owned allocation is the wrong baseline abstraction.

**Recommendation.** Phase 3 shall use **preallocated, externally owned device buffers** with stable addresses. Graph memory nodes are deferred.

---

### Decision G8 — Preserve portable interfaces, but implement CUDA-first behavior

**Sourced fact.** SPUMA emphasizes a portable programming abstraction that isolates GPU kernels in a small number of dedicated classes, and its repository tree includes both `cudaExecutor` and `hipExecutor` paths under `src/OpenFOAM/device/executor`.

**Engineering inference.** The current project is explicitly Blackwell/CUDA/NVIDIA-focused, but hard-coding execution policy directly into solver logic would make later maintenance worse.

**Recommendation.** Phase 3 interfaces shall remain backend-neutral at the header/API level. The first complete implementation may be CUDA-only behind backend guards, but HIP builds must fail cleanly or degrade to `async_no_graph`, not silently diverge.


## 3. Global assumptions and constraints

1. **Hardware target.** The primary target is a single GeForce RTX 5080 with 10,752 CUDA cores, 16 GB GDDR7, PCIe Gen 5, no NVLink, and CUDA compute capability 12.0. NVIDIA’s CUDA GPU table and the RTX 5080 product page are the source of truth for the consumer 5080 target. CUDA 12.8 adds compiler support for Blackwell architectures including `SM_120`.

2. **Toolkit pin ownership.** Phase 3 consumes the master pin manifest rather than defining its own minima. Default until superseded: CUDA 12.9.1 primary lane, CUDA 13.2 experimental lane, driver `>= 595.45.04`, `sm_120` + PTX, and NVTX3. PTX must be included in the build so that `CUDA_FORCE_PTX_JIT=1` can be used to validate forward compatibility. The CUDA 12.8 references retained elsewhere in this document are background feature-availability rationale for Blackwell/graph support, not the authoritative project pin set.

3. **Blackwell code generation.** CUDA 12.8 release notes add compiler support for `SM_100`, `SM_101`, and `SM_120`. The safest release-build policy is to include both **native cubin for the local target** and **PTX**. For workstation-local builds, `-arch=native` is acceptable only as a convenience, not as the sole release setting.

4. **Runtime baseline.** The Phase 3 execution model must first be validated on a solver that SPUMA already supports on GPU, most likely `pimpleFoam`, before it is used as the substrate for the later nozzle/VOF solver. Multiphase support is not yet part of SPUMA’s published supported scope.

5. **Source-reference caveat.** Some OpenFOAM semantic references in this specification come from OpenFOAM Foundation documentation (for example `foamRun` and `incompressibleVoF`) because they are publicly indexable. Those references are valid for control-flow intent, but exact file paths and member function names must be verified against the checked-out SPUMA/OpenFOAM-v2412 source before editing.

6. **Mesh/topology assumption.** Phase 3 assumes fixed mesh topology during a run. Dynamic mesh motion or topology change can invalidate graph reuse and is explicitly out of scope for the first implementation.

7. **Numerical assumption.** Phase 3 is not permitted to change the solver numerics. Reordered launch structure is allowed only if the dependency graph preserves the original stage ordering and data hazards.

8. **Memory assumption.** Persistent fields, topology arrays, patch maps, and scratch arenas are assumed to be allocated before the timestep hot loop starts. The Phase 3 hot loop must not perform ad hoc heap allocations.

9. **Output assumption.** CPU-side logging and file output still exist. Phase 3 may synchronize at write boundaries or fatal-error/debug boundaries, but not after every kernel or stage.

10. **Profiling assumption.** Nsight Systems is the primary timing/source-of-truth profiler for Phase 3. Nsight Compute is limited to the top few kernels; graph/node launch overhead questions are a Systems concern, not a Compute concern.


## 4. Cross-cutting risks and mitigation

| Risk | Why it matters | Detection | Mitigation |
|---|---|---|---|
| Hidden UVM migrations | Can erase any benefit from graph launch reduction | Nsight Systems UVM trace; CPU/GPU page-fault counters | Device-resident production buffers; pinned host staging only; fail CI if steady-state faults appear in production mode |
| Capture-invalidating API use | `cudaStreamSynchronize`, `cudaDeviceSynchronize`, querying captured streams/devices, or legacy stream usage invalidates capture | Capture error codes; debug asserts; Nsight trace gaps | Graph-safe launch wrappers; no legacy stream; debug-mode assertions on capture state |
| Topology instability | Whole-graph update only works for topologically identical graphs | Graph update result code; rebuild counters | Graph build fingerprint; host-controlled loop counts; reinstantiate on topology change |
| Graph memory remapping | If graph memory nodes are used later, stream changes can cause expensive remapping | First-launch spikes; Nsight trace | Do not use graph memory nodes in baseline; upload/launch on same stream |
| Third-party library graph incompatibility | Some library routines support graph capture, some do not; some still synchronize internally | Capture failures; node-level traces | Keep external library calls outside graphs until individually validated; separate graph-safe and non-graph-safe stage boundaries |
| Excessive stream count | Makes ordering and debugging harder | Nsight stream proliferation | One compute stream + one staging stream only |
| Hidden host touches in solver code | Can trigger DtoH/HtoD migrations or stale-data bugs | UVM trace; CPU backtraces; object-registry logs | Residency registry; prohibit host accessors in hot path; explicit staging rules |
| Temporary allocation churn | OpenFOAM creates/destroys many temporaries; bad for GPUs | API traces; allocator logs | Persistent scratch arena; graph-safe preallocation; no hot-path `new`/`delete` |
| Weak FP64 throughput on RTX 5080 | Some later kernels will remain latency/throughput sensitive | Nsight Compute kernel analysis | Phase 3 does not solve this directly; do not claim otherwise |
| Over-fusion / under-debuggability | Fewer kernels are good, but giant opaque kernels are hard to validate | Large regressions with poor localization | Graphs first, selective fusion later; keep stage boundaries visible in NVTX |
| Output path causing full-step sync | Writing fields each step can hide progress | Timeline shows DtoH and host idle | Write only at output intervals; stage via pinned buffers on staging stream |
| Backend divergence | CUDA-only changes can accidentally break HIP or generic build | CI compile failure | Backend-neutral interfaces; explicit CUDA/unsupported stubs |


## 5. Phase-by-phase implementation specification

## Phase 3 — Execution model overhaul

### Purpose

Replace the current GPU execution posture used by SPUMA in the targeted solver path with a **persistent, explicit, graph-capable runtime model** that:
- keeps major fields resident on the GPU,
- launches work through explicit non-legacy streams,
- removes unconditional hot-path device-wide synchronization,
- provides reusable graph-executable objects for repeated solver stages,
- supports deterministic fallback when graph capture/update is not safe,
- and becomes the execution substrate for later pressure/VOF/nozzle work.

### Why this phase exists

Without this phase, later nozzle-specific GPU development will inherit the wrong runtime behavior:

1. **Launch/synchronization overhead remains structural.** SPUMA’s published profile shows that every kernel invocation is followed by `cudaDeviceSynchronize()` in the profiled path, and those synchronizations dominate CUDA API time. This is acceptable for a development sandbox; it is not acceptable for a transient nozzle solver with repeated PIMPLE loops, pressure correctors, alpha corrections, and subcycling.

2. **Device residency cannot be trusted without a runtime contract.** SPUMA’s incremental unified-memory model is valuable for bring-up, but it intentionally tolerates incomplete ports by allowing page migration slowdowns rather than correctness failures. That is useful early, but the nozzle production path needs the stronger property: hot data stay on device unless an explicit write/stage action moves them.

3. **The solver loop is structurally repetitive enough to amortize graph setup.** OpenFOAM’s solver loop repeats the same stage ordering every timestep and every outer corrector. The VOF path adds explicit alpha subcycling. This makes the workload a strong candidate for graph reuse once loop counts are kept under host control.

4. **Later phases depend on a stable execution substrate.** Pressure backend integration, alpha/MULES porting, curvature/surface-tension kernels, and nozzle boundary-condition compaction all become materially easier if the runtime already provides stable streams, graph policies, and data-residency bookkeeping.

### Entry criteria

Phase 3 must not start until all of the following are true:

1. A CPU-reference branch on the selected SPUMA/OpenFOAM-v2412 base exists and produces acceptable results for the reference cases from Phase 0.
2. SPUMA builds and runs on the RTX 5080 with the chosen CUDA toolkit, and at least one SPUMA-supported GPU solver (preferably `pimpleFoam`) runs correctly on the workstation.
3. Nsight Systems profiling is functional, including NVTX traces and optional UVM page-fault tracing, and the team has at least one baseline `.nsys-rep` showing the pre-Phase-3 launch/sync behavior.
4. The Phase 2 memory posture is available in at least a minimal form: device allocator, pinned host allocator, and a way to register persistent field buffers with stable addresses.
   This must already be the accepted Phase 2 `productionDevice` posture on the reduced supported harness/case; Phase 3 does not reopen Phase 2 residency bring-up.
5. The selected SPUMA branch/commit is frozen for the Phase 3 work window.
6. The project has an agreed fallback mode policy:
   - `sync_debug`
   - `async_no_graph`
   - `graph_fixed`

### Exit criteria

Phase 3 exits only when all of the following are demonstrated on a supported GPU solver and the execution substrate is ready for reuse by later phases:

1. Normal-mode hot path contains **zero steady-state `cudaDeviceSynchronize()` calls**.
2. Normal-mode hot path uses only explicit non-legacy streams.
3. At least one repeated stage sequence is launched as a CUDA graph, instantiated once and reused across multiple timesteps/corrector iterations.
4. The graph path supports either whole-graph update or deterministic rebuild when launch parameters change.
5. Graph upload is used so first-launch mapping cost is removed from timed steady-state measurements.
6. Nsight Systems shows graph-level launches instead of a flood of per-kernel API calls for the graph-enabled stages.
7. UVM CPU/GPU page-fault activity in steady-state production mode is zero or near-zero for the graph-enabled path.
8. Fallback to `async_no_graph` remains functional and numerically equivalent.
9. NVTX3 ranges identify every major execution phase.
10. The coding agent has produced tests, profiling scripts, design logs, and a graph-rebuild audit trail.
11. The centralized `GraphCaptureSupportMatrix` exists and is populated for the Phase 3-supported path, including canonical stage IDs, capture-safe-now status, graph-external boundaries, host-controlled loops, required stable-address sets, forbidden operations, and fallback targets.

### Goals

1. **Correctness goal.** Preserve solver semantics while changing execution mechanics.
2. **Residency goal.** Keep persistent fields, mesh topology, patch maps, and scratch arrays resident on device across timesteps.
3. **Launch-overhead goal.** Replace repeated per-kernel launch/synchronize behavior with graph-level launches for stable stage sequences.
4. **Stream-discipline goal.** Make stream ownership explicit and centralized.
5. **Observability goal.** Make graph build/update/rebuild, sync policy, and host/device transfers visible through logs and NVTX.
6. **Fallback goal.** Provide deterministic degraded modes for debugging and unsupported operations.
7. **Extensibility goal.** Ensure the resulting runtime can host later `AlphaSubcycle`, `PressureAssembly`, `PressureSolve`, `SurfaceTension`, and nozzle-BC stage graphs.

### Non-goals

1. Porting multiphase/VOF numerics in this phase.
2. Integrating AmgX or replacing native linear solvers in this phase.
3. Writing Blackwell-specialized custom kernels in this phase.
4. Implementing multi-GPU execution or CUDA-aware MPI behavior in this phase.
5. Adding dynamic mesh/topology-change support in this phase.
6. Using device graph launch in this phase.
7. Building a fully dynamic single-graph representation of the entire future nozzle solver in this phase.

### First required graph target

The first mandatory graph demonstration for Phase 3 is intentionally narrow:

1. one reduced single-GPU supported-solver path, starting with a SPUMA-supported solver such as `pimpleFoam`;
2. host-controlled outer-corrector, pressure-corrector, and alpha-subcycle loop counts;
3. no external pressure-solver capture and no third-party library capture;
4. graph reuse across repeated timesteps or corrector iterations in that reduced path.

Later phases may widen the covered stage set, but they must consume the same stage IDs and graph support matrix rather than inventing new boundaries locally.

### Technical background

OpenFOAM’s execution model is a repeated host-side time loop with repeated solver-stage calls inside a PIMPLE loop. In public Foundation documentation, `foamRun` calls `preSolve()`, updates time, then iterates `prePredictor()`, `momentumPredictor()`, `thermophysicalPredictor()`, `pressureCorrector()`, and `postCorrector()`, then performs `postSolve()` and write. The exact SPUMA/OpenFOAM-v2412 implementation must be checked locally, but this public control-flow skeleton is sufficient to justify stage-based graph design.

For the future nozzle solver, the key additional structural repetition comes from VOF alpha handling. In the public `alphaPredictor()` reference, `nAlphaSubCycles` is computed from `nAlphaSubCyclesPtr`, `alphaSolve()` is invoked either once or inside an explicit subcycle loop, and `mixture.correct()` follows. MULES itself is documented as an explicit multidimensional limiter for bounded transport. This matters because the eventual execution substrate must handle not just sparse pressure solves but repeated explicit face-based work, limiter updates, and subcycle orchestration.

SPUMA’s paper provides the strongest direct motivation for this phase. It documents an incremental GPU porting model based on portable abstractions, isolation of GPU kernels in a small number of dedicated classes, and unified-memory-backed memory pools. It also documents the current limitation: in the profiled supported-solver path, every kernel invocation is followed by `cudaDeviceSynchronize()`, and synchronization dominates CUDA API time. In the same paper, the authors explicitly note that OpenFOAM’s lazy, allocate-on-demand behavior is inefficient on GPUs and that future work should include loop fusion, caching, and higher-intensity operations. Phase 3 operationalizes those recommendations at the runtime-execution level.

CUDA graphs are the natural runtime mechanism for repeated stage workflows. The CUDA Programming Guide states that by creating a graph that encompasses a workflow launched many times, launch overhead is paid once during instantiation and subsequent launches have very low overhead. Graphs can be built using graph APIs or stream capture; they are instantiated into executable graphs; launch parameters can be updated either by supplying a topologically identical replacement graph (`cudaGraphExecUpdate`) or by updating individual nodes; and optional conditional nodes can encode `IF`, `WHILE`, and `SWITCH` control flow entirely on device. The same documentation also states the critical restrictions: synchronizing/querying a captured stream, event, device, or context is invalid; default/legacy stream use is invalid in the wrong capture context; graph update requires topology stability; and graph memory behavior depends on stream-consistent launch and upload.

### Research findings relevant to this phase

1. **SPUMA’s current hot path is synchronization-heavy.** In the published DrivAer profile, five iterations of `simpleFoam` on one GPU produced 53,259 `cudaLaunchKernel` calls and 53,259 `cudaDeviceSynchronize()` calls, with synchronization accounting for 72.8% of dummy-pool API time and 92.0% of fixed-size-pool API time.

2. **SPUMA intentionally optimizes for incremental correctness before perfect residency.** The paper explicitly states that managed-memory pools let incomplete ports remain correct while page migrations reveal slowdowns that should later be removed.

3. **OpenFOAM’s lazy temporary-allocation model is GPU-hostile.** The SPUMA paper describes the create/destroy behavior of linear systems and temporary quantities as inefficient on GPUs.

4. **The solver workflow is structurally repetitive.** The public `foamRun` control flow and the public `alphaPredictor()` implementation both show repeated host-orchestrated stage loops and explicit subcycles.

5. **CUDA graphs are designed for repeated workflows.** NVIDIA documents low-overhead repeated launch after instantiation.

6. **Stream capture is useful but fragile.** NVIDIA documents that stream capture can wrap existing stream-based code, including library calls, but synchronizing or querying captured streams/devices/contexts is invalid and legacy-stream interactions are restricted.

7. **Graph update has a hard topology contract.** Whole-graph update is intended for a topologically identical graph with changed parameters; individual-node update is better if only a few nodes change and handles are available.

8. **Conditional nodes exist but are restricted.** Conditional body graphs must remain on a single device and may contain only a subset of node types; kernels inside them cannot use CUDA Dynamic Parallelism or Device Graph Launch.

9. **Graph memory nodes complicate performance reasoning.** NVIDIA documents remapping behavior, stream dependence, and the benefit of `cudaGraphUpload()` when upload and launch use the same stream.

10. **Graph capture support in sparse libraries is routine-specific, not universal.** cuSPARSE 12.8 documentation shows that some routines support graph capture while some routines remain blocking or explicitly do not support graph capture. That means “capture the whole pressure backend” cannot be assumed safe.

11. **Blackwell support is current, but target-specific code generation still matters.** CUDA 12.8 adds compiler support for Blackwell architectures including `SM_120`, and NVIDIA’s CUDA GPU table lists GeForce RTX 5080 as compute capability 12.0. PTX inclusion remains recommended for compatibility.

12. **Nsight Systems has graph-aware and UVM-aware tracing modes.** `--cuda-graph-trace=graph` provides lower overhead than node-level tracing, while CPU/GPU unified-memory page-fault tracing can add substantial overhead and should be used selectively. NVTX ranges are projected onto GPU activity in the timeline.

13. **NVTX v3 is mandatory going forward.** CUDA 12.8 deprecates NVTX v2, and later releases remove it. CUPTI notes that NVTX v3 initialization is embedded per binary/shared library, which matters for multi-library CFD executables.

14. **SPUMA and zeptoFOAM both favor a narrow hardware-specific backend.** SPUMA’s paper emphasizes isolation of GPU kernels in few classes; exaFOAM workshop material describes `deviceFieldExecutor`, explicit data management, `devicefvm/devicefvc`, and library-backed linear solves. That supports keeping Phase 3 implementation concentrated in the device/executor layer instead of spreading graph logic across solver numerics.

### Design decisions

#### D3.1 — Introduce an explicit execution-mode ladder

**Sourced fact.** SPUMA currently uses a synchronization-heavy path; CUDA graphs require stricter launch discipline.

**Engineering inference.** The project needs a controlled ladder of behavior for debug, baseline validation, and graph production.

**Recommendation.** Implement these modes as runtime-selectable execution policies:

- `sync_debug`  
  Original semantics plus explicit post-launch/device sync for bug localization only.

- `async_no_graph`  
  Explicit streams, no device-wide sync in hot path, no graph capture; this is the baseline fallback.

- `graph_fixed`  
  Fixed-topology stage graphs with host-controlled loop counts; this is the Phase 3 target.

- `graph_conditional_experimental`  
  Reserved, not part of Phase 3 acceptance.

These execution modes live under `gpuRuntime.execution.mode`. Stage-level fallback naming must compose with this ladder rather than create a parallel vocabulary: `GraphCaptureSupportMatrix` records the fallback target as one of these modes (normally `async_no_graph`) instead of inventing phase-local fallback labels.

**What not to do.** Do not replace the old behavior with a graph-only mode that has no fallback.

---

#### D3.2 — Centralize execution state in a long-lived `GpuExecutionContext`

**Sourced fact.** SPUMA explicitly values isolating hardware-specific behavior in a few classes.

**Engineering inference.** Stream ownership, graph cache state, residency bookkeeping, and debug counters must not be scattered across solver modules.

**Recommendation.** Create a single host-side object, `GpuExecutionContext`, owned by the solver run and valid for its entire lifetime. It shall own:
- streams,
- graph cache and rebuild policy,
- launch parameter blocks,
- residency registry view,
- fallback mode and debug flags,
- NVTX domains,
- write-boundary staging state,
- runtime counters.

---

#### D3.3 — Make stream ownership explicit and minimal

**Sourced fact.** Synchronizing captured streams/devices/contexts is invalid during capture; legacy stream usage is invalid in the wrong capture context. Capture across streams is possible when events are part of the capture graph.

**Engineering inference.** Ad hoc stream creation inside solver code is incompatible with robust graph reuse.

**Recommendation.**
- All hot-path work must launch on `computeStream`.
- Write/output staging may use `stagingStream`.
- Both streams must be created with `cudaStreamNonBlocking`.
- No kernel launcher may default to stream 0 / legacy stream.
- Cross-stream coordination shall use `cudaEventRecord` + `cudaStreamWaitEvent`; device-wide sync is forbidden in hot path.

**What not to do.** Do not allow a kernel launcher API without an explicit stream argument.

---

#### D3.4 — Remove unconditional `cudaDeviceSynchronize()` from normal-mode hot paths

**Sourced fact.** SPUMA’s published profile attributes dominant API time to `cudaDeviceSynchronize()` after every kernel.

**Engineering inference.** Hot-path sync removal is likely the largest immediate Phase 3 win.

**Recommendation.**
- In normal mode, no per-kernel or per-stage `cudaDeviceSynchronize()`.
- Allowed synchronization points:
  1. end-of-write staging,
  2. fatal-error path,
  3. initialization/teardown boundaries,
  4. optional debug assertions in `sync_debug` mode.
- Use `cudaPeekAtLastError()` and optional event fences for debugging instead of device-wide sync.

**Rollback/fallback option.** `sync_debug` may re-enable post-launch sync under a clear runtime switch.

---

#### D3.5 — Use stream-capture-built graphs first, with whole-graph update

**Sourced fact.** Stream capture creates a graph from existing stream-based APIs; whole-graph update is particularly useful when the graph topology is unknown to the caller, including graphs resulting from stream capture of library calls.

**Engineering inference.** Stream capture is the fastest path from today’s stage functions to reusable graph executables. Manual graph construction can be added later if finer-grained node updates become necessary.

**Recommendation.**
- First graph implementation shall use **stream capture** around stage lambdas/functions that already launch kernels through the explicit `computeStream`.
- Parameter changes between launches shall initially use **whole-graph update** (`cudaGraphExecUpdate`) with recapture of a topologically identical temporary graph.
- Manual graph node construction is deferred unless capture becomes a blocker.

Each captured segment must declare its covered canonical stage ID set in `GraphCaptureSupportMatrix`.

**What not to do.** Do not begin by hand-authoring a giant explicit graph for the whole solver when capture-based stage conversion has not yet been proven.

---

#### D3.6 — Use stable device-resident launch-parameter blocks

**Sourced fact.** Whole-graph update is required when parameters change but topology stays fixed; node updates can be cheaper when only a few parameters change.

**Engineering inference.** Rewriting kernel-node params every launch is unnecessary if changing scalar values can be loaded from a stable device block.

**Recommendation.**
- Allocate a small per-graph `DeviceLaunchParams` structure in persistent device memory.
- Mirror it in pinned host memory.
- Update the pinned copy each timestep/loop iteration.
- Push it to the device asynchronously before graph launch or via an explicit graph memcpy node if manual graphs are introduced later.
- Kernels receive a stable pointer to this params block; graph topology and most kernel params remain unchanged.

This reduces graph-update pressure and keeps pointer identity stable.

---

#### D3.7 — Keep variable loop counts on the host in Phase 3

**Sourced fact.** CUDA conditional nodes support `IF`, `WHILE`, and `SWITCH`, but body graphs are restricted and add another correctness surface. The future nozzle solver has variable counts for PIMPLE correctors, pressure correctors, and alpha subcycles.

**Engineering inference.** Using conditional nodes immediately would mix runtime-control innovation with solver-port innovation.

**Recommendation.**
- Host controls:
  - number of outer correctors,
  - number of pressure correctors,
  - number of alpha subcycles,
  - write interval.
- Graphs represent **one instance of a repeated stage body**.

These loop-ownership decisions are part of the centralized stage contract and must be recorded explicitly in `GraphCaptureSupportMatrix`.

- Conditional nodes are reserved for a later optimization pass once the graph/rebuild substrate is validated.

---

#### D3.8 — Treat external linear-solvers and third-party sparse calls as graph-unsafe until proven otherwise

**Sourced fact.** cuSPARSE graph support is routine-specific; some routines explicitly remain blocking or do not support graph capture. Public sources used here do not establish end-to-end AMgX capture safety for this workflow.

**Engineering inference.** Pressure solve integration is a separate risk domain from execution-model cleanup.

**Recommendation.**
- Stage graphs in Phase 3 shall stop at graph-safe boundaries under project control.
- External solver/library calls shall initially execute in `async_no_graph` mode on the same `computeStream`.
- The graph manager shall support a “graph before library / library / graph after library” decomposition.
- Only after individual validation may a third-party library call move inside a captured segment.

Rows for `pressure_solve_native` and `pressure_solve_amgx` must appear in `GraphCaptureSupportMatrix` and default to graph-external until individual capture validation exists.

**What not to do.** Do not capture through an external library because it “probably works.”

---

#### D3.9 — Use graph upload in warm-up and keep upload/launch stream consistent

**Sourced fact.** `cudaGraphUpload()` can separate mapping cost from the first launch, and changing upload/launch streams can trigger remapping costs.

**Engineering inference.** First-launch artifacts would otherwise pollute timestep measurements.

**Recommendation.**
- After instantiation, upload the graph on `computeStream`.
- Always launch that graph on `computeStream`.
- Warm up once before timing.

---

#### D3.10 — Add a graph build fingerprint and deterministic rebuild policy

**Sourced fact.** Major graph-structure changes require re-instantiation. Memory addresses/context ownership also matter for valid update.

**Engineering inference.** Graph reuse without explicit invalidation rules will produce brittle, hard-to-debug behavior.

**Recommendation.** Graph reuse shall be keyed by a `GraphBuildFingerprint` containing, at minimum:
- stage kind,
- mesh topology hash,
- patch-layout hash,
- feature mask,
- execution mode,
- backend kind,
- compile-time debug flags affecting control flow,
- pointer-stability generation counter for all referenced buffers.

A graph must be rebuilt if any of those change.

The fingerprint applies to a declared canonical stage-ID set from `GraphCaptureSupportMatrix`; a cache key may group multiple stage IDs during Phase 3 bring-up, but that grouping must be explicit rather than inferred from local code shape.

---

#### D3.11 — Keep write/output staging outside the hot compute graph

**Sourced fact.** Only a small amount of data should leave the GPU at steady state in a completed port. Nsight UVM and timeline tools make host/device movement visible.

**Engineering inference.** Capturing output behavior into the same stage graph would entangle performance measurement with I/O cadence.

**Recommendation.**
- The main compute graph ends before CPU-side output.
- If output is needed this step, record an event on `computeStream`, then perform explicit DtoH staging on `stagingStream` into pinned host buffers.
- Synchronize only the staging stream before the host file write.

---

#### D3.12 — Fail fast on accidental host access in production mode

**Sourced fact.** SPUMA’s incremental managed-memory model tolerates incomplete ports by letting migrations happen; that is useful for bring-up but unsafe as a silent production behavior.

**Engineering inference.** Silent host access to device-resident working sets will otherwise leak into production.

**Recommendation.**
- `async_no_graph` and `graph_fixed` production modes shall treat access to marked device-only fields from CPU hot-path code as a logged error or assertion failure, not as a silent migration.
- `sync_debug` may permit more permissive behavior for diagnosis.

##### Explicitly prohibited implementation patterns

The coding agent shall not do any of the following in Phase 3:

1. Reinsert `cudaDeviceSynchronize()` after every kernel as an “easy correctness fix.”
2. Launch kernels on the legacy/default stream from graph-enabled paths.
3. Allocate device buffers inside the timestep hot path.
4. Copy full fields to the CPU every timestep for logging or convergence checks.
5. Use graph memory nodes as a substitute for proper persistent allocations.
6. Capture external library calls without documented validation.
7. Build one graph per kernel.
8. Allow hidden stream creation inside solver utilities.

### Alternatives considered

#### Alternative A — Keep the current SPUMA launch model and optimize kernels later
Rejected. The published profile shows synchronization dominating API time. Launch mechanics are a first-order bottleneck, especially for smaller repeated transient workloads.

#### Alternative B — Manual graph construction for the entire solver from day one
Rejected for Phase 3. It is theoretically clean but implementation-heavy, especially before the graph-safe stage boundaries and stream discipline exist. Stream capture is the lower-risk bridge from current code to reusable graphs.

#### Alternative C — One monolithic conditional graph including all future PIMPLE/MULES logic
Rejected for Phase 3 bring-up. CUDA conditional nodes are real and useful, but the body-graph restrictions and added debug surface are not justified before fixed-stage graphs are working.

#### Alternative D — Device graph launch
Rejected. Additional restrictions, upload/update complexity, and undefined simultaneous host/device launch behavior do not help solve the immediate problem.

#### Alternative E — Keep everything under unified memory and rely on page migration heuristics
Rejected for production. Unified memory is acceptable as an incremental bridge, not as the accepted steady-state runtime for this nozzle target on discrete Blackwell hardware.

### Interfaces and dependencies

The following interfaces are required. Names are proposed and should be adapted to local SPUMA style only if the responsibility split remains unchanged.

#### Canonical stage taxonomy and graph-support ownership

Graph-safe in this program means no hot-path allocation, no host object traversal, no legacy/default stream usage, no host sync, and stable pointer identity across replays.

Stage IDs and capture support are owned by the centralized `GraphCaptureSupportMatrix`; this phase implements them. `GraphKind` below is an execution/cache grouping key, not a replacement for the canonical stage taxonomy. If one `GraphKind` covers several stage IDs during bring-up, that grouping must be declared explicitly in the matrix rather than inferred from the local implementation.

Minimum canonical stage IDs that must persist across Phases 3–7:

| Stage ID | Meaning / boundary | Phase 3 status |
|---|---|---|
| `warmup` | one-time warm-up and optional graph upload | implemented |
| `pre_solve` | graph-safe pre-solve preparation under project control | implemented target |
| `outer_iter_body` | composite reusable body for one host-controlled outer corrector | implemented target |
| `momentum_predictor` | momentum-predictor body under project control | may be grouped inside `outer_iter_body` in Phase 3 |
| `pressure_assembly` | assembly/fix-up work before the selected pressure backend | required boundary |
| `pressure_solve_native` | native SPUMA/OpenFOAM pressure solve | graph-external unless individually validated |
| `pressure_solve_amgx` | AmgX pressure solve | graph-external / uncaptured in Phase 3 baseline |
| `pressure_post` | post-solve correction / fix-up work | required boundary |
| `alpha_pre` | alpha pre-stage | forward-declared for Phase 5 |
| `alpha_subcycle_body` | one host-controlled alpha subcycle body | forward-declared for Phase 5 |
| `mixture_update` | post-alpha mixture update | forward-declared for Phase 5 |
| `nozzle_bc_update` | nozzle-specific BC update stage | forward-declared for Phase 6 |
| `write_stage` | explicit write/output staging boundary | implemented boundary; outside timed compute graph |

The `GraphCaptureSupportMatrix` artifact shall at minimum record, for each stage ID: `capture_safe_now`, `intended_phase`, `loop_owner`, `required_stable_addresses`, `forbidden_operations`, `graph_external_dependencies`, and `fallback_mode`.

#### Core enums

```cpp
namespace Foam::device::execution
{
    enum class ExecutionMode : uint8_t
    {
        SyncDebug,
        AsyncNoGraph,
        GraphFixed,
        GraphConditionalExperimental
    };

    enum class GraphKind : uint8_t
    {
        Warmup,
        PreSolve,
        PimpleOuterIter,
        PressureAssembly,
        PressurePost,
        AlphaSubcycle,
        WriteStage
    };

    enum class GraphRebuildReason : uint8_t
    {
        None,
        FirstUse,
        TopologyHashChanged,
        PatchLayoutChanged,
        FeatureMaskChanged,
        PointerGenerationChanged,
        ExecutionModeChanged,
        CaptureInvalidated,
        ExternalLibraryUnsupported,
        DebugFlagChanged,
        UserForced
    };

    enum class ResidencyClass : uint8_t
    {
        DevicePersistent,
        DeviceScratch,
        HostPinnedStage,
        ManagedDebugOnly
    };
}
```

#### `GpuExecutionConfig`

Responsibility: immutable or rarely changed runtime policy values.
The authoritative configuration lives under `gpuRuntime.execution` (with profiling-related knobs surfaced under `gpuRuntime.profiling` as needed). Legacy environment variables or per-phase control names are compatibility shims only.

Required fields:
- `ExecutionMode mode`
- `bool enableNvtx`
- `bool enableGraphUpload`
- `bool enableWholeGraphUpdate`
- `bool allowExternalLibraryCapture` (default false)
- `bool failOnHostTouch`
- `bool useStagingStream`
- `bool useGraphWarmup`
- `bool logGraphRebuilds`
- `bool debugCheckCaptureState`
- `bool debugForcePostLaunchSync`
- `unsigned maxGraphRebuildsPerRun`
- `unsigned warmupLaunchCount`

#### `ExecutionStreams`

Responsibility: create/destroy and expose the only streams allowed in production paths.

Required API:
```cpp
class ExecutionStreams
{
public:
    bool create();
    void destroy() noexcept;

    cudaStream_t compute() const noexcept;
    cudaStream_t staging() const noexcept;
    bool hasStaging() const noexcept;
};
```

Behavior:
- `computeStream` always exists.
- `stagingStream` exists only if configured.
- both created with `cudaStreamNonBlocking`.

#### `DeviceResidencyRegistry`

Responsibility: runtime registry of all buffers used by graph-enabled stages.

Required record:
```cpp
struct ResidencyRecord
{
    std::string name;
    void* devicePtr;
    size_t bytes;
    ResidencyClass residencyClass;
    uint64_t pointerGeneration;
    bool graphReferenced;
    bool hostTouchObserved;
};
```

Required API:
```cpp
class DeviceResidencyRegistry
{
public:
    void registerBuffer(std::string_view name, void* ptr, size_t bytes,
                        ResidencyClass cls, uint64_t generation);
    const ResidencyRecord& lookup(std::string_view name) const;
    void markGraphReferenced(std::string_view name);
    void markHostTouch(std::string_view name);
    void assertProductionSafe() const;
    uint64_t combinedGenerationHash(std::span<const std::string_view> names) const;
};
```

Dependency: Phase 2 allocator or device-buffer owner.

#### `DeviceLaunchParams`

Responsibility: stable per-graph device-side parameter block.

Required fields (minimum):
```cpp
struct DeviceLaunchParams
{
    double deltaT;
    double timeValue;
    int timeIndex;
    int outerCorrIndex;
    int pressureCorrIndex;
    int alphaSubcycleIndex;
    int nOuterCorr;
    int nPressureCorr;
    int nAlphaSubcycles;
    int writeNow;
    uint32_t featureMask;
};
```

Allocation:
- one persistent device allocation per graph kind or per stage family,
- one pinned mirror on host.

#### `GraphBuildFingerprint`

Responsibility: determine whether an executable graph may be reused.

Required fields:
```cpp
struct GraphBuildFingerprint
{
    GraphKind kind;
    uint64_t meshTopologyHash;
    uint64_t patchLayoutHash;
    uint64_t pointerGenerationHash;
    uint32_t featureMask;
    ExecutionMode mode;
    uint16_t backendKind;
    uint16_t reserved;
};
```

#### `GraphTemplate`

Responsibility: temporary/captured `cudaGraph_t` and metadata.

```cpp
struct GraphTemplate
{
    cudaGraph_t graph = nullptr;
    GraphBuildFingerprint fingerprint{};
    std::string debugName;
};
```

#### `GraphExecHandle`

Responsibility: persistent `cudaGraphExec_t` plus reuse/update state.

```cpp
struct GraphExecHandle
{
    cudaGraphExec_t exec = nullptr;
    GraphBuildFingerprint fingerprint{};
    GraphKind kind{};
    bool uploaded = false;
    uint64_t launchCount = 0;
    uint64_t updateCount = 0;
    uint64_t rebuildCount = 0;
    GraphRebuildReason lastRebuildReason = GraphRebuildReason::None;
    std::string debugName;
};
```

#### `GpuGraphManager`

Responsibility: graph capture, instantiate, upload, update, rebuild, launch.

Required API:
```cpp
class GpuGraphManager
{
public:
    GraphExecHandle& getOrCreate(
        GraphKind kind,
        const GraphBuildFingerprint& fp,
        std::function<void(cudaStream_t)> captureFn,
        cudaStream_t stream);

    bool tryUpdate(
        GraphExecHandle& handle,
        const GraphBuildFingerprint& fp,
        std::function<void(cudaStream_t)> captureFn,
        cudaStream_t stream);

    GraphExecHandle& rebuild(
        GraphKind kind,
        GraphRebuildReason why,
        const GraphBuildFingerprint& fp,
        std::function<void(cudaStream_t)> captureFn,
        cudaStream_t stream);

    void launch(GraphExecHandle& handle, cudaStream_t stream);
    void invalidate(GraphKind kind, GraphRebuildReason why);
    void destroyAll() noexcept;
};
```

Rules:
- build via stream capture in Phase 3 baseline;
- upload after instantiate if configured;
- if `tryUpdate()` fails, rebuild once;
- if rebuild fails, downgrade stage to `async_no_graph` and log the reason.

#### `KernelLaunchPolicy`

Responsibility: graph-safe kernel launch wrapper.

Required API:
```cpp
class KernelLaunchPolicy
{
public:
    template<class Kernel, class... Args>
    static void launchAsync(
        const char* name,
        dim3 grid,
        dim3 block,
        size_t shmem,
        cudaStream_t stream,
        Kernel kernel,
        Args... args);

    static void checkPostLaunch(const char* name, ExecutionMode mode);
};
```

Rules:
- `launchAsync()` never calls `cudaDeviceSynchronize()` in normal modes;
- `checkPostLaunch()` may call `cudaPeekAtLastError()` always and `cudaDeviceSynchronize()` only in `SyncDebug`.

#### `GpuExecutionContext`

Responsibility: top-level execution object.

Required API:
```cpp
class GpuExecutionContext
{
public:
    bool initialize(const GpuExecutionConfig& cfg,
                    DeviceResidencyRegistry* registry);
    void shutdown() noexcept;

    ExecutionStreams& streams() noexcept;
    DeviceResidencyRegistry& residency() noexcept;
    GpuGraphManager& graphs() noexcept;

    DeviceLaunchParams* deviceParams(GraphKind kind) noexcept;
    DeviceLaunchParams& hostPinnedParams(GraphKind kind) noexcept;

    const GpuExecutionConfig& config() const noexcept;
};
```

#### `HostStageController`

Responsibility: host-side orchestration of stage graphs and fallback mode.

Required API:
```cpp
class HostStageController
{
public:
    void beginTimeStep(const SolverProxy&, GpuExecutionContext&, const TimeStepMeta&);
    void runPreSolve(SolverProxy&, GpuExecutionContext&);
    void runPimpleOuterIteration(SolverProxy&, GpuExecutionContext&, int outerIter);
    void runPressureAssemblyOrFallback(SolverProxy&, GpuExecutionContext&, int pressureIter);
    void runAlphaSubcycleOrPlaceholder(SolverProxy&, GpuExecutionContext&, int alphaSubIter);
    void finishTimeStep(SolverProxy&, GpuExecutionContext&, const TimeStepMeta&);
};
```

#### External dependencies

- Phase 2 allocator/pool/device-buffer owner
- SPUMA CUDA executor layer (`src/OpenFOAM/device/executor`)
- SPUMA memory executor layer (`src/OpenFOAM/device/memoryExecutor`)
- NVTX3 headers
- Nsight Systems / Nsight Compute / Compute Sanitizer scripts
- Solver-stage adapter hooks in the selected supported solver (`pimpleFoam` first)

### Data model / memory model

#### Ownership model

**Host long-lived objects**
- `GpuExecutionContext`
- `GpuGraphManager`
- `ExecutionStreams`
- `DeviceResidencyRegistry`
- `HostStageController`
- graph rebuild logs / counters
- pinned mirrors of `DeviceLaunchParams`
- pinned output staging buffers

**Device long-lived objects**
- persistent field arrays from Phase 2
- mesh topology arrays
- patch-compaction maps
- scratch arena allocations
- `DeviceLaunchParams` blocks
- optional device diagnostics counters

**Ephemeral objects allowed in hot path**
- none on the heap
- stack-only small host temporaries
- temporary captured `cudaGraph_t` during update/rebuild (host-side graph object only)

#### Required invariants

1. Every buffer referenced by a graph must have a **stable address** across launches of that graph executable.
2. Every graph-enabled stage must consume only buffers registered in `DeviceResidencyRegistry`.
3. The hot loop may not allocate or free persistent solver buffers.
4. Data may leave the GPU only for:
   - scalar logging,
   - write/output staging at output intervals,
   - explicit debug diagnostics.
5. No CPU accessor may touch device-only field storage in production modes.
6. All graph uploads and graph launches use the same `computeStream`.

#### Production memory classes

| Class | Backing allocation | Lifecycle | Allowed in graph-enabled stages | Notes |
|---|---|---|---|---|
| `DevicePersistent` | device memory | full solver run | yes | fields, topology, patch maps, params blocks |
| `DeviceScratch` | device memory | full solver run | yes | scratch arena, zeroed/reused each stage |
| `HostPinnedStage` | pinned host memory | full solver run | no direct graph use except explicit copies | write staging, param mirrors |
| `ManagedDebugOnly` | managed memory | bring-up/debug only | not accepted in production mode | use only to bridge incomplete ports |

#### Transfer policy

**Allowed HtoD in steady state**
- small async copy of `DeviceLaunchParams` mirror to device block before stage launch, if values changed.

**Allowed DtoH in steady state**
- scalar residual/diagnostic reductions explicitly requested by the host,
- staged field output at write intervals only.

**Forbidden in steady state**
- full-field DtoH for per-iteration logging,
- implicit UVM-driven migration of registered device-persistent fields,
- any DtoH inside the graph-enabled compute hot path.

#### Synchronization policy

**Allowed**
- `cudaStreamWaitEvent()`
- `cudaEventRecord()`
- `cudaStreamSynchronize(stagingStream)` at write boundary
- `cudaDeviceSynchronize()` in `sync_debug`, fatal-error handling, shutdown

**Forbidden in normal hot path**
- `cudaDeviceSynchronize()`
- `cudaStreamSynchronize(computeStream)` between graph-enabled stages unless the host truly needs a result
- querying execution status of captured streams/events/devices/contexts during capture

#### Graph partitioning model

Phase 3 shall use **small number of reusable stage graphs**, not one monolithic full-solver graph.
These graph executables are implementation groupings. Their coverage of canonical stage IDs must be declared in `GraphCaptureSupportMatrix`, including which rows remain graph-external and which loops remain host-controlled.

Minimum graph kinds to implement or stub:

1. `Warmup`  
   One-time kernel warm-up and graph upload.

2. `PreSolve`  
   Graph-safe pre-solve preparation, scratch resets, pre-predictor steps under project control.

3. `PimpleOuterIter`  
   One outer-corrector body for the supported solver path, excluding graph-unsafe library calls.

4. `PressureAssembly`  
   Reserved boundary between assembly and solver backend; may initially remain fallback-only.

5. `AlphaSubcycle`  
   Placeholder in Phase 3; actual use begins in Phase 5.

6. `WriteStage`  
   Optional separate graph or explicit stream sequence for write staging only; not part of timed compute graph.

### Algorithms and control flow

#### Bring-up path versus production path

**Bring-up path**
- Start from a supported solver (`pimpleFoam`).
- Route all kernels through `KernelLaunchPolicy`.
- Remove post-kernel sync in normal mode.
- Create one capture-built graph for a fixed stage sequence on `computeStream`.
- Launch the graph repeatedly.
- Validate numerical equivalence against `async_no_graph`.

**Production Phase 3 path**
- Add graph cache and fingerprinting.
- Add whole-graph update or deterministic rebuild.
- Upload graphs during warm-up.
- Add write-boundary staging rules.
- Add device-residency assertions.
- Keep host-controlled loop counts.

#### High-level timestep control flow

Recommended steady-state orchestration:

```text
beginTimeStep()
    update pinned host params
    async HtoD params copy (or graph node in later manual-graph path)

runPreSolve()
    graph or async fallback

for outerCorr in [0, nOuterCorr):
    update params.outerCorrIndex
    run outer-iteration graph or async fallback

    if pressure backend not graph-safe:
        run pressure-assembly graph
        call external/native pressure solve on computeStream in async mode
        run pressure-post graph
    else:
        run pressure graph

finishTimeStep()
    if writeNow:
        record event on computeStream
        wait on stagingStream
        enqueue DtoH staging copies into pinned buffers
        synchronize stagingStream
        perform host write
```

#### Graph build/update algorithm

1. Compute `GraphBuildFingerprint`.
2. Look up existing `GraphExecHandle` by `(GraphKind, fingerprint core key)`.
3. If no handle exists, capture → instantiate → upload → cache.
4. If handle exists and pointer-generation/feature mask unchanged, reuse directly.
5. If handle exists and topology is expected to be unchanged but params changed:
   - capture a temporary topologically identical graph,
   - call `cudaGraphExecUpdate`,
   - if update succeeds, destroy temporary graph and reuse exec,
   - if update fails, rebuild once.
6. If rebuild fails, downgrade the stage to `async_no_graph` and record the reason.
7. Never silently fall back to `sync_debug`.

#### Graph-safe stage composition rules

A stage may be graph-enabled in Phase 3 only if its canonical stage ID is marked `capture_safe_now` in `GraphCaptureSupportMatrix`.
A stage may be graph-enabled only if all operations inside it satisfy all of the following:
- launches into explicit non-legacy streams only,
- no device-wide synchronization,
- no forbidden capture operations,
- no host callback needed for correctness,
- no ad hoc allocator calls in hot path,
- all referenced buffers registered and stable,
- external library calls either absent or validated for capture.

Stages failing any rule must stay in `async_no_graph` until fixed.

#### Graph rebuild triggers

A graph executable must be invalidated and rebuilt if any of the following occur:
1. mesh topology hash changes,
2. patch-layout hash changes,
3. any registered referenced buffer changes address/generation,
4. feature mask changes control-flow shape,
5. execution mode changes,
6. debug flags affecting capture behavior change,
7. capture returns an error,
8. a library call inside the stage is detected to be graph-unsafe,
9. user forces rebuild through debug option.

#### Fallback/rollback behavior

Phase 3 requires deterministic fallback:

- **Stage-level fallback:** if a specific graph kind fails capture/update/rebuild, only that stage falls back to `async_no_graph`.
- **Run-level fallback:** if more than `maxGraphRebuildsPerRun` occur, downgrade the entire run to `async_no_graph` and emit a fatal warning.
- **Debug rollback:** `sync_debug` may be enabled manually to localize race or ordering issues.

The coding agent shall implement these transitions explicitly and log them.

### Required source changes

The exact file names must be verified against the checked-out SPUMA tree, but the following changes are required.

#### 1. Device executor layer

Existing repository search results show CUDA and HIP executor code under `src/OpenFOAM/device/executor`, including `cudaExecutor.cu`, `cudaExecutor.cuh`, and `hipExecutor.hpp`. Extend this layer rather than scattering runtime policy into solver code.

Required changes:
- add stream-explicit async launch wrappers,
- add execution-mode-aware post-launch error handling,
- forbid implicit legacy-stream launches,
- add graph capture helpers and graph cache hooks.

#### 2. Device memory executor layer

Repository search results show `src/OpenFOAM/device/memoryExecutor/cudaMemoryExecutor.cu`. This layer shall expose enough allocator/pointer-generation information for graph fingerprinting and residency registration.

Required changes:
- surface buffer generation/address metadata,
- expose pinned-host staging allocation helpers,
- remove hidden syncs in memory fast paths where feasible.

#### 3. Core device headers

Search results show `src/OpenFOAM/device/deviceM.H` and `deviceUtils.H`. These shall carry:
- compile-time/runtime sync policy macros or inline helpers,
- backend guards,
- capture-state/debug assertions if consistent with local style.

#### 4. New execution-substrate files

Add new files for:
- execution context,
- streams,
- graph cache,
- graph types/fingerprints,
- device launch params,
- NVTX domains and tracing,
- graph capture utilities.

#### 5. Supported-solver adapter hooks

Patch one supported solver path first (recommended: `pimpleFoam`) so that solver-stage calls route through `HostStageController`. The adapter must be thin:
- stage entry/exit,
- update of launch params,
- graph/fallback decision,
- no numerics changes.

#### 6. Build-system changes

Required:
- compile new CUDA translation units,
- link NVTX3,
- add compile flags for native RTX 5080 target plus PTX,
- add debug build variant with `-lineinfo` and sanitizer-friendly settings.

#### 7. Profiling/test scripts

Add scripts:
- Nsight Systems collection (graph-level and node-level variants),
- Nsight Compute for top kernels,
- Compute Sanitizer runs,
- graph-mode smoke test harness.

### Proposed file layout and module boundaries

The proposed layout below is designed to fit the existing SPUMA repository shape, which publicly exposes `src/OpenFOAM/device/executor` and `src/OpenFOAM/device/memoryExecutor`. Adjust naming to local conventions only if the ownership boundaries remain unchanged.

```text
src/
  OpenFOAM/
    device/
      deviceM.H                         # existing; sync-policy hooks
      deviceUtils.H                     # existing; stream/capture assertions if appropriate

      executor/
        executor.H                      # existing; extend API surface only if needed
        cudaExecutor.cuh                # existing; add graph-safe async wrappers
        cudaExecutor.cu                 # existing; add implementation hooks
        hipExecutor.hpp                 # existing; compile-safe no-graph backend hook

        ExecutionMode.H                 # new
        GraphTypes.H                    # new: GraphKind, fingerprint, rebuild reason
        ExecutionStreams.H              # new
        ExecutionStreams.C              # new
        DeviceLaunchParams.H            # new
        GpuExecutionConfig.H            # new
        GpuExecutionContext.H           # new
        GpuExecutionContext.C           # new
        DeviceResidencyRegistry.H       # new
        DeviceResidencyRegistry.C       # new
        GpuGraphManager.H               # new
        GpuGraphManager.C               # new
        StreamCaptureUtils.H            # new
        StreamCaptureUtils.C            # new
        KernelLaunchPolicy.H            # new
        GraphDiagnostics.H              # new
        GraphDiagnostics.C              # new
        cudaGraphSupport.cuh            # new
        cudaGraphSupport.cu             # new

      memoryExecutor/
        cudaMemoryExecutor.cu           # existing; allocator metadata export
        cudaMemoryExecutor.H            # existing/new companion if needed

      profiling/
        NvtxDomains.H                   # new
        NvtxDomains.C                   # new
        ExecutionTrace.H                # new
        ExecutionTrace.C                # new

applications/
  solvers/
    ... verify exact SPUMA/v2412 path ...
      pimpleFoam/
        GpuStageHooks.H                 # new thin adapter
        GpuStageHooks.C                 # new thin adapter
        (patch existing solver main/module entry to use HostStageController)

test/
  gpuExecution/
    testExecutionStreams.C             # new
    testGraphCapture.C                 # new
    testGraphUpdate.C                  # new
    testFallbackModes.C                # new
    testResidencyRegistry.C            # new

scripts/
  profile_phase3_graph.sh             # new
  profile_phase3_node.sh              # new
  profile_phase3_uvm.sh               # new
  run_compute_sanitizer_phase3.sh     # new
```

### Pseudocode

#### Core data structures

```cpp
struct TimeStepMeta
{
    double deltaT;
    double timeValue;
    int timeIndex;
    int nOuterCorr;
    int nPressureCorr;
    int nAlphaSubcycles;
    bool writeNow;
};

struct GraphBuildInputs
{
    GraphKind kind;
    TimeStepMeta ts;
    uint32_t featureMask;
    uint64_t meshTopologyHash;
    uint64_t patchLayoutHash;
    uint64_t pointerGenerationHash;
};

struct GraphBuildResult
{
    bool ok;
    GraphRebuildReason reason;
    std::string message;
};
```

#### Initialization

```cpp
bool initializeGpuExecutionContext(
    SolverProxy& solver,
    DeviceResidencyRegistry& residency,
    const GpuExecutionConfig& cfg,
    GpuExecutionContext& ctx)
{
    if (!ctx.initialize(cfg, &residency))
    {
        return false;
    }

    // Create explicit streams first.
    if (!ctx.streams().create())
    {
        return false;
    }

    // Register all persistent device buffers required by the supported solver path.
    registerSolverPersistentBuffers(solver, ctx.residency());

    // Allocate one stable device launch-params block per graph kind we expect to use.
    allocateLaunchParamsBlocks(ctx);

    // Optional: warm up one tiny kernel on compute stream.
    if (cfg.useGraphWarmup)
    {
        launchWarmupKernel(ctx.streams().compute());
        if (cfg.mode == ExecutionMode::SyncDebug)
        {
            cudaDeviceSynchronize();
        }
    }

    return true;
}
```

#### Graph fingerprint generation

```cpp
GraphBuildFingerprint makeFingerprint(
    const GraphBuildInputs& in,
    ExecutionMode mode,
    uint16_t backendKind)
{
    GraphBuildFingerprint fp{};
    fp.kind = in.kind;
    fp.meshTopologyHash = in.meshTopologyHash;
    fp.patchLayoutHash = in.patchLayoutHash;
    fp.pointerGenerationHash = in.pointerGenerationHash;
    fp.featureMask = in.featureMask;
    fp.mode = mode;
    fp.backendKind = backendKind;
    return fp;
}
```

#### Capture helper

```cpp
GraphTemplate captureStageGraph(
    const char* debugName,
    cudaStream_t stream,
    const std::function<void(cudaStream_t)>& stageCaptureFn)
{
    GraphTemplate result{};
    result.debugName = debugName;

    // Preconditions:
    // 1. stream is computeStream or approved stagingStream
    // 2. no legacy stream launches occur inside stageCaptureFn
    // 3. no stream/device/context sync or query occurs inside capture

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    stageCaptureFn(stream);

    cudaGraph_t graph = nullptr;
    auto err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess)
    {
        result.graph = nullptr;
        return result;
    }

    result.graph = graph;
    return result;
}
```

#### Graph instantiation and upload

```cpp
bool instantiateAndUpload(
    GraphTemplate& tmpl,
    GraphExecHandle& handle,
    cudaStream_t computeStream,
    bool enableUpload)
{
    if (tmpl.graph == nullptr)
    {
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    auto err = cudaGraphInstantiate(&exec, tmpl.graph, nullptr, nullptr, 0);
    if (err != cudaSuccess)
    {
        return false;
    }

    handle.exec = exec;
    handle.debugName = tmpl.debugName;

    if (enableUpload)
    {
        err = cudaGraphUpload(handle.exec, computeStream);
        if (err != cudaSuccess)
        {
            cudaGraphExecDestroy(handle.exec);
            handle.exec = nullptr;
            return false;
        }
        handle.uploaded = true;
    }

    return true;
}
```

#### Graph get-or-create path

```cpp
GraphExecHandle& GpuGraphManager::getOrCreate(
    GraphKind kind,
    const GraphBuildFingerprint& fp,
    std::function<void(cudaStream_t)> captureFn,
    cudaStream_t stream)
{
    auto it = cache_.find(kind);

    if (it == cache_.end())
    {
        return rebuild(kind, GraphRebuildReason::FirstUse, fp, captureFn, stream);
    }

    GraphExecHandle& h = it->second;

    if (sameFingerprint(h.fingerprint, fp))
    {
        return h;
    }

    if (topologyCompatible(h.fingerprint, fp))
    {
        if (tryUpdate(h, fp, captureFn, stream))
        {
            return h;
        }
        return rebuild(kind, GraphRebuildReason::PointerGenerationChanged, fp, captureFn, stream);
    }

    return rebuild(kind, GraphRebuildReason::TopologyHashChanged, fp, captureFn, stream);
}
```

#### Whole-graph update path

```cpp
bool GpuGraphManager::tryUpdate(
    GraphExecHandle& handle,
    const GraphBuildFingerprint& fp,
    std::function<void(cudaStream_t)> captureFn,
    cudaStream_t stream)
{
    GraphTemplate tmp = captureStageGraph(handle.debugName.c_str(), stream, captureFn);
    if (tmp.graph == nullptr)
    {
        handle.lastRebuildReason = GraphRebuildReason::CaptureInvalidated;
        return false;
    }

    cudaGraphExecUpdateResult updateResult{};
    auto err = cudaGraphExecUpdate(handle.exec, tmp.graph, nullptr, &updateResult);

    cudaGraphDestroy(tmp.graph);

    if (err != cudaSuccess)
    {
        handle.lastRebuildReason = GraphRebuildReason::CaptureInvalidated;
        return false;
    }

    handle.fingerprint = fp;
    handle.updateCount++;
    return true;
}
```

#### Stage launch wrapper

```cpp
void launchStageOrFallback(
    GraphKind kind,
    const GraphBuildInputs& inputs,
    GpuExecutionContext& ctx,
    std::function<void(cudaStream_t)> captureFn,
    std::function<void(cudaStream_t)> asyncFn)
{
    auto mode = ctx.config().mode;
    auto stream = ctx.streams().compute();

    if (mode == ExecutionMode::SyncDebug)
    {
        asyncFn(stream);
        cudaDeviceSynchronize();
        return;
    }

    if (mode == ExecutionMode::AsyncNoGraph)
    {
        asyncFn(stream);
        return;
    }

    if (mode == ExecutionMode::GraphFixed)
    {
        GraphBuildFingerprint fp = makeFingerprint(inputs, mode, /*backendKind=*/0);

        try
        {
            GraphExecHandle& h = ctx.graphs().getOrCreate(kind, fp, captureFn, stream);
            ctx.graphs().launch(h, stream);
        }
        catch (...)
        {
            // deterministic stage fallback
            asyncFn(stream);
        }
        return;
    }

    // Experimental mode is not accepted in Phase 3.
    asyncFn(stream);
}
```

#### Timestep orchestration

```cpp
bool runTimestepPhase3(
    SolverProxy& solver,
    GpuExecutionContext& ctx,
    const TimeStepMeta& ts)
{
    HostStageController ctl;

    ctl.beginTimeStep(solver, ctx, ts);

    // Update per-stage launch params in pinned memory and copy to device blocks.
    updatePinnedLaunchParams(ctx, GraphKind::PreSolve, ts, /*outer=*/-1, /*p=*/-1, /*a=*/-1);
    pushLaunchParamsAsync(ctx, GraphKind::PreSolve);

    ctl.runPreSolve(solver, ctx);

    for (int outer = 0; outer < ts.nOuterCorr; ++outer)
    {
        updatePinnedLaunchParams(ctx, GraphKind::PimpleOuterIter, ts, outer, -1, -1);
        pushLaunchParamsAsync(ctx, GraphKind::PimpleOuterIter);

        ctl.runPimpleOuterIteration(solver, ctx, outer);

        // Pressure solve remains outside graph until individually validated.
        for (int p = 0; p < ts.nPressureCorr; ++p)
        {
            updatePinnedLaunchParams(ctx, GraphKind::PressureAssembly, ts, outer, p, -1);
            pushLaunchParamsAsync(ctx, GraphKind::PressureAssembly);

            ctl.runPressureAssemblyOrFallback(solver, ctx, p);
        }

        // Placeholder for future Phase 5 alpha path:
        // for (int a = 0; a < ts.nAlphaSubcycles; ++a) { ... }
    }

    ctl.finishTimeStep(solver, ctx, ts);
    return true;
}
```

#### Write-boundary staging

```cpp
void HostStageController::finishTimeStep(
    SolverProxy& solver,
    GpuExecutionContext& ctx,
    const TimeStepMeta& ts)
{
    if (!ts.writeNow)
    {
        return;
    }

    cudaEvent_t ready{};
    cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);

    cudaEventRecord(ready, ctx.streams().compute());

    if (ctx.streams().hasStaging())
    {
        cudaStreamWaitEvent(ctx.streams().staging(), ready, 0);
        enqueueWriteStagingCopies(solver, ctx.streams().staging());
        cudaStreamSynchronize(ctx.streams().staging());
    }
    else
    {
        cudaStreamWaitEvent(ctx.streams().compute(), ready, 0);
        enqueueWriteStagingCopies(solver, ctx.streams().compute());
        cudaStreamSynchronize(ctx.streams().compute());
    }

    performHostWriteFromPinnedBuffers(solver);

    cudaEventDestroy(ready);
}
```

#### Device helper kernels

```cpp
__global__ void zeroScratchKernel(double* data, size_t n)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 0.0;
}

__global__ void updateBoundaryFlagsKernel(
    DeviceLaunchParams const* params,
    PatchDeviceView patches,
    int* patchFlags)
{
    const int patchI = blockIdx.x * blockDim.x + threadIdx.x;
    if (patchI >= patches.count) return;

    // Example only:
    // mark write-time or outer-corrector-dependent patch behavior
    patchFlags[patchI] = (params->outerCorrIndex == 0) ? 1 : 0;
}

__global__ void packDiagnosticsKernel(
    DeviceFieldView residualField,
    DeviceDiagBuffer diag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= residualField.n) return;

    // Example only; final reduction kernel likely separate.
    diag.partial[i] = fabs(residualField.data[i]);
}
```

### Step-by-step implementation guide

Each step below is mandatory unless explicitly marked optional.

#### Step 1 — Inventory and gate all existing hot-path synchronization

**Modify**
- search the selected SPUMA branch for:
  - `cudaDeviceSynchronize`
  - `deviceSync`
  - `cudaStreamSynchronize`
  - macros/helpers that imply full-device sync
- add a centralized execution-mode gate so sync behavior is controlled by `ExecutionMode`.

**Why**
- capture cannot tolerate hidden sync in captured regions,
- Phase 3 acceptance requires zero steady-state hot-path device sync.

**Expected output**
- one audit document listing all sync sites,
- one central helper or wrapper controlling whether sync is emitted.

**How to verify**
- compile,
- run supported solver in `sync_debug`,
- behavior matches pre-change run.

**Likely breakages**
- missed sync sites hidden in helper macros,
- solver code relying on sync for host-side reads of GPU-written scalars.

**Fallback**
- temporarily keep those sites under `if (mode == SyncDebug)` until later steps remove host dependencies.

---

#### Step 2 — Introduce explicit stream ownership

**Modify**
- add `ExecutionStreams`,
- create `computeStream` and optional `stagingStream`,
- thread explicit stream arguments through CUDA launch helpers.

**Why**
- graph capture and async execution require explicit stream discipline.

**Expected output**
- no kernel launches on stream 0 / legacy stream from the targeted path.

**How to verify**
- Nsight Systems timeline shows only named explicit streams for the solver process.

**Likely breakages**
- existing wrappers assuming default stream,
- third-party helpers not exposing stream parameters.

**Fallback**
- wrap non-stream-aware code behind `async_no_graph` boundaries until adapted.

---

#### Step 3 — Add `GpuExecutionConfig` and execution-mode selection

**Modify**
- define runtime-selectable execution mode,
- plumb it from configuration/env/path used by project conventions.

**Why**
- staged rollout and deterministic fallback require explicit mode control.

**Expected output**
- same executable can run in `sync_debug`, `async_no_graph`, or `graph_fixed`.

**How to verify**
- startup log prints selected mode,
- mode changes behavior in a controlled way.

**Likely breakages**
- inconsistent default mode,
- solver paths bypassing the new config.

---

#### Step 4 — Create `GpuExecutionContext` and residency registry

**Modify**
- add `GpuExecutionContext`,
- add `DeviceResidencyRegistry`,
- register persistent buffers for one supported solver path.

**Why**
- graph fingerprints and production residency checks require a single source of truth.

**Expected output**
- startup log enumerates graph-relevant persistent buffers.

**How to verify**
- runtime registry dump shows stable pointers, sizes, generations.

**Likely breakages**
- unclear ownership of buffers allocated deep in SPUMA/OpenFOAM abstractions.

**Fallback**
- start by registering only the buffers actually touched in the first graph-enabled stage.

---

#### Step 5 — Convert launch wrappers to graph-safe async behavior

**Modify**
- implement `KernelLaunchPolicy::launchAsync()`,
- remove unconditional device sync in normal modes,
- keep `cudaPeekAtLastError()` or equivalent post-launch checks.

**Why**
- stream capture and async execution are impossible otherwise.

**Expected output**
- `async_no_graph` path runs without device-wide sync.

**How to verify**
- compare results against `sync_debug`,
- confirm zero `cudaDeviceSynchronize()` in normal-mode hot path via Nsight Systems.

**Likely breakages**
- latent race conditions previously masked by sync,
- host code reading device-updated scalars too early.

**Fallback**
- use `sync_debug` to localize the first race, then remove the host-side dependency.

---

#### Step 6 — Add NVTX3 domains and coarse solver-stage annotation

**Modify**
- replace any NVTX v2 usage with NVTX3,
- add domains:
  - `gpu.exec`
  - `gpu.graph.build`
  - `gpu.graph.update`
  - `gpu.graph.launch`
  - `gpu.write`
  - `gpu.debug`

**Why**
- graph launch conversion must remain observable.

**Expected output**
- Nsight timeline shows solver-stage NVTX ranges projected onto GPU activity.

**How to verify**
- run `nsys` with `--trace=cuda,nvtx`,
- confirm NVTX ranges align with kernel/graph activity.

**Likely breakages**
- multiple shared-library initialization surprises with NVTX v3.

**Fallback**
- keep domains minimal first; do not instrument sub-microsecond code regions.

---

#### Step 7 — Build the first capture-enabled stage on a supported solver

**Modify**
- choose one fixed stage sequence under project control, ideally a pre-solve or outer-iteration segment with no external solver call,
- wrap it in a `captureFn(stream)` and `asyncFn(stream)` pair,
- instantiate and upload a `GraphExecHandle`.

**Why**
- this is the first proof that the execution model can work on the actual codebase.

**Expected output**
- one graph launch visible in Nsight Systems where previously many kernel launches appeared.

**How to verify**
- compare graph-enabled and `async_no_graph` results,
- use `--cuda-graph-trace=graph`.

**Likely breakages**
- hidden capture-invalidating operations inside the stage,
- unexpected host callbacks or stream queries.

**Fallback**
- shrink the captured stage until it contains only graph-safe work, then expand incrementally.

---

#### Step 8 — Add graph fingerprinting and cache lookup

**Modify**
- implement `GraphBuildFingerprint`,
- cache graph execs by graph kind,
- log fingerprint and rebuild reason.

**Why**
- reusable graph execs are the point of the phase.

**Expected output**
- repeated timesteps reuse the same graph exec instead of rebuilding every time.

**How to verify**
- graph rebuild count stays at 1 after warm-up for stable workloads.

**Likely breakages**
- pointer-generation logic too coarse or too strict,
- false-positive rebuilds.

**Fallback**
- start with a conservative fingerprint, then relax only with profiler evidence.

---

#### Step 9 — Add whole-graph update and deterministic rebuild

**Modify**
- on parameter change, recapture a temporary graph and attempt `cudaGraphExecUpdate`,
- on update failure, rebuild once.

**Why**
- stage parameters change every timestep and iteration.

**Expected output**
- steady-state run performs updates/reuse, not constant re-instantiation.

**How to verify**
- logs show `updateCount` increasing with low or zero `rebuildCount` after warm-up.

**Likely breakages**
- topology drift across supposedly identical captures,
- pointer changes from allocations that were assumed stable.

**Fallback**
- disable update and use rebuild-only temporarily while stabilizing topology.

---

#### Step 10 — Add device launch-parameter blocks and pinned mirrors

**Modify**
- allocate per-graph `DeviceLaunchParams`,
- update pinned mirror on host,
- push to device asynchronously before launch.

**Why**
- keeps graph topology stable and minimizes per-launch graph churn.

**Expected output**
- graph-enabled stages use stable param-block pointers, not rewritten per-kernel arg lists.

**How to verify**
- logging shows params copied each step; graph fingerprint does not change because only scalar contents changed.

**Likely breakages**
- forgetting to update a field in the param block,
- stale params due to missing HtoD copy.

**Fallback**
- temporarily force param copy + stream sync in `sync_debug` to isolate stale-parameter bugs.

---

#### Step 11 — Add write-boundary staging path

**Modify**
- implement pinned host staging buffers,
- record event on `computeStream`,
- wait on `stagingStream`,
- perform DtoH copies only when write/output is due.

**Why**
- CPU I/O must not pollute the compute hot path.

**Expected output**
- no DtoH in non-write timesteps,
- only staged DtoH activity in write timesteps.

**How to verify**
- Nsight Systems shows DtoH only at write boundaries.

**Likely breakages**
- host write code still reading device memory directly,
- staging stream omitted.

**Fallback**
- temporarily synchronize `computeStream` at write boundary only, never in non-write steps.

---

#### Step 12 — Add production residency assertions

**Modify**
- in `async_no_graph` and `graph_fixed`, call `residency.assertProductionSafe()` at controlled boundaries,
- mark any host touch of device-persistent buffers.

**Why**
- silent migrations are unacceptable in production mode.

**Expected output**
- production mode fails fast if a marked device-only field is touched on host.

**How to verify**
- inject a controlled host touch in a test and confirm assertion/log fires.

**Likely breakages**
- false positives on legitimate write-time staging buffers.

**Fallback**
- correctly classify buffers as `HostPinnedStage` or `ManagedDebugOnly` instead of weakening the assertion.

---

#### Step 13 — Add profiling scripts and trace configuration

**Modify**
- add `scripts/profile_phase3_graph.sh`,
- add node-level variant for short debug runs,
- add UVM-fault variant for residency diagnosis,
- add Compute Sanitizer wrapper.

**Why**
- Phase 3 acceptance is profiler-defined.

**Expected output**
- one-command repro for graph, node, and UVM traces.

**How to verify**
- scripts generate `.nsys-rep` and sanitizer logs.

**Likely breakages**
- overly expensive default profiling options.

**Fallback**
- keep graph-level trace as the default; use node-level/UVM only on short runs.

---

#### Step 14 — Benchmark stop gate before any Phase 4 or Phase 5 work

**Modify**
- run supported solver in all three modes,
- archive metrics and logs,
- produce an engineering note comparing:
  - launch count,
  - sync count,
  - graph launches,
  - UVM faults,
  - runtime.

**Why**
- later work should not begin on an unvalidated runtime substrate.

**Expected output**
- signed-off Phase 3 performance/correctness packet.

**How to verify**
- see acceptance checklist below.

**Likely breakages**
- graph path numerically equivalent but not actually faster because rebuilds or hidden copies remain.

**Fallback**
- stop and fix graph reuse or residency before adding more solver complexity.

### Instrumentation and profiling hooks

#### NVTX3 ranges

Use NVTX3 only. Minimum domain/range schema:
Use `stage.<stage_id>` for semantic ranges whenever the implementation exposes the canonical stage directly. Composite Phase 3 labels such as `stage.outerIter` are acceptable only if `GraphCaptureSupportMatrix` declares exactly which canonical stage IDs they cover.

| Domain | Range name | When |
|---|---|---|
| `gpu.exec` | `timestep.begin` / `timestep.end` | per timestep |
| `gpu.exec` | `stage.preSolve` | pre-solve stage |
| `gpu.exec` | `stage.outerIter` | each outer corrector |
| `gpu.exec` | `stage.pressureAssembly` | each pressure assembly section |
| `gpu.exec` | `stage.alphaSubcycle` | reserved/stub now, active later |
| `gpu.graph.build` | `graph.capture.<kind>` | graph capture |
| `gpu.graph.update` | `graph.update.<kind>` | whole-graph update |
| `gpu.graph.launch` | `graph.launch.<kind>` | every graph launch |
| `gpu.write` | `write.stageCopy` / `write.hostIO` | write staging and host write |
| `gpu.debug` | `fallback.async_no_graph` / `fallback.sync_debug` | fallback events |

Do not annotate tiny sub-microsecond fragments individually; Nsight documentation warns that tracing and backtraces can add significant overhead, and NVTX is most useful for meaningful stage ranges.

#### Nsight Systems commands

Default graph-level collection:

```bash
nsys profile \
  --trace=cuda,nvtx,mpi \
  --cuda-graph-trace=graph \
  --cuda-um-cpu-page-faults=false \
  --cuda-um-gpu-page-faults=false \
  -c cudaProfilerApi \
  --cudabacktrace=all \
  ./solverExecutable -case <caseDir>
```

Debug node-level collection (short runs only):

```bash
nsys profile \
  --trace=cuda,nvtx,mpi \
  --cuda-graph-trace=node \
  --cuda-um-cpu-page-faults=false \
  --cuda-um-gpu-page-faults=false \
  ./solverExecutable -case <caseDir>
```

Residency-diagnosis collection (short runs only, high overhead expected):

```bash
nsys profile \
  --trace=cuda,nvtx,mpi \
  --cuda-graph-trace=graph \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  ./solverExecutable -case <caseDir>
```

Why these settings:
- `--cuda-graph-trace=graph` minimizes overhead compared with node tracing.
- UVM fault tracing is expensive and should be used selectively. Nsight Systems documents that CPU/GPU page-fault tracing can add substantial overhead.

#### Nsight Compute policy

Use Nsight Compute only on the top 5 kernels from the `async_no_graph` and `graph_fixed` paths after graph conversion. Do not use Nsight Compute to answer graph-overhead questions; use Nsight Systems for that.

#### Compute Sanitizer policy

Mandatory during bring-up:
```bash
compute-sanitizer --tool memcheck ./solverExecutable -case <caseDir>
compute-sanitizer --tool racecheck ./solverExecutable -case <caseDir>
```

Use `sync_debug` for the smallest failing reproduction when diagnosing data hazards.

### Validation strategy

#### Correctness checks

1. **Mode equivalence**
   - Compare `sync_debug`, `async_no_graph`, and `graph_fixed` on the same supported solver case.
   - Acceptance:
     - identical stage counts,
     - identical outer/pressure iteration counts,
     - no missing stage execution.

2. **Numerical equivalence**
   - Preferred threshold for deterministic single-stream paths:
     - bitwise identical scalar residual histories and field outputs for short tests.
   - If minor non-bitwise differences appear due to changed reduction ordering, acceptance thresholds are:
     - double-valued residuals: relative difference `< 1e-12`
     - single-precision scratch-derived diagnostics: relative difference `< 1e-6`

3. **Write-boundary correctness**
   - Output files generated by `graph_fixed` and `async_no_graph` must agree within the same thresholds.

4. **Residency correctness**
   - Production modes must show no unexpected host touches of `DevicePersistent` buffers.

#### Regression checks

1. Supported solver (`pimpleFoam`) small-case smoke test.
2. Supported solver larger-case repeated-step test.
3. Graph rebuild stress test:
   - toggle a feature flag that should force rebuild,
   - confirm rebuild occurs once and then stabilizes.
4. Forced fallback test:
   - intentionally inject a capture-invalidating operation,
   - confirm stage downgrades to `async_no_graph` with a logged reason.

#### Numerical invariants

For the supported solver path, verify at minimum:
- timestep count and write cadence preserved,
- continuity residual history preserved,
- solver convergence status preserved.

For future VOF/nozzle work, the execution substrate must also be ready to preserve:
- alpha boundedness,
- mass conservation,
- pressure-drop history,
- mass-flow history,
but those are deferred to later phases.

#### Performance checks

Minimum pass/fail thresholds for Phase 3:

1. **Hot-path sync count**
   - `graph_fixed`: zero steady-state `cudaDeviceSynchronize()` in hot path.
   - `async_no_graph`: zero steady-state `cudaDeviceSynchronize()` in hot path.

2. **Launch reduction**
   - For the graph-enabled stage(s), CUDA API launch count in steady state must drop by at least **5×** relative to the pre-Phase-3 baseline for those same stages.
   - If the stage is very small and does not achieve 5×, benchmark evidence must show why the graph boundary is still justified.

3. **Graph reuse**
   - After warm-up, graph rebuild count should remain zero for stable workloads.
   - Whole-graph update count may increase; rebuild count should not.

4. **UVM traffic**
   - Production `graph_fixed` path must show zero or near-zero steady-state UVM page faults for graph-enabled buffers.

5. **CPU API overhead**
   - CUDA API time attributed to synchronization must fall materially from the pre-Phase-3 baseline.
   - Exact speedup is workload-specific; do not require a fixed overall walltime speedup if the graph-covered portion is still small. But if sync/launch metrics do not improve, Phase 3 is not complete.

#### Profiling checks

A Phase 3 pass requires:
- graph-level nodes visible in Nsight Systems,
- NVTX ranges aligned with graph launches,
- no unexplained DtoH/HtoD bursts between stages,
- no stream-0 launches in graph-enabled path.

### Performance expectations

Do not oversell this phase.

What this phase should improve:
- CPU-side launch overhead,
- device-wide sync overhead,
- first-launch artifacts via upload/warm-up,
- observability of stage execution,
- probability that later alpha/pressure work stays device-resident.

What this phase will **not** solve by itself:
- irregular face-kernel inefficiency,
- atomics in gradient/limiter work,
- weak native GAMG GPU efficiency,
- FP64 throughput limitations of the RTX 5080,
- poor sparse-kernel data reuse.

The expected benefit on a small or medium transient nozzle-like workload is primarily **less host/API overhead and fewer forced GPU idle periods**, not magically faster sparse numerics.

### Common failure modes

1. `cudaStreamEndCapture()` fails because some code synchronized a captured stream/device/context.
2. A kernel still launches on the legacy/default stream during capture.
3. Whole-graph update fails because a “small” code change actually changed topology.
4. Graph rebuild happens every timestep because a persistent buffer address is not actually stable.
5. A host accessor touches a device-persistent field, triggering migration or stale data.
6. External library code inside a captured stage calls a blocking or graph-unsafe routine.
7. `cudaGraphUpload()` is done on one stream and launch on another, causing remapping or noisy first launches if graph memory nodes later appear.
8. Output staging accidentally occurs every timestep.
9. `sync_debug` and `graph_fixed` diverge because hidden timing dependencies were masking a data race.
10. Debug instrumentation becomes so fine-grained that profiler overhead obscures the effect being measured.
11. HIP build breaks because CUDA-specific changes leaked into shared headers without guards.
12. A captured stage contains allocator activity that changes pointer generations.
13. Graph cache invalidation rules are too weak, causing stale executable graphs to run.
14. Graph cache invalidation rules are too strong, causing constant rebuilds.
15. The supported-solver adapter becomes numerically invasive instead of being a thin execution wrapper.

### Debugging playbook

1. **Reproduce in the smallest supported solver case.** Do not start in the nozzle solver.
2. **Switch to `async_no_graph`.** If the bug persists, it is not a graph bug.
3. **Switch to `sync_debug`.** If the bug disappears only in `sync_debug`, suspect an ordering/race/host-touch issue.
4. **Enable Compute Sanitizer.** Start with `memcheck`, then `racecheck`.
5. **Enable capture-state assertions.** Assert that no forbidden sync/query occurs while a stream is capturing.
6. **Run node-level graph trace on a short case.** Use `--cuda-graph-trace=node` only for a few iterations.
7. **Check stream IDs in Nsight Systems.** Any work on stream 0 is a bug in the graph-enabled path.
8. **Inspect graph rebuild logs.** Rebuild every iteration implies unstable fingerprint inputs.
9. **Compare pointer-generation hashes.** If they change unexpectedly, allocator stability is broken.
10. **Temporarily disable write/output.** If the bug disappears, the staging boundary is wrong.
11. **Temporarily isolate the captured stage.** Shrink the stage until capture succeeds, then re-expand.
12. **Use `CUDA_FORCE_PTX_JIT=1` once per build variant.** If behavior changes, build packaging/arch targeting is suspect, not the graph logic.
13. **Use `CUDA_LAUNCH_BLOCKING=1` only as a last resort.** It changes timing enough to hide graph-related issues; use it only for localization.

### Acceptance checklist

- [ ] Supported SPUMA GPU solver runs correctly in `sync_debug`.
- [ ] Same solver runs correctly in `async_no_graph`.
- [ ] Same solver runs correctly in `graph_fixed`.
- [ ] No steady-state `cudaDeviceSynchronize()` in hot path for `async_no_graph`.
- [ ] No steady-state `cudaDeviceSynchronize()` in hot path for `graph_fixed`.
- [ ] No legacy/default stream launches in graph-enabled path.
- [ ] At least one repeated stage is launched as a CUDA graph.
- [ ] Graph upload is performed before timing steady-state runs.
- [ ] Graph reuse occurs across repeated timesteps/iterations.
- [ ] Graph update or rebuild policy is deterministic and logged.
- [ ] Rebuild count is zero after warm-up for stable workloads.
- [ ] UVM steady-state page-fault activity is zero or near-zero in production mode.
- [ ] Output staging occurs only at write boundaries.
- [ ] NVTX3 ranges cover all major execution phases.
- [ ] Nsight Systems graph trace confirms graph-level launches.
- [ ] `graph_capture_support_matrix.md` is produced and its canonical stage IDs, graph-external boundaries, and fallback targets match the implemented Phase 3 path.
- [ ] Fallback to `async_no_graph` works when a graph stage is forced to fail.
- [ ] HIP/other backend builds do not silently regress; unsupported paths fail cleanly or run no-graph.
- [ ] Tests and scripts are committed.
- [ ] Benchmark stop-gate report is produced.
- [ ] Human reviewer signs off on the Phase 3 acceptance packet.

### Future extensions deferred from this phase

1. Conditional graph nodes for pressure-corrector or alpha-subcycle loops.
2. Manual graph construction for fine-grained node update.
3. Device graph launch.
4. Graph memory nodes / `cudaMallocAsync` capture inside graphs.
5. Capturing third-party sparse-library or external-solver calls.
6. Dynamic mesh/topology support.
7. Multi-GPU graph-aware execution.
8. Automatic graph partitioning and graph-topology synthesis.
9. Aggressive kernel fusion beyond launch-graph conversion.
10. Nozzle-specific BC graphs and alpha-subcycle graphs (future Phase 5/6 work).

### Implementation tasks for coding agent

1. Add `ExecutionMode`, `ExecutionStreams`, `GpuExecutionContext`, `DeviceResidencyRegistry`, `GpuGraphManager`, and `KernelLaunchPolicy`.
2. Audit and gate all device-wide synchronization sites in the selected SPUMA branch.
3. Thread explicit streams through the targeted supported solver path.
4. Add NVTX3 instrumentation and profiling scripts.
5. Implement capture-built graph instantiation/upload/update/rebuild for one supported solver stage sequence.
6. Implement stage-level fallback to `async_no_graph`.
7. Implement pinned-host + device `DeviceLaunchParams`.
8. Implement write-boundary staging via `stagingStream`.
9. Produce tests, traces, and benchmark packet.
10. Publish `graph_capture_support_matrix.md` with canonical stage IDs, graph-external rows, host-controlled loop ownership, and fallback targets.

### Do not start until

- SPUMA branch/commit is frozen.
- One supported GPU solver runs correctly on the RTX 5080.
- Phase 2 allocator/residency primitives are available.
- Those allocator/residency primitives are already accepted as Phase 2 `productionDevice` behavior, not merely available in bring-up form.
- Nsight Systems is installed and validated.
- Human reviewer agrees on the execution-mode ladder.

### Safe parallelization opportunities

1. One engineer/agent can build `ExecutionStreams`, `ExecutionMode`, and `GpuExecutionContext`.
2. Another can implement NVTX3/profiling scripts.
3. Another can perform sync-site audit and launcher refactor.
4. Another can build graph cache/update/rebuild logic.
5. Solver-adapter hook work should proceed only after stream and launch-policy interfaces are stable.

### Governance guardrails

1. The exact SPUMA branch/commit used for a run must be recorded in `manifest_refs.json`; Phase 3 does not reopen the runtime family frozen by `master_pin_manifest.md`.
2. Executor-layer integration is the required baseline. A separate adapter library is not the milestone-1 default architecture.
3. The canonical runtime switch is `gpuRuntime.execution.mode`; environment variables, dictionary aliases, or command-line flags are compatibility shims only.
4. HIP surface compatibility remains interface-level only for this milestone; no-graph stubs are acceptable outside the CUDA primary lane.
5. Write/output synchronization at output intervals is acceptable. Timed steady-state windows continue to exclude write steps per `acceptance_manifest.md`.

### Artifacts to produce

1. `phase3_sync_audit.md`
2. `phase3_graph_design_notes.md`
3. `profile_phase3_graph.sh`
4. `profile_phase3_node.sh`
5. `profile_phase3_uvm.sh`
6. `run_compute_sanitizer_phase3.sh`
7. `phase3_benchmark_results.md`
8. `.nsys-rep` samples for baseline and graph-enabled runs
9. test logs for `sync_debug`, `async_no_graph`, and `graph_fixed`
10. `graph_capture_support_matrix.md`


## 6. Validation and benchmarking framework

This section is global to keep Phase 3 coherent with later phases, but the concrete benchmarks here are Phase-3-centric.

### Benchmark matrix

| Tier | Purpose | Solver/case | Required modes |
|---|---|---|---|
| B0 | Toolchain sanity | smallest supported SPUMA GPU solver case | `sync_debug`, `async_no_graph` |
| B1 | Execution-substrate bring-up | supported `pimpleFoam` small case | all three modes |
| B2 | Reuse stability | supported `pimpleFoam` medium repeated-step case | `async_no_graph`, `graph_fixed` |
| B3 | Residency audit | same as B2, short run with UVM tracing | `graph_fixed` |
| B4 | Write-boundary behavior | same as B2 with forced write interval | `graph_fixed` |

### Metrics to record

For every benchmark run record:
- wall-clock runtime,
- timestep runtime,
- graph launch count,
- kernel launch count,
- steady-state `cudaDeviceSynchronize()` count,
- graph rebuild count,
- graph update count,
- UVM CPU page faults,
- UVM GPU page faults,
- HtoD bytes,
- DtoH bytes,
- number of streams active,
- output write time if applicable.

### Required outputs

For each benchmark tier:
1. command line used,
2. git commit hash,
3. execution mode,
4. profiler version,
5. toolkit version,
6. driver version,
7. results table,
8. “pass / fail / blocked” status,
9. short engineering interpretation.

### Stop gates

Do not proceed to Phase 4 or Phase 5 until:
- B1 passes correctness,
- B2 passes launch/sync/reuse thresholds,
- B3 confirms production residency behavior,
- B4 confirms write-boundary staging behavior.

### Failure interpretation rules

- If correctness fails in `async_no_graph`, the problem is not graph-specific.
- If correctness fails only in `graph_fixed`, inspect capture-invalidating behavior, stale param blocks, or topology drift.
- If performance fails but correctness passes, inspect rebuild frequency, upload policy, and hidden host/device movement before changing any numerics.


## 7. Toolchain / environment specification

### Required software baseline

- **OS:** Linux x86_64 recommended for primary development.
- **GPU:** NVIDIA GeForce RTX 5080, compute capability 12.0. 
- **Master pin manifest:** Phase 3 consumes the centrally frozen toolchain pins. Default until superseded: CUDA 12.9.1 primary lane, CUDA 13.2 experimental lane, driver `>= 595.45.04`, `sm_120` + PTX, and NVTX3.
  The remaining toolkit/driver bullets in this subsection are background feature-availability references, not a second source of truth for project pins.
- **Compatibility floor reference:** CUDA 12.8 introduced Blackwell compiler support including `SM_120`; this is background compatibility context only, not the active project pin.
- **Historical minimum-driver reference for CUDA 12.8 GA:** Linux `>= 570.26`, Windows `>= 570.65`; the active project pin remains the master-pin-manifest lane.
- **Profilers:** Nsight Systems, Nsight Compute, Compute Sanitizer.
- **Instrumentation:** NVTX3 only. NVTX v2 is deprecated in CUDA 12.8 and later removed. 

### Build configuration requirements

#### Release build
Recommended NVCC target policy:
```bash
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_120,code=compute_120
```

Rationale:
- native cubin for local Blackwell RTX 5080 target,
- PTX for compatibility validation and forward resilience. CUDA documentation explicitly recommends including PTX, and `CUDA_FORCE_PTX_JIT=1` should be used to validate that release artifacts contain it.

#### Debug build
Recommended additions:
```bash
-lineinfo
-Xcompiler -fno-omit-frame-pointer
```

Do **not** use `-G` for normal profiling; reserve it for narrow debug reproductions.

### Runtime environment checks

Before Phase 3 benchmarking, validate:
1. `nvidia-smi` shows the expected driver.
2. a local `deviceQuery` or equivalent confirms compute capability 12.0.
3. `CUDA_FORCE_PTX_JIT=1` run succeeds for the target executable.
4. Nsight Systems graph-level tracing works.
5. NVTX3 ranges appear in traces.

### Runtime environment variables

Recommended debug variables:
- `CUDA_FORCE_PTX_JIT=1` — compatibility validation only, not normal benchmarking. 
- `CUDA_LAUNCH_BLOCKING=1` — last-resort debug only, not performance measurement.
- project-specific execution-mode switch, e.g. `FOAM_GPU_EXEC_MODE=graph_fixed`
- authoritative accepted setting in the normalized runtime schema, e.g. `gpuRuntime.execution.mode=graph_fixed`; environment-variable or command-line forms are compatibility shims only.

### Nsight Systems notes for Blackwell workstation runs

Nsight Systems documentation notes lower-overhead graph tracing at graph granularity and supports hardware-based CUDA trace collection on Blackwell-based GPUs for workloads with many short kernels. Use hardware-based collection only after standard graph-level tracing is stable and if the team needs lower-overhead profiling of very short kernels.


## 8. Module / file / ownership map

| Module / file set | Responsibility | Phase owner | Notes |
|---|---|---|---|
| `src/OpenFOAM/device/executor/ExecutionMode.H` | runtime execution mode enum | Phase 3 execution substrate | new |
| `src/OpenFOAM/device/executor/ExecutionStreams.*` | explicit stream ownership | Phase 3 execution substrate | new |
| `src/OpenFOAM/device/executor/GpuExecutionContext.*` | top-level runtime context | Phase 3 execution substrate | new |
| `src/OpenFOAM/device/executor/DeviceResidencyRegistry.*` | buffer registration and host-touch checks | Phase 2 + Phase 3 | new |
| `src/OpenFOAM/device/executor/GpuGraphManager.*` | capture/update/rebuild/launch | Phase 3 execution substrate | new |
| `src/OpenFOAM/device/executor/KernelLaunchPolicy.H` | graph-safe async launch wrapper | Phase 3 execution substrate | new |
| `src/OpenFOAM/device/executor/cudaExecutor.*` | backend launch implementation | existing executor layer | modify existing |
| `src/OpenFOAM/device/memoryExecutor/cudaMemoryExecutor.*` | allocator metadata / pinned staging helpers | Phase 2 memory posture | modify existing |
| `src/OpenFOAM/device/deviceM.H` / `deviceUtils.H` | sync policy hooks / backend guards | Phase 3 execution substrate | modify existing |
| `src/OpenFOAM/device/profiling/NvtxDomains.*` | NVTX3 instrumentation | Phase 3 profiling | new |
| `applications/.../pimpleFoam/GpuStageHooks.*` | thin solver adapter | Phase 3 integration | new; exact path must be verified locally |
| `test/gpuExecution/*` | unit and integration tests | Phase 3 verification | new |
| `scripts/profile_phase3_*.sh` | reproducible performance collection | Phase 3 profiling | new |
| `phase3_benchmark_results.md` | acceptance packet | human review artifact | generated |


## 9. Coding-agent execution roadmap

### Milestone M0 — Sync and stream inventory
**Depends on:** frozen SPUMA branch, working supported solver run  
**Deliverables:** sync audit, stream-launch inventory  
**Stop here and benchmark?** No. This is preparatory.

### Milestone M1 — Explicit execution modes and streams
**Depends on:** M0  
**Work:** add `ExecutionMode`, `ExecutionStreams`, explicit-stream launch signatures  
**Parallelizable:** yes, with NVTX work  
**Stop here and benchmark?** Minimal sanity only.

### Milestone M2 — Async no-graph baseline
**Depends on:** M1  
**Work:** remove unconditional normal-mode hot-path sync, preserve `sync_debug` fallback  
**Parallelizable:** partially, with residency registry scaffolding  
**Stop here and benchmark?** **Yes.** This is the first hard checkpoint. If `async_no_graph` is not correct, do not proceed.

### Milestone M3 — Runtime context and residency registry
**Depends on:** M2  
**Work:** add `GpuExecutionContext`, `DeviceResidencyRegistry`, pinned/device param blocks  
**Parallelizable:** yes, separate from graph manager implementation  
**Stop here and benchmark?** Short sanity only.

### Milestone M4 — First graph-enabled stage
**Depends on:** M2, M3  
**Work:** capture/build/upload one fixed stage on supported solver  
**Parallelizable:** no; this is the first integration risk point  
**Stop here and benchmark?** **Yes.** Validate graph launch appears and correctness holds.

### Milestone M5 — Graph cache, update, rebuild policy
**Depends on:** M4  
**Work:** fingerprinting, whole-graph update, deterministic rebuild and fallback  
**Parallelizable:** limited  
**Stop here and benchmark?** **Yes.** Reuse stability must be proven before wider rollout.

### Milestone M6 — Write-boundary staging and production residency checks
**Depends on:** M5  
**Work:** staging stream, pinned write buffers, host-touch assertions  
**Parallelizable:** yes, with profiling-script work  
**Stop here and benchmark?** **Yes.** Confirm no unexpected DtoH/HtoD in non-write steps.

### Milestone M7 — Acceptance packet
**Depends on:** M6  
**Work:** tests, profiles, benchmark packet, reviewer checklist  
**Parallelizable:** yes  
**Stop here and benchmark?** Final Phase 3 stop gate.

### Dependency graph

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7
             \         /
              \-> profiling scaffolding
```

### What can be prototyped before being productized

1. capture of a minimal supported-solver stage,
2. graph update on a toy stage before full solver-stage capture,
3. residency registry on a small subset of buffers.

### What should remain experimental in Phase 3

1. conditional graph nodes,
2. device graph launch,
3. third-party sparse-library capture,
4. graph memory nodes,
5. multi-stream compute overlap beyond staging.

### Where to stop and benchmark before proceeding

Mandatory stop-and-benchmark points:
- after M2,
- after M4,
- after M5,
- after M6.

If any of those checkpoints fail, do not move into Phase 4 or Phase 5 work.


## 10. Resolved Local Defaults and Residual Governance Notes

1. **Source-line ownership is fixed.**  
   - Phase 3 consumes the SPUMA/v2412 family frozen by `master_pin_manifest.md`. Each implementation branch must record its exact commit in `manifest_refs.json`, but the package does not reopen the runtime family locally.
2. **Executor integration posture is fixed.**  
   - Modify the existing device/executor layer directly. A separate adapter library is not the milestone-1 baseline.
3. **Cross-backend posture is fixed.**  
   - HIP compatibility is interface-level only in this milestone. The CUDA lane remains primary, and no-graph stubs are acceptable for non-CUDA backends.
4. **Execution-mode configuration is fixed.**  
   - `gpuRuntime.execution.mode` is the authoritative control surface. Any env var, CLI switch, or alias is only a compatibility shim.
5. **Topology scope is fixed.**  
   - Dynamic mesh or topology change is out of scope for the milestone-1 nozzle target. Graph reuse policy assumes static topology.
6. **Output-cadence policy is fixed.**  
   - Output synchronization at write intervals is acceptable, but timed steady-state profiling windows exclude write steps under `acceptance_manifest.md`.
7. **Initial graph boundary is fixed.**  
   - The first graph-enabled path stops at a graph-safe boundary under local control and leaves `pressure_solve_native` / `pressure_solve_amgx` outside capture until those solver-library boundaries are promoted by `graph_capture_support_matrix.md`.

---

## Human review checklist

- [ ] The implementation records its exact SPUMA commit in `manifest_refs.json`.
- [ ] The branch uses `gpuRuntime.execution.mode` as the only authoritative execution-mode contract.
- [ ] The graph partitioning matches the canonical stage boundaries and graph-external solve rows in `graph_capture_support_matrix.md`.
- [ ] The fallback and write-boundary behavior remain consistent with `acceptance_manifest.md`.
- [ ] The executor/memory-layer edits stay within the direct-integration posture described here.

## Coding agent kickoff checklist

- [ ] Check out the approved SPUMA/v2412 family and record the exact branch/commit in `manifest_refs.json`.
- [ ] Confirm supported GPU solver runs on the RTX 5080.
- [ ] Run baseline `nsys` profile and archive it.
- [ ] Inventory sync/stream launch sites.
- [ ] Implement `ExecutionMode` and `ExecutionStreams`.
- [ ] Prove `async_no_graph` correctness before any graph work.
- [ ] Implement the first capture-built stage graph on the supported solver.
- [ ] Add graph reuse/update/rebuild.
- [ ] Produce acceptance traces and logs.

## Highest risk implementation assumptions

1. **Assumption:** the selected supported solver path can be partitioned into at least one graph-safe stage without deep solver surgery.  
   **Risk if false:** Phase 3 becomes more invasive than planned.

2. **Assumption:** persistent field/topology pointers can remain stable across timesteps.  
   **Risk if false:** graph reuse collapses into rebuild churn.

3. **Assumption:** hidden host touches can be found and removed before later VOF/nozzle work starts.  
   **Risk if false:** UVM pathologies persist despite graph launch reduction.

4. **Assumption:** stream-capture-based graph construction is viable on the chosen SPUMA branch once unconditional sync is removed.  
   **Risk if false:** manual graph construction becomes necessary earlier than desired.

5. **Assumption:** the first measurable Phase 3 gain comes from reduced launch/sync overhead rather than kernel changes.  
   **Risk if false:** the chosen graph boundaries may be too narrow and need re-partitioning before later phases.
