
# Phase 8 Implementation Specification — Profiling and Performance Acceptance for a SPUMA/OpenFOAM RTX 5080 Pressure-Swirl Nozzle Port

## 1. Executive overview

This document expands **Phase 8 — profiling and performance acceptance** into an implementation-ready engineering specification for a coding agent. The rest of the port plan is preserved only as context needed to keep Phase 8 coherent; this is **not** a full expansion of Phases 0–7.

The working assumption is that Phases 0–7 have produced, or will soon produce, a runnable **single-GPU**, **SPUMA-based**, **OpenFOAM-v2412-aligned** nozzle solver path with a transient **VOF / MULES / PIMPLE** execution structure, and that the target production platform is an **NVIDIA RTX 5080** class GPU. NVIDIA’s official specifications for the RTX 5080 list **Blackwell architecture**, **16 GB GDDR7**, **960 GB/s bandwidth**, and **CUDA capability 12.0** [R17]. CUDA 12.8 added Blackwell compiler/toolchain support and Blackwell-specific compatibility guidance [R7][R8].

Phase 8 exists because this project will fail in practice if performance validation remains ad hoc. SPUMA’s published profiling shows that a representative OpenFOAM GPU path can suffer from **very high kernel-launch counts**, **`cudaDeviceSynchronize()` after each kernel**, and weaker kernel efficiency in atomic-heavy and multigrid sections [R1]. SPUMA’s own GPU-support documentation further warns that unsupported features may not fail loudly; they can instead trigger **undesired host/device copies**, so profiling must be used to detect correctness/performance hazards [R2]. For a transient nozzle solver with **PIMPLE outer loops**, **pressure solves**, **alpha corrections**, **MULES limiter logic**, and **alpha subcycling**, this is not a secondary concern. It is a first-order implementation requirement.

The specification therefore treats profiling as a **productized subsystem** with five outputs:

1. **Deterministic instrumentation** inside the solver and runtime.
2. **Repeatable capture modes** for timeline, UVM audit, sync audit, graph debug, kernel deep-dive, and sanitizer bring-up.
3. **Machine-readable acceptance artifacts** (CSV/JSON/text) generated from profiler outputs.
4. **Pass/fail gates** for device residency, synchronization discipline, launch structure, instrumentation coverage, and kernel-baseline stability.
5. **A debug playbook** that turns profiler findings into concrete next engineering actions.

This phase is intentionally opinionated. The core architectural position is:

- **Nsight Systems** is the authoritative tool for pass/fail performance acceptance, because it observes the whole solver: launch behavior, synchronization, unified-memory activity, graph behavior, and CPU/GPU boundaries [R9][R10].
- **Nsight Compute** is a targeted deep-dive tool for only the **top five kernels** per steady-state run, because its overhead is too high for broad routine use [R12][R13].
- **Compute Sanitizer** is required for bring-up and nightlies on reduced cases, but must not be confused with production performance measurement [R14][R15].
- **NVTX v3** instrumentation is mandatory. CUDA 12.8 deprecates NVTX v2, and later toolkit documentation removes it; new code must be written against NVTX v3 headers [R8][R18].

The acceptance model for this phase is built around the three ceilings already identified in the project plan:

- **Sparse algebra traffic**: DRAM/L2 movement in matrix/vector kernels.
- **Irregular-kernel efficiency**: face-based loops, atomics, boundary kernels, limiter logic, reductions.
- **Launch/synchronization overhead**: repeated orchestration work from PIMPLE, alpha correction, alpha subcycling, graph boundaries, and solver dispatch.

The deliverable is therefore not “a set of profiler commands.” It is a profiling and acceptance framework that lets a coding agent answer, with evidence, all of the following:

- Are heavy fields and matrices device-resident between timesteps?
- Are there any recurring unexpected **HtoD/DtoH** migrations or **UVM page faults** in steady-state solver execution?
- Has the port eliminated **per-kernel synchronization** in hot paths?
- Are graph launches replacing floods of micro-launches?
- Which kernels dominate GPU time in the actual nozzle workflow?
- Are regressions visible automatically before the next optimization phase starts?

The production interpretation of success in this phase is:

- **Near-zero unexpected UVM traffic** in steady-state production mode.
- **No recurring synchronization APIs** inside hot solver loops except explicit graph or write boundaries.
- **Stable NVTX coverage** over all major nozzle/VOF stages.
- **Automated profiling artifacts** for every acceptance run.
- **A locked performance baseline** before further kernel optimization begins.

---

## 2. Global architecture decisions

### GAD-1 — Treat profiling as first-class product code, not temporary debug scaffolding

- **Sourced fact:** SPUMA’s published profiling and wiki both show that unsupported or suboptimal code paths can manifest as synchronization-heavy execution and unwanted host/device traffic rather than clean failures [R1][R2].
- **Engineering inference:** If instrumentation is added only after performance problems are suspected, the coding agent will chase symptoms without a stable ground truth and will likely miss regressions introduced by unrelated refactors.
- **Recommendation:** Build Phase 8 as a permanent subsystem with compile-time wrappers, runtime configuration, scripted captures, report parsers, and acceptance thresholds checked in alongside solver code.

### GAD-2 — Use Nsight Systems as the authoritative pass/fail tool

- **Sourced fact:** Nsight Systems provides timeline tracing for CUDA, NVTX, OS runtime, unified-memory migrations/page faults, CUDA Graph tracing, and built-in statistical reports such as `cuda_api_sum`, `cuda_gpu_sum`, `cuda_kern_exec_sum`, `nvtx_sum`, `um_sum`, and `um_total_sum` [R9][R10].
- **Engineering inference:** Phase 8 must gate not only kernel efficiency but also residency, synchronization discipline, graph structure, and CPU/GPU boundaries. Nsight Compute cannot answer all of those in one run.
- **Recommendation:** All formal acceptance decisions in this phase must originate from Nsight Systems outputs or from scripts derived directly from Nsight Systems exports.

### GAD-3 — Restrict Nsight Compute to the top five kernels per steady-state run

- **Sourced fact:** Nsight Compute is the right tool for roofline, memory workload, scheduler, warp stall, occupancy, and source-correlated kernel analysis [R12][R13]. It also supports NVTX filtering and source correlation when compiled with line information [R12][R13].
- **Engineering inference:** Running Nsight Compute across the entire transient nozzle timestep will create excessive overhead and analysis noise.
- **Recommendation:** Use Nsight Systems to rank kernels by steady-state GPU time, then profile only the top five kernels (or fewer if they already cover the dominant fraction of time). Deep-dive all-kernel NCU runs are explicitly out of scope for routine acceptance.

### GAD-4 — Make NVTX v3 mandatory and design the range hierarchy before adding kernels

- **Sourced fact:** CUDA 12.8 deprecates NVTX v2 and directs users to NVTX v3; current Nsight Systems documentation supports NVTX v3 domains, categories, ranges, and marks, and notes that overhead is typically low when no profiler is attached [R8][R11].
- **Engineering inference:** If NVTX ranges are not standardized up front, later profiling data will be inconsistent across solver branches and impossible to compare automatically.
- **Recommendation:** Introduce a profiling wrapper layer with predefined domains, categories, naming rules, payload conventions, and RAII scope objects before instrumenting any hot path.

### GAD-5 — Enforce steady-state capture windows; never profile initialization by default

- **Sourced fact:** Nsight Systems supports `--capture-range=nvtx` and `--nvtx-capture` to collect only selected NVTX-delimited portions of execution [R9].
- **Engineering inference:** Mesh load, dictionary parsing, dynamic code loading, graph instantiation, startup seeding, and first-touch memory behavior will pollute runtime acceptance unless capture is restricted to a warm steady-state window.
- **Recommendation:** All baseline timeline captures must begin only after a configurable warmup phase and must target a bounded set of steady-state timesteps.

### GAD-6 — Use graph-level tracing by default; use node-level tracing only for graph debugging

- **Sourced fact:** Nsight Systems documents `--cuda-graph-trace=graph|node`, and states that graph-level tracing has significantly lower overhead while node-level tracing may introduce substantial overhead and reduce parallelism [R9][R19].
- **Engineering inference:** Phase 8 needs a production-safe default and a diagnostic fallback. The default cannot be node-level graph tracing for a transient CFD solver.
- **Recommendation:** Baseline runs use `--cuda-graph-trace=graph`. Node-level graph tracing is restricted to short diagnostic captures when debugging graph construction, missing fusion, or unexpected graph launch multiplicity.

### GAD-7 — UVM page-fault tracing is diagnostic-only, not a baseline performance mode

- **Sourced fact:** Nsight Systems can trace CPU and GPU unified-memory page faults via `--cuda-um-cpu-page-faults=true` and `--cuda-um-gpu-page-faults=true`, but NVIDIA documents that this can add significant overhead, with observed cases up to roughly 70% in internal testing [R9].
- **Engineering inference:** UVM page-fault tracing is essential to localize residency failures, but if left enabled during acceptance timing it will contaminate the very metric it is supposed to diagnose.
- **Recommendation:** Maintain two distinct timeline modes: a low-overhead baseline mode and a short diagnostic UVM audit mode.

### GAD-8 — Blackwell support is a hard requirement; toolchain choices must reflect that explicitly

- **Sourced fact:** NVIDIA’s Blackwell compatibility guide and CUDA 12.8 release notes state that CUDA 12.8 introduced support relevant to Blackwell architectures, PTX compatibility guidance, and Blackwell-target compilation support [R7][R8]. The RTX 5080 is a Blackwell part with compute capability 12.0 [R17].
- **Engineering inference:** Treating the 5080 as “just another NVIDIA GPU” risks tool mismatch, inaccurate expectations about tracing behavior, and invalid assumptions about profiler feature availability.
- **Recommendation:** The profiling framework must record toolkit version, driver version, GPU name, and whether the CUDA trace path is using the expected Blackwell-capable mode.

### GAD-9 — Prefer built-in `nsys stats` report scripts over custom SQLite queries

- **Sourced fact:** Nsight Systems exports SQLite, but NVIDIA also states that report scripts shipped with the tool are maintained to adapt to schema changes in SQLite/internal structures [R9][R20].
- **Engineering inference:** Direct SQLite parsing alone will become a maintenance burden across Nsight versions.
- **Recommendation:** The parser layer must first attempt to use shipped `nsys stats` reports; custom SQLite queries are an explicit fallback and must be schema-version-guarded.

### GAD-10 — Use Compute Sanitizer in a staged order and only on reduced cases

- **Sourced fact:** Compute Sanitizer provides `memcheck`, `initcheck`, `racecheck`, and `synccheck`; NVIDIA recommends using `memcheck` first, and the tools can slow applications significantly [R14][R15].
- **Engineering inference:** Full nozzle production cases under sanitizer are too slow and too noisy for routine use.
- **Recommendation:** The Phase 8 sanitizer workflow is limited to reduced case `R1`, kernel microtests, and nightly or pre-merge gates. Production timing runs must never use sanitizer.

---

## 3. Global assumptions and constraints

1. **Runtime base**
   - The runtime base is SPUMA on an OpenFOAM-v2412-aligned code line, not a backport into OpenFOAM 12.
   - The pressure-swirl nozzle port uses SPUMA-style GPU execution/memory abstractions and runtime selection where practical.
   - The specific VOF path is pressure-driven, transient, and based on OpenFOAM-style `incompressibleVoF` / PIMPLE / MULES semantics [R3][R4][R5][R6].

2. **Target hardware**
   - Primary target is a **single RTX 5080** workstation.
   - No multi-GPU acceptance in this phase.
   - No domain decomposition acceptance in this phase.
   - No NVLink assumptions; the 5080 does not provide it [R17].

3. **Execution model assumptions**
   - Persistent device residency is the desired production model.
   - Unified memory may exist as a bring-up path, but **production acceptance assumes device-resident hot working sets**.
   - CUDA Graphs are expected to become part of the steady-state execution path, but Phase 8 must also function during intermediate non-graph bring-up states.

4. **Build and toolchain assumptions**
   - Linux x86_64.
   - Toolchain, driver floor, and profiler versions are imported from the master pin manifest rather than restated locally. Unless that manifest is superseded, the frozen production lane is CUDA **12.9.1** with driver **>=595.45.04**, `sm_120` + PTX, and NVTX3; CUDA 13.2 remains experimental-only until promoted by that manifest. CUDA 12.8 Blackwell compatibility guidance remains the feature-floor background for this phase [R7][R8].
   - Nsight Systems, Nsight Compute, and Compute Sanitizer are installed and callable from scripts.
   - CUDA translation units can be rebuilt with source-correlation flags (`-lineinfo`) [R12][R15].

5. **Case assumptions**
   - `R0`: representative full nozzle case.
   - `R1`: reduced nozzle case for development/profiling.
   - `R1-core`: Phase-5-friendly reduced case using only the frozen generic BC/scheme subset when backend or execution-mode coverage must be exercised independently of Phase 6 nozzle specifics.
   - `R2`: generic two-phase verification case.
   - All cases used by the active acceptance manifest are numerically validated against the CPU/SPUMA reference before Phase 8 acceptance uses them.

6. **Output and I/O assumptions**
   - Output staging may legally move data to host, but only in explicit write/output phases.
   - Baseline steady-state timeline captures should exclude write phases whenever possible.
   - If a write phase must appear in a capture, it must be tagged distinctly and excluded from residency acceptance of solver-inner-loop traffic.

7. **Coding-agent constraints**
   - The coding agent must be able to implement this phase without having to infer profiler semantics from scattered documentation.
   - All scripts must fail loudly when required tool features are unavailable.
   - All thresholds must be configurable, but defaults must be provided.

8. **Non-assumptions**
   - Do **not** assume AmgX or any external solver path will automatically dominate time.
   - Do **not** assume sparse linear algebra is the only performance limiter.
   - Do **not** assume that a lack of crashes implies a valid GPU execution path.

---

## 4. Cross-cutting risks and mitigation

### Risk 1 — Silent host/device migration paths
SPUMA explicitly warns that unsupported features may degrade into unwanted CPU/GPU copies rather than failing hard [R2]. This is especially dangerous in OpenFOAM-style code because host-side object access can be triggered indirectly through runtime-selected components, patch logic, temporary fields, or diagnostics.

**Mitigation**
- Mandatory NVTX coverage of all major solver stages.
- Dedicated UVM audit mode with page-fault tracing.
- Acceptance gates that fail on unexpected steady-state `HtoD`, `DtoH`, CPU faults, or GPU faults.
- Separate output staging ranges so expected host traffic is not mistaken for solver traffic.

### Risk 2 — Profilers perturb the runtime
Nsight Systems UVM page-fault tracing and node-level graph tracing can add substantial overhead [R9][R19]. Compute Sanitizer slows applications dramatically and can trigger timeout-related behavior [R14][R15].

**Mitigation**
- Separate low-overhead baseline mode from diagnostic modes.
- Never compare wall-clock performance across different profiler modes.
- Use reduced cases for sanitizer.
- Keep capture windows short and steady-state only.

### Risk 3 — Graph tracing behavior is version-sensitive
NVIDIA’s release notes historically exposed `cuda-hw` as a Blackwell-oriented mode, while current documentation states that `--trace=cuda` uses the Hardware Event System on Blackwell and falls back to `cuda-sw` with diagnostics if required [R9][R19]. This is a version-evolution issue, not a contradiction in principle.

**Mitigation**
- Treat current Nsight Systems documentation as source of truth.
- Record actual tool version in every profiling artifact.
- Parse profiler diagnostics and flag unexpected fallback behavior.

### Risk 4 — Missing source correlation
Without `-lineinfo`, Nsight Compute and Compute Sanitizer source mapping is weaker [R12][R15]. This makes root-cause analysis of indirect, templated, or generated kernels much slower.

**Mitigation**
- Require `-lineinfo` in all CUDA translation units for profiling/sanitizer builds.
- Keep a separate high-optimization profiling build and a sanitizer build if necessary.

### Risk 5 — Runtime-selection instrumentation gaps
OpenFOAM and SPUMA-style code can enter hot paths through runtime-selected solvers, patch fields, and external solver wrappers. Instrumenting only the base class or only `foamRun` is insufficient.

**Mitigation**
- Instrument at the actual derived solver and wrapper call sites used in the nozzle path.
- Add coverage validation that fails if required NVTX ranges are missing in captured timesteps.

### Risk 6 — Over-instrumentation creates clutter and noise
NVIDIA advises caution when annotating code that usually executes in under 1 microsecond, and warns against leaving many ranges open [R11].

**Mitigation**
- Instrument solver stages, not every micro-kernel.
- Use RAII-scoped ranges for host orchestration boundaries.
- Keep domain/category naming fixed and limited.

### Risk 7 — Raw SQLite schema fragility
Nsight Systems’ `.nsys-rep` and SQLite outputs can evolve over versions; custom SQL parsers become brittle [R9][R20].

**Mitigation**
- Prefer `nsys stats` built-in reports first.
- Only use SQL for data not obtainable from built-in reports.
- Guard all SQL by schema checks and tool-version manifests.

### Risk 8 — Output staging contaminates residency gates
If a captured timestep includes field writes, explicit DtoH copies may appear and falsely trigger solver-residency alarms.

**Mitigation**
- Default acceptance capture excludes write timesteps.
- If a write timestep must be captured, classify DtoH inside `write_stage` separately and do not count it against solver-inner-loop residency.

### Risk 9 — Toolchain mismatch on Blackwell
Blackwell support begins with CUDA 12.8-era tooling [R7][R8]. Mismatched drivers, incompatible toolkit versions, or missing PTX/cubin support can produce misleading failures.

**Mitigation**
- Record driver, toolkit, GPU, and profiler version in every acceptance artifact.
- Include a preflight job with `CUDA_FORCE_PTX_JIT=1` to verify PTX readiness [R7].

### Risk 10 — Acceptance metrics become too rigid too early
Kernel-level secondary metrics such as stall reasons or occupancy vary with tool versions, compiler changes, and graph structure. Hard-failing on those too early can create false regressions.

**Mitigation**
- Use absolute pass/fail on residency, synchronization, coverage, and sanitizer cleanliness.
- Use kernel secondary metrics initially as **diagnostic baselines**, not hard fail conditions.
- Introduce stricter secondary thresholds only after baseline stabilization.

---

## 5. Phase-by-phase implementation specification

## Phase 8 — Profiling and performance acceptance

### Purpose

Build a **production-grade profiling, tracing, and performance-acceptance framework** for the SPUMA/OpenFOAM pressure-swirl nozzle GPU port, so that solver correctness/performance status is machine-verifiable on the RTX 5080 and regressions are detected before further optimization work proceeds.

### Why this phase exists

This phase exists because the project’s success criteria depend on more than “the solver runs on the GPU.” SPUMA’s own measurements show that real OpenFOAM GPU paths can be dominated by **kernel launch volume** and **synchronization overhead**, while weaker kernels occur in atomic-heavy and multigrid sections [R1]. SPUMA’s wiki also warns that unsupported features may trigger slow host/device transfers instead of hard errors [R2]. In a nozzle solver with repeated **PIMPLE outer loops**, **pressure correction**, **alpha prediction**, **MULES limiting**, **surface-tension work**, **boundary updates**, and **subcycling**, performance failures are often architectural, not local. They must be detected in the whole timestep timeline.

The phase therefore creates the operational safety net for the entire port:

- It proves that the code is actually device-resident in steady-state.
- It identifies whether launch/sync overhead is still a primary bottleneck.
- It points to the top kernels that justify custom CUDA work in later phases.
- It prevents the team from optimizing on misleading traces contaminated by initialization, writes, or diagnostic instrumentation.

Phase 8 is the sole owner of formal performance acceptance. The centralized acceptance manifest, support matrix, and graph support matrix own the supported-case scope, tolerance classes, backend comparison rules, required benchmark tuples, and allowed fallback policy; this document defines the instrumentation, capture workflows, artifact layout, and parser logic that enforce those imported decisions on real RTX 5080 hardware.

### Entry criteria

Phase 8 may start only when all of the following are true:

1. A runnable single-GPU nozzle solver path exists on the SPUMA base.
2. `R1` runs to completion on the RTX 5080 without requiring debugger intervention.
3. CPU/SPUMA reference validation artifacts exist for every case/backend tuple to be used in formal Phase 8 acceptance (`R1-core`, `R1`, `R0` as applicable), and acceptable field/QoI/restart/backend-parity deltas are documented in the centralized acceptance manifest.
4. The build system can compile CUDA translation units with profiling-friendly flags (`-lineinfo` at minimum).
5. NVTX v3 headers are available in the toolchain.
6. Nsight Systems CLI, Nsight Compute CLI, and Compute Sanitizer are installed on the development machine.
7. The coding agent has identified the actual runtime-selected solver class and hot-path methods used by the nozzle solver (not just the abstract base class).
8. A decision exists on whether the active pressure path is native SPUMA-only, AmgX-backed, or both, because the instrumentation must wrap the actual dispatch path.
9. The master pin manifest, centralized support matrix, graph support matrix, and centralized acceptance manifest are available to the harness and identify the exact case/backend/mode tuples in scope for this Phase 8 run.
10. If an external pressure backend is in scope, Phase 4 `DeviceDirect` handoff is complete for that backend; `PinnedHost` staging may still be profiled only as bring-up/correctness mode and is not eligible for production acceptance.

### Exit criteria

Phase 8 exits successfully only when all of the following are true:

1. **Instrumentation coverage**
   - All mandatory orchestration ranges appear in captured steady-state timesteps:
     - `solver/timeStep`
     - `solver/pimpleOuter`
     - `solver/steady_state`
   - All mandatory stage-level parent ranges also appear with the exact stage IDs frozen in `graph_capture_support_matrix.md` / `graph_capture_support_matrix.json`:
     - `pre_solve`
     - `outer_iter_body`
     - `alpha_pre`
     - `alpha_subcycle_body`
     - `mixture_update`
     - `momentum_predictor`
     - `pressure_assembly`
     - exactly one of `pressure_solve_native` or `pressure_solve_amgx` for the active backend
     - `pressure_post`
     - `nozzle_bc_update` whenever the Phase 6 nozzle-boundary path is active in the accepted tuple
     - `write_stage` only when a write timestep is intentionally included in the capture
   - Automation must derive the exact accepted parent-range set from `acceptance_manifest.json`: orchestration ranges come from `nvtx_contract_defaults.required_orchestration_ranges`, and each admitted tuple carries its own `required_stage_ids`. The bullets above are the human-readable summary of that machine-readable contract.
   - Finer-grained child ranges may be emitted for diagnosis, but they are optional nested diagnostics and are never acceptance keys. If present, they must nest beneath the canonical parent range that owns the work.

2. **Residency acceptance**
   - In production baseline mode, there are **no unexpected host/device transfers** inside steady-state solver ranges.
   - In UVM audit mode, the acceptance script reports:
     - zero unexpected `HtoD` migration bytes,
     - zero unexpected `DtoH` migration bytes,
     - zero CPU page faults,
     - zero GPU page faults
     inside the steady-state solver window.
   - If output staging is captured, any transfer bytes under `write_stage` are classified separately and excluded from solver-inner-loop residency gating.
   - Production acceptance additionally requires zero observed or counted:
     - host patch evaluation or CPU boundary fallback inside the timestep loop,
     - pinned-host pressure staging in any accepted production run,
     - host `setFields` or equivalent host startup field materialization in the accepted startup path,
     - unsafe functionObject field commits or other field-scale host commits inside timed steady-state windows.
   - Allowed CPU pre-run utilities (for example steady preconditioning / `potentialFoam`) must complete before the timed transient run and be recorded as pre-run artifacts rather than accepted steady-state work.

3. **Synchronization acceptance**
   - `cudaDeviceSynchronize()` count inside steady-state inner solver ranges is **zero**.
   - `cudaStreamSynchronize()` or equivalent host-blocking synchronization APIs are recorded diagnostically inside inner solver ranges in this manifest revision. They do not change the formal acceptance verdict unless a later manifest revision promotes them to a named gate, but any observed calls must still be attributed and explained in the archived report.
   - No `cudaMalloc*`, `cudaFree*`, or analogous runtime allocation APIs occur after warmup steady-state begins.

4. **Launch-structure acceptance**
   - A machine-readable summary exists of:
     - total kernel launches in the capture,
     - launches per steady-state timestep,
     - graph launches per steady-state timestep,
     - total graph instantiations/updates during the capture.
   - If graph mode is enabled in the solver, graph launches per timestep do not exceed the configured threshold.
   - If graph mode is not yet enabled, the pre-graph launch baseline is recorded and locked for regression comparison.

5. **Kernel deep-dive acceptance**
   - A ranked top-five-kernel list exists for `R1` and `R0`.
   - Nsight Compute reports exist for those kernels and include at minimum:
     - SpeedOfLight / high-level utilization,
     - MemoryWorkloadAnalysis,
     - SchedulerStats,
     - WarpStateStats,
     - Occupancy,
     - LaunchStats,
     - SourceCounters / source correlation.

6. **Sanitizer acceptance**
   - `memcheck`, `initcheck`, and `synccheck` report zero errors on `R1`, `R1-core` where required by the acceptance manifest, or the equivalent reduced microtest set.
   - `racecheck` has been run on shared-memory/barrier-using kernels or kernel microtests, and any remaining hazards are documented and approved.

7. **Correctness-manifest alignment**
   - Every profiled acceptance tuple references a centralized acceptance-manifest entry specifying case ID, backend, execution mode, kernel-family mode, tolerance class, restart/reload parity expectation, and backend-parity expectation.
   - No run may be reported as a passing Phase 8 production acceptance unless the corresponding numerical/QoI/restart/backend-parity verdict bundle exists and passes under that manifest entry.

8. **Automation and artifact acceptance**
   - Profiling and post-processing are runnable from scripts without manual GUI steps.
   - The acceptance script emits a single JSON or Markdown summary with pass/fail results, manifest hashes/revisions, and links to raw artifacts.
   - The artifact bundle includes benchmark matrix CSV/JSON, GPU-purity report, graph report, fallback counters, and the environment/version-pin bundle alongside raw profiler outputs.

### Required backend coverage

Formal Phase 8 acceptance consumes, rather than redefines, the benchmark matrix from the centralized acceptance manifest.

**Mandatory baseline coverage**
- Native SPUMA/OpenFOAM pressure path is always required for production acceptance on the validation-ladder cases marked in scope by the acceptance manifest.
- `async_no_graph` and `graph_fixed` are the execution-mode labels consumed from the centralized graph support matrix; at least one accepted non-graph baseline must remain available for regression isolation.
- Generic/fallback kernels and accepted Phase 7 custom kernels are benchmarked only for hotspot families present in the frozen support matrix and interim hotspot ranking.

**Conditional coverage**
- AmgX or any other external pressure backend is benchmarked only if:
  1. the centralized acceptance manifest marks it in scope, and
  2. the Phase 4 `DeviceDirect` bridge exists for the active case/backend tuple.
- `PinnedHost` pressure staging may be profiled only as bring-up/correctness mode. It cannot satisfy any Phase 8 production claim that says “no field-scale host transfer.”
- CPU boundary fallback, host patch execution, or host `setFields` startup are debug-only/bring-up-only modes. Any run that uses them is excluded from production timing and production acceptance even if the run is otherwise numerically useful.

**Case mapping**
- `R1-core` is the default reduced case for backend parity, execution-mode parity, and generic-kernel-versus-custom-kernel comparisons when nozzle-specific Phase 6 BCs are not required.
- `R1` is the reduced nozzle acceptance case for routine architectural baselines.
- `R0` is the representative production-shape acceptance case.
- Write-cadence variants are included only if the centralized acceptance manifest marks them in scope; otherwise write timesteps stay outside the timed steady-state window.

**Correctness coupling**
- Each benchmark tuple must carry the corresponding centralized numerical verdicts: field-level tolerances, QoI tolerances, restart/reload parity, and backend-parity expectations.
- Phase 8 does not author those tolerances locally; it fails if the tuple lacks the required verdict bundle or if the verdict bundle is non-passing.

### Goals

1. Create a reusable instrumentation layer for the GPU nozzle solver and its dependencies.
2. Produce repeatable capture workflows for baseline performance, UVM audit, synchronization audit, graph inspection, kernel deep-dive, and sanitizer runs.
3. Separate **required-for-correctness** acceptance gates from **required-for-performance** and **diagnostic-only** outputs.
4. Prevent profiler misuse by encoding correct defaults in scripts.
5. Provide enough data fidelity that the next optimization phase can target the real bottlenecks rather than guesses.
6. Make the framework robust to Nsight version drift by preferring official report scripts and version manifests.
7. Ensure all performance claims used for decision-making are reproducible on real RTX 5080 hardware.

### Non-goals

1. This phase does **not** optimize kernels by itself.
2. This phase does **not** redesign numerics, matrices, or boundary-condition algorithms.
3. This phase does **not** attempt multi-GPU profiling, MPI overlap analysis, or cluster-scale benchmarking.
4. This phase does **not** replace numerical regression testing; it depends on numerical validation from earlier phases.
5. This phase does **not** require all kernels to be individually instrumented with NVTX.
6. This phase does **not** use Nsight Compute as the primary timing tool.
7. This phase does **not** define final production throughput targets for every future mesh size; it defines measurement and gating infrastructure.

**What not to do in this phase**
- Do not capture initialization, mesh load, or dictionary parsing in baseline acceptance traces.
- Do not leave UVM page-fault tracing enabled for baseline timing runs.
- Do not instrument every micro-kernel with NVTX.
- Do not hard-code parser logic against an unversioned SQLite schema when a built-in `nsys stats` report already exists.
- Do not accept “the GUI looked fine” as a valid pass/fail artifact.
- Do not compare profiler-overhead timings across different capture modes.

### Technical background

The nozzle solver execution path follows the OpenFOAM/`foamRun` single-region time loop structure in which the runtime-selected solver executes `preSolve`, advances time, and enters the PIMPLE loop containing `prePredictor`, `momentumPredictor`, `thermophysicalPredictor` (where applicable), `pressureCorrector`, `postCorrector`, then exits to write/postsolve [R3]. For `incompressibleVoF`, additional two-phase semantics include `alphaPredictor()`, cached `rAU`, `surfaceTensionForce()`, mixture property updates, and `nAlphaSubCyclesPtr`-controlled alpha subcycling [R4][R5]. MULES is an explicit multidimensional limiter [R6], so the alpha path is not just “one sparse solve.” It is a sequence of explicit face-based and limiter-heavy operations, potentially repeated multiple times per timestep.

From a performance-analysis perspective, this solver family must be treated using a **three-ceiling model**:

1. **Sparse algebra ceiling**  
   Pressure and momentum solves, smoothers, and matrix-vector operations are often bounded by memory movement and sparse-format behavior, as widely documented in sparse linear algebra literature [R21].

2. **Irregular-kernel ceiling**  
   Gradient, flux, limiter, curvature, boundary, and face-based kernels have indirect addressing, atomics, or reductions. SPUMA’s paper specifically reports poorer efficiency for atomic-heavy gradient/RHS work and lower efficiency in GAMG pressure paths [R1].

3. **Launch/synchronization ceiling**  
   SPUMA’s profile of a representative OpenFOAM case shows over fifty thousand kernel launches in five SIMPLE iterations, with `cudaDeviceSynchronize()` dominating CUDA API time in the profiled path [R1]. In a transient nozzle run with subcycling, this can become a first-order limiter.

Nsight Systems is the correct whole-application tool for detecting ceilings (2) and (3), and for checking whether ceiling (1) is even the dominant issue in the actual steady-state timestep. Nsight Compute then explains why the top kernels behave the way they do. Compute Sanitizer provides functional validation for memory, initialization, and synchronization errors before performance claims are trusted [R14][R15].

### Research findings relevant to this phase

1. **SPUMA launch/sync pathology is real, not hypothetical.**  
   The SPUMA paper reports **53,259 kernel launches** in five SIMPLE iterations for the profiled DrivAer22M study, and states that every kernel invocation in the profiled path was followed by `cudaDeviceSynchronize()`; synchronization dominated CUDA API time [R1].

2. **SPUMA’s weaker kernels align with nozzle-solver concerns.**  
   SPUMA reports lower computational efficiency in atomic-heavy gradient/RHS kernels and in pressure/GAMG paths, which is consistent with the nozzle solver’s expected hotspots in limiter, face, and pressure-correction work [R1].

3. **SPUMA warns about silent host/device fallbacks.**  
   The SPUMA GPU-support wiki explicitly says unsupported features may not fatal-error but can cause undesired CPU/GPU copies, and recommends Nsight Systems with CUDA+NVTX tracing and UVM fault tracing to diagnose them [R2].

4. **Nsight Systems provides the required report surface.**  
   Current NVIDIA documentation confirms built-in `nsys stats` reports including `cuda_api_sum`, `cuda_gpu_sum`, `cuda_gpu_kern_sum`, `cuda_kern_exec_sum`, `nvtx_sum`, `nvtx_kern_sum`, `um_sum`, `um_total_sum`, and `um_cpu_page_faults_sum` [R9][R10].

5. **Capture-range filtering is supported and should be used.**  
   Nsight Systems supports `--capture-range=nvtx` and `--nvtx-capture` to collect only selected ranges [R9].

6. **UVM page-fault tracing is useful but expensive.**  
   NVIDIA documents that enabling CPU/GPU UVM fault tracing can significantly increase overhead, with measured cases up to about 70% [R9].

7. **Graph tracing has a recommended low-overhead mode.**  
   Current Nsight Systems documentation supports `--cuda-graph-trace=graph|node` and states that graph-level tracing has significantly less overhead, while node-level tracing can be expensive and reduce parallelism [R9][R19].

8. **Blackwell trace behavior changed across Nsight versions.**  
   Nsight Systems 2025.2 release notes described a beta `cuda-hw` mode for Blackwell, while current documentation states that `--trace=cuda` uses the Hardware Event System on Blackwell and falls back to `cuda-sw` if needed [R9][R19]. The safe interpretation is that tooling evolved; current docs should drive implementation.

9. **NVTX v3 is the supported direction.**  
   Current Nsight Systems documentation supports NVTX v3 domains, categories, ranges, marks, and recommends RAII wrappers; it also warns against annotating code that generally takes under 1 microsecond and against leaving many ranges open [R11].

10. **Nsight Compute requires line info for strong source correlation.**  
    NVIDIA documents that source correlation and imported source in Nsight Compute rely on line information (`-lineinfo`) [R12][R13].

11. **Nsight Compute’s diagnostic value is targeted, not holistic.**  
    The official profiling guide documents memory-workload, scheduler, warp-state, speed-of-light, and roofline analyses as kernel-level tools [R12].

12. **Compute Sanitizer has a recommended usage order and significant runtime overhead.**  
    NVIDIA recommends running `memcheck` first, then `initcheck`, `racecheck`, and `synccheck`, and notes substantial slowdown and limitations in some modes [R14][R15].

13. **OpenFOAM’s call structure provides natural instrumentation boundaries.**  
    `foamRun` and `incompressibleVoF` expose stable semantic stages (`preSolve`, `momentumPredictor`, `pressureCorrector`, `alphaPredictor`, `surfaceTensionForce`, etc.) that map naturally to NVTX ranges [R3][R4][R5].

14. **Blackwell-native profiling matters for this hardware.**  
    CUDA 12.8 added Blackwell compatibility/compiler support, and NVIDIA’s RTX 5080 official specifications identify compute capability 12.0 hardware with 960 GB/s memory bandwidth [R7][R8][R17].

### Design decisions

#### DD-8.1 — Introduce six explicit profiling modes

**Modes**
1. `baselineTimeline`
2. `uvmAudit`
3. `syncAudit`
4. `graphDebug`
5. `kernelDeepDive`
6. `sanitizer`

- **Sourced fact:** Nsight Systems supports the trace/report features required for timeline/UVM/graph analysis; Nsight Compute supports targeted kernel analysis; Compute Sanitizer supports memory/sync checks [R9][R12][R14].
- **Engineering inference:** One profiler configuration cannot satisfy all goals without either excessive overhead or missing required data.
- **Recommendation:** Implement mode-specific scripts and defaults. Never multiplex these concerns into one monolithic profile run.

#### DD-8.2 — Use NVTX v3 domains and categories with a fixed naming grammar

**Mandatory domains**
- `spuma.solver`
- `spuma.alpha`
- `spuma.momentum`
- `spuma.pressure`
- `spuma.surfaceTension`
- `spuma.boundary`
- `spuma.graph`
- `spuma.memory`
- `spuma.output`
- `spuma.diagnostics`

**Mandatory category identifiers**
- `timeStep`
- `steadyState`
- `pimpleOuter`
- `alphaPredictor`
- `alphaFluxAssembly`
- `mulesLimit`
- `alphaCorrection`
- `mixtureUpdate`
- `uAssembly`
- `uSolve`
- `pAssembly`
- `pSolve`
- `surfaceTension`
- `patchUpdate`
- `graphInstantiate`
- `graphUpdate`
- `graphLaunch`
- `writeStage`
- `syncBoundary`
- `uvmSuspect`

- **Sourced fact:** Nsight Systems supports NVTX domains and categories, and they can be filtered in the GUI and CLI [R11].
- **Engineering inference:** Without fixed names and domains, automated parsing of `nvtx_*` reports is unreliable and cross-run comparisons become brittle.
- **Recommendation:** Lock this naming scheme immediately. Changes after baseline lock require human sign-off.

#### DD-8.3 — Use RAII push/pop ranges for host-controlled nested phases; avoid start/end ranges unless crossing threads

- **Sourced fact:** NVTX supports both push/pop and start/end ranges; RAII wrappers are explicitly recommended in NVIDIA documentation [R11].
- **Engineering inference:** The nozzle solver’s stage hierarchy is mostly single-threaded host orchestration around asynchronous GPU work; RAII push/pop fits naturally and minimizes range-leak bugs.
- **Recommendation:** Default to push/pop RAII for all solver stage scopes. Reserve start/end ranges for rare cross-thread or asynchronous host workflows.

#### DD-8.4 — Profile only a warm steady-state window and keep write phases out of baseline runs

- **Sourced fact:** Capture-range filtering via NVTX is supported [R9].
- **Engineering inference:** Initialization and writes will generate transfers and one-time setup work that are irrelevant to steady-state solver acceptance.
- **Recommendation:** Add explicit `solver/steady_state` NVTX capture markers and configure the harness so write events do not occur during baseline acceptance traces. If that is not possible, classify write ranges separately.

#### DD-8.5 — Baseline Nsight Systems command uses `--trace=cuda,nvtx,osrt` and graph-level graph tracing

- **Sourced fact:** Current Nsight Systems docs use `--trace=cuda` for CUDA tracing, and on Blackwell that path uses the Hardware Event System when supported [R9]. Graph-level tracing is lower overhead than node-level [R9].
- **Engineering inference:** This is the minimum useful trace surface for solver-level acceptance without excessive perturbation.
- **Recommendation:** Make this the default baseline capture. Do not add UVM page-fault tracing, CPU backtraces, or node-level graph tracing to the baseline mode.

#### DD-8.6 — `--cuda-event-trace=false` by default in baseline mode

- **Sourced fact:** Nsight Systems documents CUDA event completion tracing but notes limitations when events are used in CUDA Graphs or IPC and warns it can create false dependencies across streams [R9].
- **Engineering inference:** The nozzle solver is expected to use graphs and a small number of streams; event tracing is more likely to confuse than help in baseline runs.
- **Recommendation:** Leave CUDA event tracing off by default. Enable it only in a dedicated event-orchestration debug workflow if needed.

#### DD-8.7 — UVM page-fault tracing enabled only in `uvmAudit` mode

- **Sourced fact:** UVM page-fault tracing is available, but can significantly increase overhead [R9].
- **Engineering inference:** The data is necessary to diagnose residency failures but not valid for baseline performance acceptance.
- **Recommendation:** `uvmAudit` mode should capture only one short steady-state window and should not be used for throughput comparisons.

#### DD-8.8 — Use `nsys stats` built-in reports as the primary parser surface

- **Sourced fact:** NVIDIA documents the built-in report scripts and notes that they are maintained to adapt to internal/SQLite changes [R9][R20].
- **Engineering inference:** Built-in reports reduce maintenance burden and version fragility.
- **Recommendation:** Use `nsys stats` reports first; use direct SQL only for metrics not otherwise available.

#### DD-8.9 — Define pass/fail gates in tiers

**Tier A — Hard fail**
- Missing mandatory NVTX ranges
- Unexpected steady-state UVM bytes or UVM page faults
- `cudaDeviceSynchronize()` inside inner solver ranges
- Post-warmup dynamic allocation APIs in steady-state
- Sanitizer errors in required tools

**Tier B — Soft fail / requires review**
- Increased graph launches per timestep
- Increased kernel launches per timestep
- New synchronization APIs at graph boundaries
- Top-kernel time regressions > configured threshold

**Tier C — Diagnostic-only**
- Occupancy
- Warp stall mix
- `% Peak to L2`
- sectors/request ratios
- GPU metrics sampling timelines

- **Sourced fact:** Nsight Systems and Nsight Compute provide data for all of these categories, but not all are equally stable as pass/fail gates [R9][R12].
- **Engineering inference:** Residency, synchronization, and coverage are architectural invariants. Stall reasons and occupancy are more sensitive to compiler/tool variation.
- **Recommendation:** Enforce Tier A immediately, Tier B after baseline lock, and Tier C only as analysis context.

#### DD-8.10 — Use top-five-kernel Nsight Compute sections only

**Required NCU sections**
- `SpeedOfLight`
- `MemoryWorkloadAnalysis`
- `SchedulerStats`
- `WarpStateStats`
- `Occupancy`
- `LaunchStats`
- `SourceCounters`

- **Sourced fact:** Nsight Compute documents these sections as the relevant kernel-analysis views for utilization, memory behavior, scheduler behavior, warp stalls, occupancy, launch properties, and source metrics [R12][R13].
- **Engineering inference:** These cover the main likely bottlenecks in sparse, indirect, limiter-heavy CFD kernels without the data explosion of a full default set.
- **Recommendation:** Ship this exact section set in a text file consumed by the NCU wrapper script. Allow per-branch expansion only under explicit review.

#### DD-8.11 — Use CPU sampling/backtraces only in `syncAudit`

- **Sourced fact:** Nsight Systems can collect CUDA backtraces and host sampling/backtraces, and expert rules use OS runtime blocking and other context for sync analysis [R9][R10].
- **Engineering inference:** Sampling/backtrace overhead is unnecessary in most runs but useful when a remaining synchronization call must be traced to host code.
- **Recommendation:** Only `syncAudit` mode should enable CPU sampling/backtrace options.

#### DD-8.12 — Record full environment metadata in every acceptance artifact

**Required metadata**
- Git commit SHA
- branch name
- compiler version
- CUDA toolkit version
- driver version
- Nsight Systems version
- Nsight Compute version
- Compute Sanitizer version
- GPU name
- GPU compute capability
- case ID
- solver ID
- profiling mode
- capture script version

- **Sourced fact:** Tool behavior is version-sensitive across CUDA/Nsight releases [R7][R8][R9][R19].
- **Engineering inference:** Performance artifacts without environment manifests are not reliable enough for regression comparison.
- **Recommendation:** Emit this metadata automatically into every JSON summary and profile output directory.

#### DD-8.13 — Fallback / rollback policy

- If the parser cannot extract a required metric from built-in `nsys stats`, it may fall back to version-guarded SQLite parsing.
- If graph tracing causes unacceptable overhead or incomplete traces in a particular tool version, baseline acceptance reverts temporarily to non-graph launch-count baselining while graph-debug traces are investigated.
- If compile-time memcheck instrumentation is adopted and produces incompatibilities (e.g., HMM limitations or launch-bounds pressure), revert to runtime `compute-sanitizer` for formal acceptance until the compiler/toolkit issue is resolved [R15].
- If NVTX coverage causes measurable solver perturbation due to over-annotation, remove micro-range annotations first; do not remove stage-level coverage.
- If a profiled run uses `PinnedHost` pressure staging, CPU boundary fallback, host `setFields` startup, or unsafe functionObject commits inside the timed window, the harness must classify the run as debug-only/bring-up-only, fail production acceptance, and roll benchmark comparison back to the last production-eligible fallback path.

### Alternatives considered

#### Alternative A — Manual Nsight GUI inspection only
Rejected.  
It does not produce stable machine-readable artifacts, does not scale to regressions, and is too dependent on analyst judgment.

#### Alternative B — Nsight Compute as the primary acceptance tool
Rejected.  
Nsight Compute is a kernel tool, not a whole-solver residency/synchronization/graph/UVM acceptance framework [R12].

#### Alternative C — Always-on UVM page-fault tracing
Rejected.  
The overhead is explicitly documented as potentially substantial [R9].

#### Alternative D — Always-on node-level graph tracing
Rejected.  
NVIDIA documents higher overhead and possible parallelism reduction [R9][R19].

#### Alternative E — Direct custom CUPTI collector instead of Nsight tools
Deferred.  
This could eventually produce lower-overhead custom telemetry, but Phase 8 needs a robust, documented path first.

#### Alternative F — Raw SQLite-only parser
Rejected as primary path.  
Built-in report scripts are more stable across Nsight versions [R20].

#### Alternative G — No in-code NVTX; infer phases only from kernel names
Rejected.  
Kernel names alone do not map reliably to OpenFOAM/SPUMA semantic stages, especially through templates, fusion, graphing, and runtime-selected wrappers.

#### Alternative H — Profile every kernel with NCU on every acceptance run
Rejected.  
Too slow, too intrusive, and operationally unnecessary.

### Interfaces and dependencies

#### 1. C++ instrumentation API

Create a small profiling library under `src/gpu/profiling/` with the following public interface.

```cpp
// GpuProfilingConfig.H
namespace Foam::gpu::profiling
{
    enum class ProfileMode : uint8_t
    {
        off = 0,
        baselineTimeline,
        uvmAudit,
        syncAudit,
        graphDebug,
        kernelDeepDive,
        sanitizer
    };

    struct GpuProfilingConfig
    {
        bool enabled;
        ProfileMode mode;
        bool enableNvtx;
        bool enableGraphMarkers;
        bool enableOutputMarkers;
        bool annotatePatchUpdates;
        bool annotateExternalSolver;
        int warmupTimeSteps;
        int captureTimeSteps;
        int topKernelCount;
        std::string captureDomain;      // e.g. "spuma.solver"
        std::string captureRangeName;   // e.g. "solver/steady_state"
        std::string caseId;
        std::string solverId;
    };

    struct GpuAcceptanceConfig
    {
        std::string acceptedTupleId;
        std::string acceptanceManifest;
        std::string supportMatrix;
        std::string graphSupportMatrix;
        std::string masterPinManifest;
    };

    struct GpuRuntimeConfig
    {
        GpuProfilingConfig profiling;
        GpuAcceptanceConfig acceptance;
    };

    GpuRuntimeConfig readConfig(const Foam::Time& runTime);
}
```

```cpp
// NvtxDomains.H
namespace Foam::gpu::profiling
{
    enum class DomainId : uint8_t
    {
        solver,
        alpha,
        momentum,
        pressure,
        surfaceTension,
        boundary,
        graph,
        memory,
        output,
        diagnostics
    };

    enum class CategoryId : uint16_t
    {
        timeStep,
        steadyState,
        pimpleOuter,
        alphaPredictor,
        alphaFluxAssembly,
        mulesLimit,
        alphaCorrection,
        mixtureUpdate,
        uAssembly,
        uSolve,
        pAssembly,
        pSolve,
        surfaceTension,
        patchUpdate,
        graphInstantiate,
        graphUpdate,
        graphLaunch,
        writeStage,
        syncBoundary,
        uvmSuspect
    };

    void initializeNvtx(const GpuProfilingConfig& cfg);
    void shutdownNvtx() noexcept;
}
```

```cpp
// ScopedNvtxRange.H
namespace Foam::gpu::profiling
{
    class ScopedNvtxRange
    {
    public:
        ScopedNvtxRange
        (
            DomainId domain,
            CategoryId category,
            const char* message
        ) noexcept;

        ScopedNvtxRange
        (
            DomainId domain,
            CategoryId category,
            std::string_view message,
            uint64_t payload
        ) noexcept;

        ~ScopedNvtxRange() noexcept;

        ScopedNvtxRange(const ScopedNvtxRange&) = delete;
        ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
    private:
        bool active_;
    };
}
```

```cpp
// ProfilingMarkers.H
namespace Foam::gpu::profiling
{
    void markSolverMetadata
    (
        const GpuProfilingConfig& cfg,
        int timeIndex,
        int pimpleIter,
        int alphaSubCycle
    ) noexcept;

    void markSteadyStateBegin(int timeIndex) noexcept;
    void markSteadyStateEnd(int timeIndex) noexcept;
    void markGraphInstantiate(const char* tag) noexcept;
    void markGraphUpdate(const char* tag) noexcept;
    void markGraphLaunch(const char* tag) noexcept;
}
```

**Macro layer**

```cpp
#if defined(FOAM_USE_NVTX3)
    #define SPUMA_PROFILE_SCOPE(domain, category, msg) \
        ::Foam::gpu::profiling::ScopedNvtxRange \
        _spuma_nvtx_scope_##__LINE__ \
        (domain, category, msg)

    #define SPUMA_PROFILE_SCOPE_PAYLOAD(domain, category, msg, payload) \
        ::Foam::gpu::profiling::ScopedNvtxRange \
        _spuma_nvtx_scope_##__LINE__ \
        (domain, category, msg, payload)
#else
    #define SPUMA_PROFILE_SCOPE(domain, category, msg) do {} while(0)
    #define SPUMA_PROFILE_SCOPE_PAYLOAD(domain, category, msg, payload) do {} while(0)
#endif
```

#### 2. Runtime configuration interface

Canonical runtime ownership is `system/gpuRuntimeDict`, with Phase 8 consuming `gpuRuntime.profiling` and `gpuRuntime.acceptance`. `system/gpuProfilingDict` may remain as a compatibility shim or generated subview during migration; the canonical form below shows the profiling subview and the acceptance namespace boundary that must be available in either form.

```foam
gpuRuntime
{
    profiling
    {
        enabled                 true;
        mode                    baselineTimeline;
        enableNvtx              true;
        enableGraphMarkers      true;
        enableOutputMarkers     true;
        annotatePatchUpdates    true;
        annotateExternalSolver  true;

        warmupTimeSteps         3;
        captureTimeSteps        5;
        topKernelCount          5;

        captureDomain           "spuma.solver";
        captureRangeName        "solver/steady_state";

        caseId                  "R1";
        solverId                "deviceIncompressibleVoF";
    }

    acceptance
    {
        acceptedTupleId         "P8_R1_NATIVE_GRAPH_BASELINE";
        acceptanceManifest      "acceptance_manifest.json";
        supportMatrix           "support_matrix.json";
        graphSupportMatrix      "graph_capture_support_matrix.json";
        masterPinManifest       "master_pin_manifest.md";

        // Runtime-local selectors or reporting toggles may live here, but
        // hard/soft-gate IDs and thresholds remain imported verbatim from
        // acceptance_manifest.md / acceptance_manifest.json.
    }
}
```

Compatibility-shim form only, if the branch still accepts it:

```foam
gpuProfiling
{
    enabled                 true;
    mode                    baselineTimeline;
    enableNvtx              true;
    enableGraphMarkers      true;
    enableOutputMarkers     true;
    annotatePatchUpdates    true;
    annotateExternalSolver  true;

    warmupTimeSteps         3;
    captureTimeSteps        5;
    topKernelCount          5;

    captureDomain           "spuma.solver";
    captureRangeName        "solver/steady_state";

    caseId                  "R1";
    solverId                "deviceIncompressibleVoF";
}

acceptance
{
    acceptedTupleId         "P8_R1_NATIVE_GRAPH_BASELINE";
    acceptanceManifest      "acceptance_manifest.json";
    supportMatrix           "support_matrix.json";
    graphSupportMatrix      "graph_capture_support_matrix.json";
    masterPinManifest       "master_pin_manifest.md";
}
```

**Default behavior**
- Missing `gpuRuntime.profiling` / compatibility shim → instrumentation compiled in but disabled.
- Missing key → use documented default.
- Unknown mode → fatal error at startup.
- If both `gpuRuntime.profiling` and `gpuProfilingDict` exist, the normalized `gpuRuntime` tree is authoritative; mismatches between the profiling or acceptance views are fatal.
- Local hard/soft-gate overrides are unsupported. If a branch still exposes keys such as `strictUvmZero`, `strictSyncZero`, `strictPostWarmupNoAlloc`, or local acceptance thresholds, config parsing must fail until those knobs are removed.
- If `enabled=true` but NVTX runtime unavailable → fatal error with actionable message.

#### 3. Scripted profiler interface

Provide three entry-point scripts:

1. `run_nsys_profile.sh`
2. `run_ncu_topkernels.py`
3. `run_compute_sanitizer.sh`

And two post-processors:

4. `export_nsys_stats.py`
5. `accept_profile.py`

#### 4. Artifact schema

Every profile run emits a directory:

```text
profiles/
  <date>_<case>_<mode>_<gitsha>/
    env.json
    command.txt
    run.log
    report.nsys-rep
    report.sqlite
    nsys_stats/
      cuda_api_sum.csv
      cuda_gpu_sum.csv
      cuda_gpu_kern_sum.csv
      cuda_kern_exec_sum.csv
      nvtx_sum.csv
      nvtx_gpu_proj_sum.csv
      um_sum.csv
      um_total_sum.csv
      um_cpu_page_faults_sum.csv
      osrt_sum.csv
      analyze.txt
    top_kernels.csv
    benchmark_matrix.csv
    benchmark_matrix.json
    gpu_purity_report.json
    graph_report.json
    fallback_counters.json
    manifest_refs.json
    acceptance_summary.json
    acceptance_summary.md
    ncu/
      kernel_01.ncu-rep
      kernel_01.txt
      ...
    sanitizer/
      memcheck.log
      initcheck.log
      racecheck.log
      synccheck.log
```

`acceptance_summary.json` must contain:

```json
{
  "git_sha": "abc1234",
  "case_id": "R1",
  "solver_id": "deviceIncompressibleVoF",
  "mode": "baselineTimeline",
  "tuple_id": "P8_R1_NATIVE_GRAPH_BASELINE",
  "gpu": {
    "name": "NVIDIA GeForce RTX 5080",
    "compute_capability": "12.0",
    "driver_version": "595.45.04+",
    "toolkit_version": "12.9.1"
  },
  "metrics": {
    "steady_state_steps_captured": 5,
    "kernel_launches_total": 12345,
    "kernel_launches_per_step": 2469.0,
    "graph_launches_per_step": 2.0,
    "unexpected_htod_bytes": 0,
    "unexpected_dtoh_bytes": 0,
    "cpu_um_faults": 0,
    "gpu_um_faults": 0,
    "cudaDeviceSynchronize_calls": 0,
    "cudaStreamSynchronize_calls": 0,
    "post_warmup_alloc_calls": 0,
    "mandatory_nvtx_ranges_present": true,
    "cpu_boundary_fallback_events": 0,
    "host_patch_execution_events": 0,
    "pinned_host_pressure_stage_events": 0,
    "host_setFields_startup_events": 0,
    "unsafe_functionObject_commit_events": 0,
    "top_kernel_time_regression_pct": 3.4
  },
  "scoped_metrics": {
    "cudaDeviceSynchronize_calls": {
      "steady_state_inner_ranges": 0
    },
    "pinned_host_pressure_stage_events": {
      "production_acceptance_runs": 0
    },
    "host_setFields_startup_events": {
      "accepted_startup_path": 0
    },
    "unsafe_functionObject_commit_events": {
      "timed_steady_state_windows": 0
    }
  },
  "thresholds": {
    "graph_launches_per_step": 4,
    "top_kernel_time_regression_pct": 10.0
  },
  "manifest_refs": {
    "acceptance_manifest": "sha256:...",
    "support_matrix": "sha256:...",
    "graph_capture_support_matrix": "sha256:...",
    "master_pin_manifest": "sha256:..."
  },
  "status": {
    "formal_verdict_emitted": true,
    "hard_fail": false,
    "soft_fail": false,
    "warnings": []
  }
}
```

`acceptance_summary.json` must also record the acceptance-manifest, support-matrix, graph-support-matrix, and pin-manifest revision IDs or hashes used for evaluation, plus zero/nonzero fallback counters for CPU boundary fallback, host patch execution, pinned-host pressure staging, host `setFields` startup, and unsafe functionObject commits. These fields may be embedded directly or emitted via `gpu_purity_report.json`, `fallback_counters.json`, and `manifest_refs.json`, but they are mandatory for any production acceptance bundle.

#### 5. External dependencies

- CUDA toolkit `12.9.1` on the frozen production/acceptance lane imported from `master_pin_manifest.md`; CUDA `13.2` remains verification-only, and the CUDA 12.8 references elsewhere in this spec are Blackwell feature-floor background rather than an alternative acceptance lane [R7][R8]
- Nsight Systems CLI [R9]
- Nsight Compute CLI [R12][R13]
- Compute Sanitizer [R14][R15]
- NVTX v3 header and `-ldl` link dependency [R11]
- Python 3 for post-processing scripts
- `sqlite3` availability optional but recommended for fallback queries

### Data model / memory model

#### 1. Profiling state ownership

- `GpuRuntimeConfig` is owned by the solver/runtime host process; `GpuProfilingConfig` and `GpuAcceptanceConfig` are its profiling/acceptance subviews.
- NVTX domain/category registries are process-lifetime singletons.
- No per-timestep heap allocation is permitted inside profiling wrappers after initialization.
- Profiling wrappers must not allocate device memory.
- Profiling wrappers must not trigger device-to-host field reads.

#### 2. Data flow boundaries

**Allowed host-side profiling data**
- timestep index
- PIMPLE iteration index
- alpha subcycle index
- case name
- solver name
- graph state labels
- patch-group names
- static kernel labels

**Forbidden profiling data in hot paths**
- reading device field values back to host for annotations,
- querying matrix/vector sizes every call if not already available,
- dynamic string formatting that allocates repeatedly in inner loops,
- any host-side inspection of device-resident fields for “debug counters.”

#### 3. Memory-residency acceptance semantics

Define three traffic classes:

1. **Expected solver traffic**  
   Device-only memory operations and kernel activity.  
   This is the desired state.

2. **Expected host traffic**
   - explicit output staging in `write_stage`,
   - one-time startup initialization before `solver/steady_state`,
   - one-time graph instantiation/update before steady-state if unavoidable.

3. **Unexpected host traffic**
   - any HtoD/DtoH migration or page fault inside `solver/steady_state` solver ranges excluding output/write stages,
   - post-warmup `cudaMalloc*`/`cudaFree*`,
   - API-level blocking sync inside inner solver ranges.

#### 4. Acceptance-window semantics

`solver/steady_state` is the canonical acceptance-window range.

**Rules**
- It begins after `warmupTimeSteps`.
- It spans exactly `captureTimeSteps`.
- It must not include first-touch initialization or the first graph instantiation unless the run is explicitly `graphDebug`.
- It should not include write/output unless the test case is specifically `ioProfile`.

#### 5. Baseline storage model

The system stores two baseline classes:

- **Architectural baseline**
  - NVTX coverage map
  - steady-state UVM bytes/faults
  - sync API counts
  - graph launches per timestep
  - allocations after warmup

- **Kernel baseline**
  - top-five kernel names
  - total time per kernel
  - average launch time
  - average execution time
  - NCU summary metrics

The architectural baseline is hard-gated. The kernel baseline is initially review-gated.

### Algorithms and control flow

#### 1. Runtime instrumentation control flow

1. Parse canonical `gpuRuntime.profiling` (and `gpuRuntime.acceptance` for local reporting selectors or manifest references only). If a legacy `gpuProfilingDict` compatibility shim is still accepted in this branch, normalize it into the same runtime view before continuing.
2. Initialize NVTX domains/categories if enabled.
3. Execute normal solver startup.
4. For each timestep:
   1. Open `solver/timeStep` range.
   2. If `timeIndex == warmupTimeSteps`, open `solver/steady_state` range.
   3. Open `solver/pimpleOuter` around each outer corrector.
   4. Instrument alpha, mixture, momentum, pressure, surface tension, patch, graph, and output ranges in the actual runtime-selected solver methods.
   5. If `timeIndex == warmupTimeSteps + captureTimeSteps - 1`, close `solver/steady_state`.
5. Shutdown profiling state during solver teardown.

#### 2. Baseline timeline algorithm

```text
Input: R1 or R0 case, baselineTimeline mode
Output: .nsys-rep, exported stats, acceptance summary

Algorithm:
1. Set writeInterval or harness configuration so writes do not occur during capture window.
2. Run nsys profile with trace = cuda,nvtx,osrt
3. Use graph-level graph tracing.
4. Capture only the NVTX steady_state range.
5. Export built-in stats reports.
6. Parse required metrics.
7. Fail if any hard-gate conditions are violated.
8. Generate summary JSON/Markdown.
```

#### 3. UVM audit algorithm

```text
Input: same case, uvmAudit mode
Output: UVM page-fault and migration summary

Algorithm:
1. Run a short capture with CPU/GPU UVM page-fault tracing enabled.
2. Keep capture window small (typically 1–2 steady-state steps).
3. Export um_sum, um_total_sum, um_cpu_page_faults_sum.
4. Classify migrations by NVTX range containment:
   a. inside steady_state but outside writeStage => unexpected
   b. inside writeStage => expected output traffic
   c. outside steady_state => informational only
5. Fail if unexpected migrations/faults are nonzero in production mode.
```

#### 4. Synchronization audit algorithm

```text
Input: same case, syncAudit mode
Output: synchronization API summary and host-stack context

Algorithm:
1. Run nsys with cuda,nvtx,osrt tracing plus CPU sampling/backtrace options.
2. Export cuda_api_sum and expert-system analysis.
3. Count synchronization API calls:
   - cudaDeviceSynchronize
   - cudaStreamSynchronize
   - blocking memcpys/memsets
4. Attribute each call to enclosing NVTX stage if possible.
5. If a manifest-gated sync metric exists inside its gated scope, mark hard fail. In the current manifest revision that means `cudaDeviceSynchronize_calls`; `cudaStreamSynchronize_calls` stays diagnostic unless a future manifest revision promotes it.
6. Use host backtraces to identify exact call sites.
```

#### 5. Graph debug algorithm

```text
Input: graphDebug mode
Output: graph structure visibility

Algorithm:
1. Run very short nsys capture with --cuda-graph-trace=node.
2. Restrict to one steady-state timestep.
3. Export graph launch counts and inspect node-level sequence.
4. Compare actual graph breakdown against expected solver stage grouping.
5. Use findings to reduce graph fragmentation or misplaced boundaries.
```

#### 6. Kernel deep-dive algorithm

```text
Input: baseline nsys outputs
Output: top-5 kernel NCU reports

Algorithm:
1. Parse cuda_gpu_kern_sum and cuda_kern_exec_sum to rank kernels by total GPU time.
2. Select topKernelCount kernels (default 5).
3. Run NCU for each target using NVTX filtering and/or kernel-name filtering.
4. Collect required sections only.
5. Save .ncu-rep and text summary.
6. Extract secondary metrics:
   - memory throughput balance
   - sectors/request
   - scheduler issue efficiency
   - dominant warp stalls
   - occupancy
7. Store as baseline diagnostics.
```

#### 7. Sanitizer algorithm

```text
Input: R1 reduced case or kernel microtests
Output: sanitizer logs

Algorithm:
1. Run memcheck first.
2. Run initcheck second.
3. Run synccheck third.
4. Run racecheck on reduced kernels/tests where shared memory or barriers are used.
5. Fail if required tools report errors.
6. If compile-time memcheck instrumentation is enabled, verify same-toolkit compiler/sanitizer pairing.
```

#### 8. Acceptance evaluation algorithm

```python
@dataclass
class AcceptanceEvaluationContext:
    tuple_id: str
    is_production_acceptance_run: bool
    accepted_startup_path: bool

def gate_applies(gate: GateSpec, ctx: AcceptanceEvaluationContext) -> bool:
    scope = getattr(gate, "scope", None)
    if scope == "production_acceptance_runs":
        return ctx.is_production_acceptance_run
    if scope == "accepted_startup_path":
        return ctx.accepted_startup_path
    return True

def evaluate_acceptance(
    stats: ProfileStats,
    ctx: AcceptanceEvaluationContext,
    policy: AcceptanceManifestPolicy,
) -> AcceptanceResult:
    result = AcceptanceResult()
    tuple_id = ctx.tuple_id

    if tuple_id not in policy.accepted_tuples:
        result.formal_verdict_emitted = False
        result.diagnostic_note(f"{tuple_id} is outside the accepted tuple matrix")

        if policy.disposition_rules["diagnostic_or_bringup_run_outside_accepted_tuple_matrix"]["may_report_soft_gates"]:
            for gate_id, gate in policy.soft_gates.items():
                value = stats.metric(gate_id, scope=getattr(gate, "scope", None))
                if not compare(value, gate.operator, gate.value):
                    result.soft_note(f"{gate_id} violated diagnostically: expected {gate.operator} {gate.value}")
        return result

    for gate_id, gate in policy.hard_gates.items():
        if not gate_applies(gate, ctx):
            continue

        value = stats.metric(gate_id, scope=getattr(gate, "scope", None))
        if not compare(value, gate.operator, gate.value):
            result.hard_fail(f"{gate_id} violated: expected {gate.operator} {gate.value}")

    for gate_id, gate in policy.soft_gates.items():
        if not gate_applies(gate, ctx):
            continue

        value = stats.metric(gate_id, scope=getattr(gate, "scope", None))
        if not compare(value, gate.operator, gate.value):
            result.soft_fail(f"{gate_id} violated: expected {gate.operator} {gate.value}")

    # Diagnostic-only counters that are not named by the manifest, such as
    # cudaStreamSynchronize_calls in this revision, may still be emitted in
    # reports but never change the formal verdict.

    return result
```

### Required source changes

#### 1. Add a profiling subsystem to the codebase

Create a new library/module: `src/gpu/profiling/`.

Required files:
- `GpuProfilingConfig.H`
- `GpuProfilingConfig.C`
- `NvtxDomains.H`
- `NvtxDomains.C`
- `ScopedNvtxRange.H`
- `ScopedNvtxRange.C`
- `ProfilingMarkers.H`
- `ProfilingMarkers.C`
- `ProfilingEnable.H`
- `ProfilingVersion.C`

#### 2. Instrument the top-level solver execution path

Modify the runtime-selected GPU nozzle solver (or the closest shared base class actually used by that solver) so that the following scopes exist:

- `solver/timeStep`
- `solver/steady_state`
- `solver/pimpleOuter`
- `pre_solve`
- `outer_iter_body`

If the actual control loop still lives in a `foamRun`-adjacent host file, instrument that file too, but do **not** rely on it as the only location. The derived GPU solver methods must also be instrumented.

#### 3. Instrument VOF/two-phase stage entry points

Required stage-level parent range insertions:

| Range | Required call site |
|---|---|
| `alpha_pre` | alpha pre-stage before the first subcycle body or coefficient/setup work |
| `alpha_subcycle_body` | wrapper owning one alpha subcycle body |
| `mixture_update` | mixture property / `rho` / `rhoPhi` update stage after the alpha loop |
| `momentum_predictor` | host wrapper that owns the momentum predictor stage |
| `pressure_assembly` | host wrapper for pressure-system assembly |
| `pressure_solve_native` | native pressure-solver dispatch wrapper |
| `pressure_solve_amgx` | AmgX pressure-solver dispatch wrapper when that backend is active |
| `pressure_post` | post-solve correction / fix-up stage |
| `nozzle_bc_update` | fused nozzle/boundary-update wrapper or per-patch-group parent wrapper |
| `write_stage` | explicit write/output staging path |

Optional nested diagnostic ranges:

| Range | Required parent range |
|---|---|
| `alpha/detail/fluxAssembly` | `alpha_subcycle_body` |
| `alpha/detail/MULES/limit` | `alpha_subcycle_body` |
| `alpha/detail/correction` | `alpha_subcycle_body` |
| `surfaceTension/detail/compute` | `mixture_update` |
| `momentum/detail/UAssembly` | `momentum_predictor` |
| `momentum/detail/USolve` | `momentum_predictor` |
| `pressure/detail/p_rghAssembly` | `pressure_assembly` |
| `pressure/detail/p_rghSolve` | `pressure_solve_native` or `pressure_solve_amgx` |
| `boundary/detail/patchGroup/<name>` | `nozzle_bc_update` |

#### 4. Instrument graph lifecycle points

Required markers:
- `graph/instantiate`
- `graph/update`
- `graph/launch`

Place them in the host code that owns graph creation/update/launch. If graph capture is not yet implemented, the markers may exist but remain inactive.

#### 5. Instrument external solver wrappers

If pressure or momentum solve dispatches to AmgX or another external backend via a wrapper, instrument the wrapper boundary rather than only the caller. Otherwise the trace can show an opaque hole in the most important range.

#### 6. Add post-warmup allocation detection

At minimum, count these APIs indirectly through Nsight Systems CUDA API summary:
- `cudaMalloc`
- `cudaMallocAsync`
- `cudaMallocManaged`
- `cudaFree`
- `cudaFreeAsync`

If the project uses a memory-pool wrapper that exposes its own host allocation function, add an explicit NVTX `memory/poolGrow` range or marker around the pool-grow path so it can be correlated.

#### 7. Add profiling configuration loading

Read `system/gpuRuntimeDict` at startup and expose `gpuRuntime.profiling` / `gpuRuntime.acceptance` to the solver and scripts. A legacy `system/gpuProfilingDict` may remain as a compatibility shim or generated subview during migration, but it is not the authoritative contract.

#### 8. Add build-system changes

- Add NVTX v3 include path and `-ldl` for the profiling build [R11].
- Add `-lineinfo` to CUDA translation units for profiling and sanitizer builds [R12][R15].
- Add an optional profiling define, e.g. `FOAM_USE_NVTX3`.

#### 9. Add tools and CI hooks

Required scripts and CI tasks:
- `run_nsys_profile.sh`
- `export_nsys_stats.py`
- `accept_profile.py`
- `run_ncu_topkernels.py`
- `run_compute_sanitizer.sh`

### Proposed file layout and module boundaries

```text
src/
  gpu/
    profiling/
      GpuProfilingConfig.H
      GpuProfilingConfig.C
      NvtxDomains.H
      NvtxDomains.C
      ScopedNvtxRange.H
      ScopedNvtxRange.C
      ProfilingMarkers.H
      ProfilingMarkers.C
      ProfilingEnable.H
      ProfilingVersion.C

applications/
  solvers/
    multiphase/
      deviceIncompressibleVoF/
        deviceIncompressibleVoF.C          # top-level solver instrumentation
        deviceAlphaPredictor.C             # alpha and MULES range hooks
        devicePressureCorrector.C          # p_rgh assembly/solve hooks
        deviceMomentumPredictor.C          # U assembly/solve hooks
        deviceSurfaceTension.C             # surface-tension hook
        deviceNozzleBoundary.C             # patch update hook

tools/
  profiling/
    run_nsys_profile.sh
    export_nsys_stats.py
    accept_profile.py
    run_ncu_topkernels.py
    run_compute_sanitizer.sh
    common.py

etc/
  profiling/
    gpuRuntimeDict.template
    ncu_sections_top5.txt
    nsys_reports_required.txt
    thresholds_default.json

ci/
  nightly/
    profile_r1.yml
    profile_r0.yml
```

**Module boundary rules**
- `src/gpu/profiling/` contains no solver logic.
- Solver files only call profiling wrappers; they do not know profiler CLI details.
- `tools/profiling/` owns CLI invocations, exports, parsing, and acceptance logic.
- CI files only orchestrate tools; they do not duplicate parser logic.

### Pseudocode

#### 1. NVTX initialization

```cpp
// GpuProfilingConfig.C
GpuRuntimeConfig Foam::gpu::profiling::readConfig(const Foam::Time& runTime)
{
    GpuRuntimeConfig cfg{};
    cfg.profiling.enabled = false;
    cfg.profiling.mode = ProfileMode::off;
    cfg.profiling.enableNvtx = false;
    cfg.profiling.enableGraphMarkers = true;
    cfg.profiling.enableOutputMarkers = true;
    cfg.profiling.annotatePatchUpdates = true;
    cfg.profiling.annotateExternalSolver = true;
    cfg.profiling.warmupTimeSteps = 3;
    cfg.profiling.captureTimeSteps = 5;
    cfg.profiling.topKernelCount = 5;
    cfg.profiling.captureDomain = "spuma.solver";
    cfg.profiling.captureRangeName = "solver/steady_state";

    cfg.acceptance.acceptedTupleId = "";
    cfg.acceptance.acceptanceManifest = "acceptance_manifest.json";
    cfg.acceptance.supportMatrix = "support_matrix.json";
    cfg.acceptance.graphSupportMatrix = "graph_capture_support_matrix.json";
    cfg.acceptance.masterPinManifest = "master_pin_manifest.md";

    dictionary profilingDict;
    dictionary acceptanceDict;
    if (exists(runTime.system()/"gpuRuntimeDict"))
    {
        dictionary runtimeDict(IOobject(...));
        dictionary gpuRuntime = runtimeDict.subDict("gpuRuntime");
        profilingDict = gpuRuntime.subDict("profiling");
        acceptanceDict = gpuRuntime.subDict("acceptance");
    }
    else if (exists(runTime.system()/"gpuProfilingDict"))
    {
        dictionary dict(IOobject(...));
        profilingDict = dict.subDict("gpuProfiling");
        if (dict.found("acceptance"))
        {
            acceptanceDict = dict.subDict("acceptance");
        }
    }
    else
    {
        return cfg;
    }

    cfg.profiling.enabled = profilingDict.lookupOrDefault<bool>("enabled", false);
    cfg.profiling.mode = parseProfileMode(profilingDict.lookupOrDefault<word>("mode", "off"));
    cfg.profiling.enableNvtx = profilingDict.lookupOrDefault<bool>("enableNvtx", cfg.profiling.enabled);
    cfg.acceptance.acceptedTupleId = acceptanceDict.lookupOrDefault<word>("acceptedTupleId", cfg.acceptance.acceptedTupleId);
    cfg.acceptance.acceptanceManifest = acceptanceDict.lookupOrDefault<fileName>("acceptanceManifest", cfg.acceptance.acceptanceManifest);
    cfg.acceptance.supportMatrix = acceptanceDict.lookupOrDefault<fileName>("supportMatrix", cfg.acceptance.supportMatrix);
    cfg.acceptance.graphSupportMatrix = acceptanceDict.lookupOrDefault<fileName>("graphSupportMatrix", cfg.acceptance.graphSupportMatrix);
    cfg.acceptance.masterPinManifest = acceptanceDict.lookupOrDefault<fileName>("masterPinManifest", cfg.acceptance.masterPinManifest);
    // ... load remaining keys ...
    return cfg;
}
```

```cpp
// NvtxDomains.C
void Foam::gpu::profiling::initializeNvtx(const GpuProfilingConfig& cfg)
{
    if (!cfg.enabled || !cfg.enableNvtx) return;

    registerDomain(DomainId::solver, "spuma.solver");
    registerDomain(DomainId::alpha, "spuma.alpha");
    registerDomain(DomainId::momentum, "spuma.momentum");
    registerDomain(DomainId::pressure, "spuma.pressure");
    registerDomain(DomainId::surfaceTension, "spuma.surfaceTension");
    registerDomain(DomainId::boundary, "spuma.boundary");
    registerDomain(DomainId::graph, "spuma.graph");
    registerDomain(DomainId::memory, "spuma.memory");
    registerDomain(DomainId::output, "spuma.output");
    registerDomain(DomainId::diagnostics, "spuma.diagnostics");

    nameCategory(DomainId::solver, CategoryId::timeStep, "timeStep");
    nameCategory(DomainId::solver, CategoryId::steadyState, "steadyState");
    // ... remaining categories ...
}
```

#### 2. Top-level solver orchestration

```cpp
void DeviceIncompressibleVoFSolver::solve()
{
    const auto runtimeCfg = gpu::profiling::readConfig(runTime_);
    const auto& cfg = runtimeCfg.profiling;
    gpu::profiling::initializeNvtx(cfg);

    int captureBegin = cfg.warmupTimeSteps;
    int captureEnd   = cfg.warmupTimeSteps + cfg.captureTimeSteps;

    while (pimple_.run(runTime_))
    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::solver,
                            gpu::profiling::CategoryId::timeStep,
                            "solver/timeStep");

        if (runTime_.timeIndex() == captureBegin)
        {
            gpu::profiling::markSteadyStateBegin(runTime_.timeIndex());
        }

        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::solver,
                                gpu::profiling::CategoryId::preSolve,
                                "pre_solve");
            preSolve();
        }

        while (pimple_.loop())
        {
            SPUMA_PROFILE_SCOPE_PAYLOAD(gpu::profiling::DomainId::solver,
                                        gpu::profiling::CategoryId::pimpleOuter,
                                        "solver/pimpleOuter",
                                        static_cast<uint64_t>(pimple_.corr()));

            {
                SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::solver,
                                    gpu::profiling::CategoryId::outerIterBody,
                                    "outer_iter_body");

                prePredictor();
                alphaPredictor();
                momentumPredictor();
                pressureCorrector();
                postCorrector();
            }
        }

        if (shouldWrite())
        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::output,
                                gpu::profiling::CategoryId::writeStage,
                                "write_stage");
            write();
        }

        if (runTime_.timeIndex() + 1 == captureEnd)
        {
            gpu::profiling::markSteadyStateEnd(runTime_.timeIndex());
        }
    }

    gpu::profiling::shutdownNvtx();
}
```

#### 3. Alpha path instrumentation

```cpp
void DeviceIncompressibleVoFSolver::alphaPredictor()
{
    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                            gpu::profiling::CategoryId::alphaPre,
                            "alpha_pre");
        prepareAlphaSubcycleInputsDevice();
    }

    for (int subCycle = 0; subCycle < nAlphaSubCycles(); ++subCycle)
    {
        gpu::profiling::markSolverMetadata
        (
            profilingConfig_,
            runTime_.timeIndex(),
            pimple_.corr(),
            subCycle
        );

        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                                gpu::profiling::CategoryId::alphaSubcycleBody,
                                "alpha_subcycle_body");

            {
                SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                                    gpu::profiling::CategoryId::alphaFluxAssembly,
                                    "alpha/detail/fluxAssembly");
                assembleAlphaFluxesDevice();
            }

            {
                SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                                    gpu::profiling::CategoryId::mulesLimit,
                                    "alpha/detail/MULES/limit");
                deviceMULES_.limit(alpha_, alphaPhi_, compressionPhi_);
            }

            {
                SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                                    gpu::profiling::CategoryId::alphaCorrection,
                                    "alpha/detail/correction");
                correctAlphaDevice();
            }
        }
    }

    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::alpha,
                            gpu::profiling::CategoryId::mixtureUpdate,
                            "mixture_update");
        updateMixturePropertiesDevice();

        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::surfaceTension,
                                gpu::profiling::CategoryId::surfaceTensionCompute,
                                "surfaceTension/detail/compute");
            updateSurfaceTensionDevice();
        }
    }
}
```

#### 4. Momentum and pressure instrumentation

```cpp
void DeviceIncompressibleVoFSolver::momentumPredictor()
{
    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::momentum,
                            gpu::profiling::CategoryId::momentumPredictor,
                            "momentum_predictor");

        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::momentum,
                                gpu::profiling::CategoryId::uAssembly,
                                "momentum/detail/UAssembly");
            assembleUEqnDevice();
        }

        {
            SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::momentum,
                                gpu::profiling::CategoryId::uSolve,
                                "momentum/detail/USolve");
            solveUEqnDevice();
        }
    }
}

void DeviceIncompressibleVoFSolver::pressureCorrector()
{
    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::pressure,
                            gpu::profiling::CategoryId::pAssembly,
                            "pressure_assembly");
        assemblePressureSystemDevice();
    }

    const char* solveStage =
        pressureSolver_.usesAmgx()
      ? "pressure_solve_amgx"
      : "pressure_solve_native";

    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::pressure,
                            gpu::profiling::CategoryId::pSolve,
                            solveStage);
        pressureSolver_.solve(p_rgh_);
    }

    {
        SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::pressure,
                            gpu::profiling::CategoryId::pressurePost,
                            "pressure_post");
        finalizePressureCorrectionsDevice();
    }
}
```

#### 5. Graph instrumentation

```cpp
void DeviceExecutionGraph::instantiate()
{
    SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::graph,
                        gpu::profiling::CategoryId::graphInstantiate,
                        "graph/instantiate");
    // cudaGraphInstantiate...
}

void DeviceExecutionGraph::update()
{
    SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::graph,
                        gpu::profiling::CategoryId::graphUpdate,
                        "graph/update");
    // cudaGraphExecUpdate...
}

void DeviceExecutionGraph::launch(cudaStream_t stream)
{
    SPUMA_PROFILE_SCOPE(gpu::profiling::DomainId::graph,
                        gpu::profiling::CategoryId::graphLaunch,
                        "graph/launch");
    // cudaGraphLaunch...
}
```

#### 6. Baseline `nsys` wrapper

```bash
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$1"
MODE="${2:-baselineTimeline}"
OUT_DIR="$3"

mkdir -p "$OUT_DIR"

NSYS_BASE=(
  nsys profile
  --trace=cuda,nvtx,osrt
  --cuda-graph-trace=graph
  --cuda-event-trace=false
  --capture-range=nvtx
  --nvtx-capture=solver/steady_state@spuma.solver
  --capture-range-end=repeat:1
  --output "${OUT_DIR}/report"
)

case "$MODE" in
  baselineTimeline)
    ;;
  uvmAudit)
    NSYS_BASE+=(--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true)
    ;;
  syncAudit)
    NSYS_BASE+=(--cudabacktrace=all --sample=process-tree --backtrace=fp)
    ;;
  graphDebug)
    NSYS_BASE=(nsys profile
      --trace=cuda,nvtx,osrt
      --cuda-graph-trace=node
      --capture-range=nvtx
      --nvtx-capture=solver/steady_state@spuma.solver
      --capture-range-end=repeat:1
      --output "${OUT_DIR}/report")
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac

printf '%q ' "${NSYS_BASE[@]}" "./foamRun" "-case" "${CASE_DIR}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"

"${NSYS_BASE[@]}" ./foamRun -case "${CASE_DIR}" > "${OUT_DIR}/run.log" 2>&1
```

#### 7. `nsys` export/parser

```python
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import subprocess

REQUIRED_REPORTS = [
    "cuda_api_sum",
    "cuda_gpu_sum",
    "cuda_gpu_kern_sum:nvtx-name:base",
    "cuda_kern_exec_sum:nvtx-name:base",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "um_sum",
    "um_total_sum",
    "um_cpu_page_faults_sum",
    "osrt_sum",
]

def export_reports(rep_path: Path, out_dir: Path) -> None:
    sqlite_path = rep_path.with_suffix(".sqlite")
    subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(sqlite_path), str(rep_path)], check=True)

    for report in REQUIRED_REPORTS:
        stem = report.replace(":", "_")
        out_csv = out_dir / f"{stem}.csv"
        subprocess.run([
            "nsys", "stats",
            "--report", report,
            "--format", "csv",
            "--output", str(out_csv),
            str(sqlite_path)
        ], check=True)

    analyze_txt = out_dir / "analyze.txt"
    with analyze_txt.open("w") as f:
        subprocess.run(["nsys", "analyze", str(sqlite_path)], check=True, stdout=f, stderr=subprocess.STDOUT)

@dataclass
class ProfileStats:
    mandatory_nvtx_ranges_present: bool
    unexpected_htod_bytes: int
    unexpected_dtoh_bytes: int
    cpu_um_faults: int
    gpu_um_faults: int
    cudaDeviceSynchronize_calls: int
    cudaStreamSynchronize_calls: int  # diagnostic-only unless promoted by the manifest
    post_warmup_alloc_calls: int
    cpu_boundary_fallback_events: int
    host_patch_execution_events: int
    pinned_host_pressure_stage_events: int
    host_setFields_startup_events: int
    unsafe_functionObject_commit_events: int
    graph_launches_per_step: float
    top_kernel_time_regression_pct: float
    kernel_launches_per_step: float
    steps: int
    scoped_metrics: dict[str, dict[str, float]]

    def metric(self, gate_id: str, scope: str | None = None):
        if scope is None:
            return getattr(self, gate_id)
        return self.scoped_metrics[gate_id][scope]
```

#### 8. Top-kernel selection and NCU invocation

```python
def select_top_kernels(kern_csv: Path, top_n: int) -> list[str]:
    rows = load_csv(kern_csv)
    rows = [r for r in rows if r.get("Name")]
    rows.sort(key=lambda r: parse_time_ns(r["Total Time"]), reverse=True)
    return [r["Name"] for r in rows[:top_n]]

def run_ncu_for_kernel(case_dir: Path, kernel_name: str, out_dir: Path) -> None:
    cmd = [
        "ncu",
        "--nvtx",
        "--nvtx-include", "steady_state/",
        "--section", "SpeedOfLight",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "SchedulerStats",
        "--section", "WarpStateStats",
        "--section", "Occupancy",
        "--section", "LaunchStats",
        "--section", "SourceCounters",
        "--import-source", "yes",
        "--target-processes", "application-only",
        "--kernel-name", kernel_name,
        "-o", str(out_dir / sanitize(kernel_name)),
        "./foamRun", "-case", str(case_dir)
    ]
    subprocess.run(cmd, check=True)
```

#### 9. Sanitizer wrapper

```bash
#!/usr/bin/env bash
set -euo pipefail
CASE_DIR="$1"
OUT_DIR="$2"

mkdir -p "$OUT_DIR"

compute-sanitizer --tool memcheck --leak-check full \
  ./foamRun -case "$CASE_DIR" > "${OUT_DIR}/memcheck.log" 2>&1

compute-sanitizer --tool initcheck \
  ./foamRun -case "$CASE_DIR" > "${OUT_DIR}/initcheck.log" 2>&1

compute-sanitizer --tool synccheck \
  ./foamRun -case "$CASE_DIR" > "${OUT_DIR}/synccheck.log" 2>&1
```

### Step-by-step implementation guide

#### Step 1 — Add NVTX v3 dependency and build flags

- **Modify**
  - Build system / make options / CMake equivalents.
  - Add NVTX v3 include path.
  - Add `-ldl`.
  - Add `-lineinfo` to CUDA translation units for profiling/sanitizer builds.
- **Why**
  - NVTX v3 and source correlation are prerequisites [R11][R12][R15].
- **Expected output**
  - Code compiles with profiling wrappers available.
- **How to verify success**
  - Build completes.
  - A minimal dummy `SPUMA_PROFILE_SCOPE(...)` compiles and links.
- **Likely breakages**
  - Missing NVTX include path.
  - Link failure due to missing `-ldl`.
  - Inconsistent compile flags across host/device translation units.

#### Step 2 — Create `src/gpu/profiling/` wrapper library

- **Modify**
  - Add `GpuProfilingConfig`, NVTX domain/category registry, RAII scope wrapper, and marker functions.
- **Why**
  - Centralizes profiler dependencies and prevents direct NVTX calls from spreading through solver code.
- **Expected output**
  - A small library with unit-testable behavior.
- **How to verify success**
  - A test executable creates ranges and exits cleanly.
- **Likely breakages**
  - Category/domain lifetime bugs.
  - Multiple-definition errors from global registries.

#### Step 3 — Add canonical `gpuRuntime.profiling` / `gpuRuntime.acceptance` parsing (`gpuProfilingDict` compatibility shim only if needed)

- **Modify**
  - Runtime configuration loading and validation.
- **Why**
  - Profiling behavior must be switchable without recompilation, and Phase 8 must consume the same normalized runtime tree used by the rest of the roadmap for both instrumentation controls and acceptance-manifest selection.
- **Expected output**
  - Solver starts with profiling disabled when the canonical profiling subview is absent; enabled when present.
  - Acceptance tuple/manifests are available through a typed acceptance subview.
  - If a legacy `gpuProfilingDict` shim is still accepted, it resolves to the same normalized runtime state as `gpuRuntime.profiling` plus an optional sibling `acceptance` subdict.
- **How to verify success**
  - Startup log prints resolved profiling config and accepted tuple/manifest refs once.
- **Likely breakages**
  - Invalid enum parsing.
  - Missing subdict causing silent wrong defaults.
  - Divergence between canonical `gpuRuntime` profiling/acceptance views and a legacy shim view.

#### Step 4 — Instrument top-level timestep and PIMPLE scopes

- **Modify**
  - Actual runtime-selected nozzle solver time loop and PIMPLE loop wrappers.
- **Why**
  - These are the parent ranges required for all later attribution.
- **Expected output**
  - `solver/timeStep`, `solver/steady_state`, and `solver/pimpleOuter` visible in Nsight Systems.
- **How to verify success**
  - Run a tiny profile and inspect `nvtx_sum`.
- **Likely breakages**
  - Instrumenting the wrong base class.
  - Missing range closure on exceptions or early returns.

#### Step 5 — Instrument alpha path scopes

- **Modify**
  - `alphaPredictor`, flux assembly wrapper, MULES limiter wrapper, correction wrapper, mixture update wrapper.
- **Why**
  - Alpha/MULES/subcycling is one of the main non-sparse bottlenecks in the nozzle solver.
- **Expected output**
  - Mandatory alpha ranges appear under each steady-state timestep.
- **How to verify success**
  - `nvtx_sum` shows all alpha range names with nonzero time.
- **Likely breakages**
  - Instrumenting too far down in micro-kernel wrappers, causing excessive range count.
  - Missing subcycle coverage because instrumentation was placed outside the loop.

#### Step 6 — Instrument momentum, pressure, and surface-tension scopes

- **Modify**
  - U assembly, U solve, pressure assembly, pressure solve, surface tension host wrappers.
- **Why**
  - These stages need separate baselines and acceptance attribution.
- **Expected output**
  - Momentum/pressure ranges appear distinctly.
- **How to verify success**
  - `nvtx_gpu_proj_sum` or GUI shows GPU time projected into each range.
- **Likely breakages**
  - External solver path not wrapped, leaving pressure solve unattributed.

#### Step 7 — Instrument boundary and output staging scopes

- **Modify**
  - Patch update wrappers and write/output staging path.
- **Why**
  - Boundary logic is a common hidden host/device tail; output must be distinguishable from solver traffic.
- **Expected output**
  - `nozzle_bc_update` and `write_stage` visible as the required parent ranges for boundary/output work.
- **How to verify success**
  - Trace shows write stage isolated and not nested incorrectly under pressure or alpha scopes.
- **Likely breakages**
  - Patch updates triggered from multiple code paths with partial coverage.
  - Output path called outside the expected wrapper.

#### Step 8 — Add graph lifecycle markers

- **Modify**
  - Graph instantiate/update/launch owner code.
- **Why**
  - Needed to measure graph fragmentation and launch count.
- **Expected output**
  - `graph/*` markers appear when graph mode is active.
- **How to verify success**
  - Run `graphDebug` mode and inspect graph markers.
- **Likely breakages**
  - Markers placed in dead code path because graph capture disabled.
  - Markers not correlated with actual `cudaGraphLaunch`.

#### Step 9 — Add baseline `nsys` wrapper script

- **Modify**
  - `tools/profiling/run_nsys_profile.sh`.
- **Why**
  - Prevents inconsistent manual command lines.
- **Expected output**
  - Reproducible `.nsys-rep` and `run.log`.
- **How to verify success**
  - Script runs successfully in `baselineTimeline` mode on `R1`.
- **Likely breakages**
  - Wrong `--nvtx-capture` domain/range name.
  - Capture window empty because steady-state markers never appear.

#### Step 10 — Add `nsys export` + report-generation script

- **Modify**
  - `export_nsys_stats.py`.
- **Why**
  - Formal acceptance requires machine-readable reports.
- **Expected output**
  - CSV/text exports for all required reports.
- **How to verify success**
  - Output directory contains all expected files.
- **Likely breakages**
  - Tool-version differences in report names.
  - Export path errors.

#### Step 11 — Implement parser and acceptance evaluator

- **Modify**
  - `accept_profile.py`.
- **Why**
  - Converts profiler output into pass/fail status.
- **Expected output**
  - `acceptance_summary.json` and `acceptance_summary.md`.
- **How to verify success**
  - A known-good run passes; an intentionally broken run fails.
- **Likely breakages**
  - Incorrect CSV column-name assumptions.
  - Misclassification of output-stage DtoH as solver DtoH.

#### Step 12 — Add UVM audit mode

- **Modify**
  - `run_nsys_profile.sh`, parser classification logic.
- **Why**
  - Needed to detect silent managed-memory fallbacks.
- **Expected output**
  - `um_sum` and `um_total_sum` parsed correctly.
- **How to verify success**
  - A test branch that intentionally reads a device field on host triggers failures.
- **Likely breakages**
  - UVM audit overhead too high because capture window too large.
  - Output-stage bytes not excluded.

#### Step 13 — Add sync audit mode and expert analysis capture

- **Modify**
  - `run_nsys_profile.sh`, parser, analyze capture.
- **Why**
  - Needed to locate remaining blocking sync points and pageable memcpy mistakes.
- **Expected output**
  - API summary counts and expert-rule findings available.
- **How to verify success**
  - An intentionally inserted `cudaDeviceSynchronize()` in inner loop is detected.
- **Likely breakages**
  - Missing host backtraces due to frame-pointer or sampling settings.
  - Excessive overhead from overly broad sync audit captures.

#### Step 14 — Add top-five-kernel selector and NCU wrapper

- **Modify**
  - `run_ncu_topkernels.py`, section file, environment manifest.
- **Why**
  - Deep-dive only the kernels that matter.
- **Expected output**
  - NCU reports for the top five kernels.
- **How to verify success**
  - Reports open in CLI/GUI and include source correlation.
- **Likely breakages**
  - Kernel filtering syntax mismatches.
  - No source mapping because `-lineinfo` missing.

#### Step 15 — Add sanitizer script and nightly target

- **Modify**
  - `run_compute_sanitizer.sh`, CI nightly config.
- **Why**
  - Functional GPU correctness must be checked before trusting performance traces.
- **Expected output**
  - Logs for memcheck/initcheck/synccheck (and racecheck where applicable).
- **How to verify success**
  - Known invalid memory access is caught.
- **Likely breakages**
  - Timeouts on too-large cases.
  - False assumptions about racecheck covering global-memory races.

#### Step 16 — Define default thresholds and baseline files

- **Modify**
  - `etc/profiling/thresholds_default.json`.
- **Why**
  - Acceptance must be reproducible and explicit.
- **Expected output**
  - Default hard/soft thresholds versioned in source control.
- **How to verify success**
  - Acceptance summary echoes threshold values used.
- **Likely breakages**
  - Thresholds too strict before graph mode exists.
  - Confusion between bring-up and production thresholds.

#### Step 17 — Run `R1` baseline and lock first baseline

- **Modify**
  - No code change necessarily; this is the first formal capture.
- **Why**
  - Establishes initial architectural baseline.
- **Expected output**
  - Complete profile artifact set for `R1`.
- **How to verify success**
  - All mandatory ranges, no hard fails, reports present.
- **Likely breakages**
  - Hidden host transfer or sync still present.
  - Missing range names due to instrumentation gaps.

#### Step 18 — Run `R0` acceptance and compare against `R1`

- **Modify**
  - Case harness configuration only.
- **Why**
  - Ensures the framework survives on the representative production case.
- **Expected output**
  - Full `R0` acceptance artifact set.
- **How to verify success**
  - Same hard gates pass; top-kernel list may differ but process works.
- **Likely breakages**
  - Output interval accidentally overlaps capture window.
  - Memory pressure or longer graph setup changes launch structure unexpectedly.

#### Step 19 — Add CI/nightly gates

- **Modify**
  - CI pipeline definitions.
- **Why**
  - Prevents regressions after Phase 8 completion.
- **Expected output**
  - Automated profile job on `R1` nightly or pre-merge.
- **How to verify success**
  - CI fails on injected violations.
- **Likely breakages**
  - Artifact size too large.
  - Profile duration exceeds CI budget.

#### Step 20 — Write the profiling operator guide

- **Modify**
  - `docs/profiling/README.md`.
- **Why**
  - Human reviewers need an exact operational guide.
- **Expected output**
  - Short runbook for each mode and expected outputs.
- **How to verify success**
  - Another engineer can run all modes without tribal knowledge.
- **Likely breakages**
  - Documentation drifts from scripts.

### Instrumentation and profiling hooks

#### Mandatory NVTX range inventory

| Domain | Range | Scope type | Required nesting |
|---|---|---|---|
| `spuma.solver` | `solver/timeStep` | push/pop | root of each timestep |
| `spuma.solver` | `solver/steady_state` | push/pop | wraps capture window |
| `spuma.solver` | `solver/pimpleOuter` | push/pop | inside timestep |
| `spuma.solver` | `pre_solve` | push/pop | pre-solve preparation before the first outer iteration |
| `spuma.solver` | `outer_iter_body` | push/pop | one accepted outer-iteration body |
| `spuma.alpha` | `alpha_pre` | push/pop | before the alpha subcycle loop |
| `spuma.alpha` | `alpha_subcycle_body` | push/pop | one accepted alpha subcycle body |
| `spuma.alpha` | `mixture_update` | push/pop | after alpha update |
| `spuma.momentum` | `momentum_predictor` | push/pop | wraps the momentum predictor stage |
| `spuma.pressure` | `pressure_assembly` | push/pop | inside PIMPLE before the backend solve |
| `spuma.pressure` | `pressure_solve_native` | push/pop | native pressure-dispatch wrapper |
| `spuma.pressure` | `pressure_solve_amgx` | push/pop | AmgX pressure-dispatch wrapper when active |
| `spuma.pressure` | `pressure_post` | push/pop | after the pressure solve |
| `spuma.surfaceTension` | `surfaceTension/detail/compute` | push/pop | optional child diagnostic under `mixture_update` |
| `spuma.boundary` | `nozzle_bc_update` | push/pop | around fused nozzle/boundary update |
| `spuma.graph` | `graph/instantiate` | push/pop | graph lifecycle |
| `spuma.graph` | `graph/update` | push/pop | graph lifecycle |
| `spuma.graph` | `graph/launch` | push/pop | graph lifecycle |
| `spuma.output` | `write_stage` | push/pop | explicit write only |

#### Optional NVTX markers

Use markers, not ranges, for:
- graph version IDs,
- case metadata emission,
- transition into fallback code paths,
- pool-growth events,
- one-time warning conditions.

#### Required profiler command families

1. **Baseline timeline**
   - `nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=graph --capture-range=nvtx ...`

2. **UVM audit**
   - baseline + `--cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true`

3. **Sync audit**
   - baseline + host backtrace/sampling options and `nsys analyze`

4. **Graph debug**
   - baseline with `--cuda-graph-trace=node` and extremely short capture

5. **Kernel deep-dive**
   - `ncu` on top kernels only, with NVTX filtering and required sections

6. **Sanitizer**
   - `compute-sanitizer` staged tool order

#### Required parser hooks

The parser must extract at minimum:
- presence/absence of mandatory ranges,
- total and per-step kernel launch count,
- total and per-step graph launch count,
- unexpected HtoD/DtoH bytes,
- CPU/GPU UVM page faults,
- `cudaDeviceSynchronize`/`cudaStreamSynchronize` counts,
- allocation API counts after warmup,
- top-kernel ranking,
- fallback counters for CPU boundary fallback, host patch execution, pinned-host pressure staging, host `setFields` startup, and unsafe functionObject commits,
- manifest revision/hash identifiers for the master pin manifest, centralized acceptance manifest, support matrix, and graph support matrix,
- expert-rule text containing sync/pageable-memory/GPU-starvation findings.

### Validation strategy

#### A. Instrumentation correctness validation

1. **Compile validation**
   - Profiling-enabled build must compile and link cleanly.
2. **Range-balance validation**
   - No unclosed NVTX ranges.
3. **Coverage validation**
   - All required ranges appear in `nvtx_sum` for `R1`.
4. **Nesting validation**
   - Child ranges appear under the correct parent in the timeline.
5. **Numerical non-interference**
   - Profiling-enabled build with profiler detached must reproduce reference results within existing solver tolerance.

**Pass/fail**
- Any missing mandatory range = fail.
- Any numerical deviation outside existing tolerance = fail.

#### B. Baseline residency validation

1. Configure `R1` so writes do not occur during capture.
2. Run baseline timeline.
3. Run UVM audit.
4. Parse `um_sum`, `um_total_sum`, and `um_cpu_page_faults_sum`.
5. Classify bytes/faults by NVTX window.

**Pass/fail**
- Production mode: zero unexpected steady-state UVM bytes/faults.
- Bring-up mode: any unexpected bytes/faults = fail unless explicitly whitelisted by human sign-off.

#### C. Synchronization validation

1. Run `syncAudit`.
2. Parse `cuda_api_sum`.
3. Count synchronization APIs.
4. Read `nsys analyze` output for sync and pageable-memory expert findings.

**Pass/fail**
- `cudaDeviceSynchronize()` in inner solver ranges = fail.
- Any blocking sync or pageable HtoD/DtoH in inner solver ranges = fail unless explicitly approved boundary behavior.
- Post-warmup allocation APIs in steady-state = fail.

#### D. Launch-structure validation

1. Count kernels in steady-state.
2. Count graphs and graph launches in steady-state.
3. Compare against thresholds/baseline.

**Pass/fail**
- If graph mode enabled: `graph_launches_per_step` must satisfy the imported soft gate from `acceptance_manifest.md` / `acceptance_manifest.json` for the active tuple unless that manifest later reclassifies the metric as a hard gate.
- Launch-count regressions outside the manifest-gated metric set may still be recorded diagnostically, but they do not change the formal acceptance verdict unless a later manifest revision adds them.

#### E. Kernel deep-dive validation

1. Select top five kernels by total time.
2. Run NCU.
3. Save text/rep files.
4. Extract secondary metrics.

**Pass/fail**
- Missing NCU report for a selected kernel = fail.
- Missing source correlation due to build misconfiguration = fail for profiling build quality.
- Secondary metric regressions are review-gated initially, not hard fail.

#### F. Sanitizer validation

1. `memcheck`
2. `initcheck`
3. `synccheck`
4. `racecheck` on relevant reduced tests

**Pass/fail**
- Any `memcheck`, `initcheck`, or `synccheck` error = fail.
- `racecheck` ERROR-level hazards in relevant kernels = fail.
- Warnings require review.

#### G. Regression-validation cadence

- **Per major solver refactor**: `R1` baseline timeline + acceptance parser
- **Nightly**: `R1` baseline timeline + UVM audit + sanitizer
- **Before performance claims**: `R0` baseline timeline + top-5 NCU
- **Before graph/fusion changes are merged**: `graphDebug` one-step run

### Performance expectations

1. **What this phase should improve**
   - Not raw runtime directly, but the reliability of performance decisions.
   - It should drastically reduce ambiguity about whether the current bottleneck is traffic, irregular kernels, or launch/sync overhead.

2. **Expected profiler overhead by mode**
   - `baselineTimeline`: low enough for comparative architectural gating.
   - `uvmAudit`: significantly higher overhead; not valid for timing comparisons [R9].
   - `syncAudit`: moderate to high overhead depending on sampling/backtrace.
   - `graphDebug`: high overhead if node-level tracing is enabled [R9][R19].
   - `kernelDeepDive`: high per-kernel overhead; use sparingly.
   - `sanitizer`: very high overhead; never for performance timing [R14][R15].

3. **Expected metrics by phase maturity**
   - Early bring-up may still show nonzero graph instantiations or allocations after warmup.
   - Production-ready Phase 8 should show:
     - no unexpected UVM activity,
     - no post-warmup allocations,
     - no per-kernel sync pattern,
     - stable graph launch counts,
     - a short, explainable top-kernel list.

4. **What not to interpret too strongly**
   - Occupancy alone is not a success metric.
   - Warp stall distributions are diagnostic, not sufficient for acceptance.
   - A high reported memory percentage does not automatically prove DRAM is the only limiter.

### Common failure modes

1. Mandatory ranges missing because instrumentation was added only to a base class, not the runtime-selected derived solver.
2. `solver/steady_state` capture never opens because warmup index logic is off by one.
3. Output writes occur inside baseline capture, causing false DtoH failures.
4. UVM audit mode is mistaken for baseline performance mode.
5. `cudaDeviceSynchronize()` remains hidden inside an external solver wrapper.
6. Dynamic allocations continue after warmup due to temporary-object churn or pool growth.
7. Node-level graph tracing is accidentally used for baseline mode, contaminating launch analysis.
8. `--cuda-event-trace=true` creates misleading dependencies.
9. `-lineinfo` missing, so NCU and sanitizer reports are hard to interpret.
10. Parser relies on wrong CSV column names for a different Nsight version.
11. Built-in `nsys stats` report not available in a given version and script fails without fallback.
12. Racecheck is assumed to detect global-memory races that it does not target.
13. NVTX ranges are placed around sub-microsecond helpers, producing excessive clutter.
14. Too many open NVTX ranges due to mismatched push/pop logic.
15. Graph launch count looks low, but kernel flood remains because capture happened before graph mode activated.
16. CPU-side diagnostics or logging inside steady-state cause DtoH and are misclassified as solver failures.
17. Profiling wrapper allocates strings repeatedly in the inner loop.
18. Nsight Systems falls back to `cuda-sw` on Blackwell unexpectedly and the team does not notice.

### Debugging playbook

#### Symptom 1 — Unexpected HtoD/DtoH bytes or UVM faults in steady-state
1. Re-run `uvmAudit` with the shortest possible capture.
2. Inspect `um_sum` / `um_total_sum`.
3. Correlate migration periods with NVTX ranges.
4. Check for:
   - host-side field access,
   - patch-field code running on CPU,
   - external solver wrappers copying data,
   - logging/diagnostics reading device-backed arrays,
   - managed-memory fallback code paths.
5. If bytes appear only in `write_stage`, classify them as expected output staging and move writes out of baseline capture.

#### Symptom 2 — `cudaDeviceSynchronize()` still appears in inner solver ranges
1. Run `syncAudit`.
2. Inspect `cuda_api_sum` and `nsys analyze`.
3. Use host backtrace to locate call sites.
4. Check:
   - legacy SPUMA synchronization wrappers,
   - external linear solver wrappers,
   - debug assertions,
   - error-checking macros that synchronize unconditionally,
   - graph-capture boundaries inserted too finely.

#### Symptom 3 — Graph launches per timestep are higher than expected
1. Run `graphDebug` with node-level trace on one timestep only.
2. Inspect `graph/instantiate`, `graph/update`, and `graph/launch` markers.
3. Determine whether:
   - graph capture is being rebuilt unnecessarily,
   - alpha subcycles are each launching independent graphs,
   - boundary conditions force a graph split,
   - output/write is accidentally included,
   - unsupported operations are preventing fusion.

#### Symptom 4 — Top kernels are atomic-heavy and memory-scattered
1. Run NCU on the kernel.
2. Inspect:
   - MemoryWorkloadAnalysis sectors/request,
   - `% Peak to L2` or `% Peak to SM`,
   - SchedulerStats skipped issue slots,
   - WarpStateStats stall categories,
   - SourceCounters for code lines driving atomics or scattered loads.
3. Compare against kernel role (limiter, gradient, patch update, curvature).
4. Feed results into the next-phase optimization plan: atomics reduction, reordering, fusion, or data-layout work.

#### Symptom 5 — No source lines in NCU/sanitizer
1. Confirm build uses `-lineinfo`.
2. Confirm the report actually comes from the intended build.
3. Rebuild profiling/sanitizer configuration.
4. If necessary, generate a lower-optimization debug build for root cause analysis.

#### Symptom 6 — Baseline trace shows no useful data
1. Check that canonical `gpuRuntime.profiling` enabled instrumentation, or that any accepted `gpuProfilingDict` shim normalized into the same runtime state.
2. Check `--nvtx-capture` domain/range names match exactly.
3. Confirm the steady-state range opened during the run.
4. Ensure capture window is not zero-length.
5. Ensure the solver actually executed the GPU path.

#### Symptom 7 — Expert rules flag pageable memory or GPU starvation
1. Inspect `nsys analyze`.
2. For pageable memory:
   - ensure host staging uses pinned buffers where appropriate,
   - avoid synchronous memcpys in steady-state.
3. For starvation:
   - inspect launch spacing,
   - inspect host blocking or serialization,
   - inspect graph boundaries and missing overlap opportunities.

### Acceptance checklist

- [ ] Profiling build compiles with NVTX v3 and `-lineinfo`.
- [ ] Canonical `gpuRuntime.profiling` / `gpuRuntime.acceptance` are parsed and logged correctly (`gpuProfilingDict` only as an optional compatibility shim).
- [ ] Mandatory NVTX domains/categories registered.
- [ ] `solver/timeStep`, `solver/steady_state`, `solver/pimpleOuter` visible.
- [ ] Required canonical parent ranges from `graph_capture_support_matrix.json` are visible, and any optional child diagnostics are nested beneath the correct parent stage.
- [ ] Baseline capture excludes initialization.
- [ ] Baseline capture excludes writes, or writes are distinctly tagged and excluded from solver-residency gates.
- [ ] `baselineTimeline` artifacts generated automatically.
- [ ] `uvmAudit` artifacts generated automatically.
- [ ] `syncAudit` artifacts generated automatically.
- [ ] `graphDebug` artifacts generated automatically.
- [ ] `top_kernels.csv` generated automatically.
- [ ] NCU reports for top five kernels generated.
- [ ] `memcheck`, `initcheck`, and `synccheck` clean on `R1`.
- [ ] No unexpected steady-state UVM bytes/faults.
- [ ] No `cudaDeviceSynchronize()` inside inner steady-state ranges.
- [ ] No post-warmup allocation APIs in steady-state.
- [ ] Acceptance summary JSON/Markdown emitted.
- [ ] Baseline files stored under versioned artifact directory.
- [ ] Human reviewer signed off on threshold defaults.

### Future extensions deferred from this phase

1. CUPTI-based custom collectors for lower-overhead continuous telemetry.
2. Automatic roofline dashboards across nightly runs.
3. GPU metrics sampling as a formal acceptance input rather than exploratory telemetry.
4. Multi-GPU/MPI profiling integration.
5. Automatic call-stack diffing for new synchronization points.
6. Graph node-level structural regression tests.
7. Integration of hardware counters beyond what NCU currently captures in the selected sections.
8. Automated per-patch-group boundary hotspot attribution if `nozzle_bc_update` parent coverage proves too coarse.
9. Phase-correlated power/thermals logging for workstation repeatability.

### Implementation tasks for coding agent

1. Add the profiling wrapper library in `src/gpu/profiling/`.
2. Add build flags and NVTX v3 dependency.
3. Implement canonical `gpuRuntime.profiling` / `gpuRuntime.acceptance` parsing (`gpuProfilingDict` shim only if needed).
4. Instrument the actual runtime-selected nozzle solver call sites.
5. Add graph lifecycle markers.
6. Add output-stage markers.
7. Write `run_nsys_profile.sh`.
8. Write `export_nsys_stats.py`.
9. Write `accept_profile.py`.
10. Write `run_ncu_topkernels.py`.
11. Write `run_compute_sanitizer.sh`.
12. Create default thresholds/config templates.
13. Run `R1` baseline and verify mandatory range coverage.
14. Run `R1` UVM audit and fix any unexpected transfers before proceeding.
15. Run `R1` sync audit and remove forbidden sync points.
16. Lock the first baseline.
17. Run `R0` acceptance.
18. Integrate nightly CI job.

### Do not start until

- The actual runtime-selected GPU nozzle solver class has been identified.
- `R1` runs successfully on the RTX 5080.
- Numerical reference tolerances for `R1` are agreed.
- CUDA toolkit / driver / Nsight versions are pinned for the development environment.
- The attached `acceptance_manifest.md` / `acceptance_manifest.json` revision explicitly states whether any write-cadence tuple is in scope; otherwise writes stay outside baseline capture.

### Safe parallelization opportunities

1. **Parallelizable**
   - Build-system/NVTX wrapper work.
   - Script and parser development.
   - Documentation template creation.
   - CI harness scaffolding.

2. **Conditionally parallelizable**
   - Solver-stage instrumentation and graph-marker instrumentation, after domain/category naming is fixed.
   - NCU wrapper and parser work, after `export_nsys_stats.py` can produce `top_kernels.csv`.

3. **Do not parallelize prematurely**
   - Final threshold tuning before first real `R1` capture.
   - Parser hard-gating logic before real CSV outputs are available.

### Governance guardrails

1. Hard and soft fail thresholds are owned by `acceptance_manifest.md`; this phase may not redefine them locally.
2. `graph_launches_per_step` remains a soft gate until `acceptance_manifest.md` explicitly promotes it.
3. Output-stage DtoH remains excluded from timed steady-state windows by default; any named exception must be admitted centrally through the acceptance manifest.
4. Compile-time memcheck instrumentation remains a governance/CI policy choice and does not alter the runtime profiling contract.
5. Mandatory solver/backend coverage is imported from `acceptance_manifest.md`.
6. CI runtime budget and artifact retention remain site-policy items, but artifact bundles required by this phase may not be omitted.

### Artifacts to produce

1. `src/gpu/profiling/*`
2. `system/gpuRuntimeDict` profiling/acceptance template (`gpuProfilingDict` compatibility template only if needed)
3. `tools/profiling/run_nsys_profile.sh`
4. `tools/profiling/export_nsys_stats.py`
5. `tools/profiling/accept_profile.py`
6. `tools/profiling/run_ncu_topkernels.py`
7. `tools/profiling/run_compute_sanitizer.sh`
8. `etc/profiling/thresholds_default.json` (generated mirror of `acceptance_manifest.json` for script input normalization only; never a policy source)
9. `etc/profiling/ncu_sections_top5.txt`
10. `docs/profiling/README.md`
11. First `R1` baseline artifact bundle
12. First `R0` acceptance artifact bundle

---

## 6. Validation and benchmarking framework

This section preserves enough project-wide context to keep Phase 8 operationally coherent, but it is deliberately scoped to profiling and acceptance infrastructure.

### 6.1 Benchmark case tiers

#### Tier B0 — Instrumentation smoke test
- Minimal or reduced `R1` run.
- Purpose: verify compile, NVTX visibility, parser function.
- Tools: Nsight Systems baseline timeline only.
- Gate: mandatory range presence.

#### Tier B1 — Architectural baseline
- Reduced nozzle case `R1`; add `R1-core` whenever the centralized acceptance manifest requires backend or execution-mode parity without Phase 6 nozzle-specific BCs.
- Purpose: residency, synchronization, launch structure, top-kernel identification.
- Tools:
  - baselineTimeline
  - uvmAudit
  - syncAudit
- Gate: all hard fail conditions.

#### Tier B2 — Production-shape acceptance
- Representative nozzle case `R0`.
- Purpose: ensure the same framework scales to the real workflow.
- Tools:
  - baselineTimeline
  - kernelDeepDive (top five)
- Gate: no hard failures; review top-kernel baseline.

#### Tier B3 — Functional GPU safety
- Reduced case `R1`, `R1-core` where required, plus kernel microtests.
- Purpose: memory/init/sync correctness.
- Tools: Compute Sanitizer.
- Gate: sanitizer clean.

### 6.2 Benchmark harness rules

1. Baseline performance captures must avoid write timesteps whenever possible.
2. All runs must emit:
   - environment manifest,
   - exact command,
   - stdout/stderr logs,
   - raw profiler artifacts,
   - derived summaries.
3. The acceptance harness must support both:
   - native SPUMA solver path,
   - external-solver path (e.g. AmgX), if that backend is in scope.
4. All profile directories must be immutable once published as baseline artifacts.

### 6.2.1 Benchmark protocol

1. Warm-up and timed window are defined by `warmupTimeSteps` and `captureTimeSteps` for the active manifest tuple; the Phase 8 bootstrap defaults remain 3 warm-up timesteps and 5 captured timesteps unless overridden.
2. Baseline timing and launch-structure comparisons require at least three successful repetitions per case/backend/execution/kernel-mode tuple on the same workstation. Diagnostic modes (`uvmAudit`, `syncAudit`, `graphDebug`, `sanitizer`) normally require only one short capture unless the acceptance manifest says otherwise.
3. Comparative timing runs are valid only on a single-user machine with no competing GPU workloads during the capture window. If a fixed clock/power policy is available it should be used; otherwise measured steady-state clocks/temperatures must be recorded in the environment bundle and obvious thermal-throttle outliers rejected.
4. Debug/bring-up modes are excluded from timing baselines: `PinnedHost` pressure staging, CPU boundary fallback, host `setFields` startup, explicit debug snapshots, and any run with nonzero fallback counters may still be profiled diagnostically but cannot populate the locked timing baseline.

### 6.3 Required benchmark outputs

For every acceptance run:
- `env.json`
- `command.txt`
- `run.log`
- raw `.nsys-rep`
- raw `.sqlite`
- report CSVs
- `benchmark_matrix.csv` or `benchmark_matrix.json`
- `gpu_purity_report.json`
- `graph_report.json`
- `fallback_counters.json`
- `manifest_refs.json`
- `acceptance_summary.json`
- `acceptance_summary.md`

For every kernel-deep-dive run:
- selected kernel list
- `.ncu-rep`
- text export or summary per kernel

For sanitizer:
- one log file per tool

### 6.4 Threshold import contract

Authoritative hard/soft thresholds, backend applicability, tuple coverage, and tolerance classes come from the centralized acceptance manifest. Phase 8 scripts may cache or mirror those values for runtime use, but they may not invent, override, or loosen them locally.

- Hard-gate failures always fail formal acceptance for the active tuple.
- Soft-gate failures follow the disposition frozen in `acceptance_manifest.md` / `acceptance_manifest.json`: the run may be archived diagnostically, but it is not release-eligible or baseline-lock-eligible unless an explicit waiver is recorded against the manifest revision and tuple ID.
- Diagnostic metrics such as occupancy, warp stalls, sectors/request, `% Peak to L2`, `% Peak to SM`, and GPU metrics sampling remain non-gating unless the acceptance manifest explicitly promotes them.

### 6.5 Benchmark stop points

The coding agent must stop and benchmark before proceeding past each of these checkpoints:

1. After mandatory NVTX coverage is implemented.
2. After parser/acceptance logic is implemented.
3. After first UVM-clean `R1` run.
4. After first sync-clean `R1` run.
5. After first `R0` acceptance run.
6. After any graph-capture refactor.

---

## 7. Toolchain / environment specification

### 7.1 Minimum environment

- **OS**: Linux x86_64
- **GPU**: NVIDIA RTX 5080 (Blackwell, compute capability 12.0, 16 GB GDDR7, 960 GB/s) [R17]
- **Toolchain / driver / profiler lane**: imported from the master pin manifest. Unless superseded, the frozen production lane is CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, and NVTX3.
- **Nsight Systems / Nsight Compute**: versions imported from the same pin manifest; any newer separately validated tool must be recorded as a manifest exception before it is used for acceptance.
- **Compute Sanitizer**: same toolkit family when compile-time memcheck instrumentation is used [R15]

### 7.2 Build requirements

#### Required flags
- `-lineinfo` for CUDA translation units [R12][R15]
- NVTX v3 include path and `-ldl` [R11]
- Retain PTX/native compatibility according to project build policy; verify PTX readiness with `CUDA_FORCE_PTX_JIT=1` in a preflight job [R7]

#### Recommended profiling-build flags
- Host frame pointers retained in profiling/debug builds to improve backtrace utility in `syncAudit`
- Avoid `-G` in baseline profiling builds; reserve it for deep debug/sanitizer cases if required

### 7.3 Profiler command policies

1. **Baseline timeline**
   - `--trace=cuda,nvtx,osrt`
   - `--cuda-graph-trace=graph`
   - `--capture-range=nvtx`
   - `--nvtx-capture=solver/steady_state@spuma.solver`

2. **UVM audit**
   - baseline + UVM page-fault tracing

3. **Sync audit**
   - baseline + CPU sampling/backtrace options + `nsys analyze`

4. **NCU**
   - selected sections only
   - selected kernels only
   - NVTX filtering enabled

5. **Sanitizer**
   - reduced case only

### 7.4 Environment manifest collection

`env.json` must include:
- `nvidia-smi -q` subset or equivalent queried GPU metadata
- `nvcc --version`
- `nsys --version`
- `ncu --version`
- `compute-sanitizer --version`
- git SHA
- compiler version
- master pin manifest revision/hash
- centralized acceptance manifest revision/hash
- support matrix revision/hash
- graph support matrix revision/hash

### 7.5 Explicit anti-patterns

- Do not run baseline acceptance with `CUDA_LAUNCH_BLOCKING=1`.
- Do not compare baseline runtime to sanitizer runtime.
- Do not mix tool versions silently across developers.
- Do not omit PTX compatibility validation on Blackwell-ready builds.

---

## 8. Module / file / ownership map

### 8.1 C++ runtime modules

#### `src/gpu/profiling/GpuProfilingConfig.*`
**Responsibility**
- Parse and validate canonical `gpuRuntime.profiling` / `gpuRuntime.acceptance`, with `gpuProfilingDict` accepted only as an optional compatibility shim
- Provide defaults
- Expose a typed runtime configuration

**Owner**
- Core profiling subsystem

#### `src/gpu/profiling/NvtxDomains.*`
**Responsibility**
- Domain/category registration
- Stable naming
- Domain lookup

**Owner**
- Core profiling subsystem

#### `src/gpu/profiling/ScopedNvtxRange.*`
**Responsibility**
- RAII push/pop range management
- Optional payload support
- No-throw destruction

**Owner**
- Core profiling subsystem

#### `src/gpu/profiling/ProfilingMarkers.*`
**Responsibility**
- One-shot marks for graph/pool/debug events
- Steady-state begin/end markers

**Owner**
- Core profiling subsystem

### 8.2 Solver touch points

#### `deviceIncompressibleVoF.C`
**Responsibility**
- top-level timestep/PIMPLE scopes
- steady-state range gating

#### `deviceAlphaPredictor.C`
**Responsibility**
- alpha/MULES/mixture instrumentation

#### `deviceMomentumPredictor.C`
**Responsibility**
- U assembly/solve instrumentation

#### `devicePressureCorrector.C`
**Responsibility**
- pressure assembly/solve instrumentation

#### `deviceSurfaceTension.C`
**Responsibility**
- surface-tension instrumentation

#### `deviceNozzleBoundary.C`
**Responsibility**
- patch update instrumentation

#### `ExternalSolverWrapper.*`
**Responsibility**
- attribute external backend work correctly

### 8.3 Tooling modules

#### `tools/profiling/run_nsys_profile.sh`
**Responsibility**
- one canonical entry point for Nsight Systems capture modes

#### `tools/profiling/export_nsys_stats.py`
**Responsibility**
- export `.nsys-rep` to SQLite
- generate built-in report CSVs/text

#### `tools/profiling/accept_profile.py`
**Responsibility**
- parse reports
- classify traffic and sync behavior
- evaluate thresholds
- emit summary files

#### `tools/profiling/run_ncu_topkernels.py`
**Responsibility**
- rank kernels from baseline outputs
- invoke Nsight Compute on selected kernels

#### `tools/profiling/run_compute_sanitizer.sh`
**Responsibility**
- standardize sanitizer run order and log layout

### 8.4 Configuration/data modules

#### `etc/profiling/gpuRuntimeDict.template`
**Responsibility**
- user-facing sample configuration for the canonical profiling/acceptance subviews

#### `etc/profiling/ncu_sections_top5.txt`
**Responsibility**
- authoritative section list for NCU wrapper

#### `etc/profiling/nsys_reports_required.txt`
**Responsibility**
- authoritative report list for export script

#### `etc/profiling/thresholds_default.json`
**Responsibility**
- generated machine-readable mirror of the centralized acceptance-manifest thresholds/disposition data for local script input normalization only; never edited manually as a policy source

### 8.5 Ownership rules

- Profiling C++ wrappers own no solver state.
- Solver files own instrumentation placement decisions.
- Scripts own parsing and manifest-application logic.
- `acceptance_manifest.md` / `acceptance_manifest.json` own threshold policy and hard/soft-fail disposition.
- Human reviewers own waivers and baseline approval.

---

## 9. Coding-agent execution roadmap

This roadmap is scoped to Phase 8 only.

### Milestone P8.1 — Instrumentation substrate
**Build**
- NVTX dependency
- profiling wrapper library
- configuration parsing

**Dependencies**
- none beyond compilable solver build

**Stop and benchmark**
- compile test and dummy-range smoke test

### Milestone P8.2 — Solver stage coverage
**Build**
- top-level timestep/PIMPLE markers
- alpha/momentum/pressure/surface/boundary/output markers
- graph lifecycle markers

**Dependencies**
- P8.1

**Parallel work**
- Alpha and pressure instrumentation can proceed in parallel after naming is fixed.
- Boundary/output instrumentation can proceed independently.

**Stop and benchmark**
- first `R1` timeline capture
- verify mandatory range coverage

### Milestone P8.3 — Capture scripts
**Build**
- `run_nsys_profile.sh`
- mode selection
- environment manifest
- artifact directory layout

**Dependencies**
- P8.1
- better once P8.2 exists, but can start in parallel

**Stop and benchmark**
- end-to-end baseline timeline on `R1`

### Milestone P8.4 — Report export and acceptance parser
**Build**
- `export_nsys_stats.py`
- `accept_profile.py`
- threshold file
- summary outputs

**Dependencies**
- P8.3
- sample `.nsys-rep`

**Parallel work**
- parser and docs can proceed in parallel once CSV samples exist

**Stop and benchmark**
- first formal acceptance summary on `R1`

### Milestone P8.5 — UVM/sync/graph diagnostic modes
**Build**
- `uvmAudit`
- `syncAudit`
- `graphDebug`

**Dependencies**
- P8.4

**Stop and benchmark**
- clean `R1` UVM audit
- clean `R1` sync audit

### Milestone P8.6 — Top-kernel NCU and sanitizer
**Build**
- `run_ncu_topkernels.py`
- sanitizer wrapper
- section file
- nightly reduced-case job

**Dependencies**
- P8.4
- P8.5 for stable baseline selection

**Parallel work**
- NCU wrapper and sanitizer wrapper can proceed in parallel

**Stop and benchmark**
- first top-five NCU set on `R1`
- clean sanitizer logs on `R1`

### Milestone P8.7 — Production-case acceptance
**Build**
- run the exact same framework on `R0`
- lock baseline artifacts

**Dependencies**
- P8.6

**Stop and benchmark**
- first approved `R0` acceptance bundle

### Milestone P8.8 — CI/nightly integration
**Build**
- nightly `R1`
- scheduled `R0`
- artifact retention policy

**Dependencies**
- P8.7

### Dependency graph

```text
P8.1 -> P8.2 -> P8.4 -> P8.5 -> P8.6 -> P8.7 -> P8.8
   \       \-> P8.3 -/
```

### Work that should remain experimental

- GPU metrics sampling as a gate
- compile-time memcheck instrumentation
- node-level graph regression automation
- custom SQL against Nsight SQLite tables beyond unavoidable fallback cases

### Where to stop before proceeding to the next project phase

Do not proceed to aggressive kernel optimization (future phase) until:
1. `R1` is UVM-clean in steady-state,
2. inner-loop sync calls are removed,
3. graph launch structure is measured and understood,
4. top-five kernels are identified with NCU reports,
5. sanitizer is clean on reduced tests.

---

## 10. Residual Governance Notes

Implementation-blocking Phase 8 decisions are frozen by `acceptance_manifest.md`, `support_matrix.md`, `graph_capture_support_matrix.md`, and `master_pin_manifest.md`. The remaining items are governance-only and do not change the code or profiling contract:

1. **Pin-manifest update policy**  
   Who may promote a newer CUDA/Nsight lane after the initial freeze, and what revalidation bundle is required?

2. **Compile-time memcheck adoption**  
   Should compile-time memcheck instrumentation be adopted after baseline Phase 8 completion, or deferred due to complexity and limitations [R15]?

3. **CI budget and artifact retention**  
   How much runtime and storage is available for nightly profile artifacts, especially `.nsys-rep`, `.sqlite`, and `.ncu-rep` files?

4. **Review ownership**  
   Who approves future baseline locks and acceptance-manifest revisions?

5. **Soft-fail merge policy**  
   Imported from `acceptance_manifest.md` / `acceptance_manifest.json`: a soft fail may be archived diagnostically, but it is not release-eligible or baseline-lock-eligible unless an explicit waiver is recorded against the manifest revision and tuple ID.

---

## Human review checklist

- [ ] NVTX naming hierarchy is stable and semantically correct.
- [ ] Hard/soft-fail disposition matches `acceptance_manifest.md` / `acceptance_manifest.json` and is applied consistently by the scripts.
- [ ] Output-stage DtoH classification policy is acceptable.
- [ ] Required toolchain versions are pinned and documented.
- [ ] `R1` and `R0` are the correct acceptance cases.
- [ ] External solver wrappers are instrumented where needed.
- [ ] CI cost is acceptable.

## Coding agent kickoff checklist

- [ ] Confirm actual runtime-selected solver class and touched source files.
- [ ] Add profiling wrapper library and build flags.
- [ ] Implement canonical `gpuRuntime.profiling` / `gpuRuntime.acceptance` parsing (`gpuProfilingDict` shim only if needed).
- [ ] Insert mandatory NVTX scopes.
- [ ] Add capture scripts and report export.
- [ ] Implement acceptance parser and thresholds.
- [ ] Run `R1` baseline, UVM audit, sync audit.
- [ ] Fix any hard failures before adding NCU/sanitizer automation.
- [ ] Lock first baseline artifact set.

## Highest risk implementation assumptions

1. The nozzle solver’s actual runtime-selected call path can be instrumented cleanly at semantic stage boundaries without invasive refactoring.
2. Production steady-state can reach **zero unexpected UVM bytes/faults** on the current SPUMA-based port.
3. Remaining synchronization problems can be localized and removed without destabilizing numerical behavior.
4. Nsight report naming/format will remain stable enough that built-in report scripts are sufficient for most parser needs.
5. Graphization will be mature enough soon that graph-launch counts become a meaningful gate rather than a moving target.

---

**Sources cited**

- **[R1]** SPUMA paper / arXiv HTML: https://arxiv.org/html/2512.22215v1
- **[R2]** SPUMA GPU-support wiki guidance: https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554
- **[R3]** OpenFOAM `foamRun` source / time-loop structure: https://cpp.openfoam.org/v11/foamRun_8C_source.html
- **[R4]** OpenFOAM `incompressibleVoF` class reference: https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html
- **[R5]** OpenFOAM alpha predictor / subcycling source: https://cpp.openfoam.org/v13/incompressibleMultiphaseVoF_2alphaPredictor_8C_source.html
- **[R6]** OpenFOAM MULES reference / terminology: https://cpp.openfoam.org/v13/MULES_8H.html
- **[R7]** NVIDIA Blackwell compatibility guide (CUDA 12.8 archive): https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html
- **[R8]** NVIDIA CUDA 12.8 toolkit release notes: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/
- **[R9]** NVIDIA Nsight Systems User Guide (current): https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **[R10]** NVIDIA Nsight Systems Analysis Guide: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
- **[R11]** NVIDIA Nsight Systems NVTX guidance (current and recent docs): https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- **[R12]** NVIDIA Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- **[R13]** NVIDIA Nsight Compute CLI / source-correlation guidance: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- **[R14]** NVIDIA Compute Sanitizer documentation: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- **[R15]** NVIDIA Compute Sanitizer release notes / instrumentation guidance: https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html
- **[R16]** foamExternalSolvers / AmgX4Foam README: https://gitlab.hpc.cineca.it/exafoam/foamExternalSolvers/-/blob/main/README.md?ref_type=heads
- **[R17]** NVIDIA RTX 5080 official specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
- **[R18]** CUDA 12.9 / NVTX removal context: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html
- **[R19]** Nsight Systems 2025.2 release-notes context on Blackwell / graph tracing: https://docs.nvidia.com/nsight-systems/2025.2/ReleaseNotes/index.html
- **[R20]** Nsight Systems historical user-guide note on report scripts adapting to schema changes: https://docs.nvidia.com/nsight-systems/2023.2/UserGuide/index.html
- **[R21]** Sparse SpMV performance background survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC7295357/
- **[R22]** NVIDIA compatibility/support matrix reflecting CUDA 12.8-era driver floor: https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/reference/support-matrix.html
