# 1. Executive overview

This run fully expands **Phase 1 — Blackwell bring up and tooling**. The rest of the project is carried forward only as dependency context so Phase 1 does not make choices that later block the pressure-swirl nozzle VOF/PIMPLE port.

The correct interpretation of Phase 1 is not “install CUDA and see whether SPUMA compiles.” It is: establish a **reproducible, Blackwell-native, profiler-ready, NVTX3-instrumented, PTX-verified SPUMA workstation baseline** on a single RTX 5080, then prove that baseline with **SPUMA-supported GPU solvers only**. That separation matters because SPUMA’s published paper was validated against **OpenFOAM-v2412** and used an older CUDA stack, while the same paper also states that **multiphase and heat-transfer solvers are not yet supported**. Your nozzle path therefore remains genuine new solver work for later phases; Phase 1 must isolate hardware/toolchain risk from solver-development risk. ([arXiv][1])

Phase 1 validates platform readiness only; it does not validate multiphase readiness.

I do **not** recommend taking the phrase “CUDA 12.8+” literally as the production default. CUDA 12.8 is the first toolkit that adds compiler support for **SM_120**, but CUDA 12.9.1 immediately follows with a compiler miscompilation fix for newer architectures and removes NVTX v2 from the toolkit entirely. The safer bring-up posture is a **two-lane toolchain**: **CUDA 12.9.1 as the required primary build lane**, and **CUDA 13.2 as an experimental verification lane** on the same host driver. ([NVIDIA Docs][2])

The hard deliverable for this phase is:

* a pinned SPUMA branch and environment manifest,
* a verified **`NVARCH=120` / `sm_120` + PTX** build path,
* NVTX3 wrappers and visible ranges in Nsight Systems,
* smoke cases for `laplacianFoam`, `simpleFoam`, and `pimpleFoam`,
* a Compute Sanitizer lane,
* a PTX-JIT compatibility test using `CUDA_FORCE_PTX_JIT=1`,
* archived Nsight Systems traces showing GPU execution and exposing any UVM faults,
* a machine-readable acceptance report that says whether the RTX 5080 workstation is ready for Phase 2. ([GitLab][3])

Phase 1 does **not** include `incompressibleVoF`, MULES, AmgX integration, nozzle boundary conditions, CUDA Graph capture productization, or custom CFD kernels. Those begin only after the Blackwell substrate is known-good. OpenFOAM’s VOF solver family uses a PIMPLE-based multiphase path with explicit alpha subcycling and `pressureCorrector()` / `alphaPredictor()` responsibilities, but those are Phase 5 concerns, not Phase 1 tasks. ([OpenFOAM C++][4])

# 2. Global architecture decisions

## Decision 1 — Use SPUMA on the OpenFOAM-v2412 line as the runtime base

**Sourced fact:** SPUMA’s published validation target is **OpenFOAM-v2412**. Its current GPU-support wiki lists `laplacianFoam`, `potentialFoam`, `icoFoam`, `pisoFoam`, `simpleFoam`, `SRFSimpleFoam`, and `pimpleFoam` as supported GPU solvers, and the paper states that **multiphase and heat-transfer solvers are currently not supported**. ([arXiv][1])

**Engineering inference:** The least risky way to begin is to standardize the workstation on the code line SPUMA already validates against, and to treat the nozzle solver as later feature work rather than as part of toolchain bring-up.

**Recommendation:** Freeze Phase 1 on a pinned SPUMA branch compatible with the v2412 line. Do not spend any Phase 1 time on OpenFOAM 12 backporting or on `incompressibleVoF` porting.

## Decision 2 — Primary target is native Linux, single workstation, single GPU

**Sourced fact:** NVIDIA’s current compute-capability table lists the **GeForce RTX 5080** as **compute capability 12.0**, and NVIDIA’s official RTX 5080 product page lists **10,752 CUDA cores**, **16 GB GDDR7**, and **960 GB/s** memory bandwidth. CUDA’s Linux installation guide supports Ubuntu 24.04 on x86_64, and OpenFOAM v2412 provides Ubuntu 24.04 packages. ([NVIDIA Developer][5])

**Engineering inference:** A single-GPU native Linux workstation is the lowest-variance path for early SPUMA bring-up. It avoids MPI, decomposition, and multi-GPU/UVM interactions that Phase 1 does not need.

**Recommendation:** Phase 1 is one process on one RTX 5080, pinned with `CUDA_VISIBLE_DEVICES=0`, no decomposition, no multi-GPU, no WSL, no Windows.

## Decision 3 — Use CUDA 12.9.1 as the required build lane; keep CUDA 13.2 as an experimental lane

**Sourced fact:** CUDA 12.8 is the first release that adds compiler support for **SM_100, SM_101, and SM_120**. CUDA 12.9.1 fixes a compiler miscompilation first observed starting with 12.8 on SM90/SM100, removes NVTX v2 from the toolkit, and is the final CUDA release with official Ubuntu 20.04 support. CUDA 13.2 requires Linux driver **595.45.04+** and later drivers are backward compatible with applications built by earlier toolkits. ([NVIDIA Docs][2])

**Engineering inference:** 12.8 is the first Blackwell-native transition release, but not the safest steady baseline for a new CFD port. 12.9.1 is the more conservative default. 13.2 is useful to validate forward compatibility and newer tooling, but should not gate initial success.

**Recommendation:** Required lane: **CUDA 12.9.1**. Experimental lane: **CUDA 13.2**. Recommended driver for both lanes: **595.45.04 or newer**.

## Decision 4 — Target `sm_120` natively and always include PTX

**Sourced fact:** RTX 5080 is CC **12.0**. NVIDIA’s Blackwell compatibility guide says applications should include PTX for forward compatibility, explains that `CUDA_FORCE_PTX_JIT=1` is the compatibility test, and warns that `sm_100a` / `compute_100a` PTX is not forward or backward compatible. NVIDIA also documents that `-arch=sm_XX` expands to cubin plus PTX, while explicit `-gencode` gives finer control over retained targets. ([NVIDIA Developer][5])

**Engineering inference:** The compatibility guide’s `sm_100` examples are illustrative, not product-specific. For **RTX 5080**, the correct native target is **`sm_120` / `compute_120`**, not `sm_100`.

**Recommendation:** Enforce `NVARCH=120`, inspect produced binaries, and require both native Blackwell cubin and PTX in the deliverable. Forbid `sm_100`, `sm_100a`, `compute_100a`, and any default that resolves to `80`.

## Decision 5 — NVTX3 migration is mandatory now, not later

**Sourced fact:** CUDA 12.8 deprecates NVTX v2 and instructs migration from `#include <nvtoolsext.h>` to `#include "nvtx3/nvtoolsext.h"`. CUDA 12.9.1 removes NVTX v2 from the toolkit. Nsight Systems uses NVTX ranges to show CPU regions and project them onto the GPU timeline. ([NVIDIA Docs][6])

**Engineering inference:** Deferring NVTX migration guarantees unnecessary code churn later and can break the build outright on the primary toolkit lane.

**Recommendation:** Add a thin NVTX3 wrapper in Phase 1 and replace direct NVTX2 includes immediately.

## Decision 6 — Use only SPUMA-supported solver settings in smoke tests

**Sourced fact:** SPUMA’s GPU-support wiki warns that unsupported features often do not fatal-error; they frequently manifest as simulation slowdowns caused by undesired host/device copies. The same page says `DIC` and `DILU` run on CPU and should be replaced by `aDIC` and `aDILU`; `PBiCGStab` is not supported; `PBiCG` should be used; GAMG smoothers available are `twoStageGaussSeidel`, `Richardson`, and `diagonal`. SPUMA’s paper appendix shows a validated configuration using `PBiCG` + `aDILU` for non-pressure equations and `GAMG` with `Richardson` for pressure. ([GitLab][7])

**Engineering inference:** Uncontrolled upstream tutorials are a poor bring-up vehicle because they can silently exercise unsupported CPU paths and produce misleading Nsight traces.

**Recommendation:** Build self-contained, repo-local smoke cases with audited `fvSolution` dictionaries. Default linear-solver setup should follow published SPUMA-supported patterns unless there is a strong reason not to.

## Decision 7 — Keep SPUMA’s memory pool enabled and instrument UVM behavior from day one

**Sourced fact:** SPUMA uses unified memory plus pooled allocation as part of its incremental porting model. The paper reports dramatic runtime improvement from enabling the memory pool on GPU runs and explicitly explains that incomplete ports manifest as slowdowns due to page migrations rather than incorrect answers. Nsight Systems documents CUDA UVM CPU/GPU page-fault tracing and warns that fault collection itself can add significant overhead, up to 70% in testing. ([arXiv][1])

**Engineering inference:** For Phase 1, the memory pool is baseline infrastructure, not an optimization toggle. UVM fault tracing must exist, but it must be used as a targeted diagnostic recipe, not as the default benchmark mode.

**Recommendation:** Do not disable the memory pool. Add explicit Nsight Systems recipes: one low-overhead timeline recipe, one heavy UVM-fault diagnostic recipe.

## Decision 8 — Choose a driver that also supports future graph profiling

**Sourced fact:** Nsight Compute’s graph node profiling features require driver **580+**, and device-launched graph node profiling requires **590+**. CUDA 12.8 release notes also document higher-than-normal context creation times with some **570.xx** drivers. CUDA 13.2’s toolkit driver requirement is **595.45.04+**. ([NVIDIA Docs][8])

**Engineering inference:** A “minimum driver that barely compiles” is the wrong optimization. Phase 3 will move toward CUDA Graphs, so the driver should already satisfy later tooling.

**Recommendation:** Standardize the workstation on **595.45.04 or newer** even when compiling primarily with CUDA 12.9.1.

Phase 1 is the authoritative source of the initial toolchain proposal; after Phase 1 sign-off, the accepted source/version/toolchain/profiler defaults move into the master pin manifest consumed by Phases 2–8. Later phase documents must reference that manifest rather than restating looser minima such as `CUDA 12.8+`.

# 3. Global assumptions and constraints

This specification assumes a **native Linux x86_64 workstation**, one RTX 5080, local admin rights for driver/toolkit installation, and a repo-local SPUMA build from source. It assumes Phase 1 is allowed to add tooling scripts, smoke cases, manifests, and NVTX wrappers, but not to redesign the solver numerics. Ubuntu 24.04 is the default because it sits at the clean intersection of current CUDA support and OpenFOAM v2412 packaging. ([NVIDIA Docs][9])

This phase also assumes that the nozzle solver itself is **out of scope**. That is not a convenience choice; it is a scoping guardrail. SPUMA’s supported solver list does not include the multiphase VOF solver family, and the paper states multiphase support is still future work. Later phases will target an OpenFOAM VOF/PIMPLE path with `alphaPredictor()`, `pressureCorrector()`, and explicit alpha subcycling, but Phase 1 must not conflate those responsibilities with Blackwell bring-up. ([GitLab][7])

Operational constraints for Phase 1:

* no MPI decomposition,
* no AmgX integration yet,
* no CUDA Graph implementation yet,
* no custom CFD kernels yet,
* no production performance claims,
* no dependence on large tutorial meshes,
* no acceptance based on absolute speedup against CPU.

The acceptance target is **substrate readiness**, not CFD throughput.

# 4. Cross-cutting risks and mitigation

**Risk: stale Blackwell guidance copied from the wrong source.**
The Blackwell compatibility guide uses `sm_100` examples, but the current CUDA GPU table lists RTX 5080 as **CC 12.0** and CUDA 12.8 adds **SM_120** support.
**Mitigation:** treat the compatibility guide examples as generic, not SKU-specific. Hard-fail the build if the detected GPU is 12.0 and the configured native target is not `120`. ([NVIDIA Docs][2])

**Risk: NVARCH silently stays at SPUMA’s documented default.**
SPUMA’s README still documents `export have_cuda=true` and `export NVARCH=<compute capability>`, with `80` as the default example.
**Mitigation:** `NVARCH=120` must be explicit and recorded in the manifest. The acceptance gate fails if `NVARCH` is absent or mismatched. ([GitLab][3])

**Risk: unsupported SPUMA features produce slowdown instead of failure.**
SPUMA’s wiki states this explicitly.
**Mitigation:** repo-local smoke cases only, audited `fvSolution`, Nsight UVM-fault traces on at least one transient case, and a rule that unexplained recurring page migrations during the inner solve loop are blockers. ([GitLab][7])

**Risk: driver/kernel UVM bring-up failure before solver debugging even starts.**
CUDA 12.9.1 documents Linux HMM initialization failures under certain kernels with KASLR enabled, with workarounds of disabling KASLR or disabling HMM for UVM via `options nvidia_uvm uvm_disable_hmm=1`.
**Mitigation:** add a managed-memory probe before SPUMA build and document both workarounds. Treat UVM init failure as an environment blocker, not a solver bug. Decision path: if the managed-memory probe fails with symptoms matching the documented HMM/KASLR issue, stop Phase 1 immediately, choose exactly one sanctioned workaround (`nokaslr` or `options nvidia_uvm uvm_disable_hmm=1`), record that choice in the host/toolchain manifest and acceptance report, rerun the native and managed probes, and resume SPUMA debugging only after both probes pass. ([NVIDIA Docs][10])

**Risk: profiler overhead corrupts conclusions.**
Nsight Systems documents significant overhead for UVM page-fault tracing, up to 70% in testing.
**Mitigation:** separate low-overhead timeline runs from fault-diagnostic runs; never use UVM-fault traces for timing acceptance. ([NVIDIA Docs][11])

**Risk: current SPUMA sync-heavy behavior is mistaken for a Blackwell regression.**
SPUMA’s paper shows tens of thousands of kernel launches over a few iterations and states that every kernel invocation in the profiled path is followed by `cudaDeviceSynchronize()`.
**Mitigation:** document this as expected Phase 1 baseline behavior. Do not try to “fix performance” in this phase; only verify that the behavior is observable and reproducible. ([arXiv][1])

**Risk: minimum supported driver is chosen instead of a future-proof one.**
570.xx has a documented context-creation issue in CUDA 12.8 notes, while later graph profiling needs 590+.
**Mitigation:** standardize on 595.45.04+ now. ([NVIDIA Docs][6])

# 5. Phase-by-phase implementation specification

The full project still consists of the plan phases you provided:

* **Phase 0 — freeze the reference problem**
* **Phase 1 — Blackwell bring up and tooling**
* **Phase 2 — replace SPUMA’s default memory posture for production use**
* **Phase 3 — execution model overhaul**
* **Phase 4 — linear algebra path**
* **Phase 5 — port the VOF core**
* **Phase 6 — nozzle-specific kernels and boundary conditions**
* **Phase 7 — custom CUDA kernels where needed**
* **Phase 8 — profiling and performance acceptance**

Only **Phase 1** is expanded below. The dependency constraints from the rest of the plan are:

* Phase 0 freeze artifacts provide the canonical CPU/source-version comparison line on the SPUMA/v2412 path; Phase 1 consumes that line rather than redefining it.
* Phase 2 will replace hot working sets with explicit DEVICE allocation, but Phase 1 keeps current SPUMA allocator behavior intact and observable.
* Phase 3 will remove today’s launch/sync-heavy execution, so Phase 1 must select a driver/tooling stack that supports later graph profiling.
* Phase 4 will compare native SPUMA linear solves and foamExternalSolvers/AmgX, but Phase 1 does not integrate them yet.
* Phase 5 onward will target a multiphase VOF/PIMPLE path that SPUMA does not currently support, so Phase 1 must not pretend solver readiness where only platform readiness exists. ([GitLab][12])

## Phase 1 — Blackwell bring up and tooling

### Purpose

Create a **reproducible, Blackwell-native development substrate** for SPUMA on an RTX 5080 so later solver-porting work starts from a known-good host, driver, compiler, profiler, and binary configuration.

### Why this phase exists

Without this phase, later failures become uninterpretable. A multiphase CFD port could fail because of:

* wrong native architecture target,
* PTX missing from binaries,
* NVTX2 header breakage on modern CUDA,
* unsupported SPUMA solver settings causing host/device copies,
* UVM/HMM kernel issues,
* profiler/tool mismatch,
* current SPUMA synchronization overhead being mistaken for a new regression.

Phase 1 exists to turn those unknowns into explicit, testable artifacts.

### Entry criteria

Required:

1. Phase 0 freeze artifacts are available so this phase stays attached to the canonical SPUMA/v2412 comparison line.
2. Physical access to a workstation containing one **GeForce RTX 5080**.
3. Ability to install or verify an NVIDIA Linux driver and CUDA toolkit.
4. A clean SPUMA checkout with a pinned branch or SHA.
5. Python 3 available for wrapper scripts and report generation.
6. Disk space for build artifacts, raw profiler traces, and temporary case copies.
7. Agreement that **no nozzle solver work** is part of Phase 1.

Recommended:

8. Ability to reboot the workstation if a driver/kernel module change is required.
9. Permission to alter boot/module settings if the documented HMM/KASLR issue appears. ([NVIDIA Developer][5])

### Exit criteria

Phase 1 exits successfully only if **all** of the following are true:

1. Host manifest records:

   * driver version,
   * toolkit version,
   * GCC version,
   * OS release,
   * kernel version,
   * Nsight Systems / Nsight Compute / Compute Sanitizer versions,
   * detected GPU name,
   * detected compute capability = `12.0`.
2. SPUMA builds in the required CUDA 12.9.1 lane with `have_cuda=true` and `NVARCH=120`.
3. At least one produced SPUMA-linked binary is inspected and shown to contain:

   * native Blackwell cubin for `sm_120`,
   * PTX.
4. A dedicated PTX-JIT run with `CUDA_FORCE_PTX_JIT=1` succeeds.
5. Repo-local smoke cases for `laplacianFoam`, `simpleFoam`, and `pimpleFoam` complete under the GPU build.
6. NVTX3 ranges appear in Nsight Systems for at least one smoke case.
7. Compute Sanitizer `memcheck` passes on at least the smallest smoke case.
8. Nsight Systems captures at least one UVM-fault diagnostic run and one low-overhead timeline run.
9. Any recurring host/device migration in the inner solve loop is either absent or fully explained and documented.
10. An acceptance report is generated, marked `PASS`, and records the accepted Phase 1 source/version/toolchain/profiler proposal that later phases will consume through the master pin manifest.
11. The archived artifact bundle is complete and includes, at minimum, the toolkit/driver matrix artifact, CUDA probe results, fatbinary inspection artifacts, PTX-JIT smoke artifacts, smoke-case logs/result JSONs, Nsight Systems smoke artifacts, and Compute Sanitizer smoke artifacts.

### Goals

1. Standardize the workstation on one primary CUDA lane.
2. Standardize the GPU target to `sm_120`.
3. Prove PTX inclusion and Blackwell compatibility.
4. Migrate instrumentation to NVTX3.
5. Establish a reusable script-driven bring-up pipeline.
6. Build small, deterministic smoke cases that remain valid through later phases.
7. Create profiling recipes that are safe to reuse later.
8. Capture raw artifacts now so future regressions have a baseline.
9. Avoid touching solver numerics except where instrumentation or supported settings require it.

### Non-goals

1. No nozzle mesh or nozzle case migration.
2. No `incompressibleVoF` port.
3. No MULES, alpha subcycling, surface tension, or pressure-swirl boundary work.
4. No AmgX integration.
5. No DEVICE-allocation redesign.
6. No CUDA Graph capture implementation.
7. No custom CFD kernel fusion.
8. No multi-GPU support.
9. No absolute performance claims against CPU or A100-class results.

### Technical background

The RTX 5080 is a consumer Blackwell GPU listed by NVIDIA as **compute capability 12.0**. CUDA 12.8 is the first toolkit release to add compiler support for **SM_120**, and NVIDIA’s compatibility guidance for Blackwell states that applications should include PTX so they remain runnable via JIT on future-compatible hardware. Blackwell support is therefore not just “new driver, old build.” The binary contents matter. ([NVIDIA Docs][2])

SPUMA is an OpenFOAM GPU-porting fork built around a portable programming abstraction and pooled memory. Its paper explicitly positions unified memory plus pooling as an incremental-porting mechanism: incomplete GPU support often still returns correct answers, but page migrations can dominate performance. That is useful for development, but dangerous for bring-up because a case can “run” while silently falling off the intended device-resident path. Phase 1 therefore treats profiling as part of correctness, not just performance analysis. ([arXiv][1])

Phase 1 is intentionally solver-light. SPUMA’s current public GPU-support guidance covers steady and transient single-phase solvers such as `simpleFoam` and `pimpleFoam`, while the paper states multiphase remains unsupported. That makes Phase 1 the correct place to validate the workstation using **supported paths only**, so later VOF failures are attributable to new solver work rather than to Blackwell/toolchain defects. ([GitLab][7])

### Research findings relevant to this phase

1. **RTX 5080 target facts.** NVIDIA lists GeForce RTX 5080 as **CC 12.0**, and its official product page lists **10,752 CUDA cores**, **16 GB GDDR7**, and **960 GB/s** memory bandwidth. ([NVIDIA Developer][5])

2. **Blackwell compiler support first appears in CUDA 12.8.** The CUDA 12.8 features archive explicitly adds compiler support for **SM_120** and adds `ELSE` / `SWITCH` support for CUDA Graphs. ([NVIDIA Docs][2])

3. **PTX is the compatibility floor.** NVIDIA’s Blackwell compatibility guide says PTX is forward compatible, recommends including PTX in all applications, and defines `CUDA_FORCE_PTX_JIT=1` as the runtime test for compatibility. It also warns that `sm_100a` / `compute_100a` PTX is not forward or backward compatible. ([NVIDIA Docs][13])

4. **CUDA 12.8 is not the safest default baseline.** CUDA 12.9.1 fixes a compiler miscompilation first observed starting with 12.8 on newer architectures and removes NVTX v2 from the toolkit. NVIDIA also documents higher-than-normal context creation times with some 570.xx drivers in the 12.8 release notes. ([NVIDIA Docs][6])

5. **Driver choice should exceed the minimum.** CUDA 12.9.1 requires Linux driver **575.57.08+**; CUDA 13.2 requires **595.45.04+**; later drivers are backward compatible. Nsight Compute graph-node profiling for device-launched graphs requires **590+**. ([NVIDIA Docs][14])

6. **Ubuntu 24.04 is a clean host baseline.** NVIDIA’s installation guide lists Ubuntu 24.04.3 x86_64 with GCC 13.3 as a supported platform, with x86_64 host-compiler support through GCC 15.x. OpenFOAM v2412 publishes Ubuntu 24.04 packages. CUDA 12.9 is also the last release with official Ubuntu 20.04 support, making 24.04 the less transient baseline for new work. ([NVIDIA Docs][9])

7. **NVTX3 is mandatory.** NVTX v2 is deprecated in CUDA 12.8 and removed in CUDA 12.9.1. Nsight Systems relies on NVTX ranges to make CPU regions visible and project them onto the GPU timeline. ([NVIDIA Docs][6])

8. **Profiler stack maturity on Blackwell is sufficient.** CUPTI 12.8 adds Blackwell support and introduces experimental hardware event timestamping on Blackwell. Nsight Compute 2025.3 adds or improves Blackwell support and node-level graph profiling. Compute Sanitizer added Blackwell architecture support in 2024.4 and additional Blackwell GPU support in 2025.1. ([NVIDIA Docs][15])

9. **SPUMA operational guidance is current but incomplete.** The public GPU-support wiki updated June 10, 2025 lists supported GPU solvers and explicitly warns that unsupported features often degrade to undesired host/device copies instead of fatal errors. It also gives recommended `fvSolution` constraints and an `nsys` profiling recipe including UVM fault tracing. ([GitLab][7])

10. **Current SPUMA execution is sync-heavy.** The paper reports **53,259 kernel launches** over five profiled `simpleFoam` iterations on one GPU and states every kernel invocation is followed by `cudaDeviceSynchronize()`. This means a sync-dominated CUDA API trace in Phase 1 is consistent with upstream SPUMA behavior, not evidence of a broken Blackwell port. ([arXiv][1])

11. **Memory pool is non-optional baseline behavior.** SPUMA’s paper shows 1-GPU runtime on a reference case dropping from **822.68 s** without a pool to roughly **141 s** with fixed-size/Umpire pooling. Phase 1 should therefore inherit the memory pool as baseline infrastructure. ([arXiv][1])

12. **Source conflict to handle explicitly.** SPUMA’s published software stack used CUDA 12.1 on earlier cluster hardware, which is stale for RTX 5080. Separately, the Blackwell compatibility guide’s examples emphasize `compute_100`, while current NVIDIA product tables list consumer RTX 5080 at **12.0**. Those sources are not actually contradictory; they operate at different abstraction levels. The correct consumer-Blackwell native target for this workstation is `sm_120`. ([arXiv][1])

### Design decisions

#### 1. Primary host baseline

**Sourced fact:** Ubuntu 24.04.3 x86_64 and GCC 13.3 are in NVIDIA’s current Linux support matrix, and OpenFOAM v2412 publishes Ubuntu 24.04 packages. ([NVIDIA Docs][9])

**Engineering inference:** This is the lowest-friction host combination for a Linux-native SPUMA workstation in 2026.

**Recommendation:** Default Phase 1 host = **Ubuntu 24.04 LTS** native, not WSL, not Windows.

#### 2. Primary toolkit lane

**Sourced fact:** CUDA 12.8 first adds SM_120 support; CUDA 12.9.1 fixes a compiler issue observed starting with 12.8 and removes NVTX v2. ([NVIDIA Docs][2])

**Engineering inference:** 12.9.1 is the safer baseline for a new Blackwell CFD port.

**Recommendation:** Build and gate Phase 1 on **CUDA 12.9.1**.

#### 3. Experimental forward lane

**Sourced fact:** CUDA 13.2 requires driver **595.45.04+** and bundles current Nsight tools. ([NVIDIA Docs][14])

**Engineering inference:** A second lane catches future compatibility drift without destabilizing the primary lane.

**Recommendation:** Maintain **CUDA 13.2** as compile-and-smoke experimental lane only.

#### 4. Driver baseline

**Sourced fact:** 12.9.1 needs 575.57.08+, 13.2 needs 595.45.04+, later drivers remain backward compatible, and graph node profiling for device-launched graphs requires 590+. 570.xx also has a documented context-creation issue in 12.8 notes. ([NVIDIA Docs][14])

**Engineering inference:** The minimum driver is not the right target.

**Recommendation:** Standardize on **595.45.04 or newer**.

#### 5. Native architecture target

**Sourced fact:** RTX 5080 is CC 12.0; CUDA 12.8 adds SM_120 support; PTX should be included; `100a` targets are special-case and not forward/backward compatible. ([NVIDIA Developer][5])

**Engineering inference:** Any `sm_100` or `80` carry-over is a bring-up defect.

**Recommendation:** Build for **`sm_120` + `compute_120` PTX** and fail hard on anything else.

#### 6. Instrumentation layer design

**Sourced fact:** NVTX2 is removed in the primary toolkit lane; NVTX ranges are the mechanism Nsight Systems uses to correlate CPU and GPU activity. ([NVIDIA Docs][10])

**Engineering inference:** Direct scattered NVTX calls across the codebase will be brittle.

**Recommendation:** Add a thin wrapper layer:

* `ScopedNvtxRange`
* `nvtxMark`
* optional thread/stream naming helpers
  and compile them behind a single `SPUMA_ENABLE_NVTX` switch.

#### 7. Smoke-case philosophy

**Sourced fact:** Unsupported SPUMA features can degrade to host/device copies, and unsupported preconditioners/smoothers exist. Published SPUMA validation settings use `PBiCG` + `aDILU` and `GAMG` + `Richardson`. ([GitLab][7])

**Engineering inference:** Upstream tutorials are too noisy and too unstable for reproducible bring-up.

**Recommendation:** Vendor three small, deterministic, repo-local cases and audit their solver settings manually.

#### 8. UVM probe separation

**Sourced fact:** SPUMA uses UM plus pooling, and incomplete ports can degrade via page migrations. CUDA 12.9.1 also documents HMM/UVM initialization problems on some Linux kernels. ([arXiv][1])

**Engineering inference:** A first probe that uses only `cudaMalloc` is useful to isolate raw CUDA bring-up from UVM pathologies.

**Recommendation:** Implement one **native CUDA probe** and one **managed-memory probe** before building SPUMA.

#### 9. Profiling recipe split

**Sourced fact:** SPUMA recommends `nsys` with UVM fault tracing; Nsight Systems documents significant overhead and up to 70% overhead in testing for UVM fault collection. ([GitLab][7])

**Engineering inference:** One profiling mode cannot serve both observability and timing.

**Recommendation:** Ship two recipes:

* `timeline_basic`
* `timeline_um_fault`

#### 10. Binary inspection is mandatory

**Sourced fact:** Blackwell compatibility depends on embedded cubin/PTX content, not just build success. NVIDIA explicitly recommends PTX verification via `CUDA_FORCE_PTX_JIT=1`. ([NVIDIA Docs][13])

**Engineering inference:** A green compile without binary inspection can still be wrong.

**Recommendation:** Binary inspection and PTX-JIT execution are acceptance-gate steps, not optional diagnostics.

#### 11. CPU-only SPUMA sanity lane ownership

**Engineering inference:** A very small CPU-only SPUMA sanity lane in Phase 1 reduces ambiguity around the SPUMA/v2412 CPU comparison line without turning this phase into the owner of CPU reference freezing.

**Recommendation:** Add an optional but recommended CPU-only build-and-smoke sanity run when Phase 0 Baseline B is not already green. Archive it only as de-risking evidence. Phase 0 remains the owner of canonical CPU reference bundles, tolerances, and comparison sign-off.

### Alternatives considered

**Alternative: standardize on CUDA 12.8 because it is the first Blackwell-native release.**
Rejected. It is the first Blackwell-native compiler, but its immediate successor carries correctness fixes for newer-architecture code generation and removes deprecated NVTX2. 12.8 is acceptable as a historical minimum, not as the preferred 2026 bring-up default. ([NVIDIA Docs][2])

**Alternative: standardize only on CUDA 13.2.**
Rejected as the primary lane. It is useful and should be tested, but it is a larger divergence from SPUMA’s historical stack than necessary for first workstation bring-up. Keep it experimental until the 12.9.1 lane is green. ([arXiv][1])

**Alternative: rely on SPUMA’s documented `NVARCH` example and avoid touching build logic.**
Rejected. The documented example defaults to `80`; that is wrong for RTX 5080. Build scripts may also silently omit PTX or ignore the arch variable. ([GitLab][3])

**Alternative: use upstream tutorials directly.**
Rejected. Silent CPU fallbacks and unsupported settings make them poor acceptance cases. Repo-local smoke cases are more reproducible.

**Alternative: skip NVTX until later phases.**
Rejected. NVTX2 removal in CUDA 12.9.1 turns that delay into a build risk. ([NVIDIA Docs][10])

**Alternative: treat UVM-fault tracing as the default profiling mode.**
Rejected. The overhead is too high for normal timing. ([NVIDIA Docs][11])

**Alternative: bring up on WSL first.**
Rejected for Phase 1. CUDA tool components exist for WSL, but native Linux is the lower-variance platform for SPUMA/OpenFOAM, and CUPTI’s Blackwell HES path is not supported on WSL. ([NVIDIA Docs][14])

### Unresolved questions

Only residual policy overrides belong here. The default pins above stand unless an explicit override is approved and recorded in the master pin manifest or operations notes.

1. Whether local operations policy permits boot-parameter changes (`nokaslr`) if the documented HMM issue appears.
2. Whether the team wants to patch SPUMA build scripts directly or keep Phase 1 largely wrapper-based.
3. Whether the experimental CUDA 13.2 lane is promoted from recommended verification to a required pre-Phase-2 gate; absent that explicit promotion, a green CUDA 12.9.1 primary lane remains sufficient.

### Interfaces and dependencies

#### External dependencies

* NVIDIA Linux driver, **595.45.04+** recommended.
* CUDA toolkit **12.9.1** required.
* CUDA toolkit **13.2** optional experimental.
* GCC 13.x on Ubuntu 24.04 preferred.
* SPUMA repository checkout.
* Python 3.
* Nsight Systems.
* Nsight Compute.
* Compute Sanitizer.
* `cuobjdump`.
* `nvidia-smi`. ([NVIDIA Docs][14])

#### Internal interfaces to add

1. **Environment validation CLI**

   * `tools/bringup/env/check_host_env.sh`
   * output: `host_env.json`

2. **CUDA runtime probe**

   * `tools/bringup/src/validate_cuda_runtime.cpp`
   * output: `cuda_probe.json`

3. **Build wrapper**

   * `tools/bringup/build/build_spuma_cuda.sh`
   * input: `toolchain lane`, `build mode`, `NVARCH`
   * outputs: build log, linked binary list, manifest

4. **Binary inspection**

   * `tools/bringup/build/inspect_fatbinary.sh`
   * inputs: binary path
   * outputs: `sass.txt`, `ptx.txt`, `fatbinary_report.json`

5. **Smoke runner**

   * `tools/bringup/run/run_smoke_case.sh`
   * inputs: case path, solver, profile mode
   * outputs: log, copied case directory, result JSON

6. **Profiler wrappers**

   * `tools/bringup/run/run_nsys.sh`
   * `tools/bringup/run/run_ncu.sh`

7. **Sanitizer wrapper**

   * `tools/bringup/run/run_compute_sanitizer.sh`

8. **PTX-JIT compatibility runner**

   * `tools/bringup/run/check_ptx_jit.sh`

9. **Acceptance gate**

   * `tools/bringup/python/acceptance_gate.py`

10. **Instrumentation API**

* `src/gpu/common/NvtxScope.H`
* optional `src/gpu/common/NvtxScope.C`

#### Dependency order

`check_host_env` → `validate_cuda_runtime` → `build_spuma_cuda` → `inspect_fatbinary` → `run_smoke_case` → `run_compute_sanitizer` → `run_nsys` → `check_ptx_jit` → `acceptance_gate`

### Data model / memory model

#### Tooling-side host data structures

Use JSON for all machine-readable outputs.

**`ToolchainManifest`**

```json
{
  "hostname": "ws-rtx5080-01",
  "os_release": "Ubuntu 24.04.3 LTS",
  "kernel": "6.8.0-83-generic",
  "driver_version": "595.45.04",
  "cuda_toolkit": "12.9.1",
  "gcc_version": "13.3.0",
  "nsys_version": "...",
  "ncu_version": "...",
  "compute_sanitizer_version": "...",
  "spuma_git_sha": "...",
  "have_cuda": true,
  "nvarch": 120
}
```

**`CudaProbeResult`**

```json
{
  "device_index": 0,
  "device_name": "NVIDIA GeForce RTX 5080",
  "cc_major": 12,
  "cc_minor": 0,
  "total_global_mem_bytes": 17179869184,
  "managed_memory": true,
  "concurrent_managed_access": true,
  "unified_addressing": true,
  "native_kernel_ok": true,
  "managed_memory_probe_ok": true
}
```

**`SmokeCaseSpec`**

```json
{
  "case_name": "channelTransient",
  "solver": "pimpleFoam",
  "profile_modes": ["none", "nsys_basic", "nsys_um_fault"],
  "required_checks": {
    "exit_code": 0,
    "no_nan_inf": true,
    "gpu_activity_present": true
  }
}
```

**`AcceptanceReport`**

```json
{
  "phase": "phase1_blackwell_bringup",
  "status": "PASS",
  "blockers": [],
  "warnings": [],
  "binaries": [...],
  "cases": [...],
  "profiles": [...],
  "ptx_jit": {"passed": true}
}
```

#### Runtime-side memory model

Phase 1 deliberately **does not redesign SPUMA’s runtime memory model**.

Rules:

1. Existing SPUMA solver allocations remain as-is.
2. The new **CUDA runtime probe** must use **plain `cudaMalloc`** for its native-kernel test to separate raw CUDA execution from UVM behavior.
3. The same probe may then run a small **managed-memory test** to detect environment-level HMM/UVM failures early.
4. The NVTX wrapper must be host-only and must not allocate dynamic GPU memory.
5. No new host/device synchronization points may be inserted into SPUMA core numerics in this phase, other than normal process boundaries and explicit probe tests.
6. Data leaving the GPU in Phase 1 is limited to:

   * normal OpenFOAM field writes,
   * solver logs,
   * profiler trace data,
   * explicit probe result copies,
   * any unintended UVM migrations that profiling is meant to reveal.

### Algorithms and control flow

Phase 1 is a deterministic pipeline.

#### Control-flow states

1. **Environment discovery**
2. **Raw CUDA validation**
3. **Managed-memory validation**
4. **SPUMA build**
5. **Binary inspection**
6. **Smoke execution**
7. **Sanitizer execution**
8. **Profiler execution**
9. **PTX-JIT compatibility execution**
10. **Acceptance gating**

#### State machine

* Fail immediately on driver/toolkit mismatch.
* Fail immediately if detected GPU is not CC 12.0 on the target machine.
* Fail immediately if `NVARCH != 120`.
* Fail immediately if binary inspection finds no PTX.
* Fail immediately if PTX-JIT run fails.
* Fail immediately if NVTX ranges are absent in the profile.
* Fail immediately if Compute Sanitizer memcheck reports actionable errors.
* Warn, but do not fail, if the experimental CUDA 13.2 lane fails after the required lane has passed.

#### Orchestration pseudocode

```python
def phase1_bringup(cfg):
    host = check_host_env(cfg)
    assert host.driver_version >= cfg.min_driver
    assert host.cuda_toolkit == cfg.required_toolkit

    probe = run_cuda_probe(cfg)
    assert probe.device_name_contains("RTX 5080")
    assert (probe.cc_major, probe.cc_minor) == (12, 0)
    assert probe.native_kernel_ok
    assert probe.managed_memory_probe_ok

    build = build_spuma_cuda(cfg)
    assert build.have_cuda is True
    assert build.nvarch == 120

    fatbin = inspect_binaries(build.binary_paths, expected_sm=120)
    assert fatbin.ptx_present
    assert fatbin.native_sm_present

    cases = []
    for case in cfg.required_smoke_cases:
        warmup(case, build)
        result = run_smoke_case(case, build, profile="none")
        assert result.exit_code == 0
        assert result.no_nan_inf
        assert result.gpu_activity_present
        cases.append(result)

    sanitizer = run_compute_sanitizer(cfg.smallest_case, build)
    assert sanitizer.memcheck_errors == 0

    nsys_basic = run_nsys(cfg.timeline_case, build, recipe="basic")
    assert nsys_basic.nvtx_ranges_present
    assert nsys_basic.gpu_kernels_present

    nsys_um = run_nsys(cfg.timeline_case, build, recipe="um_fault")
    assert nsys_um.trace_captured

    ptx_jit = run_ptx_jit(cfg.smallest_case, build)
    assert ptx_jit.passed

    report = acceptance_gate(host, probe, build, fatbin, cases, sanitizer,
                             nsys_basic, nsys_um, ptx_jit)
    assert report.status == "PASS"
    return report
```

#### NVTX wrapper pseudocode

```cpp
// src/gpu/common/NvtxScope.H
#pragma once

#ifdef SPUMA_ENABLE_NVTX
#include "nvtx3/nvtoolsext.h"
#endif

namespace spuma::profiling
{
class ScopedNvtxRange
{
public:
    explicit ScopedNvtxRange(const char* name)
    {
#ifdef SPUMA_ENABLE_NVTX
        nvtxEventAttributes_t attr{};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        nvtxRangePushEx(&attr);
#endif
    }

    ~ScopedNvtxRange()
    {
#ifdef SPUMA_ENABLE_NVTX
        nvtxRangePop();
#endif
    }
};

inline void mark(const char* name)
{
#ifdef SPUMA_ENABLE_NVTX
    nvtxMarkA(name);
#endif
}
}  // namespace spuma::profiling

#define SPUMA_NVTX_RANGE(name) \
    ::spuma::profiling::ScopedNvtxRange spuma_nvtx_range_##__LINE__(name)
```

#### CUDA runtime probe pseudocode

```cpp
// tools/bringup/src/validate_cuda_runtime.cpp
int main(int argc, char** argv)
{
    // Force context creation early.
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) return fail("cudaFree(0)", err);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 1) return fail("No CUDA devices", cudaErrorNoDevice);

    const int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);

    if (!(p.major == 12 && p.minor == 0))
        return fail("Unexpected compute capability");

    // Native CUDA path: no managed memory.
    float* d = nullptr;
    cudaMalloc(&d, 1024 * sizeof(float));
    launch_fill_kernel<<<4, 256>>>(d, 1024, 1.0f);
    cudaDeviceSynchronize();
    cudaFree(d);

    // Managed-memory probe: isolates HMM/UVM issues before SPUMA build.
    float* m = nullptr;
    cudaMallocManaged(&m, 1024 * sizeof(float));
    launch_fill_kernel<<<4, 256>>>(m, 1024, 2.0f);
    cudaDeviceSynchronize();
    volatile float checksum = 0.0f;
    for (int i = 0; i < 1024; ++i) checksum += m[i];
    cudaFree(m);

    write_json_manifest(p, checksum, /*managed_ok=*/true);
    return 0;
}
```

### Required source changes

#### Must change now

1. **Add NVTX3 wrapper**

   * Create a single wrapper header under a GPU/common utility path.
   * Replace any direct inclusion of `<nvtoolsext.h>` with the wrapper or with `"nvtx3/nvtoolsext.h"` through the wrapper.
   * Add `SPUMA_ENABLE_NVTX` build toggle.

2. **Add Blackwell bring-up env script**

   * Must export:

     * `have_cuda=true`
     * `NVARCH=120`
     * explicit `CUDA_HOME`
     * `PATH` and `LD_LIBRARY_PATH` pointing at the intended toolkit
     * `CUDA_VISIBLE_DEVICES=0`
   * Optional:

     * `OMP_NUM_THREADS=1`
     * debug/profiling flags

3. **Add environment and runtime probes**

   * Shell host probe.
   * C++ CUDA probe.

4. **Add binary inspection scripts**

   * PTX presence check.
   * native cubin target check.

5. **Add repo-local smoke cases**

   * `laplacianFoam` linear scalar case.
   * `simpleFoam` steady channel-like case.
   * `pimpleFoam` transient channel-like case.

6. **Add wrapper scripts for Nsight and Compute Sanitizer**

   * avoid ad hoc per-engineer commands.

7. **Add acceptance gate**

   * JSON output required.

#### Change only if needed

8. **Patch SPUMA build logic** if current `NVARCH` handling does not produce `sm_120` + PTX.

   * Preferred patch: explicit `-gencode=arch=compute_120,code=sm_120`
   * and `-gencode=arch=compute_120,code=compute_120`
   * only if the existing build logic cannot be trusted to retain PTX. NVIDIA documents that `-arch=sm_XX` is shorthand for native cubin plus PTX, but verify, do not assume. ([NVIDIA Docs][13])

9. **Patch minimal solver entry points for NVTX ranges**

   * only where no common execution hook exists.

#### Do not change in this phase

10. No modification of `incompressibleVoF`, `MULES`, nozzle boundary conditions, pressure-swirl logic, or later device-allocator design.

### Proposed file layout and module boundaries

This is a **proposed** layout. If the repository already has equivalent directories, modify in place rather than creating duplicates.

```text
repo-root/
  tools/
    bringup/
      env/
        check_host_env.sh
        gpu_blackwell_env.sh
        collect_env_manifest.py
      src/
        validate_cuda_runtime.cpp
      build/
        build_spuma_cuda.sh
        inspect_fatbinary.sh
      run/
        run_smoke_case.sh
        run_compute_sanitizer.sh
        run_nsys.sh
        run_ncu.sh
        check_ptx_jit.sh
      python/
        acceptance_gate.py
        parse_solver_log.py
        parse_probe_manifest.py
  src/
    gpu/
      common/
        NvtxScope.H
        DeviceInfo.H         (optional)
        DeviceInfo.C         (optional)
  cases/
    bringup/
      laplacianFoam/
        cubeLinear/
          system/
          constant/
          0/
          acceptance.json
      simpleFoam/
        channelSteady/
          system/
          constant/
          0/
          acceptance.json
      pimpleFoam/
        channelTransient/
          system/
          constant/
          0/
          acceptance.json
  docs/
    bringup/
      phase1_blackwell.md
      profiling_recipes.md
```

#### Module boundaries

* `tools/bringup/env`: host/toolchain discovery only.
* `tools/bringup/src`: native CUDA probe binaries only.
* `tools/bringup/build`: build and fatbinary inspection only.
* `tools/bringup/run`: execution wrappers only.
* `tools/bringup/python`: parsing and acceptance logic only.
* `src/gpu/common`: reusable runtime instrumentation only.
* `cases/bringup`: deterministic acceptance cases only.

No Phase 1 tooling file may depend on later-phase multiphase/nozzle code.

### Pseudocode

#### `check_host_env.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

out="${1:-host_env.json}"

gpu_csv=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)
nvcc_ver=$(nvcc --version || true)
gcc_ver=$(gcc --version | head -n1)
nsys_ver=$(nsys --version || true)
ncu_ver=$(ncu --version || true)
cs_ver=$(compute-sanitizer --version || true)
os_rel=$(grep PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '"')
kernel=$(uname -r)

python3 - <<PY
import json, os
data = {
  "gpu_csv": ${gpu_csv@Q},
  "nvcc_version": ${nvcc_ver@Q},
  "gcc_version": ${gcc_ver@Q},
  "nsys_version": ${nsys_ver@Q},
  "ncu_version": ${ncu_ver@Q},
  "compute_sanitizer_version": ${cs_ver@Q},
  "os_release": ${os_rel@Q},
  "kernel": ${kernel@Q},
}
json.dump(data, open(${out@Q}, "w"), indent=2)
PY
```

#### `build_spuma_cuda.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

lane="${1:-cuda12.9.1}"
mode="${2:-relwithdebinfo}"

source tools/bringup/env/gpu_blackwell_env.sh "$lane"

# Prefer repo-native environment setup.
if [[ -f etc/bashrc ]]; then
  source etc/bashrc
fi

export have_cuda=true
export NVARCH=120

# Optional build-mode flags injected through repo-native mechanisms.
case "$mode" in
  debug)
    export SPUMA_EXTRA_NVCC_FLAGS="-O0 -g -lineinfo"
    export SPUMA_EXTRA_CXX_FLAGS="-O0 -g -fno-omit-frame-pointer"
    ;;
  relwithdebinfo)
    export SPUMA_EXTRA_NVCC_FLAGS="-O3 -g -lineinfo"
    export SPUMA_EXTRA_CXX_FLAGS="-O3 -g -fno-omit-frame-pointer"
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    exit 2
    ;;
esac

# Use repo-native build entrypoint; do not invent a second build system.
if [[ -x ./Allwmake ]]; then
  ./Allwmake > "artifacts/build_${lane}_${mode}.log" 2>&1
else
  echo "Missing repo-native build entrypoint" >&2
  exit 3
fi
```

#### `check_ptx_jit.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

binary="$1"
solver="$2"
case_dir="$3"

# First verify PTX is embedded.
cuobjdump --dump-ptx "$binary" > ptx.txt
if ! grep -q ".target" ptx.txt; then
  echo "PTX missing" >&2
  exit 1
fi

# Then force PTX JIT at runtime.
export CUDA_FORCE_PTX_JIT=1
"$solver" -case "$case_dir" > ptx_jit.log 2>&1
unset CUDA_FORCE_PTX_JIT
```

### Step-by-step implementation guide

#### Step 1 — Pin the repository state and create a Phase 1 branch

* **Modify:** create a dedicated branch such as `phase1-blackwell-bringup`; record git SHA in `docs/bringup/phase1_blackwell.md`.
* **Why:** all later profiler artifacts must be attributable to one exact code state.
* **Expected output:** branch exists; manifest template includes repo SHA.
* **Verify:** `git rev-parse HEAD` captured in docs and manifest.
* **Likely breakages:** uncommitted local changes, detached HEAD, submodule drift.

#### Step 2 — Add environment discovery tooling

* **Modify:** add `check_host_env.sh` and `collect_env_manifest.py`.
* **Why:** later failures are often environment mismatches, not code defects.
* **Expected output:** `host_env.json`.
* **Verify:** JSON contains driver, toolkit, GCC, OS, kernel, profiler versions.
* **Likely breakages:** missing tools in `PATH`, conflicting multiple CUDA installs.

#### Step 3 — Add native CUDA and managed-memory probe binary

* **Modify:** add `tools/bringup/src/validate_cuda_runtime.cpp`.
* **Why:** isolate raw CUDA viability before touching SPUMA, then detect UVM/HMM failure explicitly.
* **Expected output:** `cuda_probe.json`.
* **Verify:** reports `device_name` contains RTX 5080; `cc_major=12`, `cc_minor=0`; both native and managed probes pass.
* **Likely breakages:** old driver, missing toolkit headers/libs, HMM init failure, stale `LD_LIBRARY_PATH`.

#### Step 4 — Add Blackwell environment wrapper

* **Modify:** add `gpu_blackwell_env.sh`.
* **Why:** SPUMA’s README requires `have_cuda=true` and `NVARCH=<compute capability>`; Phase 1 must make `NVARCH=120` impossible to forget. ([GitLab][3])
* **Expected output:** a single sourceable script that selects toolkit lane and exports `NVARCH=120`.
* **Verify:** `echo $NVARCH` prints `120`; `which nvcc` points at the intended toolkit.
* **Likely breakages:** environment overridden by previously sourced CUDA/OpenFOAM scripts.

#### Step 5 — Add NVTX3 wrapper and audit old includes

* **Modify:** create `src/gpu/common/NvtxScope.H`; grep the repo for `<nvtoolsext.h>` and remove direct usage.
* **Why:** CUDA 12.9.1 removes NVTX2. ([NVIDIA Docs][10])
* **Expected output:** wrapper builds cleanly in both primary and experimental lanes.
* **Verify:** repo grep shows no remaining direct NVTX2 includes.
* **Likely breakages:** include-path assumptions, C/C++ namespace collisions, duplicate wrapper definitions.

#### Step 6 — Identify minimal NVTX insertion points

* **Modify:** insert coarse-grain ranges into:

  * process startup,
  * case setup,
  * main iteration/timestep loop,
  * linear solve section,
  * write/output section.
* **Why:** Phase 1 needs visible ranges in Nsight without broad code churn.
* **Expected output:** at least 4–5 named ranges visible in Nsight Systems.
* **Verify:** `.nsys-rep` shows ranges.
* **Likely breakages:** patching too many files, touching runtime-selection registration paths, name collisions.

#### Step 7 — Add build wrapper and build modes

* **Modify:** add `build_spuma_cuda.sh`; define `debug` and `relwithdebinfo`.
* **Why:** repeated manual builds are error-prone; Phase 1 requires repeatability.
* **Expected output:** build logs per lane and mode.
* **Verify:** successful SPUMA build in required lane; binaries archived.
* **Likely breakages:** repo-native build entrypoint differences, unsupported host compiler, wrong toolkit symlink.

#### Step 8 — Inspect produced binaries for native target and PTX

* **Modify:** add `inspect_fatbinary.sh`.
* **Why:** build success alone is insufficient for Blackwell compatibility.
* **Expected output:** `fatbinary_report.json`, `ptx.txt`, `sass.txt`.
* **Verify:** report says `native_sm_present: [120]`, `ptx_present: true`.
* **Likely breakages:** build scripts ignore `NVARCH`; no PTX retained; binary inspection script parsing too brittle.

#### Step 9 — Add three repo-local smoke cases

* **Modify:** create:

  * `cubeLinear` for `laplacianFoam`,
  * `channelSteady` for `simpleFoam`,
  * `channelTransient` for `pimpleFoam`.
* **Why:** deterministic, low-cost acceptance substrate.
* **Expected output:** self-contained blockMesh-driven cases and `acceptance.json` files.
* **Verify:** each case runs from a copied scratch directory with no external asset dependencies.
* **Likely breakages:** tutorial inheritance, missing meshes, unsupported solver settings.

#### Step 10 — Audit `fvSolution` in smoke cases

* **Modify:** set supported GPU solver settings explicitly:

  * `PBiCG` instead of `PBiCGStab`,
  * `aDIC` / `aDILU` instead of `DIC` / `DILU`,
  * if using `GAMG`, use `Richardson`, `twoStageGaussSeidel`, or `diagonal`.
* **Why:** SPUMA’s wiki warns unsupported options can silently fall back or migrate data. ([GitLab][7])
* **Expected output:** audited dictionaries committed with cases.
* **Verify:** `fvSolution` matches supported set.
* **Likely breakages:** copied CPU-era dictionaries, hidden include files, coarse-level solver defaults.

#### Step 11 — Run unprofiled smoke cases in the primary lane

* **Modify:** add `run_smoke_case.sh`.
* **Why:** prove basic execution before instrumentation-heavy runs.
* **Expected output:** solver logs, copied case dirs, result JSON.
* **Verify:** exit code 0; no NaN/Inf in logs; fields written; GPU activity later visible in profile.
* **Likely breakages:** wrong `FOAM_APPBIN`, stale environment, runtime dictionary issues.

#### Step 12 — Run Compute Sanitizer memcheck on the smallest case

* **Modify:** add `run_compute_sanitizer.sh`.
* **Why:** early correctness triage before deep profiling.
* **Expected output:** memcheck log.
* **Verify:** zero actionable memcheck errors.
* **Likely breakages:** benign third-party noise, lineinfo absent, debug symbols missing.

#### Step 13 — Run low-overhead Nsight Systems profile

* **Modify:** add `run_nsys.sh` recipe `basic`.
* **Why:** verify GPU kernels, NVTX ranges, and overall timeline shape.
* **Expected output:** `.nsys-rep` and text summary.
* **Verify:** GPU kernels present; NVTX ranges visible; trace completes.
* **Likely breakages:** Nsight not installed, permissions, overly broad trace categories, display-attached GPU noise.

#### Step 14 — Run UVM-fault diagnostic Nsight Systems profile

* **Modify:** add `run_nsys.sh` recipe `um_fault`.
* **Why:** supported-path bring-up must explicitly surface unexpected page migrations.
* **Expected output:** `.nsys-rep` with UVM fault info.
* **Verify:** fault data present; recurring steady-state faults are either absent or documented.
* **Likely breakages:** severe trace overhead, backtrace requirements, CPU sampling missing when backtrace enabled. ([NVIDIA Docs][11])

#### Step 15 — Run PTX-JIT compatibility test

* **Modify:** add `check_ptx_jit.sh`.
* **Why:** this is the definitive Blackwell-compatibility test recommended by NVIDIA. ([NVIDIA Docs][13])
* **Expected output:** successful smoke-case completion under `CUDA_FORCE_PTX_JIT=1`.
* **Verify:** solver exits 0; logs archived.
* **Likely breakages:** PTX absent, binary built for wrong arch, JIT-only path exposing latent toolchain issues.

#### Step 16 — Produce acceptance report and stop

* **Modify:** add `acceptance_gate.py`.
* **Why:** Phase 2 must not start from ambiguous Phase 1 status.
* **Expected output:** `phase1_acceptance_report.json`, `phase1_acceptance_report.md`, and `phase1_acceptance_bundle_index.json`.
* **Verify:** report marked `PASS`.
* **Likely breakages:** parser too fragile, missing artifact paths, inconsistent naming.

### Instrumentation and profiling hooks

#### Required NVTX range names

Use stable names now so later phases can extend them without churn.

Minimum required range names:

* `phase1:init`
* `phase1:caseSetup`
* `phase1:solveLoop`
* `phase1:iteration`
* `phase1:linearSolve`
* `phase1:write`

For `pimpleFoam` smoke specifically, add:

* `phase1:pimple:outerLoop`

#### Nsight Systems recipes

**Recipe A — `basic`**

* trace CUDA + NVTX + OS runtime only,
* no UVM-fault collection,
* no backtrace unless explicitly debugging,
* used for baseline timeline capture.

**Recipe B — `um_fault`**

* trace CUDA + NVTX + OS runtime,
* enable `--cuda-um-cpu-page-faults=true`,
* enable `--cuda-um-gpu-page-faults=true`,
* enable `--cudabacktrace=all` only when CPU sampling is enabled,
* used only on the smallest transient smoke case. ([NVIDIA Docs][11])

For single-GPU Phase 1, omit `mpi` from the default recipe. Keep an alternate recipe that mirrors SPUMA’s wiki (`cuda,nvtx,mpi`) for later decomposed runs.

#### Nsight Compute hook

Phase 1 uses Nsight Compute only to prove that kernel-level profiling works on this workstation.

Required behavior:

* profile one smoke case only,
* target one or a few dominant kernels,
* archive the session output,
* do not use Nsight Compute timing as a Phase 1 pass/fail criterion.

#### Compute Sanitizer hook

Required:

* `memcheck` on smallest smoke case.

Optional:

* `synccheck` if a suspicious synchronization defect appears later.

#### CUPTI note

CUPTI 12.8 adds Blackwell support and experimental hardware event timestamps on Blackwell. Treat those timestamps as useful but not authoritative for acceptance; repeatability across runs matters more in Phase 1. ([NVIDIA Docs][15])

### Validation strategy

#### Correctness checks

1. **Host correctness**

   * toolkit lane matches requested lane,
   * driver meets minimum,
   * GPU detected as RTX 5080 / CC 12.0.

2. **Binary correctness**

   * PTX present,
   * native `sm_120` present,
   * no accidental `sm_80`-only build.

3. **Runtime correctness**

   * native CUDA probe passes,
   * managed-memory probe passes.

4. **Smoke-case correctness**

   * exit code 0,
   * no NaN/Inf in logs,
   * required writes/functionObject outputs present.

5. **Instrumentation correctness**

   * NVTX ranges visible in Nsight Systems,
   * GPU kernels visible in Nsight Systems.

6. **Compatibility correctness**

   * PTX-JIT run passes.

7. **Memory-path correctness**

   * no unexplained recurring host/device page migrations in the steady inner loop.

#### Regression checks

Store and compare:

* probe JSON,
* build manifest,
* binary inspection report,
* smoke-case result JSON,
* profiler artifacts,
* acceptance report.

Any later change to driver/toolkit/build logic must re-run the same Phase 1 suite.

#### Numerical invariants

For `cubeLinear`:

* analytic linear scalar solution exists,
* relative L2 error against analytic profile should be **≤ 1e-8**,
* final residual should be **≤ 1e-8**.

For `channelSteady`:

* no NaN/Inf,
* outlet mass flow and inlet mass flow match to **≤ 1e-6 relative imbalance**,
* pressure and velocity residuals decrease by at least one order of magnitude over the configured short run.

For `channelTransient`:

* no NaN/Inf,
* solver completes all configured timesteps,
* continuity error remains finite,
* monitored velocity remains bounded and non-explosive.

These thresholds are deliberately conservative; Phase 1 is screening for substrate failures, not validating nozzle physics.

#### Performance checks

Required:

* warm up once before any measured run,
* do not use PTX-JIT run for timing,
* do not use UVM-fault trace for timing,
* capture one unprofiled wall-time for each smoke case after warm-up.

Advisory:

* repeated unprofiled runs should vary by less than **10%** wall time on the same workstation.

#### Profiling checks

Pass conditions:

* `.nsys-rep` generated successfully,
* NVTX ranges visible,
* CUDA kernels present,
* UVM-fault trace captured on one case,
* Compute Sanitizer memcheck runs successfully.

### Performance expectations

Do not set a hard speedup target in Phase 1.

Expected behaviors:

* First run after build may be slower because of context creation and PTX JIT.
* UVM fault tracing may add severe overhead; that is normal. ([NVIDIA Docs][11])
* Current SPUMA timelines are likely to be synchronization-heavy because upstream profiling shows `cudaDeviceSynchronize()` after each kernel in the profiled path. ([arXiv][1])
* `GAMG` on current SPUMA may look worse than ideal because the paper shows low efficiency in that path; Phase 1 should not over-interpret this. ([arXiv][1])

The Phase 1 performance expectation is simply: **the workstation runs supported SPUMA GPU cases correctly and observably**.

### Common failure modes

1. Driver older than required toolkit.
2. Wrong toolkit selected because `PATH` points at another CUDA install.
3. `NVARCH` left unset or left at `80`.
4. PTX absent from binaries.
5. Native target compiled for `sm_100` or `sm_80` instead of `sm_120`.
6. NVTX2 include still present and build fails on 12.9.1. ([NVIDIA Docs][10])
7. HMM/UVM initialization failure due to kernel/KASLR interaction. ([NVIDIA Docs][10])
8. Smoke case uses `DIC`, `DILU`, or `PBiCGStab` and silently falls off the intended GPU path. ([GitLab][7])
9. Nsight Systems run is misread as a performance result even though UVM fault tracing was enabled.
10. Sync-heavy timeline is misdiagnosed as a new Blackwell failure even though it matches current SPUMA behavior.
11. Binary inspection script checks the wrong binary or misses shared objects.
12. Case relies on upstream tutorial includes that changed layout across releases.

### Debugging playbook

1. **Start with host manifest.**
   Confirm driver, toolkit, GCC, OS, and PATH.

2. **Run native CUDA probe before SPUMA.**
   If this fails, stop. The issue is environment, not CFD.

3. **Run managed-memory probe next.**
   If this fails, check for the documented HMM/KASLR problem and apply one of NVIDIA’s workarounds. ([NVIDIA Docs][10])

4. **Rebuild only in the primary lane.**
   Do not debug two toolchain lanes at once.

5. **Inspect fatbinary before running any solver.**
   If PTX is missing, fix build flags first.

6. **Run `cubeLinear` unprofiled.**
   It is the smallest, easiest case to reason about.

7. **Run Compute Sanitizer memcheck on `cubeLinear`.**
   Fix any real memory errors before profiling.

8. **Run Nsight Systems `basic` on `cubeLinear` or `channelSteady`.**
   Confirm GPU kernels and NVTX ranges.

9. **Run Nsight Systems `um_fault` on `channelTransient`.**
   Check for recurring migrations during steady solve sections.

10. **Only after the primary lane passes, try the experimental 13.2 lane.**

11. **If current SPUMA execution is slow but correct, do not “optimize” in Phase 1.**
    Document it and stop. Execution-model fixes belong to Phase 3.

### Acceptance checklist

* [ ] Host manifest exists and is complete.
* [ ] Detected GPU is RTX 5080, CC 12.0.
* [ ] Primary lane toolchain is CUDA 12.9.1.
* [ ] Driver is 595.45.04 or newer.
* [ ] `have_cuda=true` and `NVARCH=120` recorded.
* [ ] SPUMA builds in required lane.
* [ ] Binary inspection confirms native `sm_120`.
* [ ] Binary inspection confirms PTX present.
* [ ] `CUDA_FORCE_PTX_JIT=1` run succeeds.
* [ ] `laplacianFoam` smoke case passes.
* [ ] `simpleFoam` smoke case passes.
* [ ] `pimpleFoam` smoke case passes.
* [ ] NVTX3 ranges visible in Nsight Systems.
* [ ] GPU kernels visible in Nsight Systems.
* [ ] Compute Sanitizer memcheck passes on smallest case.
* [ ] UVM-fault trace captured on one case.
* [ ] No unexplained recurring page migrations remain in the supported smoke path.
* [ ] `phase1_acceptance_report.json` marked `PASS`.

### Future extensions deferred from this phase

* Explicit DEVICE allocator strategy.
* Persistent device residency redesign.
* CUDA Graph capture and graph-conditional control flow.
* foamExternalSolvers / AmgX integration.
* Native GAMG/AmgX comparison.
* `incompressibleVoF` and MULES port.
* Nozzle-specific BC compaction.
* Device-resident startup seeding.
* Custom CUDA kernels for alpha/curvature/patches.
* Multi-GPU.

### What not to do

* Do not compile for `sm_80`, `sm_100`, `sm_100a`, or `compute_100a` on this workstation. ([NVIDIA Developer][5])
* Do not leave `NVARCH` at SPUMA’s example default.
* Do not use NVTX2 includes.
* Do not use `DIC`, `DILU`, or `PBiCGStab` in Phase 1 smoke cases. ([GitLab][7])
* Do not benchmark using UVM-fault tracing.
* Do not treat build success as compatibility proof.
* Do not start nozzle/VOF work before PTX-JIT and profiler checks pass.
* Do not modify runtime-selection names or solver registration machinery just to add instrumentation.
* Do not disable the memory pool for bring-up. ([arXiv][1])
* Do not use `nvprof` or Visual Profiler; use Nsight tooling. ([NVIDIA Docs][6])

### Rollback / fallback options

1. **If CUDA 12.9.1 build fails for toolchain reasons only:**
   try the experimental CUDA 13.2 lane on the same driver to separate repo/build logic problems from toolkit problems.

2. **If managed-memory probe fails because of HMM/KASLR issue:**
   apply one of NVIDIA’s documented workarounds:

   * boot with `nokaslr`, or
   * set `options nvidia_uvm uvm_disable_hmm=1`. ([NVIDIA Docs][10])

3. **If native `sm_120` cubin generation is blocked by build logic:**
   temporarily prove PTX-only compatibility on the probe binary, then patch build logic before declaring Phase 1 complete.

4. **If NVTX integration blocks the build:**
   allow `SPUMA_ENABLE_NVTX=0` only as a temporary diagnostic workaround. Phase 1 cannot exit successfully until NVTX3 is restored.

5. **If `simpleFoam` or `pimpleFoam` smoke case fails unexpectedly:**
   fall back to `cubeLinear`, then re-enable one solver at a time with the audited dictionaries.

### Implementation tasks for coding agent

1. Create the Phase 1 branch and manifest template.
2. Add host environment discovery scripts.
3. Add native + managed CUDA runtime probe binary.
4. Add Blackwell environment wrapper with `NVARCH=120`.
5. Add NVTX3 wrapper and remove NVTX2 includes.
6. Add build wrapper for required and experimental lanes.
7. Add binary inspection script for PTX/native-target validation.
8. Create three repo-local smoke cases.
9. Audit `fvSolution` in those cases for SPUMA-supported settings.
10. Add smoke execution wrapper and per-run scratch-copy logic.
11. Add Compute Sanitizer wrapper.
12. Add Nsight Systems wrapper with `basic` and `um_fault` recipes.
13. Add PTX-JIT compatibility runner.
14. Add acceptance parser and PASS/FAIL gate.
15. Write `docs/bringup/phase1_blackwell.md`.

### Do not start until

* Phase 0 freeze artifacts are available,
* workstation access is confirmed,
* driver installation authority is confirmed,
* the intended SPUMA/v2412 work branch is identified and its exact branch/SHA will be recorded in the Phase 1 artifact bundle,
* the team agrees that nozzle/VOF work is out of scope for this phase.

### Safe parallelization opportunities

* Smoke-case creation can proceed in parallel with NVTX wrapper implementation.
* Host env tooling can proceed in parallel with CUDA probe implementation.
* Acceptance parser can proceed in parallel with build wrapper work.
* Documentation can proceed in parallel with all non-code tasks.

Keep these serialized:

* build-system patching,
* binary inspection validation,
* primary-lane acceptance.

### Requires human sign-off on

* host OS if not Ubuntu 24.04,
* driver installation/change,
* boot/module changes for HMM/KASLR workarounds,
* whether CUDA 13.2 experimental lane stays recommended-only or is promoted to a required pre-Phase-2 gate,
* whether the optional CPU-only SPUMA sanity lane should be exercised on this workstation when Phase 0 Baseline B is not already green.

### Artifacts to produce

These artifacts are mandatory Phase 1 outputs and must be archived together. The accepted host manifest set serves as the toolkit/driver matrix artifact for the required lane, and should be lane-keyed if the experimental lane is exercised.

* `host_env.json`
* `cuda_probe.json`
* build logs per lane/mode
* `fatbinary_report.json`
* smoke-case logs and result JSONs
* `.nsys-rep` traces
* Compute Sanitizer logs
* PTX-JIT logs
* `phase1_acceptance_report.json`
* `phase1_acceptance_report.md`
* `phase1_acceptance_bundle_index.json`
* `docs/bringup/phase1_blackwell.md`

# 6. Validation and benchmarking framework

Phase 1 should leave behind a reusable validation framework, not just one successful workstation session.

## Validation tiers

**Tier A — environment validation**

* host manifest,
* CUDA probe,
* managed-memory probe.

**Tier B — build validation**

* required lane build,
* optional experimental lane build,
* fatbinary inspection.

**Tier C — smoke validation**

* three repo-local cases,
* unprofiled runs only.

**Tier D — correctness instrumentation**

* Compute Sanitizer memcheck.

**Tier E — observability**

* Nsight Systems `basic`,
* Nsight Systems `um_fault`,
* optional Nsight Compute sample run.

**Tier F — compatibility**

* PTX-JIT smoke run.

## Benchmarking policy

Phase 1 benchmarking is **health benchmarking**, not solver-performance benchmarking.

Rules:

1. Always perform one warm-up run before timing.
2. Never compare PTX-JIT runs to normal runs.
3. Never compare UVM-fault-traced runs to normal runs.
4. Store unprofiled warm-run wall times for trend tracking only.
5. Archive raw profiler traces; do not rely only on text summaries.

## Artifact naming

Use a deterministic run ID:

`<date>-<host>-<gpu>-<driver>-<toolkit>-<gitsha>-<case>-<mode>`

Example:

`2026-03-11-ws01-rtx5080-595.45.04-cuda12.9.1-a1b2c3d-channelTransient-basic`

## Reference policy

For Phase 1 itself:

* `cubeLinear` uses analytic reference.
* `channelSteady` and `channelTransient` use committed acceptance thresholds plus repeated-run consistency.

For later phases:

* this exact framework should be reused against Phase 0 CPU references.

# 7. Toolchain / environment specification

This section is the authoritative Phase 1 source of the initial toolchain proposal. After Phase 1 sign-off, these defaults move into the master pin manifest consumed by Phases 2–8, and later phase documents should reference that manifest instead of reopening the baseline with looser minima.

## Required baseline

* **OS:** Ubuntu 24.04 LTS native x86_64.
* **Driver:** NVIDIA Linux driver **595.45.04 or newer**.
* **Primary toolkit:** **CUDA 12.9.1**.
* **Experimental toolkit:** **CUDA 13.2**.
* **Compiler:** GCC 13.x preferred on Ubuntu 24.04.
* **GPU target:** `NVARCH=120`; native `sm_120` plus PTX.
* **Instrumentation:** NVTX3 only.
* **Profilers:** Nsight Systems, Nsight Compute, Compute Sanitizer.
* **SPUMA backend switch:** `have_cuda=true`. ([NVIDIA Docs][9])

## Recommended lane definitions

### Lane A — required

* toolkit: CUDA 12.9.1
* build mode: `relwithdebinfo`
* purpose: primary bring-up, smoke, Nsight, PTX-JIT

### Lane B — required

* toolkit: CUDA 12.9.1
* build mode: `debug`
* purpose: probe + Compute Sanitizer + first-failure triage

### Lane X — experimental

* toolkit: CUDA 13.2
* build mode: `relwithdebinfo`
* purpose: forward-compatibility compile and limited smoke

## Compiler / flag policy

Required:

* no `-use_fast_math`
* include debug symbols
* include `-lineinfo`
* keep frame pointers in host code for better backtraces

Preferred if build system patching is needed:

* explicit `-gencode=arch=compute_120,code=sm_120`
* explicit `-gencode=arch=compute_120,code=compute_120`

Acceptable only if verified by binary inspection:

* `-arch=sm_120` shorthand, because NVIDIA documents it expands to native cubin plus PTX. ([NVIDIA Docs][13])

## Environment variables

Mandatory in Phase 1 runs:

```bash
export have_cuda=true
export NVARCH=120
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Optional but recommended:

```bash
export OMP_NUM_THREADS=1
export SPUMA_ENABLE_NVTX=1
```

Dedicated compatibility test only:

```bash
export CUDA_FORCE_PTX_JIT=1
```

Unset immediately after that test. NVIDIA explicitly recommends unsetting it after compatibility testing. ([NVIDIA Docs][13])

## Kernel/boot note

If managed-memory probe or CUDA initialization fails in a way consistent with NVIDIA’s documented HMM/KASLR issue, Phase 1 should apply one documented workaround before any solver debugging begins, record the chosen path (`nokaslr` or `uvm_disable_hmm=1`) in the manifest/report, rerun both probes, and treat any continued failure as an environment blocker rather than a solver bug. ([NVIDIA Docs][10])

# 8. Module / file / ownership map

This is a functional ownership map, not a personnel assignment.

## Build and toolchain domain

Owns:

* `tools/bringup/env/*`
* `tools/bringup/build/*`

Responsibilities:

* environment selection,
* toolkit lane selection,
* build orchestration,
* fatbinary inspection,
* manifest creation.

## Runtime instrumentation domain

Owns:

* `src/gpu/common/NvtxScope.H`
* any minimal insertion points in supported solver entry paths

Responsibilities:

* NVTX3 wrapper,
* stable range names,
* zero semantic impact on solver numerics,
* no runtime-selection churn.

## Validation and smoke-case domain

Owns:

* `cases/bringup/*`
* `tools/bringup/run/run_smoke_case.sh`
* `tools/bringup/python/parse_solver_log.py`

Responsibilities:

* deterministic cases,
* accepted solver settings,
* acceptance thresholds,
* repeatable case copies.

## Profiling and sanitizer domain

Owns:

* `tools/bringup/run/run_nsys.sh`
* `tools/bringup/run/run_ncu.sh`
* `tools/bringup/run/run_compute_sanitizer.sh`

Responsibilities:

* standard recipes,
* artifact capture,
* separation of low-overhead vs diagnostic traces.

## Acceptance and release-readiness domain

Owns:

* `tools/bringup/python/acceptance_gate.py`
* `docs/bringup/*`

Responsibilities:

* PASS/FAIL logic,
* blocker/warning classification,
* human-readable summary,
* future-phase handoff quality.

# 9. Coding-agent execution roadmap

## Milestone M0 — Workspace freeze

Deliverables:

* branch,
* pinned SHA,
* doc stub.

Dependency:

* none.

Stop here if repo state is unstable.

## Milestone M1 — Host and CUDA substrate

Deliverables:

* `host_env.json`
* `cuda_probe.json`

Dependencies:

* M0

Stop here if:

* driver mismatch,
* CC not 12.0,
* managed-memory probe fails.

## Milestone M2 — Blackwell build enablement

Deliverables:

* primary-lane build,
* build logs,
* `fatbinary_report.json`

Dependencies:

* M1

Stop here if:

* `NVARCH` mismatch,
* PTX missing,
* native `sm_120` missing.

## Milestone M3 — Smoke-case bring-up

Deliverables:

* three repo-local cases,
* passing unprofiled smoke logs.

Dependencies:

* M2

Stop here if:

* any smoke case fails,
* unsupported solver setting is detected.

## Milestone M4 — Correctness tooling

Deliverables:

* passing memcheck on smallest case.

Dependencies:

* M3

Stop here if:

* actionable memcheck errors appear.

## Milestone M5 — Profiling substrate

Deliverables:

* `basic` Nsight Systems trace,
* `um_fault` Nsight Systems trace,
* visible NVTX ranges.

Dependencies:

* M4

Stop here if:

* NVTX ranges absent,
* no GPU kernels visible,
* recurring unexplained UVM migrations remain.

## Milestone M6 — Blackwell compatibility proof

Deliverables:

* successful PTX-JIT run.

Dependencies:

* M5

Stop here if:

* PTX-JIT fails.

## Milestone M7 — Acceptance freeze

Deliverables:

* `phase1_acceptance_report.json`
* `phase1_acceptance_report.md`
* `phase1_acceptance_bundle_index.json`

Dependencies:

* M6

Do **not** start Phase 2 until M7 is green.

## Parallel work

Can be done in parallel after M0:

* env scripts,
* probe binary,
* smoke-case setup,
* docs stub,
* acceptance parser scaffold.

Must remain serial:

* primary-lane build,
* binary inspection,
* smoke pass,
* memcheck,
* Nsight trace,
* PTX-JIT,
* final acceptance.

## Prototype before productize

Prototype first:

* NVTX insertion points,
* probe binary,
* smoke-case dictionaries,
* parser regexes.

Productize only after M3:

* acceptance gate,
* docs,
* full artifact naming policy.

## Experimental only

Keep experimental until after M7:

* CUDA 13.2 lane,
* Nsight Compute deep-kernel runs,
* any attempt to compare wall-time across toolchains.

# 10. Site Governance Notes

1. **Host baseline is fixed.**
   - Ubuntu 24.04 LTS on Linux x86_64 is the canonical workstation baseline for Phase 1. Any site-local deviation must be recorded in the Phase 1 artifact bundle, not improvised silently.
2. **Driver baseline is fixed.**
   - The required workstation lane uses driver `>=595.45.04` as frozen in `master_pin_manifest.md`.
3. **Kernel workaround policy is fixed.**
   - If the documented HMM/KASLR issue appears, Phase 1 may apply `nokaslr` or `uvm_disable_hmm=1`, but the chosen workaround must be recorded in the host manifest and acceptance report. ([NVIDIA Docs][10])
4. **Repo patching policy is fixed.**
   - Wrapper-driven bring-up is preferred, but direct SPUMA build-logic patching is allowed when `NVARCH=120` / PTX emission cannot be corrected externally.
5. **Experimental-lane policy is fixed.**
   - CUDA 13.2 remains verification-only until `master_pin_manifest.md` is explicitly revised. A green CUDA 12.9.1 primary lane is sufficient to start Phase 2.
6. **CPU sanity-lane policy is fixed.**
   - The optional CPU-only SPUMA sanity lane is recommended only when Phase 0 Baseline B is not already green or when workstation/toolchain drift must be isolated.
7. **Artifact-retention policy is site-local.**
   - Raw `.nsys-rep`, PTX-JIT, and sanitizer logs must remain discoverable from `phase1_acceptance_report.json` / `.md`, even if long-term storage location differs by site.

## Human review checklist

* Verify the workstation matches the frozen Ubuntu 24.04 + `>=595.45.04` baseline, or that any deviation is explicitly recorded in the artifact bundle.
* Verify that Phase 1 scope excludes nozzle and VOF work.
* Verify that `sm_120` / PTX remains the required binary target.
* Verify that NVTX3 migration is complete for the Phase 1 bring-up path.
* Verify that repo-local smoke cases remain the accepted bring-up harness instead of upstream tutorials.
* Verify that CUDA 13.2 remains verification-only unless `master_pin_manifest.md` is revised. ([NVIDIA Developer][5])

## Coding agent kickoff checklist

* Create the Phase 1 branch.
* Add host/env scripts.
* Add CUDA probe binary.
* Add NVTX3 wrapper.
* Add Blackwell env wrapper with `NVARCH=120`.
* Build required lane.
* Inspect fatbinary for `sm_120` + PTX.
* Add three smoke cases.
* Run smoke, memcheck, Nsight, PTX-JIT.
* Generate acceptance report.

## Highest risk implementation assumptions

1. SPUMA’s current build system can be made to emit correct `sm_120` + PTX binaries with limited patching.
2. Current SPUMA supported single-phase solvers remain operational on the latest workstation driver/toolkit combination.
3. The workstation can be moved to a modern driver without organizational friction.
4. Managed-memory probe failures, if they occur, are environment issues and can be resolved with NVIDIA’s documented workarounds.
5. Current sync-heavy SPUMA execution will remain acceptable as a Phase 1 baseline and will not be mistaken for a correctness defect. ([GitLab][3])

[1]: https://arxiv.org/html/2512.22215v1 "https://arxiv.org/html/2512.22215v1"
[2]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html"
[3]: https://gitlab.hpc.cineca.it/exafoam/spuma/-/blob/develop/README.md "https://gitlab.hpc.cineca.it/exafoam/spuma/-/blob/develop/README.md"
[4]: https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html "https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html"
[5]: https://developer.nvidia.com/cuda/gpus "https://developer.nvidia.com/cuda/gpus"
[6]: https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/ "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/"
[7]: https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554 "https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554"
[8]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"
[9]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
[10]: https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html "https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html"
[11]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "https://docs.nvidia.com/nsight-systems/UserGuide/index.html"
[12]: https://gitlab.hpc.cineca.it/exafoam/foamExternalSolvers "https://gitlab.hpc.cineca.it/exafoam/foamExternalSolvers"
[13]: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/ "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/"
[14]: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
[15]: https://docs.nvidia.com/cupti/release-notes/release-notes.html "https://docs.nvidia.com/cupti/release-notes/release-notes.html"
