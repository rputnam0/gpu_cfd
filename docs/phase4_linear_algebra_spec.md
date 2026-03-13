
# 1. Executive overview

This document expands **only Phase 4 — Linear algebra path** into implementation-level detail. All other phases are intentionally compressed to the minimum context required to keep the Phase 4 design coherent and executable.

Exact SPUMA commit, OpenFOAM flavor/release mapping, `foamExternalSolvers` revision, AmgX revision/config family, toolchain lane, support-matrix decisions, GPU operational contract, and graph-capture support status are consumed from the centralized project artifacts (master pin manifest, semantic source map, support matrix, and graph support matrix). Phase 4 does not reopen those decisions locally.

The target is a **single-GPU, single-rank pressure-solver integration path** for a future SPUMA-based pressure-swirl nozzle solver running on an NVIDIA GeForce RTX 5080 (Blackwell, compute capability 12.0, 16 GB GDDR7, 10,752 CUDA cores, up to 960 GB/s bandwidth, no NVLink). SPUMA v0.1-v2412 is based on the OpenFOAM v2412 line, and SPUMA’s published results and current support material confirm that multiphase/VOF remains unsupported and that `pimpleFoam` is GPU-supported while `incompressibleVoF` is not. That makes Phase 4 a real integration phase, not a configuration phase. [R1][R2][R3][R15][R16]

Phase 4 is **not** the whole nozzle port. It is the pressure linear-algebra substrate that later phases will depend on. The deliverable in this phase is:

1. a persistent **LDU-to-CSR topology cache** keyed by mesh topology and solver configuration,
2. an **AmgX-backed `lduMatrix::solver` implementation** for scalar pressure equations,
3. a **native fallback path** that remains available at runtime,
4. a **validation harness** based on captured pressure-system snapshots, and
5. a memory/execution design that is safe now and does not block later evolution toward persistent device residency.

The critical architectural nuance is this:

- **Bring-up path required now:** matrix assembly remains in the existing OpenFOAM/SPUMA host path; the solver packs current LDU coefficient values into pinned host staging buffers, uploads them to AmgX, runs the solve on device, and downloads the updated pressure field.
- **Production path / required bridge that must be designed now and completed before any later no-field-scale-host-transfer production claim that uses AmgX:** Phase 4 defines persistent device staging buffers and the backend upload abstraction; later device-resident pressure assembly/update code writes values and vectors into those buffers, and the AmgX upload path uses device pointers, eliminating PCIe/UVM traffic except for fields that the rest of the solver still requires on the host.

`PinnedHost` satisfies Phase 4 correctness and snapshot-replay acceptance only; it is not final production acceptance for later phases that claim no field-scale host transfer while using AmgX. Phase 4 owns the linear-solver boundary, persistent topology/cache, staging/upload abstraction, and replay/telemetry harness. Phase 5 owns the device-resident pressure assembly/update semantics that populate the `DeviceDirect` buffers, and the AmgX pressure solve remains graph-external in Phase 4.

That split is deliberate. A plugin-only linear-solver offload is not the final architecture for the nozzle workflow because assembly, alpha/MULES, curvature, and boundary handling also matter. However, Phase 4 should still start with a solver-only bring-up because it de-risks the hardest interoperability boundary first. The OGL/Ginkgo paper and exaFOAM workshop material both make the Amdahl-law limitation of solver-only offload explicit; SPUMA’s own paper makes the complementary case that a full-solver approach is better but far more invasive. [R3][R5][R6]

The guiding rule for this phase is therefore:

> Build a pressure linear-solver path that is correct, measurable, cached, fallback-safe, and future-proof for device residency, while explicitly **not** pretending that it alone solves the full nozzle GPU-port problem.

---

# 2. Global architecture decisions

## G1 — Base runtime

- **Sourced fact:** SPUMA v0.1-v2412 is based on the OpenFOAM 2412 line and is the active exaFOAM/SPUMA GPU-porting branch. SPUMA’s public support wiki lists `pimpleFoam` among supported GPU solvers, but not `incompressibleVoF`. [R1][R2]
- **Engineering inference:** Re-basing the nozzle work onto SPUMA/v2412 first reduces version-drift risk and preserves compatibility with the existing GPU-porting abstractions and exaFOAM solver ecosystem.
- **Recommendation:** Phase 4 must target **SPUMA/v2412 + foamExternalSolvers + AMGX** as the baseline integration environment. Do not backport to OpenFOAM 12 and do not invent a parallel GPU runtime stack.

## G2 — Solver-backend strategy

- **Sourced fact:** `foamExternalSolvers` exists specifically as an interface to external linear algebra package(s), currently AmgX, for SPUMA/OpenFOAM. SPUMA’s paper reports strong scaling advantages for AmgX at larger GPU counts, but also a case where changing coefficients prevented effective coarse-grid caching and made the pressure solve 2.5x slower than GAMG at 2 GPUs. [R4][R3]
- **Engineering inference:** A transient `p_rgh` pressure system in a VOF/PIMPLE nozzle solver will also have changing coefficients and repeated non-orthogonal correctors, so AmgX cannot be assumed to dominate native GAMG for all cases or all matrix sequences.
- **Recommendation:** Implement **AmgX as the primary Phase 4 external backend**, but keep **native OpenFOAM/SPUMA solver selection available at runtime** and benchmark both on captured pressure snapshots and live reduced cases.

## G3 — Persistent topology caching

- **Sourced fact:** The OGL/Ginkgo integration paper preserves the sparsity pattern and stores persistent accelerator-side matrix structures in the OpenFOAM object registry because solver objects are otherwise short-lived. It updates only coefficients across solves when the mesh topology is unchanged. [R5]
- **Engineering inference:** The same persistence pattern is mandatory for Phase 4. Rebuilding row pointers, column indices, and LDU-to-CSR maps on every corrector would waste CPU time, increase allocation churn, and obscure actual solver costs.
- **Recommendation:** Phase 4 must build a **mesh-resident persistent pressure-matrix cache** in the object registry. The cache must outlive individual solver objects and must rebuild only on topology change or solver-configuration change.

## G4 — Matrix format

- **Sourced fact:** AmgX’s public C API accepts CSR uploads, and the OGL/Ginkgo paper uses CSR/COO-style sparse conversion as its persistent linear-solver bridge. Sparse-kernel literature confirms that format selection matters, but CSR is the standard baseline for general unstructured FVM systems. [R5][R12][R14]
- **Engineering inference:** For a first implementation that must interoperate with AmgX and be debugged against OpenFOAM LDU semantics, CSR is the lowest-risk format.
- **Recommendation:** Phase 4 supports **CSR only**. Do not attempt ELL/HYB/SELL experiments until correctness and reuse are proven.

## G5 — Boundary/interface scope

- **Sourced fact:** In OpenFOAM’s segregated `fvMatrix::solve` path, `addBoundaryDiag`, `addBoundarySource`, and `updateMatrixInterfaces` modify the diagonal/source and interface coefficient data before the selected `lduMatrix::solver` is invoked. Coupled interfaces remain explicit inputs to the solver object. [R7]
- **Engineering inference:** An AmgX backend that ignores `lduInterfaceField` data is safe only when no coupled interfaces are present in the equation being solved.
- **Recommendation:** Phase 4 explicitly supports **single-rank, non-coupled scalar pressure systems** only. If coupled interfaces are detected (`cyclic`, `AMI`, processor interfaces, or any non-empty effective `lduInterfaceField` set), the solver must **fallback to native** unless a specific later extension implements those interfaces correctly.

## G6 — Precision policy

- **Sourced fact:** AmgX mode encodes host/device, matrix precision, vector precision, and index type; `AMGX_mode_dDDI` is a valid mode for double-precision matrix/vector data with 32-bit indices on device. SPUMA examples in the paper show `mode dDDI` for AmgX. [R3][R14]
- **Engineering inference:** A GeForce RTX 5080 is a consumer Blackwell GPU, so FP64 throughput is unlikely to be a performance strength relative to datacenter GPUs. That does not change the numerical-risk profile of pressure solves in early bring-up.
- **Recommendation:** Use **`dDDI` only** in Phase 4. Mixed/single precision are deferred until the pressure path is correct and benchmarked. If the OpenFOAM/SPUMA build is not using double-precision `scalar`, Phase 4 should refuse to enable AmgX until explicit support is added.

## G7 — Graph-capture posture

- **Sourced fact:** CUDA 12.8 adds Blackwell support and richer conditional graph nodes, but graph capture has restrictions around synchronous APIs and stream synchronization. No authoritative source found in this research pass guarantees that AmgX setup/solve calls are graph-capture-safe. [R13][R17][R18]
- **Engineering inference:** Attempting to capture AmgX calls in this phase would add failure modes with unclear upside.
- **Recommendation:** **Do not capture AmgX in CUDA Graphs in Phase 4.** Structure APIs so later phases can isolate the pressure solve as a graph boundary, but treat full graph integration as deferred experimental work.

## G8 — Memory posture

- **Sourced fact:** SPUMA currently uses unified memory plus a memory pool as a porting strategy. exaFOAM workshop material reports that OpenMP+UVM on discrete GPUs can spend more than 65% of time in page migrations. The AmgX reference manual states that uploads are synchronous, user buffers may be host or device, and pinned host buffers are recommended when host buffers are used. [R3][R6][R14]
- **Engineering inference:** Unified memory is the wrong primary memory model for the pressure bridge. It would hide transfer boundaries, trigger page-fault noise, and make Phase 4 profiles difficult to trust.
- **Recommendation:** Phase 4 must use **explicit device allocations** for persistent GPU objects and **pinned host staging** for bring-up transfers. No unified memory in the AmgX bridge path.

---

# 3. Global assumptions and constraints

1. **Repository baseline**
   - SPUMA/v2412 is the active runtime base. [R1]
   - `foamExternalSolvers` is available as the AmgX integration layer or extension point. [R4]

2. **Hardware baseline**
   - Primary development hardware is a single NVIDIA GeForce RTX 5080.
   - Official NVIDIA sources list the RTX 5080 as Blackwell, compute capability 12.0, with 16 GB GDDR7 and no NVLink. [R15][R16]

3. **CUDA/toolchain baseline**
   - Use the frozen toolchain lane from the master pin manifest. The CUDA 12.8.x references in this document remain Blackwell-era compatibility rationale, not a substitute for the project pin. [R17][R18]
   - Driver and profiler versions must satisfy the master pin manifest rather than a locally restated minimum.

4. **Execution scope**
   - Single node, single GPU, single MPI rank only in Phase 4.
   - No distributed AmgX matrix communication maps.
   - No processor decomposition support.
   - No graph-captured pressure solve.

5. **Mesh and matrix scope**
   - Static mesh topology during a run.
   - Scalar pressure equation only (`p` or `p_rgh` style LDU scalar system).
   - Block size 1 only.
   - No coupled interfaces in supported Phase 4 cases.
   - `nnz`, `nRows`, and all AmgX-facing row/column indices must fit into 32-bit integers.

6. **Numerical scope**
   - Pressure equation remains assembled by existing OpenFOAM/SPUMA numerics in this phase.
   - Reference-cell/reference-value handling remains the caller’s responsibility (`setReference` before solve).
   - Boundary/source/diag modifications already performed by the OpenFOAM `fvMatrix` path must not be duplicated in the AmgX backend. [R7][R11]

7. **Performance scope**
   - Phase 4 correctness acceptance does **not** require AmgX to beat native GAMG on all cases.
   - Phase 4 performance acceptance **does** require elimination of unnecessary topology rebuilds, avoidance of unified-memory page migration in the bridge path, and a clean measurement framework.

8. **Repository and build assumptions**
   - The coding agent can modify SPUMA and `foamExternalSolvers`.
   - The coding agent can add lightweight test utilities and benchmark scripts.
   - The coding agent should not introduce a new large testing framework if OpenFOAM-style executable tests are sufficient.

---

# 4. Cross-cutting risks and mitigation

## RISK-C1 — Amdahl ceiling from solver-only offload

- **Issue:** A solver-only offload can never deliver full-nozzle speedups because assembly and non-linear updates remain outside the accelerated region. OGL/Ginkgo and exaFOAM explicitly call this out. [R5][R6]
- **Mitigation:** Treat the Phase 4 host-pack/upload path as a bring-up step only. Design persistent caches and device staging interfaces now so later phases can remove host transfers.

## RISK-C2 — AmgX hierarchy setup cost on changing matrices

- **Issue:** AmgX setup must be called again whenever the matrix changes, and SPUMA reports that pressure matrices with changing coefficients may not benefit enough from caching to beat native GAMG at low GPU counts. [R3][R14]
- **Mitigation:** Separate and measure:
  - value-pack time,
  - upload time,
  - setup time,
  - solve time,
  - download time.
  Always benchmark against native fallback on captured snapshots before claiming a win.

## RISK-C3 — Silent performance collapse from memory choices

- **Issue:** Unified memory and pageable host buffers can produce page faults or poor PCIe throughput. [R6][R14]
- **Mitigation:** Use explicit pinned host buffers and explicit device-resident AmgX objects only. Run Nsight Systems with UVM page-fault tracing and require zero steady-state UVM events in the AmgX bridge path.

## RISK-C4 — Incorrect LDU-to-CSR mapping

- **Issue:** A single owner/neighbour or upper/lower mapping mistake will produce numerically plausible but wrong answers.
- **Mitigation:** Add:
  - a synthetic matrix test,
  - an `A*x` equivalence validator against OpenFOAM `lduMatrix::Amul`,
  - per-row duplicate-column checks in debug mode,
  - optional symmetry checks.

## RISK-C5 — Coupled interface misuse

- **Issue:** The segregated OpenFOAM path still passes interface coefficient structures to the solver. Ignoring them on a case with coupled patches is wrong. [R7]
- **Mitigation:** Detect unsupported interfaces and fallback immediately with a precise log message that includes patch names.

## RISK-C6 — Version drift between OpenFOAM API families

- **Issue:** Public searchable API sources span OpenFOAM Foundation and OpenCFD documentation. High-level solve flow matches, but local signatures and file layout can drift by release.
- **Mitigation:** Treat the local SPUMA/v2412 source tree as final authority during implementation. Use this document for control-flow requirements and invariants, not to override local compile-time truth.

## RISK-C7 — Mixed-precision source conflict

- **Issue:** The AMGX README still advertises mixed precision as a feature, while the v2.5.0 release notes say the old `DISABLE_MIXED_PRECISION` path was removed during cuSPARSE API modernization. [R12][R13]
- **Mitigation:** Do not depend on mixed-precision behavior in Phase 4. Use `dDDI` only.

## RISK-C8 — Opaque external-library behavior and graphs

- **Issue:** AmgX internal kernels, streams, allocations, and graph-capture behavior are opaque from the OpenFOAM side.
- **Mitigation:** Keep the pressure solve outside graph capture in Phase 4. Add NVTX ranges around external library boundaries so profiles show exact time ownership.

---

# 5. Phase-by-phase implementation specification

## Phase 4 — Linear algebra path

### Purpose

Build a reusable, benchmarkable, fallback-safe **pressure linear-solver backend** for SPUMA/OpenFOAM that:

1. converts scalar LDU pressure systems into persistent CSR topology once,
2. updates only coefficient values and vectors on subsequent solves,
3. solves on the RTX 5080 via AmgX in device mode,
4. preserves numerical correctness relative to the native solver path, and
5. leaves a clean seam for later device-resident pressure assembly.

### Why this phase exists

This phase exists because the nozzle solver’s pressure path is both numerically central and operationally self-contained enough to be the first external GPU linear-algebra integration target.

It is the right first linear-algebra phase because:

- `foamExternalSolvers` already exists as the AmgX bridge for SPUMA/OpenFOAM. [R4]
- Pressure solves dominate enough runtime to matter, but are still only part of the total workflow; Phase 4 deliberately solves the backend interoperability problem without pretending that it solves the full VOF nozzle port. [R3][R5][R6]
- OpenFOAM pressure equations are built on LDU matrices with fixed sparsity on static meshes, making them a good fit for persistent topology caching. [R5][R9][R10]
- SPUMA’s current profile shows that pressure/GAMG paths can be weak, but also that naive GPU offload can suffer from launch/synchronization overhead and hierarchy rebuild effects. [R3]

### Entry criteria

Phase 4 must not start until all of the following are true:

1. **SPUMA/v2412 baseline is frozen**
   - The CPU reference branch on the SPUMA/v2412 base builds and runs the reduced nozzle case.
   - Version-migration discrepancies versus the original production nozzle workflow are understood well enough that Phase 4 regressions will not be confused with migration issues.

2. **Toolchain bring-up is complete**
   - The master pin manifest and semantic source map for SPUMA/v2412 + `foamExternalSolvers` + AmgX are frozen and available before implementation PRs begin.
   - The pinned CUDA/driver/Nsight lane from the master pin manifest is installed.
   - AMGX builds and its example(s) run on the RTX 5080.
   - Nsight Systems and Compute Sanitizer are operational. [R12][R13][R17][R18]

3. **Pressure-system capture path exists**
   - There is a way to dump one or more representative pressure matrices (`diag`, `lower`, `upper`, `source`, `psi`, `lowerAddr`, `upperAddr`) from the CPU reference branch.

4. **Native solver baseline is available**
   - The exact native fallback solver and settings to compare against are pinned for snapshot replay and reduced-case runs.

5. **Case scope is compatible**
   - Initial target cases use single rank and do not require coupled interfaces in the pressure equation.

### Exit criteria

Phase 4 is complete only when all of the following are true:

1. An `AmgX` solver can be selected in `fvSolution` for a scalar pressure equation.
2. The implementation detects unsupported cases according to the centralized support matrix and falls back to native deterministically.
3. Topology conversion happens once per mesh/configuration, not once per solve.
4. Repeated solves on the same topology use:
   - `AMGX_matrix_replace_coefficients` for coefficients,
   - repeated vector uploads for `b` and optional `x`,
   - repeated `AMGX_solver_setup` as required by AmgX semantics. [R14]
5. Snapshot replay validates `A*x` equivalence and solve correctness versus native.
6. Nsight Systems confirms:
   - no steady-state UVM faults in the AmgX path,
   - no repeated row/column uploads after first setup,
   - pinned-host or device-direct explicit transfer behavior only.
7. Reduced-case integration runs complete without solver failures or silent fallback.
8. Benchmark logs separate:
   - pattern build,
   - value pack,
   - upload,
   - setup,
   - solve,
   - download,
   - fallback count.
9. `PinnedHost` staging is accepted here only as the Phase 4 correctness and snapshot-replay exit path.
10. The Phase 4 exit report explicitly states that any later phase that wants AmgX to satisfy a no-field-scale-host-transfer production claim must first implement and validate the `DeviceDirect` bridge; until then, the AmgX path remains correctness-only under the global GPU operational contract.

### Goals

#### Required for correctness

1. Preserve OpenFOAM pressure-equation semantics exactly at the solver boundary.
2. Map LDU coefficients to CSR correctly for asymmetric scalar matrices.
3. Ensure boundary-adjusted diagonal and source are used exactly once.
4. Preserve native fallback as a first-class runtime path.
5. Detect unsupported interface cases before wrong answers occur.
6. Support repeated solves on the same topology without rebuilding row/column structures.

#### Required for performance

1. Reuse sparsity pattern and mapping arrays across all correctors/timesteps on static meshes.
2. Use pinned host staging buffers for the bring-up path.
3. Avoid any unified memory in the linear-algebra bridge.
4. Avoid allocation churn after first warm-up solve.
5. Keep all AmgX-internal matrix/vector storage on device (`dDDI`).

#### Required for future phases

1. Define interfaces that can later accept device-resident `csrValues`, `rhs`, and `x`.
2. Keep cache ownership independent of a single solver invocation.
3. Preserve instrumentation boundaries that later graph-level execution can use.

### Non-goals

Phase 4 explicitly does **not** do the following:

1. Port pressure assembly itself to device.
2. Port `alpha`, MULES, surface tension, or mixture-property updates.
3. Support multi-GPU or MPI decomposition.
4. Support processor interfaces, cyclic/cyclicAMI, or other coupled patches in the AmgX path.
5. Capture AmgX operations in CUDA Graphs.
6. Introduce alternative sparse formats beyond CSR.
7. Enable mixed precision or single precision.
8. Replace native SPUMA/OpenFOAM linear solvers globally.
9. Claim end-to-end nozzle speedup.
10. Remove the need for later pressure-assembly device residency.

### Technical background

#### OpenFOAM pressure solve boundary

For segregated `fvMatrix` solves, the caller modifies the diagonal and source before constructing the runtime-selected `lduMatrix::solver`. In the OpenCFD API path, `fvMatrix::solveSegregated`:

1. copies the current internal field into a scalar solve field,
2. applies `addBoundaryDiag(diag(), cmpt)`,
3. builds the component source,
4. performs coupled-boundary explicit updates via `initMatrixInterfaces` / `updateMatrixInterfaces`,
5. invokes `lduMatrix::solver::New(...)->solve(psiCmpt, sourceCmpt, cmpt)`,
6. restores the saved diagonal afterwards. [R7]

For Phase 4, this means:

- the AmgX backend must use **the matrix state it receives at solve time**,
- it must **not** try to reapply `addBoundaryDiag` or `addBoundarySource`,
- it must treat `source` as already boundary-adjusted for the current solve.

#### LDU storage semantics relevant to CSR conversion

OpenFOAM LDU stores:

- diagonal coefficients in `diag()`,
- one coefficient per internal face in `lower()`,
- one coefficient per internal face in `upper()`,
- addressing arrays exposed through `lduAddressing`.

From OpenFOAM sources:

- `lowerAddr()` is the owner-address list,
- `upperAddr()` is the neighbour-address list,
- in `lduMatrix::Amul`, the owner row uses `upper[face]` at the neighbour column,
- the neighbour row uses `lower[face]` at the owner column. [R9][R10]

For an internal face `f` with
- `owner = lowerAddr[f]`,
- `neigh = upperAddr[f]`,

the full CSR matrix entries are therefore:

- row `owner`, col `owner`  <- `diag[owner]`
- row `owner`, col `neigh`  <- `upper[f]`
- row `neigh`, col `neigh`  <- `diag[neigh]`
- row `neigh`, col `owner`  <- `lower[f]`

That mapping is the single most error-prone part of Phase 4.

#### AmgX upload/setup semantics

Relevant AmgX facts from the reference manual:

- `AMGX_matrix_upload_all` uploads a CSR matrix and copies the data into the matrix object.
- `AMGX_matrix_replace_coefficients` updates coefficients without changing sparsity and requires a prior `upload_all`.
- If the diagonal is already present in CSR, `diag_data` must be null. Duplicating diagonal entries in both locations is undefined behavior.
- User buffers may be on host or device; host buffers should be pinned for performance.
- Upload calls are synchronous.
- `AMGX_solver_setup` must be called again whenever matrix entries change; repeated setup calls are allowed, and some algorithms may reuse cached computations when only values change. [R14]

#### Why persistent topology matters here

The OGL/Ginkgo paper shows that because OpenFOAM solver objects are short-lived, persistent matrix mappings must be stored in the object registry if one wants to reuse sparsity structure across repeated solves. Pressure-solver use on a static mesh is exactly that use case. [R5]

### Research findings relevant to this phase

1. **SPUMA base and solver support**
   - SPUMA v0.1-v2412 is tied to the OpenFOAM 2412 line. [R1]
   - SPUMA’s current GPU-support material lists `pimpleFoam` among supported GPU solvers and explicitly recommends Nsight profiling with CPU/GPU page-fault tracing because unsupported features may manifest as unintended host/device copies. [R2]

2. **Pressure solver may not be the easiest GPU win**
   - SPUMA profiling shows weaker efficiency in pressure/GAMG paths and significant synchronization overhead in current profiled flows. [R3]
   - Therefore a successful Phase 4 design must make the pressure path measurable and fallback-safe rather than assuming automatic acceleration.

3. **AmgX is appropriate but not guaranteed to win**
   - `foamExternalSolvers` exists to connect SPUMA/OpenFOAM to AmgX. [R4]
   - SPUMA’s paper reports that AmgX scaled well at higher GPU counts but lost to GAMG in one low-GPU case because coefficient changes prevented beneficial hierarchy reuse. [R3]
   - This directly applies to transient nozzle `p_rgh` solves with changing coefficients.

4. **Persistent sparsity and object-registry storage are established patterns**
   - The OGL/Ginkgo work uses persistent registry-based storage for converted matrix structures because the underlying solver object is not persistent. [R5]
   - This is the correct pattern for Phase 4.

5. **UVM is a trap for this bridge**
   - exaFOAM workshop material shows severe page-migration costs on discrete GPUs for an OpenFOAM-style workload under unified memory. [R6]
   - SPUMA’s wiki explicitly recommends UVM fault tracing because copies may appear instead of clean failures. [R2]
   - Therefore explicit pinned/device memory must be used here.

6. **AMGX version selection needs care**
   - The AMGX README still contains older, broad feature claims such as mixed precision support and older build flags. [R12]
   - The AMGX v2.5.0 release notes state Blackwell support, CUDA 13 support, new minimum CUDA 12.0, `CMAKE_CUDA_ARCHITECTURES` usage, NVTX3 adoption, and removal of certain older cuSPARSE mixed-precision paths. [R13]
   - Recommendation: treat v2.5.0 release notes as more authoritative for Blackwell-era build decisions than older README statements.

7. **Blackwell-specific toolchain facts**
   - Official NVIDIA compute-capability tables list GeForce RTX 5080 as compute capability 12.0. [R15]
   - CUDA 12.8 is the first toolkit family with explicit Blackwell support. [R17][R18]

8. **OpenFOAM solve flow is cross-checked, but local source remains final authority**
   - OpenCFD latest API documentation provides the solve path structure used here. [R7]
   - Foundation API pages provide the LDU mapping logic and addressing references. [R9][R10]
   - Implementation must still verify exact signatures against the local SPUMA/v2412 source tree.

#### Stale or conflicting source notes

- **AMGX mixed precision:** README says mixed precision exists; v2.5.0 release notes say older mixed-precision build path was removed. Treat mixed-precision support as uncertain for this project and do not rely on it. [R12][R13]
- **OpenFOAM API sources:** This document cross-references both OpenCFD and Foundation documentation to reconstruct stable semantics; do not assume source-file names or constructor macros are identical without checking the local tree. [R7][R9][R10][R11]

### Design decisions

#### Decision 4.1 — Use existing `foamExternalSolvers` as the extension point

- **Sourced fact:** `foamExternalSolvers` exists specifically as an interface to external linear algebra package (AmgX) for SPUMA and OpenFOAM. [R4]
- **Engineering inference:** Extending the existing AmgX bridge is less risky than creating a parallel pressure-solver adapter in SPUMA.
- **Recommendation:** Modify/extend the existing `foamExternalSolvers` AmgX module first. Only create a parallel module in SPUMA if repository/API drift makes extension impractical.

#### Decision 4.2 — Cache topology in the object registry

- **Sourced fact:** OGL/Ginkgo stores persistent conversion data in the object registry because solver instances are short-lived. [R5]
- **Engineering inference:** OpenFOAM `lduMatrix::solver` objects are also short-lived at the point where `solve()` is called.
- **Recommendation:** Introduce a persistent `PressureMatrixCache` registry object keyed by region, field name, solver-backend name, mode, and config hash.

#### Decision 4.3 — Build a full CSR with diagonal embedded

- **Sourced fact:** AmgX accepts CSR uploads and states that if the diagonal is contained in the matrix itself, `diag_data` must be null; duplicating diagonal entries between CSR data and `diag_data` is undefined. [R14]
- **Engineering inference:** Using a full CSR with embedded diagonal is the safest representation for an OpenFOAM LDU matrix because it avoids a second diagonal ownership path.
- **Recommendation:** CSR must include the diagonal entry in each row, and all AmgX calls must pass `diag_data = nullptr`.

#### Decision 4.4 — Preserve asymmetric storage semantics

- **Sourced fact:** OpenFOAM `lduMatrix::Amul` uses `upper` on the owner row and `lower` on the neighbour row. [R9]
- **Engineering inference:** Even for pressure systems that are often symmetric in practice, the bridge must support asymmetric coefficient arrays because the LDU abstraction permits it and the safest mapping is the fully general one.
- **Recommendation:** Always build the full asymmetric CSR:
  - owner row gets `upper[f]`,
  - neighbour row gets `lower[f]`.
  Do not assume `lower == upper`.

#### Decision 4.5 — Support only uncoupled scalar systems in Phase 4

- **Sourced fact:** Coupled interface data is still part of the segregated solve path. [R7]
- **Engineering inference:** Getting coupled interfaces wrong is worse than falling back.
- **Recommendation:** In Phase 4, if any effective coupled interfaces are present, log the reason and fallback to native. This is non-negotiable for the first implementation.

#### Decision 4.6 — Use `dDDI` only

- **Sourced fact:** `dDDI` is a valid AmgX mode for device-resident double-precision matrix/vector data with 32-bit integer indices. [R14]
- **Engineering inference:** Pressure-correction stability is not where precision experiments should begin on a GeForce-class GPU.
- **Recommendation:** Accept only `mode dDDI` in Phase 4. Reject other modes with a clear message unless explicit later support is added.

#### Decision 4.7 — Use explicit pinned-host staging in the bring-up path

- **Sourced fact:** AmgX uploads are synchronous; pinned host buffers are recommended for faster transfer when using host buffers. [R14]
- **Engineering inference:** The bring-up path inevitably starts from host-resident OpenFOAM matrices and fields.
- **Recommendation:** Allocate persistent pinned-host staging buffers and reuse them across solves. Do not pin/unpin transient heap allocations per solve.

#### Decision 4.8 — Separate bring-up and production paths in the API now

- **Sourced fact:** The AmgX C API accepts both host and device user buffers. [R14]
- **Engineering inference:** Later phases can eliminate PCIe traffic only if Phase 4 already defines a staging abstraction that can switch from host to device sources without redesigning the solver interface.
- **Recommendation:** Implement a staging abstraction with two modes:
  - `PinnedHost` (implemented first and accepted only for Phase 4 correctness/replay),
  - `DeviceDirect` (mandatory follow-on bridge work before any Phase 5/6/8 production claim of no field-scale host transfer with AmgX).

  Phase 4 owns the staging/upload abstraction and device-resident buffer contract; Phase 5 owns the device-side pressure assembly/update semantics that fill those buffers.

#### Decision 4.9 — Always call `AMGX_solver_setup` after coefficient replacement

- **Sourced fact:** The AmgX reference manual explicitly states that whenever the matrix changes, `AMGX_solver_setup` must be called again; repeated calls are allowed and some algorithms may reuse cached computations automatically. [R14]
- **Engineering inference:** Skipping setup after value replacement is unsafe and backend-dependent.
- **Recommendation:** After each `AMGX_matrix_replace_coefficients`, call `AMGX_solver_setup`. Measure setup cost explicitly.

#### Decision 4.10 — Do not sort CSR rows in the first implementation

- **Sourced fact:** CSR row ordering requirements are not stated as a correctness constraint in the AmgX manual. [R14]
- **Engineering inference:** Sorting rows increases implementation complexity because all LDU-to-CSR index maps would need remapping after sort.
- **Recommendation:** For Phase 4 bring-up, insert CSR entries in deterministic order:
  1. diagonal first,
  2. off-diagonals in native internal-face order.
  Add duplicate-column checks in debug mode. Revisit sorting only if profiling justifies it.

#### Decision 4.11 — First production config should be Krylov+AMG but not symmetry-assuming

- **Sourced fact:** AmgX supports multiple Krylov and AMG-based configurations; the README example uses `FGMRES_AGGREGATION.json`. [R12]
- **Engineering inference:** Pressure matrices are often SPD in practice, but symmetry assumptions can be invalidated by discretization details and unsupported boundary/interface cases.
- **Recommendation:** Start with a non-symmetry-assuming AmgX outer Krylov configuration (e.g., FGMRES + AMG preconditioner) for bring-up. Treat CG-specific tuning as a later optimization behind symmetry validation.

#### Decision 4.12 — Native solver fallback is mandatory, not optional

- **Sourced fact:** SPUMA’s own results show a case where AmgX lost to GAMG at low GPU count when coefficients changed. [R3]
- **Engineering inference:** Runtime fallback is required both for unsupported features and for performance reality.
- **Recommendation:** Add explicit fallback settings and counters. The project must be able to continue with native solves when AmgX is unsupported or fails.

#### Bring-up path versus production path

**Bring-up path (must do now):**
- build persistent pattern on host,
- pack values/RHS/guess into pinned host buffers,
- `upload_all` once for topology,
- `replace_coefficients` + vector uploads + setup + solve each solve,
- download solution to host.

**Production path / mandatory bridge (must be designed now and completed before any later AmgX production claim of no field-scale host transfer):**
- persistent device staging buffers exist,
- pressure assembly kernels write directly into `csrValuesDev`, `rhsDev`, and optionally `xDev`,
- AmgX upload calls use device pointers,
- host traffic is reduced to only what the rest of the solver still requires,
- but note: AmgX still performs an internal copy into its own matrix/vector storage, so later phases should measure the cost of device-to-device upload as well as solver time. [R14]

### Alternatives considered

#### Alternative A — Native SPUMA/OpenFOAM GAMG only

- **Why considered:** It is already present, simpler, and may outperform AmgX on some changing-coefficient matrices.
- **Why not chosen as the only path:** It does not answer the Phase 4 requirement to establish a Blackwell-ready external device linear-algebra path. It remains the fallback/baseline.

#### Alternative B — Ginkgo/OGL-style new plugin instead of AmgX

- **Why considered:** The OGL paper is highly relevant and demonstrates persistent structure reuse. [R5]
- **Why not chosen now:** `foamExternalSolvers` already exists around AmgX in the SPUMA ecosystem, and adding Ginkgo would introduce another major dependency and validation burden.
- **Status:** Rejected for Phase 4, but the persistent-cache pattern from OGL is adopted conceptually.

#### Alternative C — Custom cuSPARSE + custom AMG

- **Why considered:** It could reduce opaque-library behavior and potentially enable tighter graph integration.
- **Why not chosen now:** Too much implementation risk for the first linear-algebra phase. Not compatible with the “minimal additional interpretation” requirement for the coding agent.

#### Alternative D — Use separate external diagonal (`diag_data`) in AmgX

- **Why considered:** It resembles OpenFOAM’s diagonal-separated storage.
- **Why not chosen:** The manual explicitly warns about duplicate diagonal ambiguity and undefined behavior when diagonal entries appear in both places. [R14]
- **Status:** Rejected.

#### Alternative E — Sort rows immediately

- **Why considered:** Some sparse kernels prefer sorted columns.
- **Why not chosen now:** Adds a remapping step and another place to introduce bugs. Solve correctness is more important than row-order cosmetics in Phase 4.

#### Alternative F — Attempt graph capture immediately

- **Why considered:** Later phases aim to reduce launch overhead.
- **Why not chosen now:** Capture-safety of AmgX is unverified in this research pass, and CUDA capture restrictions around synchronous APIs make this a poor first move. [R17]
- **Status:** Deferred.

### Interfaces and dependencies

## External dependencies

1. **SPUMA / OpenFOAM v2412 base**
   - Source of `lduMatrix`, `fvMatrix`, runtime selection, object registry, and field data structures. [R1][R7]

2. **foamExternalSolvers**
   - Preferred repository/module to extend for AmgX solver integration. [R4]

3. **AMGX**
   - External solver backend library and C API. [R12][R13][R14]

4. **CUDA runtime/tooling**
   - Required for AMGX and Blackwell-compatible compilation. [R15][R17][R18]

## Internal module interfaces to implement

### `PressureMatrixCache`

Purpose:
- persistent ownership of topology mapping, staging buffers, AmgX handles, and telemetry.

Expected API:
```cpp
class PressureMatrixCache : public regIOobject
{
public:
    static PressureMatrixCache& getOrCreate
    (
        objectRegistry& db,
        const word& regionName,
        const word& fieldName,
        const dictionary& solverControls
    );

    bool compatibleWith
    (
        const lduMatrix& matrix,
        const lduInterfaceFieldPtrsList& interfaces,
        word& reason
    ) const;

    void ensurePatternBuilt(const lduMatrix& matrix);
    void ensureAmgXObjectsCreated();
    void packHostStaging
    (
        const lduMatrix& matrix,
        const scalarField& source,
        const scalarField& psi,
        const bool reuseInitialGuess
    );
    void uploadOrReplace();
    void setupSolver();
    void solve(const bool zeroInitialGuess);
    void downloadSolution(scalarField& psi);
    void invalidate(const word& reason);

    const PressureSolveTelemetry& telemetry() const;
};
```

### `LduCsrPatternBuilder`

Purpose:
- build one-time row pointers, column indices, and position maps.

Expected API:
```cpp
struct LduCsrPattern
{
    int nRows;
    int nnz;
    List<int> rowPtr;
    List<int> colInd;
    List<int> diagToCsr;
    List<int> upperToCsr;
    List<int> lowerToCsr;
    uint64 topologyHash;
};

LduCsrPattern buildLduCsrPattern
(
    const lduAddressing& addr,
    const label nCells
);
```

### `CsrValuePacker`

Purpose:
- take current OpenFOAM diagonal/lower/upper/source/psi values and pack them into reusable staging buffers.

Expected API:
```cpp
void packCsrValuesHost
(
    const lduMatrix& matrix,
    const scalarField& source,
    const scalarField& psi,
    const LduCsrPattern& pattern,
    scalar* csrValuesHost,
    scalar* rhsHost,
    scalar* xHost,
    const bool reuseInitialGuess
);
```

### `AmgXContext`

Purpose:
- RAII management of config/resources/matrix/vectors/solver handles, plus upload/setup/solve/download wrappers.

Expected API:
```cpp
class AmgXContext
{
public:
    void create(const dictionary& solverControls);
    void destroy();

    void uploadPatternAndValuesOnce
    (
        const LduCsrPattern& pattern,
        const scalar* csrValues,
        const scalar* rhs,
        const scalar* x
    );

    void replaceValues
    (
        int nRows,
        int nnz,
        const scalar* csrValues
    );

    void uploadVectors
    (
        int nRows,
        const scalar* rhs,
        const scalar* x,
        const bool reuseInitialGuess
    );

    void setup();
    void solve(const bool zeroInitialGuess);
    void downloadSolution(int nRows, scalar* xOut);

    AMGX_SOLVE_STATUS lastStatus() const;
    int lastIterations() const;
    double lastFinalResidual() const;
};
```

### `AmgXSolver`

Purpose:
- runtime-selected `lduMatrix::solver` subclass that integrates the above components and exposes OpenFOAM `solverPerformance`.

Expected API shape:
```cpp
class AmgXSolver final : public lduMatrix::solver
{
public:
    TypeName("AmgX");

    AmgXSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const override;
};
```

## Dependency sequencing

- `LduCsrPatternBuilder` depends on OpenFOAM addressing only.
- `CsrValuePacker` depends on `LduCsrPatternBuilder`.
- `AmgXContext` depends on AMGX and CUDA only.
- `PressureMatrixCache` depends on all three.
- `AmgXSolver` depends on `PressureMatrixCache` and native fallback utilities.

### Data model / memory model

## Ownership and lifetime model

| Object | Owner | Lifetime | Location in bring-up path | Location in later production path | Notes |
|---|---|---:|---|---|---|
| `diag`, `lower`, `upper`, `source`, `psi` from OpenFOAM | existing solver/matrix objects | per solve call / timestep | host | eventually mirrored or replaced by device-resident fields in later phases | not owned by Phase 4 |
| `PressureMatrixCache` | object registry | mesh/config lifetime | host object with host/device members | same | persistent |
| `rowPtr`, `colInd` | cache | topology lifetime | pinned host or standard host (one-time upload source) | optional device mirror plus host debug mirror | uploaded once |
| `diagToCsr`, `upperToCsr`, `lowerToCsr` | cache | topology lifetime | host | optional device mirror later | reused every solve |
| `csrValuesHost` | cache | run lifetime | pinned host | optional debug mirror only | reused each solve in bring-up |
| `rhsHost`, `xHost` | cache | run lifetime | pinned host | optional debug mirror only | reused each solve in bring-up |
| `csrValuesDev`, `rhsDev`, `xDev` | cache | cache lifetime once `DeviceDirect` is implemented | optional during initial `PinnedHost` bring-up; required before AmgX production handoff | device | populated by later Phase 5 assembly/update code |
| `AMGX_matrix`, `AMGX_vector b`, `AMGX_vector x`, `AMGX_solver` | AmgX context | cache lifetime | device (internal to AMGX) | device | must persist |
| telemetry counters/timers | cache | run lifetime | host | host | persistent |

## Mandatory memory rules

1. **No unified memory** in Phase 4 bridge buffers.
2. **Pinned host buffers** only for bring-up transfer sources/destinations.
3. **No per-solve allocation** after first warm-up unless topology or configuration changes.
4. **32-bit AMGX indices** only; explicit range-check before allocation.
5. **Diagonal must live in CSR values**, never in separate `diag_data`.
6. **No repeated row/column uploads** after first successful pattern upload.

## What lives where and why

### Bring-up path (implemented now)

- **Host only**
  - OpenFOAM matrix data (`diag/lower/upper/source/psi`)
  - mapping arrays (`diagToCsr`, `upperToCsr`, `lowerToCsr`)
  - pinned host staging buffers (`csrValuesHost`, `rhsHost`, `xHost`)
- **Device only**
  - AmgX internal matrix and vector objects

**Allowed transfers per solve after warm-up:**
- Host → Device:
  - `csrValues` (`8 * nnz` bytes for double precision)
  - `rhs` (`8 * nRows`)
  - `x` (`8 * nRows`) if reusing initial guess
- Device → Host:
  - `x` (`8 * nRows`)
- **Not allowed after warm-up:**
  - row pointer upload
  - column index upload
  - UVM fault-driven migration

### Production path / mandatory bridge (defined now; completed before any later AmgX production claim of no field-scale host transfer)

- **Device-resident external staging**
  - `csrValuesDev`, `rhsDev`, `xDev`
- **Phase ownership**
  - Phase 4 owns these buffers and the backend upload path.
  - Phase 5 owns the device-resident pressure assembly/update semantics that populate them.
- **Assembly path**
  - later pressure assembly writes directly into these device buffers
- **AmgX bridge**
  - uses device pointers for `replace_coefficients` and vector upload
- **Remaining unavoidable copy**
  - AmgX still copies staged buffers into its own internal structures during upload/replace. This will be device-to-device instead of host-to-device, but it is still not zero-copy. [R14]

## Memory sizing formulas

For a scalar internal-cell matrix with no coupled interface rows:

- `nRows = nCells`
- `nnz = nCells + 2 * nInternalFaces`

Approximate persistent external storage (not including AmgX internal hierarchy) with double precision and 32-bit indices:

- row pointers: `4 * (nRows + 1)`
- column indices: `4 * nnz`
- values staging: `8 * nnz`
- RHS and solution staging: `16 * nRows`
- mapping arrays:
  - `4 * nRows` (`diagToCsr`)
  - `4 * nInternalFaces` (`upperToCsr`)
  - `4 * nInternalFaces` (`lowerToCsr`)

**Engineering inference:** For multi-million-cell cases, this external staging footprint is material but usually much smaller than the total later VOF field set plus AmgX multigrid hierarchy. Do not budget GPU memory only from the fine-grid matrix; measure peak memory usage under the chosen AmgX config.

### Algorithms and control flow

## Control-flow overview

### First use on a compatible matrix

1. Parse solver controls.
2. Obtain or create `PressureMatrixCache` from object registry.
3. Validate compatibility:
   - scalar system,
   - no coupled interfaces,
   - counts fit 32-bit,
   - supported mode/config.
4. Build topology pattern if absent:
   - compute row lengths,
   - build `rowPtr`,
   - insert columns deterministically,
   - record `diagToCsr`, `upperToCsr`, `lowerToCsr`.
5. Allocate persistent staging buffers.
6. Create AmgX handles/config/resources/matrix/vectors/solver.
7. Pack current values and vectors from LDU into staging buffers.
8. Call `AMGX_matrix_upload_all(..., rowPtr, colInd, values, nullptr)`.
9. Upload vectors.
10. Call `AMGX_solver_setup`.
11. Call solve.
12. Download solution.
13. Populate `solverPerformance`.

### Subsequent solves on same topology/config

1. Reuse cache.
2. Re-pack only current values and vectors.
3. Call `AMGX_matrix_replace_coefficients`.
4. Upload current RHS and optionally initial guess.
5. Call `AMGX_solver_setup`.
6. Solve.
7. Download solution.
8. Update telemetry and `solverPerformance`.

### Unsupported or failed path

1. Detect unsupported condition or AmgX failure.
2. Increment fallback counters.
3. Create/dispatch native fallback solver using configured controls.
4. Return native `solverPerformance`.

## Pattern-build algorithm

### Input
- `nCells`
- `lowerAddr` (owner list)
- `upperAddr` (neighbour list)

### Output
- CSR row pointers
- CSR column indices
- maps from LDU positions to CSR positions

### Deterministic insertion policy

For each row:
1. insert diagonal entry first,
2. insert off-diagonals in internal-face order as encountered.

This preserves a simple 1:1 mapping and reproducible debug behavior.

### Detailed algorithm

1. Range-check:
   - `nCells <= INT_MAX`
   - `nInternalFaces <= INT_MAX`
   - `nnz = nCells + 2*nInternalFaces <= INT_MAX`

2. Count row lengths:
   - initialize `rowLen[c] = 1` for every cell (diagonal)
   - for each face `f`:
     - `rowLen[owner[f]]++`
     - `rowLen[neigh[f]]++`

3. Exclusive scan:
   - `rowPtr[0] = 0`
   - `rowPtr[c+1] = rowPtr[c] + rowLen[c]`

4. Initialize row cursors:
   - `cursor = rowPtr` copy

5. Insert diagonal entries:
   - for each cell `c`:
     - `idx = cursor[c]++`
     - `colInd[idx] = c`
     - `diagToCsr[c] = idx`

6. Insert off-diagonals:
   - for each face `f`:
     - `o = lowerAddr[f]`
     - `n = upperAddr[f]`
     - `idxOwner = cursor[o]++`
     - `colInd[idxOwner] = n`
     - `upperToCsr[f] = idxOwner`
     - `idxNeigh = cursor[n]++`
     - `colInd[idxNeigh] = o`
     - `lowerToCsr[f] = idxNeigh`

7. Debug validations:
   - `cursor[c] == rowPtr[c+1]` for all `c`
   - no duplicates per row
   - `diagToCsr[c]` points to column `c`
   - `upperToCsr`/`lowerToCsr` indices inside row bounds

## Value-pack algorithm

Given the current matrix state as passed into `solve()`:

1. `csrValuesHost[diagToCsr[c]] = matrix.diag()[c]` for each cell.
2. `csrValuesHost[upperToCsr[f]] = matrix.upper()[f]` for each face.
3. `csrValuesHost[lowerToCsr[f]] = matrix.lower()[f]` for each face.
4. `rhsHost[c] = source[c]`
5. If `reuseInitialGuess`:
   - `xHost[c] = psi[c]`
   else
   - leave `xHost` unchanged or zero locally if the implementation wants a debug copy,
   - actual solve call should use `AMGX_solver_solve_with_0_initial_guess`.

**Important semantic note:** `matrix.diag()` at this point already contains any boundary-diagonal contributions added by the caller’s solve path. Do not use cached or constructor-time diagonal values.

## Fallback algorithm

Fallback is triggered if any of the following occurs:

- coupled interfaces detected,
- unsupported mode,
- unsupported matrix type,
- index overflow risk,
- AmgX object creation failure,
- upload/setup failure,
- solve status not successful and fallback-on-failure enabled,
- topology changed but rebuild disabled,
- local build not using double precision.

Fallback procedure:

1. Build a fallback solver-controls dictionary:
   - start from `solverControls`,
   - replace `solver` with `fallbackSolver`,
   - remove/ignore AmgX-specific keys.
2. Call native `lduMatrix::solver::New(...)->solve(...)`.
3. Record fallback reason and counters.

## Host/device synchronization policy

- The Phase 4 implementation must not call `cudaDeviceSynchronize()` around every library step.
- Treat AmgX upload/setup/solve/download as blocking host API boundaries unless explicit AMgX documentation proves otherwise.
- Use NVTX ranges for profiling, not global device synchronizations.

### Required source changes

## Repository 1: `foamExternalSolvers` (preferred primary implementation site)

### Must change

1. **AmgX solver class**
   - Extend or replace the existing AmgX solver implementation so it uses a persistent pressure-matrix cache instead of rebuilding conversion metadata every solve.

2. **AmgX wrapper/context**
   - Add or extend RAII wrappers for:
     - config creation/destruction,
     - resources,
     - matrix handle,
     - vector handles,
     - solver handle,
     - status/iteration/residual queries.

3. **Runtime controls parser**
   - Parse Phase 4 controls:
     - `solver AmgX`
     - `matrixType CSR`
     - `dataLocation device`
     - `mode dDDI`
     - `fallbackSolver`
     - `reuseInitialGuess`
    - `allowCoupledInterfaces` (must default false and may not override the centralized support matrix)
    - `stagingMode`
    - config file/string/dictionary source

   Project-normalized ownership of these controls is under `gpuRuntime.pressure`; existing `fvSolution` keys or phase-local dictionaries are compatibility shims/generated views, not independent contracts.

4. **Persistent cache object**
   - Add a registry-backed cache object type.

5. **Topology builder and packer**
   - Add explicit modules for LDU→CSR mapping and value packing.

6. **Validation utilities**
   - Add debug-only or test-only validators for matrix equivalence.

### Should change

1. Add NVTX3 ranges around all major AmgX bridge phases.
2. Add per-solve telemetry reporting hooks.
3. Add a matrix snapshot replay utility.

## Repository 2: `spuma`

### Must change

1. **Build integration**
   - Ensure the AmgX-enabled `foamExternalSolvers` library is built and linked for SPUMA/v2412.
2. **Case configuration templates**
   - Provide an example `fvSolution` pressure-solver stanza using the new AmgX path.
3. **Test harness integration**
   - Add a reduced case or replay workflow to CI or at least reproducible scripts.

### Should change

1. Add optional function-object or utility support for dumping representative pressure matrices.
2. Add profiling wrapper scripts.

## Not required in Phase 4

- Modifying `incompressibleVoF` itself.
- Modifying alpha, surface-tension, or nozzle BC code.

### Proposed file layout and module boundaries

The recommended file layout assumes the preferred extension point is the existing `foamExternalSolvers` AmgX module. If the local repository layout differs, keep the same module boundaries and adapt paths.

```text
foamExternalSolvers/
  src/
    AmgX4Foam/
      AmgXSolver.H                  # runtime-selected lduMatrix::solver subclass
      AmgXSolver.C                  # solve orchestration + fallback bridge
      AmgXContext.H                 # RAII wrapper around AMGX handles
      AmgXContext.C
      PressureMatrixCache.H         # objectRegistry-backed persistent cache
      PressureMatrixCache.C
      LduCsrPatternBuilder.H        # topology-only conversion
      LduCsrPatternBuilder.C
      CsrValuePacker.H              # value/vector packing
      CsrValuePacker.C
      PressureFallback.H            # fallback reason enum + helper
      PressureFallback.C
      PressureTelemetry.H           # counters/timers/NVTX labels
      PressureTelemetry.C
      AmgXError.H                   # error macros / status conversion
      AmgXConfigParser.H            # config parsing from OpenFOAM dict/file/string
      AmgXConfigParser.C
      Make/
        files
        options

  applications/
    utilities/
      phase4DumpPressureSystem/
        phase4DumpPressureSystem.C  # dump LDU pressure snapshots
        Make/files
        Make/options
      phase4ReplayPressureSystem/
        phase4ReplayPressureSystem.C # load snapshot, compare native vs AmgX
        Make/files
        Make/options
      phase4ValidateLduCsr/
        phase4ValidateLduCsr.C      # synthetic/unit-style validation executable
        Make/files
        Make/options

spuma/
  tutorials/ or cases/
    phase4/
      reducedPressureReplay/
      reducedNozzleR1/
  scripts/
    profilePhase4.sh
    benchmarkPhase4.sh
```

## Module boundary rules

1. `LduCsrPatternBuilder` must not depend on AMGX.
2. `CsrValuePacker` must not know about runtime selection or fallback.
3. `AmgXContext` must not know about OpenFOAM addressing details.
4. `PressureMatrixCache` is the only component allowed to own both OpenFOAM-side mappings and AmgX-side objects.
5. `AmgXSolver` is orchestration only; it should not manually build patterns or call raw AMGX C functions directly.
6. Test utilities must exercise the library code, not reimplement it.

### Pseudocode

#### 1. Cache lookup and solve orchestration

```cpp
solverPerformance AmgXSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance perf(typeName, fieldName_);
    perf.nIterations() = 0;

    if (cmpt != 0)
    {
        // Scalar pressure path only. For scalar fields cmpt is normally 0.
        WarningInFunction
            << "Unexpected component " << cmpt
            << ". Continuing as scalar solve." << nl;
    }

    word reason;
    objectRegistry& db = const_cast<objectRegistry&>(matrix().mesh().thisDb());

    PressureMatrixCache& cache =
        PressureMatrixCache::getOrCreate
        (
            db,
            matrix().mesh().name(),
            fieldName_,
            solverControls_
        );

    if (!cache.compatibleWith(matrix(), interfaces_, reason))
    {
        return fallbackSolve(psi, source, cmpt, reason);
    }

    try
    {
        NVTXRange rangeTotal("pressure.amgx.total");

        cache.ensurePatternBuilt(matrix());
        cache.ensureAmgXObjectsCreated();

        const bool reuseInitialGuess =
            solverControls_.getOrDefault<bool>("reuseInitialGuess", true)
         && !solverControls_.getOrDefault<bool>("forceZeroInitialGuess", false);

        cache.packHostStaging(matrix(), source, psi, reuseInitialGuess);

        cache.uploadOrReplace();   // upload_all first time, replace_coefficients later
        cache.setupSolver();       // mandatory after matrix changes
        cache.solve(!reuseInitialGuess);
        cache.downloadSolution(psi);

        perf = cache.makeSolverPerformance(fieldName_);
        return perf;
    }
    catch (const AmgXRecoverableError& e)
    {
        cache.noteFailure(e.what());
        return fallbackSolve(psi, source, cmpt, e.what());
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "Unrecoverable AmgX bridge failure: " << e.what()
            << exit(FatalError);
        return perf;
    }
}
```

#### 2. Pattern builder

```cpp
LduCsrPattern buildLduCsrPattern
(
    const lduAddressing& addr,
    const label nCells
)
{
    const labelUList& owner = addr.lowerAddr(); // owner list
    const labelUList& neigh = addr.upperAddr(); // neighbour list
    const label nFaces = owner.size();

    // ---- Range checks for AMGX int-based API ----
    const long long nnz64 = static_cast<long long>(nCells) + 2LL * nFaces;
    if (nCells > INT_MAX || nFaces > INT_MAX || nnz64 > INT_MAX)
    {
        throw UnsupportedMatrix("Matrix exceeds AMGX 32-bit local index limits");
    }

    LduCsrPattern pat;
    pat.nRows = static_cast<int>(nCells);
    pat.nnz   = static_cast<int>(nnz64);

    List<int> rowLen(nCells, 1); // one diagonal per row
    forAll(owner, f)
    {
        rowLen[owner[f]]++;
        rowLen[neigh[f]]++;
    }

    pat.rowPtr.setSize(nCells + 1);
    pat.rowPtr[0] = 0;
    for (label c = 0; c < nCells; ++c)
    {
        pat.rowPtr[c + 1] = pat.rowPtr[c] + rowLen[c];
    }

    pat.colInd.setSize(pat.nnz);
    pat.diagToCsr.setSize(nCells);
    pat.upperToCsr.setSize(nFaces);
    pat.lowerToCsr.setSize(nFaces);

    List<int> cursor(pat.rowPtr); // working copy

    // Insert diagonal first.
    for (label c = 0; c < nCells; ++c)
    {
        const int idx = cursor[c]++;
        pat.colInd[idx] = static_cast<int>(c);
        pat.diagToCsr[c] = idx;
    }

    // Insert off-diagonals.
    forAll(owner, f)
    {
        const label o = owner[f];
        const label n = neigh[f];

        const int idxOwner = cursor[o]++;
        pat.colInd[idxOwner] = static_cast<int>(n);
        pat.upperToCsr[f] = idxOwner;

        const int idxNeigh = cursor[n]++;
        pat.colInd[idxNeigh] = static_cast<int>(o);
        pat.lowerToCsr[f] = idxNeigh;
    }

    // Debug checks.
    #ifdef FULLDEBUG
    for (label c = 0; c < nCells; ++c)
    {
        if (cursor[c] != pat.rowPtr[c+1])
        {
            throw LogicError("CSR row fill mismatch");
        }
        validateNoDuplicateColumnsInRow(pat.rowPtr, pat.colInd, c);
        if (pat.colInd[pat.diagToCsr[c]] != c)
        {
            throw LogicError("Diagonal mapping corrupted");
        }
    }
    #endif

    pat.topologyHash = hashAddressing(owner, neigh);
    return pat;
}
```

#### 3. Value packer

```cpp
void packCsrValuesHost
(
    const lduMatrix& matrix,
    const scalarField& source,
    const scalarField& psi,
    const LduCsrPattern& pat,
    scalar* csrValuesHost,
    scalar* rhsHost,
    scalar* xHost,
    const bool reuseInitialGuess
)
{
    const scalarField& diag  = matrix.diag();
    const scalarField& upper = matrix.upper();
    const scalarField& lower = matrix.lower();

    // Fill diagonal entries
    forAll(diag, c)
    {
        csrValuesHost[pat.diagToCsr[c]] = diag[c];
        rhsHost[c] = source[c];
        if (reuseInitialGuess)
        {
            xHost[c] = psi[c];
        }
    }

    // Fill off-diagonal entries
    forAll(upper, f)
    {
        csrValuesHost[pat.upperToCsr[f]] = upper[f];
        csrValuesHost[pat.lowerToCsr[f]] = lower[f];
    }

    #ifdef FULLDEBUG
    validateFinite(csrValuesHost, pat.nnz, "csrValuesHost");
    validateFinite(rhsHost, pat.nRows, "rhsHost");
    if (reuseInitialGuess)
    {
        validateFinite(xHost, pat.nRows, "xHost");
    }
    #endif
}
```

#### 4. AmgX upload / replace / solve boundary

```cpp
void PressureMatrixCache::uploadOrReplace()
{
    if (!uploadedOnce_)
    {
        NVTXRange r("pressure.amgx.upload_all");
        amgxContext_.uploadPatternAndValuesOnce
        (
            pattern_,
            csrValuesHostPtr_,
            rhsHostPtr_,
            xHostPtr_
        );
        uploadedOnce_ = true;
        telemetry_.patternUploadCount++;
        telemetry_.bytesPatternHtoD += patternBytes(pattern_);
        telemetry_.bytesValuesHtoD  += valuesBytes(pattern_);
        telemetry_.bytesRhsHtoD     += rhsBytes(pattern_);
        telemetry_.bytesGuessHtoD   += xBytes(pattern_);
    }
    else
    {
        NVTXRange r1("pressure.amgx.replace_coefficients");
        amgxContext_.replaceValues(pattern_.nRows, pattern_.nnz, csrValuesHostPtr_);
        telemetry_.replaceCount++;
        telemetry_.bytesValuesHtoD += valuesBytes(pattern_);

        NVTXRange r2("pressure.amgx.upload_vectors");
        amgxContext_.uploadVectors
        (
            pattern_.nRows,
            rhsHostPtr_,
            xHostPtr_,
            reuseInitialGuess_
        );
        telemetry_.bytesRhsHtoD += rhsBytes(pattern_);
        if (reuseInitialGuess_)
        {
            telemetry_.bytesGuessHtoD += xBytes(pattern_);
        }
    }
}

void PressureMatrixCache::setupSolver()
{
    NVTXRange r("pressure.amgx.setup");
    amgxContext_.setup();
    telemetry_.setupCount++;
}

void PressureMatrixCache::solve(const bool zeroInitialGuess)
{
    NVTXRange r("pressure.amgx.solve");
    amgxContext_.solve(zeroInitialGuess);
    telemetry_.solveCount++;

    const AMGX_SOLVE_STATUS st = amgxContext_.lastStatus();
    if (st == AMGX_SOLVE_FAILED || st == AMGX_SOLVE_DIVERGED)
    {
        throw AmgXRecoverableError
        (
            st == AMGX_SOLVE_FAILED
          ? "AMGX solve failed"
          : "AMGX solve diverged"
        );
    }
}

void PressureMatrixCache::downloadSolution(scalarField& psi)
{
    NVTXRange r("pressure.amgx.download_solution");
    amgxContext_.downloadSolution(pattern_.nRows, xHostPtr_);
    telemetry_.bytesSolutionDtoH += xBytes(pattern_);

    forAll(psi, c)
    {
        psi[c] = xHostPtr_[c];
    }
}
```

#### 5. `A*x` equivalence validator

```cpp
bool validateCsrEquivalentToLdu
(
    const lduMatrix& matrix,
    const LduCsrPattern& pat,
    const scalar* csrValues,
    const scalar tolRel,
    scalar& relErrOut
)
{
    scalarField x(matrix.diag().size(), 0.0);
    fillDeterministicPseudoRandom(x);

    scalarField yLdu(x.size(), 0.0);
    scalarField yCsr(x.size(), 0.0);

    matrix.Amul(yLdu, x, FieldField<Field, scalar>(), lduInterfaceFieldPtrsList(), 0);

    for (int row = 0; row < pat.nRows; ++row)
    {
        scalar sum = 0;
        for (int k = pat.rowPtr[row]; k < pat.rowPtr[row+1]; ++k)
        {
            sum += csrValues[k] * x[pat.colInd[k]];
        }
        yCsr[row] = sum;
    }

    relErrOut = normInf(yLdu - yCsr) / max(SMALL, normInf(yLdu));
    return relErrOut <= tolRel;
}
```

#### 6. DeviceDirect staging interface (required pressure-backend bridge after bring-up)

```cpp
// Phase 4 lands this in two steps: reserve the interface during `PinnedHost` bring-up,
// then complete the `DeviceDirect` bridge before any later production claim that says
// AmgX pressure correction avoids field-scale host transfer. Phase 4 owns only the
// staging/upload boundary; Phase 5 owns the device-side pressure assembly/update
// semantics that populate DevicePressureFields.
void PressureMatrixCache::packDeviceStaging
(
    const DevicePressureFields& deviceFields,
    cudaStream_t stream
)
{
    // Mandatory bridge work:
    // 1. Launch kernel to scatter diag/upper/lower into csrValuesDev using mapping arrays
    // 2. Launch kernel to copy RHS and optional initial guess into rhsDev/xDev
    // 3. Call AmgX replace/upload with device pointers
}
```

### Step-by-step implementation guide

Each step below includes: what to modify, why, expected output, how to verify success, and likely breakages.

#### Step 1 — Freeze dependency versions and add a standalone AmgX smoke target

- **What to modify**
  - Pin AMGX version in the build environment.
  - Add a tiny standalone build or utility that creates an identity matrix in AmgX and solves it.
- **Why**
  - Confirms the Blackwell/CUDA/driver/AMGX stack before touching OpenFOAM integration.
- **Expected output**
  - Executable runs on the RTX 5080 and returns the input RHS as solution.
- **How to verify success**
  - `AMGX_solver_get_status == AMGX_SOLVE_SUCCESS`.
  - Solution matches exactly for identity matrix.
- **Likely breakages**
  - Wrong AMGX version or build flags for Blackwell.
  - NVTX2/NVTX3 mismatch.
  - bad `CMAKE_CUDA_ARCHITECTURES`.

#### Step 2 — Add a pressure-system snapshot dump utility

- **What to modify**
  - Add `phase4DumpPressureSystem` utility or equivalent hook in the CPU reference branch.
- **Why**
  - Snapshot replay is the safest prototype boundary before in-solver replacement.
- **Expected output**
  - A dump directory containing:
    - `diag`
    - `lower`
    - `upper`
    - `source`
    - `psi`
    - `lowerAddr`
    - `upperAddr`
    - metadata (`fieldName`, `time`, `nCells`, `nFaces`, `hasInterfaces`, `solverControlsHash`)
- **How to verify success**
  - Reload snapshot and reconstruct matrix sizes exactly.
- **Likely breakages**
  - Dumping at the wrong stage of the solve (before boundary/reference modifications).
- **Important**
  - Capture snapshots **at the exact point the native solver is invoked**, not earlier in equation assembly.

#### Step 3 — Implement the topology-only LDU→CSR builder

- **What to modify**
  - Add `LduCsrPatternBuilder`.
- **Why**
  - Isolates the most error-prone mapping logic.
- **Expected output**
  - A reusable `LduCsrPattern` object with row pointers, columns, and mapping arrays.
- **How to verify success**
  - Use a synthetic 3x3 or 4x4 matrix with hand-checked owner/neighbour layout.
  - Assert `nnz = nCells + 2*nFaces`.
- **Likely breakages**
  - Owner/neighbour reversed.
  - `upper`/`lower` mapping swapped.
  - duplicate or missing diagonal entries.

#### Step 4 — Implement a pure-host CSR value packer

- **What to modify**
  - Add `CsrValuePacker`.
- **Why**
  - Separates coefficient/value ownership from topology ownership.
- **Expected output**
  - `csrValuesHost`, `rhsHost`, and optional `xHost` arrays for a current matrix state.
- **How to verify success**
  - Run `A*x` equivalence against the native LDU matrix for deterministic random vectors.
- **Likely breakages**
  - Using stale diagonal values from constructor time.
  - incorrect mapping of `upper`/`lower`.
  - forgetting to pack boundary-adjusted `source`.

#### Step 5 — Implement `PressureMatrixCache` with persistent host staging buffers

- **What to modify**
  - Add cache object in object registry.
  - Add pinned-buffer allocation and reuse.
- **Why**
  - Avoids repeated conversion/allocation across correctors.
- **Expected output**
  - Cache survives across repeated solves and reuses its buffers.
- **How to verify success**
  - Warm-up once, then run two consecutive solves and confirm:
    - pattern build count remains 1,
    - allocation count remains 1.
- **Likely breakages**
  - Bad cache key causing accidental sharing across incompatible matrices.
  - cache not invalidated when config changes.

#### Step 6 — Implement `AmgXContext` RAII wrapper

- **What to modify**
  - Wrap config/resources/matrix/vectors/solver creation and destruction.
- **Why**
  - Keeps raw C API calls out of the solver logic and centralizes error handling.
- **Expected output**
  - One safe place for:
    - `AMGX_initialize`,
    - config creation,
    - resources creation,
    - matrix/vector/solver handle lifecycle.
- **How to verify success**
  - Leak-free repeated create/destroy in a smoke test.
- **Likely breakages**
  - resource ordering or double-destruction.
  - mismatched mode/config strings.

#### Step 7 — Implement first-call upload path

- **What to modify**
  - In cache/context path, call `AMGX_matrix_upload_all` once with full CSR including diagonal.
- **Why**
  - Establishes AmgX internal storage and binding.
- **Expected output**
  - First solve completes on device.
- **How to verify success**
  - Identity and Poisson snapshot replay tests succeed.
- **Likely breakages**
  - Passing non-null `diag_data` while diagonal is already inside CSR.
  - wrong rowPtr/colInd sizes.
  - non-pinned host buffers masking transfer cost.

#### Step 8 — Implement repeated-solve replace/setup/solve path

- **What to modify**
  - Add `AMGX_matrix_replace_coefficients`, vector uploads, repeated `AMGX_solver_setup`, and `solve`.
- **Why**
  - This is the actual hot-path behavior for transient pressure correction.
- **Expected output**
  - Consecutive solves on the same topology do not rebuild or reupload row/column arrays.
- **How to verify success**
  - Telemetry shows:
    - `patternUploadCount == 1`
    - `replaceCount == numberOfAdditionalSolves`
- **Likely breakages**
  - forgetting `AMGX_solver_setup` after replace.
  - stale vectors due to missing RHS or guess upload.

#### Step 9 — Implement runtime-selected `AmgXSolver` with fallback

- **What to modify**
  - Register `AmgX` solver in runtime selection tables.
  - Add fallback controls and unsupported-case detection.
- **Why**
  - Needed for live OpenFOAM integration.
- **Expected output**
  - `solver AmgX;` in `fvSolution` activates the backend.
- **How to verify success**
  - Native fallback triggers cleanly when unsupported interfaces are injected.
- **Likely breakages**
  - recursion if fallback solver also resolves to `AmgX`.
  - not stripping AmgX-only keys from fallback dictionary.

#### Step 10 — Add debug validators and snapshot replay executable

- **What to modify**
  - `phase4ReplayPressureSystem` utility using both native and AmgX paths.
- **Why**
  - Gives a repeatable correctness harness independent of full nozzle runtime.
- **Expected output**
  - Tool can load a snapshot and solve it with either backend.
- **How to verify success**
  - Native and AmgX solutions match within tolerance on small snapshots.
- **Likely breakages**
  - snapshot captured at wrong solve stage.
  - mismatch between live and replay solver controls.

#### Step 11 — Add NVTX3 ranges and telemetry counters

- **What to modify**
  - Wrap each bridge phase with NVTX3.
  - Add counters to cache.
- **Why**
  - Required to separate setup from solve and catch repeated topology upload bugs.
- **Expected output**
  - Nsight timeline shows distinct regions:
    - `pressure.pattern_build`
    - `pressure.value_pack`
    - `pressure.amgx.upload_all`
    - `pressure.amgx.replace_coefficients`
    - `pressure.amgx.upload_vectors`
    - `pressure.amgx.setup`
    - `pressure.amgx.solve`
    - `pressure.amgx.download_solution`
    - `pressure.fallback.native`
- **How to verify success**
  - Timeline contains exactly those ranges in expected order.
- **Likely breakages**
  - NVTX2 headers accidentally used under the frozen Blackwell-compatible CUDA lane.

#### Step 12 — Integrate with reduced live case and benchmark against native

- **What to modify**
  - Reduced-case `fvSolution`.
  - benchmark script.
- **Why**
  - Ensures replay correctness translates into live solve behavior.
- **Expected output**
  - Reduced case runs with AmgX backend and emits timing/telemetry logs.
- **How to verify success**
  - No solver crashes.
  - fallback count zero for intended supported case.
- **Likely breakages**
  - hidden coupled patch/interface in case setup.
  - reference value not set, causing singular pressure matrix.

#### Step 13 — Complete the `DeviceDirect` pressure bridge

- **What to modify**
  - Implement the `DeviceDirect` staging mode using persistent device-resident `csrValuesDev`, `rhsDev`, and `xDev` buffers plus backend upload/replace paths that accept device pointers.
- **Why**
  - This is the required bridge before AmgX can support any Phase 5/6/8 production claim of no field-scale host pressure transfer.
- **Expected output**
  - A runtime-selectable `DeviceDirect` path validated on replay and reduced live cases; `PinnedHost` remains available for correctness-only bring-up.
- **How to verify success**
  - Nsight Systems shows no field-scale host pressure staging in production mode and the acceptance report records the `PinnedHost` vs `DeviceDirect` handoff status explicitly.
- **Likely breakages**
  - stale device-buffer ownership, assembly-to-solver handoff mismatch, accidental fallback to host staging.

### Instrumentation and profiling hooks

## NVTX3 ranges to add

These ranges are mandatory:

1. `pressure.pattern_build`
2. `pressure.pattern_validate`
3. `pressure.value_pack`
4. `pressure.amgx.create`
5. `pressure.amgx.upload_all`
6. `pressure.amgx.replace_coefficients`
7. `pressure.amgx.upload_vectors`
8. `pressure.amgx.setup`
9. `pressure.amgx.solve`
10. `pressure.amgx.download_solution`
11. `pressure.fallback.native`
12. `pressure.cache.invalidate`

## Telemetry counters to persist

```cpp
struct PressureSolveTelemetry
{
    uint64_t patternBuildCount;
    uint64_t patternUploadCount;
    uint64_t replaceCount;
    uint64_t setupCount;
    uint64_t solveCount;
    uint64_t fallbackCount;

    uint64_t bytesPatternHtoD;
    uint64_t bytesValuesHtoD;
    uint64_t bytesRhsHtoD;
    uint64_t bytesGuessHtoD;
    uint64_t bytesSolutionDtoH;

    double msPatternBuild;
    double msValuePack;
    double msUpload;
    double msSetup;
    double msSolve;
    double msDownload;
};
```

## Required profiling commands

### Nsight Systems

Use a reduced test or replay utility first:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --sample=none \
  -o phase4_pressure_replay \
  phase4ReplayPressureSystem -snapshot <snapshotDir> -solver AmgX
```

For SPUMA full-case profiling, follow SPUMA’s documented UVM page-fault tracing approach and add MPI tracing only if/when MPI is involved. [R2]

### Compute Sanitizer

```bash
compute-sanitizer --tool memcheck phase4ReplayPressureSystem -snapshot <snapshotDir> -solver AmgX
```

### Nsight Compute

Use only on replay or reduced cases. Focus on top kernels and on API-to-kernel ratio, not on the whole nozzle case at once.

## Metrics to log per solve

- topology reused? (`yes/no`)
- pattern upload performed? (`yes/no`)
- setup time
- solve time
- final residual
- iteration count
- bytes HtoD / DtoH
- fallback reason if any

### Validation strategy

Validation is split into **unit-level**, **snapshot-replay**, and **live reduced-case** validation.

## 1. Unit-level correctness checks

### U1 — Hand-constructed matrix mapping

Build a tiny synthetic matrix where the exact full matrix is known. Verify:
- rowPtr
- colInd
- diag positions
- owner/neighbour off-diagonal placement

**Pass threshold:** exact integer match.

### U2 — `A*x` equivalence test

For a deterministic random vector `x`, compare:
- OpenFOAM `lduMatrix::Amul`
- host CSR SpMV built from the Phase 4 conversion

**Pass threshold:** relative infinity-norm error `< 1e-13` for double-precision builds.

### U3 — Duplicate-column validator

For each row, confirm no duplicate column indices.

**Pass threshold:** zero duplicates.

### U4 — Diagonal ownership validator

Confirm every row contains exactly one diagonal entry and `diag_data` is always null in AmgX calls.

**Pass threshold:** always true.

## 2. Snapshot-replay validation

Use captured pressure-system snapshots from representative cases:

- startup / early transient snapshot
- mid-run representative snapshot
- worst non-orthogonal-corrector snapshot

### S1 — Native replay sanity

The snapshot replay utility using native solver should reproduce the original solution within normal iterative tolerance drift.

**Pass threshold:** relative L2 difference `< 1e-12` on deterministic single-snapshot replay when using the same solver controls.

### S2 — AmgX replay correctness

Compare native and AmgX replay solutions on the same snapshot.

**Required pass threshold for small deterministic snapshots:**
- relative L2 difference `< 1e-9`
- relative infinity-norm difference `< 1e-8`

**Recommended pass threshold for real nozzle snapshots:**
- relative L2 difference `< 1e-7`
- residual histories may differ, but final solver convergence should satisfy the configured tolerance.

### S3 — Pattern reuse across multiple snapshot solves

Repeated replay of multiple snapshots with identical topology must show:
- pattern build count = 1
- pattern upload count = 1
- coefficient replace count = number of later solves

**Pass threshold:** exact counter values.

### S4 — No UVM traffic

Nsight Systems should show zero recurring UVM page-fault activity in the AmgX bridge path.

**Pass threshold:** zero steady-state UVM HtoD/DtoH fault events attributable to the solver bridge.

## 3. Live reduced-case validation

### L1 — Reduced case runs to completion

Run reduced nozzle or reduced pressure-only case for a fixed number of timesteps.

**Pass threshold:** no crashes, no unsupported-case fallback for the intended supported case.

### L2 — Macroscopic metric comparison

Compare against native baseline on the reduced case:
- integrated continuity error
- pressure drop
- mass flow rate
- any case-specific scalar invariant chosen by the human reviewer

**Recommended threshold:** within 0.1%–0.5% depending on metric sensitivity.  
**Human sign-off required** on exact production thresholds.

### L3 — Solver path logging

Ensure telemetry indicates:
- topology upload only once,
- repeated coefficient replacement later,
- setup time measured separately from solve time.

### Validation order

1. Unit tests
2. Snapshot replay
3. Reduced live case
4. Only then use the backend in longer transient nozzle runs

### Performance expectations

## What Phase 4 should improve immediately

1. **Topology work amortization**
   - pattern build and row/column construction should disappear after the first compatible solve.

2. **Transfer clarity**
   - the remaining per-solve cost should be explicit value/vector upload and solution download, not hidden UVM movement.

3. **Allocation stability**
   - no repeated pinned-buffer allocation after warm-up.

## What Phase 4 should not promise

1. End-to-end nozzle speedup.
2. AmgX beating GAMG on every pressure snapshot.
3. Zero-copy pressure solve.
4. Graph-level launch reduction.

## Quantitative expectations for the bring-up path

For each repeated solve after warm-up, expected external transfer volume is approximately:

- `values`: `8 * nnz`
- `rhs`: `8 * nRows`
- `initial guess`: `8 * nRows` if reused
- `solution download`: `8 * nRows`

One-time topology upload:
- `rowPtr`: `4 * (nRows + 1)`
- `colInd`: `4 * nnz`

These are engineering expectations from the chosen data model, not backend-internal guarantees.

## Success criteria for performance in this phase

1. `patternBuildCount == 1` for a static-mesh benchmark sequence.
2. `patternUploadCount == 1` for that same sequence.
3. Zero UVM page-fault traffic in the AmgX bridge path.
4. Per-solve timing decomposition is stable and reproducible.
5. Native fallback remains measurable and selectable.

## When to stop and reassess before proceeding to later phases

Stop and benchmark before integrating further if any of the following occur:

1. `AMGX_solver_setup` dominates solve time by a large margin on representative snapshots and no case shows benefit.
2. Peak memory usage threatens 16 GB headroom on reduced cases.
3. UVM traffic appears despite explicit allocations.
4. Correctness mismatch exceeds thresholds and mapping validators fail.
5. Supported cases silently fall back.

### Common failure modes

1. **Owner/neighbour mapping reversed**
   - Symptom: solution converges to wrong field; `A*x` validator fails badly.

2. **`upper`/`lower` coefficient placement swapped**
   - Symptom: asymmetric systems look transposed or mirrored; `A*x` mismatch localized to off-diagonals.

3. **Boundary terms applied twice**
   - Symptom: pressure field shifted or over-damped; diagonal too strong; native vs AmgX mismatch even on simple cases.

4. **Diagonal duplicated in both CSR and `diag_data`**
   - Symptom: undefined behavior, divergence, or unexplained wrong answers. [R14]

5. **Pattern rebuilt every solve**
   - Symptom: high CPU overhead, repeated row/column uploads, profiler shows persistent topology traffic.

6. **Pageable host buffers accidentally used**
   - Symptom: lower transfer throughput, noisy PCIe timings.

7. **UVM introduced accidentally**
   - Symptom: Nsight shows page faults in the pressure path despite explicit intent.

8. **Unsupported interface case slips through**
   - Symptom: incorrect results on cyclic/coupled cases instead of fallback.

9. **OpenFOAM precision build mismatch**
   - Symptom: `scalar` is not double, but AmgX path assumes `dDDI`.

10. **32-bit overflow**
    - Symptom: crashes, negative indices, or corrupted CSR on larger meshes.

11. **Skipped `AMGX_solver_setup` after coefficient change**
    - Symptom: stale hierarchy/metadata, convergence anomalies, or solver failure.

12. **Recursive fallback**
    - Symptom: fallback solver name resolves back to `AmgX`.

13. **Reference pressure not set**
    - Symptom: singular pressure solve, divergence, or inconsistent nullspace behavior.

14. **AmgX config too aggressive for the matrix family**
    - Symptom: repeated divergence or unexpectedly high iteration counts on certain snapshots.

### Debugging playbook

#### Case 1 — `A*x` mismatch

1. Run the synthetic matrix validator.
2. Print first failing row:
   - row range in CSR,
   - expected owner/neighbour columns,
   - `diag`, `upper`, `lower` source values,
   - packed values.
3. Check:
   - `owner = lowerAddr[f]`
   - `neigh = upperAddr[f]`
   - owner row got `upper[f]`
   - neighbour row got `lower[f]`
4. Re-run with a single-face tiny matrix.

#### Case 2 — Native and AmgX solutions differ on snapshots

1. Confirm snapshot was captured **after** boundary/reference modifications and immediately before native solve.
2. Compare:
   - matrix diagonal sums,
   - RHS norm,
   - initial guess norm.
3. Force zero initial guess on both paths.
4. Tighten native solver tolerance temporarily.
5. If mismatch persists, run `A*x` validation again on the live matrix.

#### Case 3 — AmgX diverges

1. Retrieve `AMGX_solver_get_status`.
2. Log iterations and final residual if available.
3. Retry once with zero initial guess.
4. Compare snapshot with native baseline to exclude mapping error.
5. Switch to a more conservative config (e.g., from symmetry-assuming to FGMRES+AMG).
6. If still failing, fallback and record snapshot for analysis.

#### Case 4 — Pattern uploads happen repeatedly

1. Inspect cache key:
   - region
   - field name
   - config hash
   - topology hash
2. Confirm config hash excludes transient values like tolerances that can be read frequently unless change should invalidate the solver.
3. Check whether the code destroys and recreates cache on each solve accidentally.

#### Case 5 — UVM page faults appear

1. Inspect all bridge allocations.
2. Verify pinned host allocator is used.
3. Confirm no `new[]` or `Foam::Field` memory is being passed directly to AmgX as if it were pinned.
4. Confirm no managed memory allocator leaked into the bridge through SPUMA defaults.

#### Case 6 — Live case falls back unexpectedly

1. Log patch names and interface types.
2. Determine whether the case contains cyclic/coupled patches.
3. If yes, fallback is correct; do not “fix” by suppressing the check.
4. If no, inspect interface detection logic.

#### Case 7 — Build fails on the frozen Blackwell-compatible CUDA lane

1. Check AMGX version.
2. Prefer v2.5.0 or a verified commit that is compatible with the frozen primary lane and Blackwell support. [R13]
3. Ensure `CMAKE_CUDA_ARCHITECTURES=120` (or equivalent build-system setting) is used where applicable.
4. Ensure NVTX3 headers are used.

### Acceptance checklist

#### Correctness

- [ ] Synthetic LDU→CSR mapping test passes exactly.
- [ ] `A*x` equivalence passes with relative inf-norm error `< 1e-13`.
- [ ] Small snapshot replay matches native within required tolerances.
- [ ] Reduced live case runs without AmgX-specific numerical failures.
- [ ] Unsupported cases fallback cleanly and visibly.

#### Performance/engineering

- [ ] No topology rebuild after warm-up on static mesh.
- [ ] No repeated row/column uploads after warm-up.
- [ ] No UVM page faults in the AmgX bridge path.
- [ ] No per-solve allocation churn after warm-up.
- [ ] Setup and solve times are logged separately.

#### Integration

- [ ] `solver AmgX` works through runtime selection.
- [ ] `fallbackSolver` path works and does not recurse.
- [ ] `mode dDDI` is enforced.
- [ ] `diag_data` is always null.
- [ ] Cache invalidation on config change is implemented.

#### Documentation/artifacts

- [ ] Example `fvSolution` stanza exists.
- [ ] Snapshot replay utility exists.
- [ ] Profiling script exists.
- [ ] Benchmark report template exists.

### Future extensions deferred from this phase

1. **Further optimization of the `DeviceDirect` bridge after the required handoff exists**
   - fusing scatter/copy kernels, reducing device-to-device upload overhead inside the AmgX bridge, and tightening staging ownership beyond the baseline bridge. The `DeviceDirect` bridge itself is not deferred because later production no-host-transfer claims depend on it.

2. **Coupled interface support**
   - only after a clear plan exists for cyclic/AMI/processor coupling.

3. **Graph integration**
   - only after AmgX capture-safety is verified or pressure solve is isolated as a graph-external node.

4. **Mixed/single precision**
   - only after correctness and performance baselines are stable.

5. **Alternative matrix formats**
   - only if native/custom kernels, not AmgX, become the dominant direction.

6. **Multi-GPU**
   - requires `matrix_comm_from_maps*`, vector binds/partition handling, and pure device pointer discipline. [R14]

7. **Symmetry-specialized pressure configs**
   - only after a robust symmetry validator is in place and human sign-off is obtained.

### Implementation tasks for coding agent

1. Add standalone AmgX smoke executable and verify RTX 5080 toolchain.
2. Add `phase4DumpPressureSystem` utility.
3. Implement `LduCsrPatternBuilder` plus synthetic mapping validator.
4. Implement `CsrValuePacker` plus `A*x` equivalence validator.
5. Implement `PressureMatrixCache` with persistent pinned host staging.
6. Implement `AmgXContext` RAII wrapper.
7. Implement first-call `upload_all` and repeated `replace_coefficients + setup + solve`.
8. Implement runtime-selected `AmgXSolver`.
9. Implement native fallback policy and controls parsing.
10. Add NVTX3 ranges and telemetry counters.
11. Add `phase4ReplayPressureSystem` utility.
12. Benchmark snapshot replay vs native fallback.
13. Integrate into reduced live case.
14. Implement the `DeviceDirect` pressure bridge required for AmgX production handoff, while retaining `PinnedHost` as a correctness-only bring-up mode.

### Do not start until

- [ ] SPUMA/v2412 baseline is frozen.
- [ ] AMGX version is pinned.
- [ ] RTX 5080 build environment is validated.
- [ ] Pressure snapshot capture plan exists.
- [ ] Native fallback solver settings are pinned.
- [ ] Accepted reduced cases remain inside the centralized support-matrix envelope and therefore contain no coupled interfaces.

### Safe parallelization opportunities

1. `LduCsrPatternBuilder` and synthetic validators can be implemented independently of AmgX context work.
2. Snapshot dump utility can be built in parallel with RAII wrapper work.
3. NVTX/telemetry instrumentation can be added in parallel after the cache and solver class skeletons exist.
4. Replay utility can be developed in parallel with live solver integration once snapshot format is frozen.

### Governance guardrails

1. Reduced-case pressure/nozzle thresholds are imported from `acceptance_manifest.md`; this phase may not redefine them locally.
2. Coupled pressure interfaces remain out of milestone-1 scope unless `support_matrix.md` is revised centrally.
3. The exact AmgX / SPUMA / `foamExternalSolvers` revisions must be covered by `master_pin_manifest.md` plus `semantic_source_map.md`; if the implementation branch needs a different reviewed tuple, update those authority docs before coding.
4. A correctness-only Phase 4 checkpoint may stop at `PinnedHost`, but it must remain explicitly labeled replay/correctness-only until `DeviceDirect` is validated.
5. The initial AmgX configuration stays conservative (FGMRES-like / non-symmetry-assuming) until symmetry is demonstrated on real snapshots.

### Artifacts to produce

1. Source patches in `foamExternalSolvers` and minimal SPUMA integration changes.
2. `phase4DumpPressureSystem` utility.
3. `phase4ReplayPressureSystem` utility.
4. `phase4ValidateLduCsr` utility.
5. Example `fvSolution` snippet and AmgX config file.
6. Nsight Systems profile of replay utility.
7. Reduced-case benchmark log comparing AmgX vs native.
8. One-page acceptance report summarizing:
   - correctness,
   - fallback behavior,
   - transfer behavior,
   - timing decomposition,
   - `PinnedHost` vs `DeviceDirect` handoff status.
9. `DeviceDirect` bridge validation note (or explicit blocker note) showing whether AmgX has cleared the no-field-scale-host-transfer production gate for later phases.

---

# 6. Validation and benchmarking framework

This section is intentionally broader than the Phase 4 validation subsection because the coding agent needs a concrete harness, not just local tests.

## Benchmark assets to create

### B1 — Synthetic matrix suite

A tiny in-tree suite of manually checkable scalar LDU systems:
- identity
- 1D Poisson
- 2D/3D 7-point-style structured Laplacian projected into LDU form
- one deliberately asymmetric matrix

Purpose:
- validate mapping and basic AmgX solve flow.

### B2 — Pressure snapshot suite

At minimum three captured snapshots:
1. **startup** — early transient pressure system
2. **mid-run** — representative developed solve
3. **stress** — a timestep/corrector with the largest expected non-orthogonal or coefficient variation

Each snapshot must carry metadata:
- case name
- time
- timestep index
- corrector index
- pressure field name
- reference cell/value metadata if relevant
- topology hash

### B3 — Reduced live case

A reduced nozzle or pressure-only case that runs fast enough for iterative profiling and regression.

## Benchmark outputs required

For each backend (`native`, `AmgX`) and each snapshot/live run, log:

- solver name
- initial residual
- final residual
- iteration count
- setup time
- solve time
- total bridge time
- bytes HtoD/DtoH
- fallback reason (if any)
- peak GPU memory if measurable

## Benchmark decision rules

1. **Correctness gate comes first**
   - No performance comparison is meaningful until snapshot correctness passes.

2. **Topology reuse gate**
   - If repeated snapshot replay still rebuilds topology, stop and fix that before deeper profiling.

3. **Transfer gate**
   - If UVM appears, stop and fix allocation policy before deeper profiling.

4. **Backend comparison gate**
   - Compare native and AmgX on the same snapshot suite before live-case claims.

## Suggested benchmark sequence

1. `phase4ValidateLduCsr`
2. `phase4ReplayPressureSystem -solver native`
3. `phase4ReplayPressureSystem -solver AmgX`
4. same replay under `nsys`
5. reduced live case with native
6. reduced live case with AmgX
7. reduced live case with profiler

## Regression storage

Store, at minimum:
- golden snapshot outputs for synthetic tests
- last-known-good replay residual and iteration summaries
- reduced-case macro metrics

Do not require bitwise-equal floating-point fields across backends, but do require metric thresholds to hold.

---

# 7. Toolchain / environment specification

## Required baseline

1. **GPU**
   - NVIDIA GeForce RTX 5080 (compute capability 12.0). [R15]
2. **Driver**
   - Consume the driver floor from the master pin manifest.
3. **CUDA**
   - Consume the primary and experimental CUDA lanes from the master pin manifest. The CUDA 12.8.x references in this document remain minimum Blackwell-era compatibility context, not the project pin. [R17][R18]
4. **AMGX**
   - Use the exact pinned AmgX revision and config family from the master pin manifest and semantic source map. The v2.5.0 release line remains the documented Blackwell-era reference point for this phase. [R13]
5. **SPUMA**
   - Use the exact pinned SPUMA commit and OpenFOAM flavor/release mapping from the master pin manifest and semantic source map. [R1]
6. **foamExternalSolvers**
   - Use the exact pinned `foamExternalSolvers` revision paired with SPUMA and AmgX in the master pin manifest. [R4]

## AMGX build recommendations

- Use `sm_120` + PTX / `CMAKE_CUDA_ARCHITECTURES` exactly as specified by the master pin manifest and the pinned AmgX build. [R13][R15]
- Prefer a non-MPI build for Phase 4 if the site policy allows it, because this phase is single-GPU/single-rank. [R12]
- Use NVTX3-compatible headers/libraries. [R13][R17]

## OpenFOAM/SPUMA build recommendations

- Inherit the local SPUMA/v2412 compiler standard and ABI choices; do not force a different C++ standard in the AmgX bridge unless the local tree already does so.
- Link against the pinned AMGX build through the existing OpenFOAM/SPUMA external-solver linkage mechanism.
- Avoid optional dependencies not required for Phase 4.

## Environment variables / runtime flags to support

- `AMGX_CONFIG_FILE` (optional convenience override)
- `CUDA_VISIBLE_DEVICES` for deterministic single-GPU runs
- profiling wrappers via scripts rather than buried in code

## Profiling environment

- Nsight Systems installed and usable
- Compute Sanitizer usable
- Optional Nsight Compute for replay utility deep dives

## Explicit non-requirements in Phase 4

- MPI runtime for multi-GPU execution
- CUDA Graph capture support for AmgX
- NCCL/NVLink assumptions

---

# 8. Module / file / ownership map

This section assigns ownership by **responsibility domain**, not by person. The coding agent is expected to implement all domains unless the human reviewer assigns them separately.

| Responsibility domain | Primary module(s) | Primary owner | Review focus |
|---|---|---|---|
| LDU→CSR conversion correctness | `LduCsrPatternBuilder.*`, `phase4ValidateLduCsr` | Coding agent | senior reviewer checks mapping logic |
| Value/vector packing | `CsrValuePacker.*` | Coding agent | reviewer checks boundary-semantic correctness |
| Persistent cache + memory ownership | `PressureMatrixCache.*` | Coding agent | reviewer checks lifetime/invalidation rules |
| AmgX C API integration | `AmgXContext.*`, `AmgXError.H`, config parser | Coding agent | reviewer checks version/build assumptions |
| Runtime selection and fallback | `AmgXSolver.*`, `PressureFallback.*` | Coding agent | reviewer checks unsupported-case detection |
| Test and replay harness | `phase4DumpPressureSystem`, `phase4ReplayPressureSystem` | Coding agent | reviewer checks reproducibility |
| Profiling and telemetry | `PressureTelemetry.*`, scripts | Coding agent | reviewer checks completeness of timings |

## Ownership rules

1. Mapping logic must not be spread across multiple files.
2. Raw AMGX calls must be centralized in the context/wrapper layer.
3. Fallback policy must be centralized so all unsupported reasons are logged uniformly.
4. Test utilities must remain thin and reuse library code.
5. Profile scripts belong outside core solver code.

---

# 9. Coding-agent execution roadmap

This is the concrete build order for execution.

## Milestone M4.1 — Dependency freeze and AMgX smoke

**Depends on:** toolchain bring-up  
**Work:** pin SPUMA commit, `foamExternalSolvers` commit, AMGX version; build standalone AmgX smoke test.  
**Stop and benchmark before proceeding:** yes.  
**Success condition:** identity-system solve succeeds on RTX 5080.

## Milestone M4.2 — Pressure snapshot dump

**Depends on:** M4.1  
**Work:** add pressure snapshot utility/hook to CPU reference branch.  
**Parallelizable with:** M4.3.  
**Success condition:** representative pressure snapshots captured from reduced case.

## Milestone M4.3 — Prototype LDU→CSR conversion outside live solver

**Depends on:** M4.2 for realistic snapshots, but synthetic tests can start earlier.  
**Work:** implement topology builder + value packer + `A*x` validator.  
**Parallelizable with:** M4.4 partly.  
**Stop and benchmark before proceeding:** yes.  
**Success condition:** synthetic and snapshot `A*x` equivalence passes.

## Milestone M4.4 — AmgX context wrapper and replay utility

**Depends on:** M4.1 and M4.3  
**Work:** build replay utility that loads a snapshot and solves it with AmgX.  
**Why this should be prototyped before productization:** it isolates AMGX/OpenFOAM matrix bridging from full live solver orchestration.  
**Stop and benchmark before proceeding:** yes.  
**Success condition:** snapshot replay produces correct solutions and clean timing decomposition.

## Milestone M4.5 — Persistent cache object and runtime-selected solver

**Depends on:** M4.4  
**Work:** implement `PressureMatrixCache` and `AmgXSolver` runtime selection plus fallback.  
**Parallelizable with:** telemetry work.  
**Success condition:** `solver AmgX;` works in reduced live case or falls back cleanly.

## Milestone M4.6 — Telemetry, profiling, and reduced-case validation

**Depends on:** M4.5  
**Work:** add NVTX3 ranges, counters, benchmark scripts, reduced-case runs.  
**Stop and benchmark before proceeding:** yes.  
**Success condition:** no repeated topology upload; no UVM traffic; native comparison recorded.

## Milestone M4.7 — `DeviceDirect` pressure bridge

**Depends on:** M4.6  
**Work:** implement device-resident `csrValuesDev` / `rhsDev` / `xDev` staging plus device-pointer upload/replace paths and validate them on replay and reduced live case.  
**Why now:** this is the mandatory handoff bridge before any later phase can claim AmgX pressure correction without field-scale host transfer.  
**Success condition:** `DeviceDirect` runs correctly, `PinnedHost` remains available for bring-up only, and profiling shows no field-scale host pressure staging in production mode.

## Work that can be done in parallel

- Snapshot dump utility and synthetic validator.
- AmgX context wrapper and test harness scripts.
- NVTX/telemetry after module boundaries are frozen.
- Example `fvSolution` + config documentation once runtime-control parsing is frozen.

## Work that should remain experimental after Phase 4

- Graph capture around or through AmgX.
- mixed/single precision.
- coupled-interface support.
- multi-GPU.
- matrix-format alternatives.

## Where to stop and benchmark before proceeding to later phases

1. After synthetic `A*x` equivalence is correct.
2. After snapshot replay is correct.
3. After reduced-case live integration is correct.
4. Before any device-direct staging or graph work begins.

If any one of those stops fails, do not proceed to later phases.

---

# 10. Resolved Local Defaults and Residual Governance Notes

1. **Coupled-interface scope is fixed.**
   - The milestone-1 accepted case family imported from `support_matrix.md` excludes coupled pressure interfaces (`cyclic`, `AMI`, `processor`, and similar patch families).
2. **Imported package authorities are fixed.**
   - `master_pin_manifest.md` and `semantic_source_map.md` are the required Phase 4 authorities for the reviewed SPUMA/v2412 + `foamExternalSolvers` + AmgX lane.
3. **The correctness-only checkpoint policy is fixed.**
   - A Phase 4 checkpoint may stop at the `PinnedHost` bring-up path as a replay/correctness milestone, but it may not be labeled production-ready or satisfy any no-field-scale-host-transfer claim.
4. **Threshold ownership is fixed.**
   - Reduced-case hard/soft gates come from `acceptance_manifest.md`; this document's local ranges are engineering guidance only.
5. **Initial solver-config posture is fixed.**
   - Start with a conservative AmgX configuration (FGMRES+AMG-like / non-symmetry-assuming). Symmetry-specialized configs are follow-on optimization work only after validation.
6. **Precision posture is fixed.**
   - Phase 4 assumes double-precision `scalar`. If the local SPUMA build does not satisfy that assumption, do not enable the AmgX lane until explicit support is added.
7. **Replay-utility retention is fixed.**
   - Snapshot dump/replay utilities remain in-tree through at least Phase 5 because later phases depend on them for parity and bridge validation.

## Human review checklist

- [ ] Verify the implementation keeps coupled pressure interfaces out of the accepted Phase 4 case set.
- [ ] Verify the branch uses the dependency/build lane already frozen in `master_pin_manifest.md` and `semantic_source_map.md`.
- [ ] Verify reduced-case and snapshot selection match the central acceptance policy.
- [ ] Verify fallback settings and benchmark methodology preserve the correctness-only status of `PinnedHost`.
- [ ] Verify no hidden requirement for immediate multi-GPU or graph-captured AmgX support slipped into the scope.

## Coding agent kickoff checklist

- [ ] Pin SPUMA / `foamExternalSolvers` / AMGX versions.
- [ ] Build and run standalone AmgX smoke test on RTX 5080.
- [ ] Implement snapshot dump utility.
- [ ] Implement topology builder and `A*x` validator.
- [ ] Implement replay utility before live solver integration.
- [ ] Implement persistent cache and runtime-selected solver.
- [ ] Add native fallback and log unsupported reasons.
- [ ] Add NVTX3 ranges and telemetry counters.
- [ ] Profile replay utility for UVM faults and repeated uploads.
- [ ] Integrate into reduced case only after replay correctness passes.

## Highest risk implementation assumptions

1. The intended Phase 4 cases truly do **not** require coupled pressure interfaces.
2. The local SPUMA/v2412 API is close enough to the documented OpenFOAM solve flow that the described runtime-selection integration applies with minor signature adjustments only.
3. AMGX v2.5.0 (or the exact approved equivalent pinned by `master_pin_manifest.md`) builds cleanly and behaves correctly on the RTX 5080 on the frozen primary CUDA lane.
4. The `PinnedHost` bring-up path is acceptable only as the first replay/correctness milestone; production device-resident pressure architecture still requires `DeviceDirect`.
5. Pressure snapshot capture can be inserted at the exact pre-solve point so replay fidelity is trustworthy.

## References

- **[R1]** SPUMA project root (v0.1-v2412 / OpenFOAM 2412 base):  
  https://gitlab-hpc.cineca.it/exafoam/spuma

- **[R2]** SPUMA GPU-support wiki (supported solvers, profiling guidance, unsupported-feature caveats):  
  https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554

- **[R3]** SPUMA paper / preprint (full-port rationale, profiling, AmgX vs GAMG observations, current multiphase status):  
  https://arxiv.org/html/2512.22215v1

- **[R4]** foamExternalSolvers project (AmgX interface for SPUMA/OpenFOAM):  
  https://gitlab-hpc.cineca.it/exafoam/foamExternalSolvers

- **[R5]** OGL/Ginkgo plugin paper (persistent structure reuse, object-registry persistence, Amdahl limitation of solver-only offload):  
  https://link.springer.com/article/10.1007/s11012-024-01806-1

- **[R6]** exaFOAM / zeptoFoam GPU-porting workshop slides (explicit data management, unified-memory pitfalls, plugin speedup ceiling):  
  https://exafoam.eu/wp-content/uploads/2ndWorkshop/exaFOAM_2ndPublicWorkshop_Porting_to_GPUs.pdf

- **[R7]** OpenFOAM latest API `fvMatrixSolve.C` (segregated solve flow, boundary/source handling, runtime-selected solver invocation):  
  https://www.openfoam.com/documentation/guides/latest/api/fvMatrixSolve_8C_source.html

- **[R8]** OpenFOAM latest API `lduMatrix.H` / solver interface reference:  
  https://www.openfoam.com/documentation/guides/latest/api/lduMatrix_8H_source.html

- **[R9]** OpenFOAM Foundation `LduMatrixATmul.C` (owner/neighbour, upper/lower application in `Amul`):  
  https://cpp.openfoam.org/v12/LduMatrixATmul_8C_source.html

- **[R10]** OpenFOAM `lduAddressing` reference (addressing semantics and links to owner/neighbour use):  
  https://cpp.openfoam.org/v13/classFoam_1_1lduAddressing.html

- **[R11]** OpenFOAM `incompressibleVoF` / `twoPhaseSolver` pressure-corrector references:  
  https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html  
  https://cpp.openfoam.org/v13/twoPhaseSolver_2pressureCorrector_8C_source.html

- **[R12]** NVIDIA AMGX README (features, build overview, example config usage):  
  https://github.com/NVIDIA/AMGX/blob/main/README.md

- **[R13]** NVIDIA AMGX releases (v2.5.0 Blackwell/CUDA 13 support, NVTX3, `CMAKE_CUDA_ARCHITECTURES`, build changes):  
  https://github.com/NVIDIA/AMGX/releases

- **[R14]** NVIDIA AMGX Reference Manual PDF (mode semantics, upload/replace semantics, pinned-host recommendation, repeated `solver_setup` requirement):  
  https://raw.githubusercontent.com/NVIDIA/AMGX/main/doc/AMGX_Reference.pdf

- **[R15]** NVIDIA CUDA GPU compute capability table (RTX 5080 = compute capability 12.0):  
  https://developer.nvidia.com/cuda/gpus

- **[R16]** NVIDIA RTX 5080 product / Blackwell public specs (architecture, memory, bandwidth, no NVLink on GeForce 5080) and Blackwell whitepaper:  
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/  
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

- **[R17]** CUDA 12.8 release notes (Blackwell support, driver version guidance, toolkit context):  
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/

- **[R18]** CUDA Blackwell compatibility guide:  
  https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html
