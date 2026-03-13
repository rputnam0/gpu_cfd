# Phase 6 implementation specification — nozzle-specific kernels and boundary conditions

## 1. Executive overview

This document expands **only Phase 6 — nozzle-specific kernels and boundary conditions** into an implementation-ready engineering specification for a coding agent, while preserving enough context from the surrounding program to keep the design coherent.

The target is a **single-GPU, RTX 5080, SPUMA-based OpenFOAM-v2412-class transient pressure-swirl nozzle workflow** using a **VOF / MULES / PIMPLE** formulation with **persistent device residency** and **minimal host/device transfer**. The output of this phase is not “GPU support in general.” It is a concrete subsystem that makes nozzle cases actually runnable without boundary-condition and startup logic dragging execution back to the CPU hot path. [R1][R2][R3][R4][R5][R15][R16][R17][R18]

This phase exists because the generic GPU port of sparse algebra and even the generic VOF port are insufficient for the target application. The nozzle workflow requires:

1. **Pressure-driven swirl inlet patch groups** with liquid injection and tangential/radial swirl superposition.
2. **Wall and open-boundary handling** that remains device-resident during alpha, momentum, and pressure stages.
3. **Startup seeding / air-core conditioning** that can be used repeatedly in DOE-style runs without host-side tooling becoming a blocking dependency.

### Scope boundary for this document

This document **does not** fully expand Phases 0–5 or 7–8. It assumes they exist at the level described in the baseline plan and only supplies the minimum contextual constraints needed so that Phase 6 can be implemented coherently.

### Bottom-line implementation stance

- **Must do now for correctness**
  - Build a **device-resident patch-compacted boundary subsystem**.
  - Implement a **custom pressure-driven swirl inlet velocity boundary condition** for nozzle inlet patches.
  - Implement device-side support for the pressure and alpha boundary types needed by the reference nozzle cases.
  - Implement a **device-resident startup seeding path**.
  - Enforce **fail-fast validation** for unsupported patch/function configurations.

- **Must do now for performance**
  - Avoid host-side `fvPatchField::updateCoeffs()` in the hot path for supported patches.
  - Avoid per-patch and per-face micro-kernel launch patterns.
  - Allocate all persistent boundary metadata and scratch in **device memory**, not managed memory, for production.
  - Ensure boundary stages are **graph-capture-safe** even if full CUDA Graph integration is completed later. [R1][R2][R15][R17]

- **Can defer**
  - Generalized support for arbitrary OpenFOAM patch-field types.
  - Full `setFieldsDict` compatibility.
  - Contact-angle wall models if not already implemented by the Phase 5 surface-tension path.
  - Multi-GPU boundary exchange and processor patches.
  - Dynamic mesh / AMI / cyclic support.

## 2. Global architecture decisions

The table below distinguishes **sourced fact**, **engineering inference**, and **recommendation** for the major decisions that shape Phase 6.

| Topic | Sourced fact | Engineering inference | Recommendation |
|---|---|---|---|
| Runtime base | SPUMA is the active GPU-porting fork in this program context, its published comparisons are against OpenFOAM-v2412, and its documented supported GPU solver list excludes the incompressible VOF path required here. [R1][R2] | The nozzle workflow is real new solver work, not a configuration-only exercise. | Implement Phase 6 against the SPUMA runtime base and treat multiphase/nozzle BC support as new code owned by this project. |
| Boundary execution model | SPUMA’s maintainers warn that unsupported features often do not hard-fail; they can instead degrade into unwanted host/device copies. SPUMA profiling also shows a very large kernel count and synchronization after each kernel in the profiled path. [R1][R2] | Boundary-condition logic that remains polymorphic and host-driven is highly likely to reintroduce migration and launch-overhead pathologies. | Replace hot-path BC execution for supported nozzle patches with an explicit device executor. Do not rely on host patch-object evaluation inside the timestep loop for supported patches. |
| Need for a custom nozzle inlet BC | Standard `swirlFlowRateInletVelocity` is flow-rate-driven; standard `swirlInletVelocity` prescribes velocity directly; standard `pressureDirectedInletOutletVelocity` derives inflow from flux and an inlet direction but does not provide swirl-profile superposition as required; `pressureInletOutletVelocity` can add tangential velocity but derives inflow from the internal-cell normal component, not the flux-driven normal component required here. [R7][R8][R9][R10] | No stock boundary condition cleanly provides “pressure-driven normal component + prescribed tangential/radial swirl profile” in one implementation with the semantics required by the nozzle case. | Introduce one custom U-boundary condition for nozzle inlets: `gpuPressureSwirlInletVelocity`. Keep pressure on those patches in standard `p_rgh` boundary types where possible. |
| Patch metadata layout | OpenFOAM patch fields are dictionary-selected runtime objects with per-patch update/evaluate lifecycles. [R6] | A per-patch object-dispatch model is the wrong execution substrate for GPU boundary stages. The natural GPU execution model is a compact face manifest grouped by semantic role. | Build a **patch-compacted device manifest** keyed by semantic role and BC kind, not by individual patch-object dispatch. |
| Pressure BC handling | `fixedFluxPressure` requires `updateCoeffs(const scalarField& snGradp)` to be called before `updateCoeffs()` / `evaluate()`. `constrainPressure` computes `snGradp` from `phiHbyA`, `rho`, `U`, `Sf`, `magSf`, and `rhorAU`. [R13][R14] | Pressure-boundary handling cannot be an afterthought; it must be explicitly sequenced inside pressure-correction control flow. | Give the device boundary executor an explicit **pressure-boundary stage** that computes and stores `snGradp` before pressure assembly/solve. |
| Startup seeding | Standard OpenFOAM initialization utilities are dictionary-driven and host-oriented. The baseline plan makes startup conditioning a first-class variable for DOE. | Requiring external host tooling for each startup variant will become an operational bottleneck and a graph-capture obstacle. | Implement a dedicated device-side seeding subsystem with a deliberately constrained dictionary grammar. Keep host `setFields` as fallback only. |
| Memory policy | SPUMA uses unified memory plus pooling as an incremental porting mechanism, but its paper and wiki emphasize the need to profile for unwanted CPU-GPU faults and page migration. [R1][R2] | Managed memory is too risky for production boundary state because unsupported host touches can silently fault pages. | Allocate persistent Phase 6 manifests, profile tables, and scratch in **device memory only**. Use pinned host memory only for tiny runtime parameters and diagnostics staging. |
| Blackwell/CUDA target | NVIDIA lists GeForce RTX 5080 as Blackwell, 16 GB GDDR7, CUDA capability 12.0, and CUDA 12.8 adds Blackwell compiler support, conditional-graph improvements, and recommends NVTX v3. [R15][R16][R17][R18] | The project should not hide behind generic portability where a simpler CUDA implementation is safer and faster to land. | Use direct CUDA kernels behind a stable host-side interface. Preserve SPUMA-style abstraction at the module boundary, not inside every Phase 6 kernel. |

## 3. Global assumptions and constraints

These assumptions are binding for Phase 6 unless a human reviewer changes them.

### 3.1 Assumptions that are treated as fixed in this phase

1. **Single GPU only**. No domain decomposition, no processor patches, no multi-rank orchestration in Phase 6.
2. **Static mesh topology**. No dynamic mesh, no AMI, no overset, no topological changes.
3. **Single-region solver path**. The nozzle use case is treated as a single-region incompressible VoF run with an optional short external near-field.
4. **Device-resident field core already exists** from earlier phases for:
   - `U`
   - `p_rgh`
   - `phi`
   - `alpha` / phase fraction(s)
   - `rho`
   - `rhoPhi`
   - `rAU` / `rhorAU`
5. **Gravity is constant** over the run.
6. **MRF / rotating-frame support is not required** for the reference nozzle path in this phase.
7. **Boundary patch names are stable** across the reference cases:
   - `swirlInletA`
   - `swirlInletB`
   - one or more wall patches
   - one or more ambient/open patches
   - optional symmetry patches
8. **Graph capture is a target constraint** even if the final graph integration lands in a later phase.

### 3.2 Constraints inherited from upstream/OpenFOAM/SPUMA behavior

1. OpenFOAM patch fields are runtime-selected and rely on `updateCoeffs()` / `evaluate()` lifecycle semantics, with `updated()` guarding repeated evaluation. [R6]
2. Pressure-driven BC semantics in OpenFOAM depend on flux sign conventions:
   - positive flux = outflow
   - negative flux = inflow  
   for the standard inlet/outlet pressure-coupled BC families cited here. [R7][R8]
3. `fixedFluxPressure` has a hard ordering requirement: boundary gradient must be supplied before normal `updateCoeffs()` / evaluation. [R13]
4. `prghPressure` and `prghTotalHydrostaticPressure` have explicit hydrostatic formulations that depend on `rho`, `g`, `hRef`, and, for total hydrostatic pressure, `U`. [R11][R12]
5. SPUMA’s current documented GPU support does not include the nozzle multiphase path. Unsupported features may degrade into copies rather than producing a clean fatal error. [R1][R2]

### 3.3 Known version drift / source-quality caveats

The technical semantics cited here come from a mix of:

- SPUMA paper + SPUMA wiki,
- OpenFOAM Foundation source/API pages (v11–v13),
- OpenCFD/OpenFOAM-v2412 API search snippets,
- NVIDIA official CUDA and GPU documentation.

This is acceptable for design, but there is a real risk of **runtime type-name drift**, especially for some `p_rgh` boundary conditions between Foundation and OpenCFD trees. The coding agent **must** verify exact type names and constructor signatures against the **actual SPUMA target tree and commit** before wiring runtime selection or dictionary validation. [R2][R11][R12]

## 4. Cross-cutting risks and mitigation

| Risk | Why it matters in Phase 6 | Mitigation required now | Rollback / fallback |
|---|---|---|---|
| Silent CPU fallback | SPUMA wiki explicitly warns that unsupported features can manifest as slow host/device traffic instead of clean failure. [R2] | Default policy is `unsupportedPatchAction fail`. Emit a startup support report and abort unless fallback is explicitly enabled. | Per-patch CPU fallback may be enabled only with `allowCpuBoundaryFallback yes` for bring-up. It is not acceptable for production acceptance. |
| Boundary launch explosion | SPUMA profiling shows extremely high kernel counts and synchronization overhead in the profiled path. [R1] | Launch count must scale with **BC family** and **solver stage**, not with patch count. Batch faces by role/kind. | If fusion becomes numerically risky, keep at most one kernel per field-family/stage, not per patch. |
| Wrong flux sign handling | All pressure-coupled inlet/outlet semantics depend on correct sign convention. [R7][R8] | Codify sign conventions in one shared utility. Unit-test them independently. | None. A wrong sign convention is a correctness bug, not a performance issue. |
| `fixedFluxPressure` ordering bug | Pressure correction will diverge or silently misbehave if `snGradp` is stale or absent. [R13][R14] | Create an explicit pressure-boundary stage and validate that it runs before pressure assembly every corrector iteration. | CPU reference path for one iteration can be used to compare `snGradp`; do not ship without direct regression. |
| Tangential swirl corrupts mass flux | If the tangential/radial velocity addition is not orthogonalized relative to the patch normal, it changes `phi`. | Orthogonalize swirl additions against `nHat` before composing `U_b`. Unit-test with non-axis-aligned patch normals. | None. This is a must-fix correctness defect. |
| Axis singularity | At or near the axis, radial direction is ill-conditioned. | Handle `r < epsilon` explicitly and force radial/tangential contribution to zero unless a specific regularization is supplied. | Use deterministic fallback basis + zero swirl at the axis. |
| Version/name drift in patch types | `prgh*` naming differs across docs and distributions. | Build a compile-time or startup validator against the actual linked runtime types. | Keep a small compatibility map in the validator only. Do not scatter string checks across kernels. |
| Boundary-state duplication | Maintaining a shadow boundary store separate from the actual field boundary storage invites drift and stale data. | Extend device field abstractions to expose flat boundary spans. Boundary kernels must write into the real device field storage where possible. | Only if impossible in the first iteration, use a temporary mirror plus explicit scatter, then remove it before acceptance. |
| Startup seeding mismatch | If device seeding differs from the CPU reference initialization logic, end-to-end comparisons become ambiguous. | Keep a deterministic, constrained grammar and regression-test against a CPU reference implementation of the same grammar. | Use host `setFields` only as a temporary fallback, not as the permanent primary path. |
| Graph capture breakage | Host dictionary lookups or allocations inside the timestep path will make later graph integration expensive or impossible. | No allocation, no runtime selection, no host introspection inside boundary stage methods. | Boundary path may run uncaptured during bring-up, but the code must already be capture-safe by construction. |

## 5. Phase-by-phase implementation specification

Only Phase 6 is expanded in this run.

## Phase 6 — Nozzle-specific kernels and boundary conditions

### Purpose

Implement a **device-resident nozzle boundary and startup subsystem** that supports the reference pressure-swirl nozzle cases without re-entering host-driven OpenFOAM patch evaluation inside the hot path.

Phase 6 implements only the patch kinds frozen in the centralized support matrix; it does not discover support during implementation. Any CPU fallback is forbidden in production acceptance and exists only as an explicit bring-up mode under the global GPU operational contract.

This phase must deliver:

1. A patch-compacted device boundary manifest.
2. A custom GPU-capable **pressure-driven swirl inlet velocity** boundary condition.
3. Device-side support for the pressure and alpha boundary semantics required by the nozzle reference cases.
4. A device-resident **startup seeding / air-core initialization** path.
5. Strict validation, telemetry, and fallback controls that prevent silent performance failure.

### Why this phase exists

The nozzle workload is not a generic VOF benchmark. It adds three case-specific requirements that standard “sparse linear algebra on GPU” work does not solve:

1. **Swirl inlet patch groups** whose physics are pressure-driven in the normal direction but profile-driven in tangential/radial directions.
2. **Open boundary handling** for a short external plume region, where sign-dependent inflow/outflow semantics matter.
3. **Startup conditioning** that affects transient behavior and therefore must be treated as part of the solver, not as external pre-processing.

Without this phase, the program will likely end up in one of two failure modes:

- numerically correct but operationally unusable because the hot loop is still invoking host patch logic, or
- fast-looking but numerically wrong because boundary semantics were “simplified” too aggressively.

### Entry criteria

Do not start Phase 6 implementation until **all** of the following are true:

1. The CPU reference nozzle case on the SPUMA/v2412 baseline is frozen, and the Phase 5 generic-core `R1-core` case has already passed its generic VOF gate.
2. The mesh, patch naming conventions, and patch-role map for the acceptance nozzle cases are frozen.
3. Device-resident field storage exists and is stable for `U`, `p_rgh`, `phi`, `alpha`, `rho`, `rhoPhi`, and `rAU`/`rhorAU`, including the Phase 5 old-time / prev-iteration state that later alpha and pressure stages assume exists.
4. The Phase 5 VOF path can advance at least one timestep on device with temporary/simple host-side BC handling for the `R1-core` reduced verification case.
5. The named Phase 5 pressure-boundary-state contract is available: pressure assembly and solve can consume device-side `snGradp` / boundary-gradient state without forcing a host copy. If AmgX is to be included in any production claim of “no field-scale host transfer,” the Phase 4 `DeviceDirect` bridge is already present.
6. If the frozen support matrix marks wall wetting/contact-angle as required, Phase 5 interface semantics are already ported and exported; otherwise the milestone-1 default remains zero-gradient wall alpha.
7. Toolchain and profiling support are working under the master pin manifest:
   - current primary lane = CUDA 12.9.1,
   - current experimental lane = CUDA 13.2,
   - driver floor `>=595.45.04`,
   - Nsight Systems,
   - Compute Sanitizer,
   - NVTX v3. [R15][R16]
8. The centralized support matrix has already frozen the authoritative inlet `p_rgh` BC kind, the ambient/open velocity-BC subset, the wall alpha/contact-angle requirement, and the allowed startup-seeding grammar for the accepted case family.

### Exit criteria

Phase 6 exits only when all of the following are satisfied:

1. Supported nozzle patches execute through the device boundary path with **zero required host patch evaluation** inside the timestep loop.
2. The following patch roles are supported and validated in the reference nozzle cases:
   - swirl inlet patches (`swirlInletA`, `swirlInletB`)
   - wall patches
   - ambient/open patches
   - alpha inlet/open/wall semantics required by those patches
3. Device startup seeding can initialize the reference case without host `setFields`.
4. Steady-state Nsight traces show **no recurring UVM HtoD/DtoH migrations attributable to boundary execution**. [R2]
5. Boundary-kernel launch count scales with stage family, not patch count.
6. CPU/GPU boundary outputs match reference tolerances on component tests.
7. End-to-end nozzle metrics are within agreed thresholds.
8. The subsystem is graph-capture-safe by construction:
   - no hot-path allocations,
   - no dictionary parsing in stage execution,
   - no host object traversal during stage execution.

### Goals

#### Required for correctness

1. Preserve OpenFOAM-consistent sign conventions and patch semantics for the supported BCs.
2. Maintain pressure/velocity consistency at pressure-driven nozzle inlets.
3. Maintain alpha boundedness and correct patch values during MULES subcycling.
4. Ensure `fixedFluxPressure` gradients are recomputed in the correct place in every pressure-correction loop.
5. Ensure startup seeding is deterministic and reproducible.

#### Required for performance

6. Keep boundary metadata and scratch arrays device-resident across timesteps.
7. Remove per-patch host dispatch from the supported nozzle path.
8. Use patch-compacted kernels with stable launch counts.
9. Make all phase entry points non-allocating and graph-capture-safe.

#### Required for operability

10. Emit a clear support report at startup.
11. Fail fast on unsupported configurations unless fallback is explicitly requested.
12. Provide CPU fallback only as an opt-in bring-up mode with loud telemetry.

### Non-goals

Phase 6 will **not** attempt to solve the following:

1. Full generic support for arbitrary OpenFOAM `fvPatchField` types.
2. Full generic support for all `Function1` / `Function2` subclasses.
3. Dynamic mesh, AMI, cyclic, overset, or processor patches.
4. Multi-GPU boundary exchange.
5. General STL/triSurface-driven startup seeding.
6. Contact-angle wall models unless they are already handled by the Phase 5 surface-tension path.
7. MRF-aware pressure boundary correction.
8. Direct writer/output redesign.

### Technical background

#### 1. Where boundary stages live in the solver control flow

The flexible PIMPLE solver loop calls, in order, `preSolve`, `prePredictor`, `momentumPredictor`, `thermophysicalPredictor`, `pressureCorrector`, and `postCorrector` inside the main PIMPLE loop. [R3]

For the incompressible VOF path, `alphaPredictor()` performs explicit MULES-style work with optional alpha subcycling; `pressureCorrector()` and the `p_rgh` path depend on pressure boundary handling; mixture properties are updated after alpha work. [R4][R5]

**Implication:** boundary updates are not one monolithic step. Phase 6 must place boundary work at solver-stage boundaries that are numerically consistent with those algorithms.

#### 2. Why alpha BCs matter here

The VOF alpha path constructs `alphaPhi`, applies `MULES::limit`, applies `MULES::explicitSolve`, optionally subcycles, and then updates `mixture.correct()`. `alphaPredictor()` determines the number of explicit subcycles using `nAlphaSubCyclesPtr->value(alphaCoNum)`. [R5]

**Implication:** alpha boundary values for inlet/open/wall patches must be valid **before each alpha subcycle**, not just once per timestep.

#### 3. Why pressure BCs need their own stage

`fixedFluxPressure` is not a simple fixed value. It stores a patch-normal gradient and its implementation explicitly requires `updateCoeffs(const scalarField& snGradp)` to be called before normal update/evaluate; otherwise it throws a fatal error. `constrainPressure` shows the formula used to compute the boundary gradient from `phiHbyA`, `rho`, `U`, `Sf`, `magSf`, and `rhorAU`. [R13][R14]

**Implication:** pressure BC evaluation has a precise place in the pressure-correction algorithm and cannot be folded into an unrelated generic “boundary refresh” call.

#### 4. Why standard swirl/open BCs are insufficient as-is

- `swirlFlowRateInletVelocity` prescribes the normal component from mass/volumetric flow rate and adds radial/tangential profiles. It is **flow-rate-driven**, not pressure-driven. [R9]
- `swirlInletVelocity` prescribes axial/radial/tangential velocities directly. [R10]
- `pressureDirectedInletOutletVelocity` derives inflow from flux and a specified inlet direction, but does not natively superimpose swirl profiles. [R8]
- `pressureInletOutletVelocity` can optionally specify tangential velocity, but its inflow normal component comes from the internal-cell normal component, not directly from the flux. [R7]

**Implication:** the nozzle inlet requires a dedicated hybrid BC, not an awkward composition of stock types.

#### 5. Why host patch polymorphism is a trap in this project

`fvPatchField` is a fat-interface base class, selected at runtime, with virtual `updateCoeffs()`, `evaluate()`, `snGrad()`, and related semantics. [R6] SPUMA’s maintainers warn that unsupported features may degrade into slow copy behavior instead of clean failure, and SPUMA profiling shows heavy synchronization and launch overhead in the profiled path. [R1][R2]

**Implication:** for supported nozzle patches, runtime selection should stop at startup. The hot path must operate on prevalidated, flattened device manifests.

### Research findings relevant to this phase

#### Sourced facts

1. **SPUMA does not currently provide the needed multiphase nozzle path out of the box.** Its documented supported GPU solver list excludes the incompressible VoF family needed here. [R1][R2]
2. **SPUMA warns that unsupported features can silently slow down runs via unwanted transfers.** [R2]
3. **SPUMA profiling shows a launch/synchronization-heavy execution pattern.** This makes per-patch micro-kernel strategies especially dangerous. [R1]
4. **Standard OpenFOAM BCs cover only parts of the nozzle inlet requirement**:
   - flow-rate-driven swirl,
   - directly prescribed swirl,
   - pressure-coupled inflow/outflow with direction,
   - pressure-coupled inflow/outflow with optional tangential velocity. [R7][R8][R9][R10]
5. **`prghPressure` and `prghTotalHydrostaticPressure` have explicit hydrostatic formulas** that must be preserved if those patch types are supported. [R11][R12]
6. **`fixedFluxPressure` has strict evaluation ordering requirements** and `constrainPressure` shows exactly how `snGradp` is computed. [R13][R14]
7. **RTX 5080 is an NVIDIA Blackwell consumer GPU with 16 GB GDDR7 and compute capability 12.0**, and CUDA 12.8 adds Blackwell compiler support and newer CUDA Graph control-flow features. [R15][R16][R17][R18]

#### Engineering inferences

1. **Boundary handling is not likely to be DRAM-bandwidth-dominated.** For the nozzle boundary path, branch divergence, launch count, and hidden host activity are higher risks than raw memory bandwidth.
2. **Generic device interpretation of arbitrary OpenFOAM dictionary function objects is not a good first landing path.** The implementation risk is too high relative to the real case requirements.
3. **The highest-value optimization is not “faster arithmetic” but “never leaving the device.”**
4. **A small number of semantically rich kernels is preferable to maximum fusion.** Full fusion would increase branch divergence and implementation risk; per-patch dispatch would increase launch overhead and host coupling.

#### Recommendations

1. Build **one custom nozzle inlet U BC**, not a full custom BC ecosystem.
2. Reuse **standard pressure BC semantics** where they already match the needed physics.
3. Support only a **constrained set of device-evaluable profile forms** in Phase 6.
4. Make startup seeding a **small, explicit DSL**, not a full `setFields` clone.
5. Treat CPU fallback as a **bring-up valve**, not a design target.

### Design decisions

The following decisions are binding unless a human reviewer overrides them.

| ID | Decision | Classification | Basis |
|---|---|---|---|
| D1 | Introduce `gpuPressureSwirlInletVelocity` as a new patch type for `U` on nozzle inlet patches. | Recommendation | Existing stock BCs do not cleanly provide pressure-driven normal flow plus swirl-profile superposition. [R7][R8][R9][R10] |
| D2 | Freeze the accepted nozzle-case pressure tuple to standard semantics: swirl inlet `p_rgh = prghTotalHydrostaticPressure`, wall `p_rgh = fixedFluxPressure`, ambient/open `p_rgh = prghPressure`, and ambient/open `U = pressureInletOutletVelocity`. | Recommendation | This closes the final case-family ambiguity while still relying on explicit upstream semantics rather than inventing a second custom pressure/open BC family. [R11][R12][R13][R14] |
| D3 | Do **not** execute supported patch updates via host `fvPatchField::updateCoeffs()` inside the timestep loop. | Recommendation | SPUMA warns about hidden copies and already shows launch/sync overhead sensitivity. [R1][R2] |
| D4 | Flatten supported boundary faces into a **device-resident manifest** grouped by semantic role and BC kind. | Recommendation | GPU execution wants contiguous, role-based traversal rather than per-object polymorphism. |
| D5 | Extend the device field abstraction to expose **flat contiguous boundary spans** for real field storage. | Recommendation | A second shadow boundary store is a stale-data trap. |
| D6 | Compute the **normal component** of inlet velocity from the current boundary flux and add only the **orthogonalized tangential/radial swirl component**. | Recommendation | This preserves flux consistency while allowing swirl superposition. |
| D7 | Support only a constrained device profile grammar in Phase 6: `constant`, `radialTable`, `radialPolynomial`, and separable `timeScale × radialShape + offset`. | Recommendation | This is sufficient for practical nozzle work while avoiding a generic function interpreter. |
| D8 | Device startup seeding will use a dedicated dictionary grammar that mirrors only the high-value subset of `setFields`: `defaultFieldValues` and ordered `regions`. | Recommendation | This preserves usability while keeping parsing and device execution tractable. |
| D9 | Runtime policy defaults to `unsupportedPatchAction fail` and `allowCpuBoundaryFallback no`. | Recommendation | Silent fallback is too dangerous in SPUMA. [R2] |
| D10 | All Phase 6 boundary kernels remain in solver precision (`Foam::scalar`, typically `double`). | Recommendation | Boundary metadata is small relative to field data; mixed precision here adds complexity for little gain. |
| D11 | All persistent Phase 6 state is allocated from the **device memory pool**; tiny runtime parameter blocks use pinned host memory and async copies. | Recommendation | This is the safest production memory posture for the nozzle hot path. |
| D12 | Phase 6 code must be **graph-capture-safe from the first implementation**, even if uncaptured execution is used during initial debugging. | Recommendation | Retrofitting capture safety later is more expensive than doing it from day one. |

#### Supported boundary-condition matrix

This matrix is subordinate to the centralized support matrix. It records the local Phase 6 implementation envelope, but the production-required subset is the already-frozen case family. Phase 6 implements only the patch kinds frozen in that support matrix; it does not discover support during implementation.

| Patch role | Field | Supported kind(s) in Phase 6 | Status |
|---|---|---|---|
| Swirl inlet (`swirlInletA/B`) | `U` | `gpuPressureSwirlInletVelocity` | Required |
| Swirl inlet (`swirlInletA/B`) | `p_rgh` | `prghTotalHydrostaticPressure` | Required (frozen `R1` / `R0` production kind) |
| Swirl inlet (`swirlInletA/B`) | `alpha.water` | `fixedValue` | Required |
| Wall | `U` | `noSlip` as the frozen support-matrix token; zero-vector `fixedValue` accepted only as a compatibility alias if normalized to `noSlip` before support checking | Required |
| Wall | `p_rgh` | `fixedFluxPressure` | Required |
| Wall | `alpha.water` | `zeroGradient` for the milestone-1 default; contact-angle BC only if the support matrix explicitly requires it and Phase 5 already exports the semantics | Required / conditional |
| Ambient/open | `U` | `pressureInletOutletVelocity` | Required (frozen `R1` / `R0` production kind) |
| Ambient/open | `p_rgh` | `prghPressure` | Required (frozen `R1` / `R0` production kind) |
| Ambient/open | `alpha.water` | `inletOutlet` | Required |
| Symmetry / empty | all relevant fields | pass-through / no-op semantics | Optional but low-cost |
| Cyclic / AMI / processor | all | unsupported in Phase 6 | Fail |

#### Explicit default choices for ambiguous areas

1. **Default inlet alpha value** on liquid swirl inlets: `1.0` for the liquid phase fraction.
2. **Default ambient/open inlet alpha value** during backflow: `0.0` for the liquid phase fraction.
3. **Default backflow handling for `gpuPressureSwirlInletVelocity`**: treat backflow as zero-gradient and zero swirl.
4. **Default profile regularization near the axis**: if `r < 1e-12 * Lref` for the patch, set radial and tangential contributions to zero.
5. **Default seed-region precedence**: later regions override earlier regions (`lastWins`).
6. **Default seeding application**: apply once before the first timestep; do not reapply on restart unless `forceReseed yes`.

### Alternatives considered

#### Alternative A — use stock OpenFOAM patch classes directly in the hot path

**Rejected.**

Why:
- host-side polymorphism is the wrong execution model for the GPU hot path,
- SPUMA warns about silent copy slowdowns,
- it does not solve the missing nozzle-inlet BC semantics. [R1][R2][R6][R7][R8][R9][R10]

#### Alternative B — build a generic device interpreter for arbitrary `fvPatchField` types

**Rejected for Phase 6.**

Why:
- too broad,
- too fragile across OpenFOAM variants,
- unnecessary for the reference nozzle cases.

This may be a longer-term platform goal, but it is not the safe first implementation.

#### Alternative C — use `swirlFlowRateInletVelocity` for nozzle inlets

**Rejected.**

Why:
- it is flow-rate-driven, not pressure-driven. [R9]

#### Alternative D — use `pressureDirectedInletOutletVelocity` with a separate swirl correction field

**Rejected.**

Why:
- the composition is ambiguous,
- it complicates flux consistency,
- it splits one physical BC across two implementation sites.

#### Alternative E — use `pressureInletOutletVelocity` with `tangentialVelocity` at the nozzle inlet

**Rejected for nozzle inlet; retained for ambient/open boundaries where appropriate.**

Why:
- it can superimpose tangential velocity, but its inflow normal component comes from the internal-cell normal component, not the flux-driven normal component required for the nozzle inlet. [R7]

#### Alternative F — keep host `setFields` as the permanent startup path

**Rejected as production architecture; accepted as a bring-up fallback.**

Why:
- it keeps initialization outside the solver and outside the device execution model,
- it is awkward for DOE repetition,
- it does not align with graph-friendly execution.

#### Alternative G — flatten boundary values into a separate mirror array and scatter/gather each stage

**Rejected as the target architecture; allowed only as a temporary bring-up hack if field abstractions are insufficient.**

Why:
- creates two sources of truth,
- invites stale data,
- adds avoidable data motion.

### Interfaces and dependencies

#### 1. Module boundaries

Phase 6 introduces four primary modules:

1. **Boundary configuration/validation module**
   - host-side only
   - parses dictionaries
   - classifies patches
   - emits support report
   - compiles device profile tables

2. **Boundary manifest module**
   - owns immutable flattened patch metadata
   - lives on host + device
   - exposes device views

3. **Boundary executor module**
   - owns mutable boundary scratch and tiny runtime parameter buffers
   - launches device kernels for alpha/U/p_rgh boundary stages
   - binds to and fills the named Phase 5 `PressureBoundaryState` contract, including device-side `snGradp`, for consumption by pressure assembly

4. **Startup seeding module**
   - parses seeding grammar
   - owns seed-region manifests
   - applies field initialization on device before the time loop

#### 2. Concrete interface definitions

The coding agent should implement the following host-side classes and data types.

```cpp
namespace Foam::gpuNozzle
{

enum class PatchRole : uint8_t
{
    SwirlInletA,
    SwirlInletB,
    Wall,
    AmbientOpen,
    Symmetry,
    Empty,
    Unsupported
};

enum class VelocityBcKind : uint8_t
{
    GpuPressureSwirlInlet,
    PressureInletOutlet,
    PressureDirectedInletOutlet,
    FixedZero,          // noSlip / fixedValue (0 0 0)
    ZeroGradient,       // represented as owner copy in executor
    PassThrough,
    Unsupported
};

enum class PressureBcKind : uint8_t
{
    PrghPressure,
    PrghTotalHydrostaticPressure,
    FixedFluxPressure,
    FixedValue,
    PassThrough,
    Unsupported
};

enum class AlphaBcKind : uint8_t
{
    FixedValue,
    InletOutlet,
    ZeroGradient,
    PassThrough,
    Unsupported
};

enum class FluxKind : uint8_t
{
    VolumetricPhi,
    MassFlux
};

enum class UnsupportedPatchAction : uint8_t
{
    Fail,
    WarnAndCpuFallback
};

enum class BackflowMode : uint8_t
{
    ZeroGradient,
    ClampToDirection
};

enum class ProfileKind : uint8_t
{
    Constant,
    RadialTable,
    RadialPolynomial
};

struct DeviceProfile1D
{
    ProfileKind kind;
    int coeffOffset;      // offset into coeff/value arrays
    int coeffCount;
    Foam::scalar offset;
    Foam::scalar clampMin;
    Foam::scalar clampMax;
};

struct DeviceSeparableProfile
{
    DeviceProfile1D timeScale;
    DeviceProfile1D radialShape;
};

struct PressureSwirlParams
{
    Foam::vector origin;
    Foam::vector axisHat;                  // normalized on host
    DeviceSeparableProfile tangential;
    DeviceSeparableProfile radial;
    FluxKind fluxKind;
    BackflowMode backflowMode;
    bool orthogonalizeSwirl;
    bool zeroSwirlOnBackflow;
    int rhoFieldSlot;                      // -1 if not needed
};

struct FixedValuePressureParams
{
    Foam::scalar uniformValue;
};

struct PrghPressureParams
{
    Foam::scalar pStaticUniform;           // default supported mode
};

struct PrghTotalHydrostaticParams
{
    Foam::scalar phRghUniform;             // default supported mode
};

struct AlphaFixedValueParams
{
    Foam::scalar inletValue;
};

struct AlphaInletOutletParams
{
    Foam::scalar inletValue;
};

struct PatchRange
{
    int patchId;
    int faceStart;
    int faceCount;
    PatchRole role;
    VelocityBcKind uKind;
    PressureBcKind pKind;
    AlphaBcKind alphaKind;
    int uParamIndex;
    int pParamIndex;
    int alphaParamIndex;
};

struct BoundaryRuntimeParams
{
    Foam::scalar timeValue;
    Foam::scalar deltaTValue;
    Foam::label pimpleIter;
    Foam::label pressureIter;
    Foam::label alphaSubIter;
    Foam::vector g;
    Foam::scalar hRef;
};

struct BoundarySupportReport
{
    bool ok;
    Foam::wordList hardErrors;
    Foam::wordList softWarnings;
    Foam::label nSupportedFaces;
    Foam::label nSwirlFaces;
    Foam::label nWallFaces;
    Foam::label nOpenFaces;
    Foam::label nFallbackPatches;
};

class BoundaryRegistry
{
public:
    BoundaryRegistry(const fvMesh& mesh);

    BoundarySupportReport build
    (
        const volVectorField& U,
        const volScalarField& p_rgh,
        const volScalarField& alpha1,
        const surfaceScalarField& phi,
        const dictionary& gpuNozzleBCDict,
        UnsupportedPatchAction action
    );

    void allocateAndUpload(DeviceAllocator& alloc);
    const BoundarySupportReport& report() const;

    // Device manifest view
    const DeviceBoundaryManifestView& deviceView() const;

    // Host debug / testing
    const List<PatchRange>& patchRanges() const;
};

class BoundaryExecutor
{
public:
    BoundaryExecutor
    (
        const BoundaryRegistry& registry,
        DeviceFieldBundle& fields,
        DeviceAllocator& alloc
    );

    void updateRuntimeParams
    (
        const BoundaryRuntimeParams& params,
        cudaStream_t stream
    );

    void applyAlphaBoundaries(cudaStream_t stream);
    void applyVelocityPreMomentum(cudaStream_t stream);
    void applyPressureBoundaryState
    (
        const DevicePressureInputs& inputs,
        cudaStream_t stream
    );
    void applyVelocityPostPressure(cudaStream_t stream);

    // Access for pressure assembly
    const DeviceFixedFluxGradientView& fixedFluxGradients() const;

    bool captureReady() const;
};

class StartupSeeder
{
public:
    StartupSeeder(const fvMesh& mesh);

    void build(const dictionary& gpuStartupSeedDict);
    void allocateAndUpload(DeviceAllocator& alloc);

    void apply(DeviceFieldBundle& fields, cudaStream_t stream);

    const SeedReport& report() const;
};

} // namespace Foam::gpuNozzle
```

#### 3. Phase 6 dependencies on previous phases

Phase 6 depends on the following interfaces existing before coding begins:

| Dependency | Required from earlier phase | Why Phase 6 depends on it |
|---|---|---|
| Device field views | Flat access to internal and boundary storage for `U`, `p_rgh`, `alpha`, `phi`, `rho`, `rAU`/`rhorAU` | Boundary kernels must write directly into device field storage. |
| Pressure-boundary-state contract | Named Phase 5 `PressureBoundaryState` / pressure-corrector interface that consumes device-side `snGradp` or equivalent boundary-gradient state | `fixedFluxPressure` cannot remain host-owned in the hot path, and Phase 6 must not invent a second ownership model. |
| Device mesh geometry views | Face centers, face areas, magnitudes, owner cell map, cell centers | Needed for patch flattening and startup seeding. |
| Mixture-property update hook | Device-side `mixture.correct()` equivalent or its inputs/outputs | Startup seeding changes alpha and therefore rho/viscosity state. |
| Allocator / pool | Stable device allocator with no hot-path allocation | Phase 6 must preallocate everything. |
| NVTX wrapper | NVTX v3 range helpers | Required for profiling hooks. [R16] |

| State-initialization hook | Ability to propagate seeded current fields into device-resident `oldTime` / `prevIter` state and any previous-correction mirrors | Startup seeding must initialize every state later alpha, momentum, and pressure stages assume exists. |


#### 4. Hidden coupling points that must be called out explicitly

1. **Pressure assembly consumes the named pressure-boundary-state contract**. If Phase 5 pressure assembly still expects host patch objects to own `snGradp`, Phase 6 cannot be correct without modifying that interface; Phase 6 must not create a second shadow ownership path.
2. **Alpha BCs couple to mixture property updates**. If `alpha` boundary values are stale before `mixture.correct()`, `rho` and `rhoPhi` can become inconsistent.
3. **`phi` sign conventions couple velocity and alpha BC logic**. One shared implementation of “inflow or outflow” must be used across U and alpha.
4. **Wall wetting/contact-angle handling is owned elsewhere**. Phase 6 must not silently override that logic.
5. **Flat boundary storage must be stable across mesh reload/restart**. If field storage is reallocated, manifest/device views must be rebound.

6. **Startup seeding must also initialize old-time / prev-iteration state**. If the first alpha or pressure stage reads stale mirrors after seeding, the regression signal becomes ambiguous immediately.


### Data model / memory model

#### 1. Ownership and lifetime

| Object | Owner | Allocation time | Update frequency | Leaves GPU? |
|---|---|---|---|---|
| `BoundaryRegistry` host config | Solver object | Startup after field creation | Immutable | No |
| `DeviceBoundaryManifestView` + backing arrays | `BoundaryRegistry` | Startup after validation | Immutable until mesh reload | No |
| Profile coefficient arrays | `BoundaryRegistry` | Startup | Immutable | No |
| `BoundaryExecutor` scratch (`snGradp`, counters, small staging) | `BoundaryExecutor` | Startup | Per pressure iter / per stage | Diagnostics only |
| `BoundaryRuntimeParams` device buffer | `BoundaryExecutor` | Startup | Every stage call | No |
| Startup seed manifest | `StartupSeeder` | Startup | Immutable | No |
| Optional debug masks / sampled dumps | `StartupSeeder` / executor | Debug mode only | On demand | Yes, in debug only |
| Field boundary storage views | Phase 5 field system | Before Phase 6 binding | Every stage write | No |

#### 2. Required immutable device arrays

All of the following must be persistent, device-resident, and allocated from the device memory pool.

```cpp
struct DeviceBoundaryManifestView
{
    // patch-range table
    const PatchRange* ranges;
    int nRanges;

    // flattened face metadata for supported faces only
    const int* ownerCell;          // [nSupportedFaces]
    const int* patchId;            // [nSupportedFaces]
    const int* patchLocalFace;     // [nSupportedFaces]

    // geometry in SoA form
    const scalar* Cf_x;
    const scalar* Cf_y;
    const scalar* Cf_z;
    const scalar* Sf_x;
    const scalar* Sf_y;
    const scalar* Sf_z;
    const scalar* magSf;
    const scalar* nHat_x;
    const scalar* nHat_y;
    const scalar* nHat_z;
    const scalar* deltaCoeff;      // if available / required by assembly

    // cylindrical basis for swirl faces
    const scalar* radius;          // 0 for non-swirl faces
    const scalar* eR_x;
    const scalar* eR_y;
    const scalar* eR_z;
    const scalar* eTheta_x;
    const scalar* eTheta_y;
    const scalar* eTheta_z;
    const scalar* eAxial_x;
    const scalar* eAxial_y;
    const scalar* eAxial_z;

    // parameter tables
    const PressureSwirlParams* swirlParams;
    int nSwirlParams;
    const FixedValuePressureParams* fixedValuePressureParams;
    const PrghPressureParams* prghPressureParams;
    const PrghTotalHydrostaticParams* prghTotalHydrostaticParams;
    const AlphaFixedValueParams* alphaFixedParams;
    const AlphaInletOutletParams* alphaInletOutletParams;

    // profile coefficient pools
    const scalar* profileX;        // abscissa for table profiles
    const scalar* profileY;        // ordinate for table profiles
    const scalar* polyCoeff;       // concatenated polynomial coefficients

    // field-boundary flat offsets (must align with actual field storage)
    const int* UBoundaryFaceIndex;       // [nSupportedFaces]
    const int* pRghBoundaryFaceIndex;    // [nSupportedFaces]
    const int* alphaBoundaryFaceIndex;   // [nSupportedFaces]
    const int* phiBoundaryFaceIndex;     // [nSupportedFaces]
    const int* rhoBoundaryFaceIndex;     // [nSupportedFaces]
    const int* rhorAUBoundaryFaceIndex;  // [nSupportedFaces]
};
```

#### 3. Mutable device state

```cpp
struct DeviceFixedFluxGradientView
{
    scalar* snGradp;           // one entry for each supported pressure face that uses fixedFluxPressure
    int*    mapToSupported;    // optional compact mapping
    int     size;
};

struct DeviceBoundaryCounters
{
    unsigned int* nBackflowFaces;
    unsigned int* nAxisRegularizedFaces;
    unsigned int* nNaNFaces;
    unsigned int* nCpuFallbackHits;
};
```

#### 4. Data-layout decisions

1. **Geometry metadata uses SoA**, not AoS.
   - Reason: boundary kernels repeatedly access one component at a time and benefit from contiguous arrays.
2. **Field values use the native device field layout**.
   - Reason: avoid shadow copies.
3. **Patch ranges are small AoS records**.
   - Reason: patch-range traversal is coarse-grained and branch-driven; readability matters more than raw packing.
4. **Profile tables are packed contiguously**.
   - Reason: makes async upload and index-based access simple.

#### 5. How profiles are represented

Phase 6 will not attempt a full generic `Function1` / `Function2` interpreter. Instead, it uses the following device-evaluable grammar:

##### 5.1 `constant`

```text
type constant;
value <scalar>;
```

Evaluation:
```text
f(x) = value
```

##### 5.2 `radialPolynomial`

```text
type radialPolynomial;
coefficients (a0 a1 a2 ... aN);
clampMin <optional>;
clampMax <optional>;
```

Evaluation:
```text
f(r) = clamp(a0 + a1*r + a2*r^2 + ... + aN*r^N)
```

##### 5.3 `radialTable`

```text
type radialTable;
points ((r0 v0) (r1 v1) ... (rM vM));
outOfRange clamp;   // only supported mode in Phase 6
```

Evaluation:
- monotonic `r` required,
- linear interpolation,
- clamp below/above range.

##### 5.4 `separableProfile`

This is the only 2D time-radius form supported in Phase 6.

```text
type separableProfile;
offset <scalar>;              // default 0
timeScale
{
    type constant | radialTable;   // interpreted against time, not radius
    ...
}
radialShape
{
    type constant | radialPolynomial | radialTable;
    ...
}
```

Evaluation:
```text
f(t, r) = offset + timeScale(t) * radialShape(r)
```

This support set is deliberately narrow. It is sufficient for most nozzle startup and inlet swirl prescriptions while keeping the device evaluator deterministic and small.

#### 6. Dictionary sources and ownership

Phase 6 uses three logical dictionary sources, but their canonical runtime ownership is under the normalized `gpuRuntime` tree:

1. Existing field boundary dictionaries:
   - `0/U`
   - `0/p_rgh`
   - `0/alpha.<phase>`
2. Boundary configuration under `gpuRuntime.boundary`
   - legacy `system/gpuNozzleBCDict` remains accepted only as a compatibility shim / generated subview
3. Startup seeding configuration under `gpuRuntime.startupSeed`
   - legacy `system/gpuStartupSeedDict` remains accepted only as a compatibility shim / generated subview

The new dictionaries exist to avoid overloading generic OpenFOAM function-object grammars with half-supported device semantics, but they are not independent long-term contracts outside the centralized runtime schema.

#### 7. Exact case-configuration policy

##### 7.1 Required boundary dictionary for nozzle inlet `U`

Example:

```foam
swirlInletA
{
    type                gpuPressureSwirlInletVelocity;
    phi                 phi;                // optional, default phi
    rho                 rho;                // required only if phi is mass flux
    normalMode          fromFlux;           // only supported mode in Phase 6

    origin              (0 0 0);
    axis                (0 0 1);

    tangentialProfile
    {
        type            separableProfile;
        offset          0;
        timeScale
        {
            type        constant;
            value       1;
        }
        radialShape
        {
            type        radialTable;
            points      ((0 0) (2.0e-4 10) (4.0e-4 18));
        }
    }

    radialProfile
    {
        type            constant;
        value           0;
    }

    backflowMode        zeroGradient;
    zeroSwirlOnBackflow yes;
    orthogonalizeSwirl  yes;

    value               uniform (0 0 0);    // OpenFOAM patch-field requirement
}
```

`swirlInletB` uses the same grammar.

##### 7.2 Allowed `p_rgh` configurations on inlet/open patches

The centralized support matrix freezes the authoritative inlet/open `p_rgh` kind before coding. Phase 6 does not choose among these at implementation time; it imports the frozen selection. For the accepted `R1` / `R0` case family, the required production tuple is:

- swirl inlet `p_rgh`: `prghTotalHydrostaticPressure` with **uniform hydrostatic total reference value**
- ambient/open `p_rgh`: `prghPressure` with **uniform static pressure value**
- wall `p_rgh`: `fixedFluxPressure` with device-resident `snGradp` update owned by the pressure-boundary stage

Do **not** attempt full arbitrary-field or function-driven `p_rgh` support in Phase 6.

##### 7.2A Accepted ambient/open velocity configuration

For the accepted `R1` / `R0` case family, the required ambient/open `U` kind is `pressureInletOutletVelocity`. `pressureDirectedInletOutletVelocity` remains design background only and is not part of the milestone-1 acceptance tuple.

##### 7.3 Alpha patch policies

- nozzle liquid inlet: `fixedValue 1`
- ambient/open boundary: `inletOutlet` with `inletValue 0`
- wall: `zeroGradient` for milestone 1 unless the centralized support matrix explicitly marks wall wetting/contact-angle as required and Phase 5 already exports that path

##### 7.4 `gpuNozzleBCDict`

Canonical form under `system/gpuRuntimeDict`:

```foam
gpuRuntime
{
    boundary
    {
        enabled                     yes;
        unsupportedPatchAction      fail;       // fail | warnAndCpuFallback
        allowCpuBoundaryFallback    no;
        requireZeroUvmBoundaryFaults yes;
        debug                       0;
    }
}
```

Canonical ownership is under `gpuRuntime.boundary`. The legacy `gpuNozzleBCDict` block below is the compatibility-shim form. Any `warnAndCpuFallback` / `allowCpuBoundaryFallback yes` setting is bring-up only and is forbidden in production acceptance.

```foam
gpuNozzleBC
{
    enabled                     yes;
    unsupportedPatchAction      fail;       // fail | warnAndCpuFallback
    allowCpuBoundaryFallback    no;
    requireZeroUvmBoundaryFaults yes;
    debug                       0;
}
```

##### 7.5 `gpuStartupSeedDict`

Canonical form under `system/gpuRuntimeDict`:

```foam
gpuRuntime
{
    startupSeed
    {
        enabled yes;
        forceReseed no;
        precedence lastWins;

        // The frozen inner grammar below is identical to the compatibility-shim
        // payload shown in `gpuStartupSeedDict`.
    }
}
```

Canonical ownership is under `gpuRuntime.startupSeed`. The legacy `gpuStartupSeedDict` block below is the compatibility-shim form. The allowed seeding grammar is owned centrally by `support_matrix.md` / `support_matrix.json`; Phase 6 imports and implements that DSL and intentionally does not grow it during implementation.

```foam
gpuStartupSeedDict
{
    enabled yes;
    forceReseed no;
    precedence lastWins;

    defaultFieldValues
    (
        volScalarFieldValue alpha.water 0
        volVectorFieldValue U (0 0 0)
        volScalarFieldValue p_rgh 0
    );

    regions
    (
        cylinderToCell
        {
            p1 (0 0 0);
            p2 (0 0 0.01);
            radius 1.5e-4;
            fieldValues
            (
                volScalarFieldValue alpha.water 1
            );
        }

        frustumToCell
        {
            p1 (0 0 0.002);
            p2 (0 0 0.008);
            radius1 1.0e-4;
            radius2 2.5e-4;
            fieldValues
            (
                volScalarFieldValue alpha.water 0
            );
        }
    );
}
```

Supported region types in Phase 6:
- `cylinderToCell`
- `frustumToCell`
- `boxToCell`
- `sphereToCell`
- `halfSpaceToCell`

Any additional region or field-value form is out of scope unless the centralized support matrix / startup-seed plan is revised before coding.


### Algorithms and control flow

#### 1. High-level orchestration

Phase 6 introduces the following solver-stage calls:

1. **Startup**
   - build boundary registry
   - upload manifests
   - optionally run startup seeding
   - apply initial boundary state
   - compute initial derived fields

2. **Before each alpha solve / subcycle**
   - apply alpha boundary values

3. **Before momentum predictor**
   - apply velocity boundary values for nozzle inlets and open boundaries

4. **During each pressure corrector**
   - update fixed-value `p_rgh` boundaries
   - compute `fixedFluxPressure` gradients
   - provide pressure boundary state to pressure assembly

5. **After each pressure solve (if needed)**
   - refresh pressure-coupled velocity boundaries using the updated boundary flux

#### 2. Stage ordering relative to PIMPLE and alpha subcycling

Required order for one timestep:

```text
startup (once before first time loop)
    -> seed fields (optional)
    -> apply initial alpha/U/p_rgh BCs
    -> update mixture properties / derived fields

for each timestep:
    for each PIMPLE outer iteration:
        prePredictor:
            for each alpha subcycle:
                applyAlphaBoundaries()
                alphaSolve()
            mixture.correct()

        preMomentum:
            applyVelocityPreMomentum()

        momentumPredictor()

        pressureCorrector:
            for each pressure corrector iteration:
                applyPressureBoundaryState()
                assemble pressure matrix using current p_rgh BC values + snGradp
                solve pressure
                applyVelocityPostPressure()   // if using final phi to refresh pressure-coupled U
```

#### 3. Detailed algorithm for `gpuPressureSwirlInletVelocity`

For each supported face on a nozzle inlet patch:

Inputs:
- boundary flux `phi_b`
- optional `rho_b` if using mass flux
- face normal `nHat` and area magnitude `magSf`
- cylindrical basis `(eAxial, eR, eTheta)` precomputed from `origin`, `axis`
- current time
- profile parameters
- owner/internal velocity `U_owner` (for zero-gradient fallback on backflow)

Algorithm:

1. Determine inflow/outflow:
   - `isInflow = (phi_b < 0)` using OpenFOAM sign convention. [R8]
2. If outflow/backflow handling is `zeroGradient` and `phi_b >= 0`:
   - set `U_b = U_owner`
   - if `zeroSwirlOnBackflow == true`, do not add swirl
   - return
3. Compute normal speed:
   - if `fluxKind == VolumetricPhi`:
     - `Un = phi_b / magSf`
   - else if `fluxKind == MassFlux`:
     - `Un = phi_b / (rho_b * magSf)`
4. Compute swirl profiles at current `(t, r)`:
   - `Ut = tangentialProfile(t, r)`
   - `Ur = radialProfile(t, r)`
5. Compose swirl vector:
   - `Uswirl = Ur * eR + Ut * eTheta`
6. If `orthogonalizeSwirl`:
   - `Uswirl = Uswirl - nHat * dot(nHat, Uswirl)`
7. Compose final boundary velocity:
   - `U_b = Un * nHat + Uswirl`
8. Optional safety clamps:
   - if any component is NaN/Inf, increment debug counter and hard-fail in debug builds

Important numerical note:
- `Un * nHat` preserves the flux sign convention because `phi = Sf · U = magSf * (nHat · U)`.
- Orthogonalization is mandatory because the precomputed cylindrical basis may not be perfectly orthogonal to the face normal for non-planar or imperfectly meshed patches.

#### 4. Detailed algorithm for ambient/open-boundary velocity BCs

The accepted `R1` / `R0` milestone-1 tuple uses `pressureInletOutletVelocity` for ambient/open `U`. The directed variant below is preserved as background/reference only and is not required for milestone-1 signoff.

##### 4.1 `pressureDirectedInletOutletVelocity`

For each supported face:

1. `isInflow = (phi_b < 0)` [R8]
2. If outflow:
   - `U_b = U_owner`  // zero-gradient representation
3. If inflow:
   - compute inflow speed from flux:
     - `Un = phi_b / magSf` for volumetric flux
     - `Un = phi_b / (rho_b * magSf)` for mass flux
   - `dir = normalized(inletDirection)`
   - `U_b = Un * dir`

##### 4.2 `pressureInletOutletVelocity`

For each supported face:

1. `isInflow = (phi_b < 0)` [R7]
2. If outflow:
   - `U_b = U_owner`
3. If inflow:
   - `Un = dot(U_owner, nHat)`
   - `U_b = Un * nHat + U_tSpecified`
   - if no tangential specification is present, `U_tSpecified = 0`

This BC is supported for ambient/open patches only, not for nozzle swirl inlets.

#### 5. Detailed algorithm for alpha BCs

##### 5.1 `fixedValue`

```text
alpha_b = inletValue
```

##### 5.2 `zeroGradient`

```text
alpha_b = alpha_owner
```

##### 5.3 `inletOutlet`

Using the same `phi` sign convention:

- outflow (`phi_b >= 0`): `alpha_b = alpha_owner`
- inflow (`phi_b < 0`): `alpha_b = inletValue`

For the reference nozzle cases, default ambient/open inflow value is liquid alpha = `0`.

#### 6. Detailed algorithm for `prghPressure`

Using the upstream hydrostatic relation: [R11]

```text
p_rgh = p - rho * g * (h - hRef)
```

Phase 6 supports the uniform-static-pressure subset. For each supported face:

1. Obtain `rho_b`
2. Compute `gDotCf = g · Cf`
3. Compute `ghRef = -|g| * hRef` if using the same sign convention as the referenced implementation, or an equivalent stable precomputed form
4. Set:
   ```text
   p_rgh_b = pStaticUniform - rho_b * (gDotCf - ghRef)
   ```

Implementation note:
- compute and store exactly one consistent formulation; do not mix multiple sign conventions across code paths.

#### 7. Detailed algorithm for `prghTotalHydrostaticPressure`

Using the upstream hydrostatic-total relation: [R12]

```text
p_rgh = ph_rgh - 0.5 * rho * |U|^2
```

For each supported face:

1. Obtain `rho_b`
2. Obtain current `U_b`
3. Set:
   ```text
   p_rgh_b = phRghUniform - 0.5 * rho_b * magSqr(U_b)
   ```

Use the current boundary velocity after the current U-boundary stage, not stale owner velocity.

#### 8. Detailed algorithm for `fixedFluxPressure` gradient update

Use the same structure as `constrainPressure`, simplified for the no-MRF Phase 6 scope. [R14]

For each supported `fixedFluxPressure` face:

Inputs:
- `phiHbyA_b`
- `rho_b`
- `U_b`
- `Sf`
- `magSf`
- `rhorAU_b`

Compute:
```text
snGradp_b = (phiHbyA_b - rho_b * (Sf · U_b)) / (magSf * rhorAU_b)
```

Store this into the persistent `snGradp` device buffer for use by pressure assembly.

Important:
- This must happen **every pressure corrector iteration before pressure assembly**.
- If future MRF support is required, the missing `MRF.relative(...)` term must be added in a later phase. It is intentionally out of scope now.

#### 9. Detailed algorithm for startup seeding

Phase 6 startup seeding is cell-centered and deterministic.

Per run:

1. Parse `defaultFieldValues`.
2. Parse ordered `regions`.
3. Allocate and upload a compact shape manifest.
4. Launch one kernel over all cells:
   - initialize each target field from `defaultFieldValues`
   - for each region in order:
     - test whether cell center is inside region
     - if yes, apply region `fieldValues`
   - region precedence = `lastWins`
5. After the seed kernel:
   - apply alpha/U/p_rgh boundary values
   - update mixture properties and any dependent fields
   - recompute `phi` if required by the solver startup path
   - propagate the resulting seeded state into any device-resident `oldTime`, `prevIter`, and previous-correction mirrors that the first alpha, momentum, or pressure stages may read

Supported inclusion tests:

- `cylinderToCell`: distance to axis line segment <= radius, projection within segment bounds
- `frustumToCell`: projection within axis segment, local interpolated radius comparison
- `boxToCell`: axis-aligned bounds
- `sphereToCell`: Euclidean distance
- `halfSpaceToCell`: signed plane distance

#### 10. Graph-capture-safe control flow

This section is subordinate to the centralized `GraphCaptureSupportMatrix`. Phase 6 boundary stages own only the nozzle-BC rows in that matrix; any graph-unsafe pressure-backend boundary remains declared there rather than inferred locally.

The following are allowed inside captured boundary stages:

- reading persistent manifests
- reading field views
- writing boundary field storage
- writing `snGradp`
- reading tiny runtime param buffer already copied to device

The following are **forbidden** inside captured boundary stages:

- allocation
- dictionary lookup
- runtime selection / patch-object discovery
- `new` / `delete`
- host reductions
- any CPU fallback
- any debug dump that synchronizes the device

### Required source changes

This section lists the concrete source changes the coding agent shall make.

#### 1. New files to add

```text
src/gpuNozzleBC/
    Make/files
    Make/options

    gpuNozzleBCConfig.H
    gpuNozzleBCConfig.C

    BoundaryRegistry.H
    BoundaryRegistry.C

    BoundaryManifest.H
    BoundaryManifest.C

    BoundaryProfiles.H
    BoundaryProfiles.C
    BoundaryProfilesDevice.cu

    BoundaryExecutor.H
    BoundaryExecutor.C
    BoundaryExecutorKernels.cu

    PressureBoundaryState.H
    PressureBoundaryState.C

    StartupSeeder.H
    StartupSeeder.C
    StartupSeederKernels.cu

    BoundarySupportReport.H
    BoundarySupportReport.C

    gpuPressureSwirlInletVelocityFvPatchVectorField.H
    gpuPressureSwirlInletVelocityFvPatchVectorField.C

    gpuNozzleBCDebug.H
    gpuNozzleBCDebug.C
```

#### 2. Existing solver files to modify

The exact paths must be verified against the SPUMA target tree, but the coding agent should expect to modify:

```text
applications/modules/incompressibleVoF/<solver>.H
applications/modules/incompressibleVoF/<solver>.C
applications/modules/incompressibleVoF/alphaPredictor.C
applications/modules/incompressibleVoF/pressureCorrector.C
applications/modules/incompressibleVoF/createFields.H
```

Expected modifications:
- instantiate `BoundaryRegistry`, `BoundaryExecutor`, and `StartupSeeder`
- bind them to the solver object lifetime
- insert stage calls at the correct points
- expose device boundary spans for fields if not already available

#### 3. Existing device-field / pressure-assembly files to modify

Likely modules from earlier phases that must be extended:

```text
src/deviceFields/...
src/deviceFiniteVolume/devicefvc/...
src/deviceFiniteVolume/devicefvm/...
src/devicePressure/...
```

Required changes:
- expose flat boundary spans and patch offsets
- expose a pressure-assembly hook for `snGradp`
- ensure no hidden host patch-object reads remain on the supported nozzle path

#### 4. Build-system changes

- add CUDA compilation for new `.cu` files
- link NVTX v3 instrumentation where build options require it [R16]
- add any required include paths for the new patch type and executor modules
- ensure device code is built for the target Blackwell architecture and/or includes PTX fallback consistent with CUDA 12.8 guidance [R15][R18]

#### 5. Case dictionary changes

Reference nozzle cases must be updated to:
- use `gpuPressureSwirlInletVelocity` on the inlet U patches
- provide `gpuRuntime.boundary` (with `gpuNozzleBCDict` accepted only as a compatibility shim / generated subview during transition)
- optionally provide `gpuRuntime.startupSeed` (with `gpuStartupSeedDict` accepted only as a compatibility shim / generated subview during transition)

#### 6. Explicit “what not to modify” guidance

Do **not** modify the following unless a blocking integration issue is proven:

1. core OpenFOAM `fvPatchField` base semantics
2. upstream `fixedFluxPressure` implementation
3. unrelated pressure solver numerics
4. write/output subsystem

Phase 6 should be additive and localized.

### Proposed file layout and module boundaries

#### 1. Ownership map

| Module | Files | Responsibility | Primary owner role |
|---|---|---|---|
| BC config + validation | `gpuNozzleBCConfig.*`, `BoundarySupportReport.*`, `BoundaryRegistry.*` | parse dictionaries, classify patches, produce support report, compile profiles | CFD solver integration engineer |
| Device manifest | `BoundaryManifest.*` | host/device flattened patch metadata | GPU data-layout engineer |
| Profile evaluator | `BoundaryProfiles.*`, `BoundaryProfilesDevice.cu` | parse and evaluate supported profiles | GPU numerics engineer |
| Device executor | `BoundaryExecutor.*`, `BoundaryExecutorKernels.cu`, `PressureBoundaryState.*` | launch stage kernels, maintain `snGradp`, counters, runtime params | CUDA kernel engineer |
| Custom inlet BC | `gpuPressureSwirlInletVelocityFvPatchVectorField.*` | runtime-selected BC object, CPU fallback implementation, parameter accessor | OpenFOAM BC engineer |
| Startup seeding | `StartupSeeder.*`, `StartupSeederKernels.cu` | device initialization grammar and kernels | GPU solver bring-up engineer |
| Solver integration | edits in solver sources | insert stage calls and lifetime ownership | Solver architect |
| Tests / regression harness | under `tests/` or project harness | component and end-to-end validation | QA / performance engineer |

#### 2. Module-boundary rules

1. The custom patch class is **not** allowed to own device memory.
   - It owns parsed configuration only.
2. The boundary registry is the sole owner of immutable device manifests.
3. The executor is the sole owner of mutable boundary scratch.
4. The startup seeder is separate from the boundary executor.
   - Reason: initialization is operationally distinct from per-timestep boundary handling.
5. Solver integration code may orchestrate stage calls but must not contain BC math.

### Pseudocode

#### 1. Host-side startup integration

```cpp
// In solver constructor or createFields path
void GpuPressureSwirlVoFSolver::initializeNozzleBoundarySubsystem()
{
    const dictionary& bcDict =
        lookupGpuRuntimeBoundaryDict(mesh().time().system());
        // canonical location = gpuRuntime.boundary;
        // legacy gpuNozzleBCDict is a compatibility shim only

    boundaryRegistry_.reset(new gpuNozzle::BoundaryRegistry(mesh()));

    gpuNozzle::BoundarySupportReport report =
        boundaryRegistry_->build(U_, p_rgh_, alpha1_, phi_, bcDict, unsupportedPatchAction_);

    logBoundarySupportReport(report);

    if (!report.ok)
    {
        FatalErrorInFunction
            << "Unsupported nozzle boundary configuration. "
            << "See startup support report." << exit(FatalError);
    }

    boundaryRegistry_->allocateAndUpload(deviceAllocator_);

    boundaryExecutor_.reset
    (
        new gpuNozzle::BoundaryExecutor
        (
            *boundaryRegistry_,
            deviceFieldBundle_,
            deviceAllocator_
        )
    );

    if (gpuStartupSeedEnabled_)
    {
        const dictionary& seedDict =
            lookupGpuRuntimeStartupSeedDict(mesh().time().system());
            // canonical location = gpuRuntime.startupSeed;
            // legacy gpuStartupSeedDict is a compatibility shim only

        startupSeeder_.reset(new gpuNozzle::StartupSeeder(mesh()));
        startupSeeder_->build(seedDict);
        startupSeeder_->allocateAndUpload(deviceAllocator_);

        nvtxRangePush("startupSeed");
        startupSeeder_->apply(deviceFieldBundle_, stream_);
        nvtxRangePop();

        // Seed changes alpha/U/p_rgh, so boundary state and derived fields must be refreshed
        gpuNozzle::BoundaryRuntimeParams rt = initialRuntimeParams();
        boundaryExecutor_->updateRuntimeParams(rt, stream_);

        boundaryExecutor_->applyAlphaBoundaries(stream_);
        boundaryExecutor_->applyVelocityPreMomentum(stream_);
        boundaryExecutor_->applyPressureBoundaryState(initialPressureInputs(), stream_);

        deviceMixtureCorrect(stream_);
        deviceRecomputePhiIfNeeded(stream_);

        // Seeded startup state must also initialize any old-time / prev-iteration
        // mirrors that the first alpha or pressure stage may read.
        deviceInitializeOldTimeAndPrevIterState(stream_);
    }
}
```

#### 2. Alpha stage integration

```cpp
void GpuPressureSwirlVoFSolver::alphaPredictor()
{
    const label nAlphaSubCycles = computeNAlphaSubCycles(); // existing logic

    for (label alphaSubIter = 0; alphaSubIter < nAlphaSubCycles; ++alphaSubIter)
    {
        gpuNozzle::BoundaryRuntimeParams rt = runtimeParams();
        rt.alphaSubIter = alphaSubIter;
        boundaryExecutor_->updateRuntimeParams(rt, stream_);

        nvtxRangePush("bc.alpha");
        boundaryExecutor_->applyAlphaBoundaries(stream_);
        nvtxRangePop();

        deviceAlphaSolve(stream_); // Phase 5 path
    }

    deviceMixtureCorrect(stream_);
}
```

#### 3. Momentum stage integration

```cpp
void GpuPressureSwirlVoFSolver::momentumPredictor()
{
    gpuNozzle::BoundaryRuntimeParams rt = runtimeParams();
    boundaryExecutor_->updateRuntimeParams(rt, stream_);

    nvtxRangePush("bc.velocity.preMomentum");
    boundaryExecutor_->applyVelocityPreMomentum(stream_);
    nvtxRangePop();

    deviceMomentumPredictor(stream_);
}
```

#### 4. Pressure stage integration

```cpp
void GpuPressureSwirlVoFSolver::pressureCorrector()
{
    while (pimple_.correct())
    {
        gpuNozzle::BoundaryRuntimeParams rt = runtimeParams();
        rt.pressureIter = currentPressureIter_;
        boundaryExecutor_->updateRuntimeParams(rt, stream_);

        DevicePressureInputs inputs;
        inputs.phiHbyA = deviceFieldBundle_.phiHbyA();
        inputs.rhorAU = deviceFieldBundle_.rhorAU();
        inputs.rho    = deviceFieldBundle_.rho();
        inputs.U      = deviceFieldBundle_.U();
        inputs.p_rgh  = deviceFieldBundle_.p_rgh();

        nvtxRangePush("bc.pressure");
        boundaryExecutor_->applyPressureBoundaryState(inputs, stream_);
        nvtxRangePop();

        devicePressureAssembler_.setFixedFluxGradients
        (
            boundaryExecutor_->fixedFluxGradients()
        );

        devicePressureAssembler_.assembleAndSolve(stream_);

        nvtxRangePush("bc.velocity.postPressure");
        boundaryExecutor_->applyVelocityPostPressure(stream_);
        nvtxRangePop();
    }
}
```

#### 5. Device kernel: swirl inlet velocity

```cpp
__global__
void applyPressureSwirlInletVelocityKernel
(
    DeviceBoundaryManifestView mf,
    BoundaryRuntimeParams rt,
    DeviceFieldView<vector> UBoundary,
    DeviceFieldView<vector> UInternal,
    DeviceFieldView<scalar> phiBoundary,
    DeviceFieldView<scalar> rhoBoundary,
    PatchRange range,
    DeviceBoundaryCounters counters
)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= range.faceCount) return;

    const int i = range.faceStart + local;
    const int bU = mf.UBoundaryFaceIndex[i];
    const int bPhi = mf.phiBoundaryFaceIndex[i];
    const int owner = mf.ownerCell[i];

    const PressureSwirlParams p = mf.swirlParams[range.uParamIndex];

    const scalar phi = phiBoundary[bPhi];
    const scalar magSf = mf.magSf[i];

    vector nHat(mf.nHat_x[i], mf.nHat_y[i], mf.nHat_z[i]);
    vector eR(mf.eR_x[i], mf.eR_y[i], mf.eR_z[i]);
    vector eTheta(mf.eTheta_x[i], mf.eTheta_y[i], mf.eTheta_z[i]);

    const bool inflow = (phi < 0);

    if (!inflow && p.backflowMode == BackflowMode::ZeroGradient)
    {
        UBoundary[bU] = UInternal[owner];
        return;
    }

    scalar rho = 1;
    if (p.fluxKind == FluxKind::MassFlux)
    {
        const int bRho = mf.rhoBoundaryFaceIndex[i];
        rho = rhoBoundary[bRho];
    }

    scalar Un = (p.fluxKind == FluxKind::VolumetricPhi)
        ? phi / magSf
        : phi / (rho * magSf);

    const scalar t = rt.timeValue;
    const scalar r = mf.radius[i];

    scalar Ur = evaluateSeparableProfile(p.radial, t, r, mf);
    scalar Ut = evaluateSeparableProfile(p.tangential, t, r, mf);

    if (r <= axisRegularizationEpsilon())
    {
        Ur = 0;
        Ut = 0;
        if (counters.nAxisRegularizedFaces) atomicAdd(counters.nAxisRegularizedFaces, 1u);
    }

    vector Uswirl = Ur * eR + Ut * eTheta;

    if (p.orthogonalizeSwirl)
    {
        Uswirl -= nHat * (nHat & Uswirl);
    }

    vector Ub = Un * nHat + Uswirl;

    if (!isfinite(Ub.x()) || !isfinite(Ub.y()) || !isfinite(Ub.z()))
    {
        if (counters.nNaNFaces) atomicAdd(counters.nNaNFaces, 1u);
#ifdef GPU_NOZZLE_BC_DEBUG
        asm("trap;");
#endif
        return;
    }

    UBoundary[bU] = Ub;
}
```

#### 6. Device kernel: alpha boundary

```cpp
__global__
void applyAlphaBoundaryKernel
(
    DeviceBoundaryManifestView mf,
    BoundaryRuntimeParams rt,
    DeviceFieldView<scalar> alphaBoundary,
    DeviceFieldView<scalar> alphaInternal,
    DeviceFieldView<scalar> phiBoundary,
    PatchRange range
)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= range.faceCount) return;

    const int i = range.faceStart + local;
    const int bA = mf.alphaBoundaryFaceIndex[i];
    const int bPhi = mf.phiBoundaryFaceIndex[i];
    const int owner = mf.ownerCell[i];

    const scalar phi = phiBoundary[bPhi];

    switch (range.alphaKind)
    {
        case AlphaBcKind::FixedValue:
        {
            const AlphaFixedValueParams p = mf.alphaFixedParams[range.alphaParamIndex];
            alphaBoundary[bA] = p.inletValue;
            return;
        }

        case AlphaBcKind::ZeroGradient:
        {
            alphaBoundary[bA] = alphaInternal[owner];
            return;
        }

        case AlphaBcKind::InletOutlet:
        {
            const AlphaInletOutletParams p = mf.alphaInletOutletParams[range.alphaParamIndex];
            alphaBoundary[bA] = (phi >= 0) ? alphaInternal[owner] : p.inletValue;
            return;
        }

        default:
            return;
    }
}
```

#### 7. Device kernel: fixed-flux pressure gradient

```cpp
__global__
void computeFixedFluxPressureGradientKernel
(
    DeviceBoundaryManifestView mf,
    BoundaryRuntimeParams rt,
    DeviceFieldView<scalar> phiHbyABoundary,
    DeviceFieldView<scalar> rhoBoundary,
    DeviceFieldView<vector> UBoundary,
    DeviceFieldView<scalar> rhorAUBoundary,
    DeviceFixedFluxGradientView gradView,
    PatchRange range
)
{
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= range.faceCount) return;

    const int i = range.faceStart + local;
    const int bPhi = mf.phiBoundaryFaceIndex[i];
    const int bRho = mf.rhoBoundaryFaceIndex[i];
    const int bU   = mf.UBoundaryFaceIndex[i];
    const int bRAU = mf.rhorAUBoundaryFaceIndex[i];

    const vector Sf(mf.Sf_x[i], mf.Sf_y[i], mf.Sf_z[i]);
    const scalar magSf = mf.magSf[i];

    const scalar phiHbyA = phiHbyABoundary[bPhi];
    const scalar rho = rhoBoundary[bRho];
    const vector Ub = UBoundary[bU];
    const scalar rhorAU = rhorAUBoundary[bRAU];

    const scalar snGrad =
        (phiHbyA - rho * (Sf & Ub)) / (magSf * rhorAU);

    const int gradIdx = gradView.mapToSupported ? gradView.mapToSupported[i] : i;
    gradView.snGradp[gradIdx] = snGrad;
}
```

#### 8. Device kernel: startup seeding

```cpp
__global__
void seedFieldsKernel
(
    DeviceSeedManifestView sm,
    DeviceMeshView mesh,
    DeviceFieldView<scalar> alpha1,
    DeviceFieldView<vector> U,
    DeviceFieldView<scalar> p_rgh
)
{
    int celli = blockIdx.x * blockDim.x + threadIdx.x;
    if (celli >= mesh.nCells) return;

    vector C = mesh.cellCenter(celli);

    // Initialize defaults
    scalar alpha = sm.defaultAlpha1;
    vector vel   = sm.defaultU;
    scalar prgh  = sm.defaultPrgh;

    // Ordered region application, lastWins
    for (int regioni = 0; regioni < sm.nRegions; ++regioni)
    {
        if (pointInsideRegion(C, sm.regions[regioni]))
        {
            applyRegionValues(sm.regions[regioni], alpha, vel, prgh);
        }
    }

    alpha1[celli] = alpha;
    U[celli]      = vel;
    p_rgh[celli]  = prgh;
}
```

### Step-by-step implementation guide

The coding agent should execute the steps below in order. Each step includes what to change, why, how to verify, and likely breakages.

#### Step 1 — Add startup validation and support reporting

**Modify**
- add `BoundarySupportReport`
- add `BoundaryRegistry::build(...)`
- add startup logging

**Why**
- prevent silent CPU fallback and unsupported patch drift

**Expected output**
- at solver startup, a printed support report listing:
  - total supported faces
  - patch-role classification
  - unsupported patch list
  - fallback policy

**Verify**
- run on reference case and confirm all expected nozzle patches are recognized
- run on a deliberately unsupported case and confirm startup aborts cleanly

**Likely breakages**
- runtime type-name mismatch across OpenFOAM variants
- missing patch names in case dictionaries

#### Step 2 — Expose flat boundary spans in the device field layer

**Modify**
- extend device field abstractions to expose contiguous boundary spans and patch-to-boundary offsets

**Why**
- boundary kernels must write into actual device field storage, not mirrors

**Expected output**
- test utility can write a simple value into one boundary face and read it back without host copies

**Verify**
- unit test: mutate one boundary face on device and compare field boundary storage after synchronization
- inspect Nsight: no HtoD/DtoH traffic caused by test

**Likely breakages**
- hidden non-contiguous storage assumptions in existing device field wrappers
- patch ordering mismatches

#### Step 3 — Implement host-side patch manifest builder

**Modify**
- `BoundaryManifest.*`
- host geometry flattening code in `BoundaryRegistry`

**Why**
- convert patch objects into device-friendly ranges and SoA arrays

**Expected output**
- host manifest with correct `faceStart/faceCount` ranges and parameter indices

**Verify**
- dump manifest summary for the reference case
- cross-check per-patch face counts against mesh boundary metadata

**Likely breakages**
- incorrect mapping from patch-local faces to flat boundary indices
- wrong owner-cell mapping on boundary faces

#### Step 4 — Implement constrained profile parser/compiler

**Modify**
- `BoundaryProfiles.*`

**Why**
- custom inlet BC requires device-evaluable swirl profiles

**Expected output**
- parsed/compiled profile tables for constant, radialTable, radialPolynomial, separableProfile

**Verify**
- unit tests:
  - constant profile exact value
  - polynomial at known radii
  - table interpolation
  - separable time-radius composition

**Likely breakages**
- non-monotonic table input
- unsupported profile keywords accepted accidentally

#### Step 5 — Add the custom patch type `gpuPressureSwirlInletVelocity`

**Modify**
- add `gpuPressureSwirlInletVelocityFvPatchVectorField.*`
- register runtime selection

**Why**
- make nozzle inlet intent explicit in case dictionaries and provide a CPU fallback/reference implementation

**Expected output**
- case can parse the new patch type
- CPU fallback implementation can run on a small synthetic case

**Verify**
- startup on a CPU-only build with the new patch type succeeds
- CPU boundary values on a frozen field snapshot match a small hand calculation

**Likely breakages**
- constructor signature drift in SPUMA target tree
- missing `clone` / `write` methods causing runtime-selection issues

#### Step 6 — Implement alpha boundary kernels

**Modify**
- `BoundaryExecutorKernels.cu`
- `BoundaryExecutor::applyAlphaBoundaries`

**Why**
- alpha BCs must execute before every subcycle

**Expected output**
- device kernel supports `fixedValue`, `zeroGradient`, `inletOutlet`

**Verify**
- synthetic tests with forced inflow/outflow signs
- compare GPU face values vs CPU reference implementation on frozen fields

**Likely breakages**
- using wrong `phi` sign convention
- stale owner cell reads due incorrect flat mapping

#### Step 7 — Implement ambient/open velocity kernels

**Modify**
- device kernels for `pressureInletOutletVelocity`

**Why**
- the short external plume requires pressure-coupled open boundaries

**Expected output**
- correct inflow/outflow switching for the frozen ambient/open tuple

**Verify**
- one synthetic patch test for the accepted BC kind
- compare against CPU reference values on frozen fields

**Likely breakages**
- accidental use of mass-flux formula on volumetric flux

#### Step 8 — Implement the custom swirl inlet kernel

**Modify**
- `applyPressureSwirlInletVelocityKernel`
- executor dispatch for inlet patch ranges

**Why**
- this is the core nozzle-specific BC

**Expected output**
- inflow normal component from flux
- tangential/radial superposition from profiles
- zero-gradient backflow mode

**Verify**
- annular-patch synthetic test with known profile values
- integrated normal flux must match input `phi`
- tangential component must not alter integrated normal flux beyond tolerance

**Likely breakages**
- forgotten orthogonalization
- incorrect cylindrical basis sign (right-handedness error)
- axis regularization not triggered

#### Step 9 — Implement `prghPressure` and `prghTotalHydrostaticPressure` kernels

**Modify**
- device pressure-value kernels
- executor dispatch for pressure-value patch ranges

**Why**
- inlet/open pressure values must be device-resident before pressure assembly

**Expected output**
- correct per-face `p_rgh` values on supported fixed-value pressure patches

**Verify**
- hydrostatic column test
- compare GPU patch values vs CPU reference on frozen fields

**Likely breakages**
- sign errors in hydrostatic correction
- stale `U_b` used for total-hydrostatic calculation

#### Step 10 — Implement fixed-flux pressure gradient kernel and assembly hook

**Modify**
- `computeFixedFluxPressureGradientKernel`
- pressure assembly interface to consume `snGradp`

**Why**
- wall/nozzle pressure boundary correctness depends on this

**Expected output**
- `snGradp` available to pressure assembly every pressure corrector iteration

**Verify**
- compare GPU `snGradp` against CPU `constrainPressure` on frozen fields
- run one pressure-correction iteration and confirm no fatal path or divergence attributable to stale gradient

**Likely breakages**
- using final `phi` instead of `phiHbyA` at the wrong stage
- wrong `rhorAU` face indexing
- missing update on inner pressure iterations

#### Step 11 — Wire stage calls into alpha predictor

**Modify**
- solver `alphaPredictor()`

**Why**
- alpha BCs must execute before each subcycle

**Expected output**
- stage call appears once per alpha subcycle

**Verify**
- NVTX trace shows one `bc.alpha` range per subcycle
- alpha boundary face values update correctly across subcycles

**Likely breakages**
- stage called once per timestep instead of per subcycle
- graph-safety violated by on-demand allocation in the new path

#### Step 12 — Wire stage calls into momentum predictor

**Modify**
- solver `momentumPredictor()`

**Why**
- U BCs must be consistent before momentum assembly

**Expected output**
- `bc.velocity.preMomentum` appears in trace before momentum assembly

**Verify**
- compare boundary U values with CPU snapshot just before momentum assembly

**Likely breakages**
- U BC stage using stale runtime params
- missing update of ambient/open patches

#### Step 13 — Wire stage calls into pressure corrector

**Modify**
- solver `pressureCorrector()`
- pressure assembly interface

**Why**
- p_rgh values and `snGradp` must be ready before each pressure assembly

**Expected output**
- `bc.pressure` appears before pressure assembly every pressure iter
- post-pressure velocity refresh appears after solve if enabled

**Verify**
- Nsight Systems timeline
- `snGradp` buffer changes across pressure iterations on transient cases

**Likely breakages**
- stage called outside the inner pressure loop
- stale `U_b` used in gradient evaluation

#### Step 14 — Implement startup seeding grammar and host parser

**Modify**
- `StartupSeeder.*`

**Why**
- make startup conditioning device-resident and reproducible

**Expected output**
- parser supports `defaultFieldValues` and ordered `regions`

**Verify**
- parse-only unit tests for all supported region types
- invalid dictionaries fail with actionable errors

**Likely breakages**
- field-name mapping errors (`alpha.water` vs generic `alpha1` research shorthand)
- unsupported field value type silently accepted

#### Step 15 — Implement device seed kernel

**Modify**
- `StartupSeederKernels.cu`

**Why**
- actual field initialization must happen on the device

**Expected output**
- fields updated in one cell-parallel kernel pass

**Verify**
- synthetic geometry tests:
  - cylinder
  - frustum
  - box
  - sphere
  - half-space
- compare seeded cell masks against CPU reference implementation of the same grammar

**Likely breakages**
- shape inclusion math bugs
- incorrect precedence handling

#### Step 16 — Add post-seeding refresh sequence

**Modify**
- solver startup path

**Why**
- seeding changes field state; boundary and derived values must be refreshed before the first timestep

**Expected output**
- after seeding: alpha/U/p_rgh boundaries refreshed, mixture properties corrected, phi recomputed if required

**Verify**
- run one timestep from a seeded case and confirm no uninitialized-field warnings or NaNs

**Likely breakages**
- forgetting to recompute derived fields after seeding
- stale boundary values on the first timestep

#### Step 17 — Add NVTX, counters, and debug controls

**Modify**
- `gpuNozzleBCDebug.*`
- all stage entry points

**Why**
- Phase 6 is at high risk of hidden migration and staging bugs

**Expected output**
- NVTX ranges around each boundary stage and startup seeding
- counters for backflow, axis regularization, NaNs, and fallback

**Verify**
- Nsight Systems trace
- debug counters visible in logs when enabled

**Likely breakages**
- accidental synchronization in debug code
- counters allocated on host instead of device

#### Step 18 — Add component tests

**Modify**
- test harness

**Why**
- end-to-end nozzle failures are too expensive for first-line debugging

**Expected output**
- independent tests for:
  - profile evaluation
  - cylindrical basis
  - swirl inlet BC
  - ambient/open BCs
  - pressure BCs
  - fixed-flux gradient
  - startup seeding

**Verify**
- all component tests pass before end-to-end nozzle run

**Likely breakages**
- test harness not using the same sign conventions as production path
- CPU reference implementation diverges from device evaluator semantics

#### Step 19 — Add end-to-end reference-case regression

**Modify**
- harness scripts / CI steps

**Why**
- verify actual nozzle metrics, not just isolated patch math

**Expected output**
- R1 reduced nozzle case regression
- R0 representative nozzle case regression

**Verify**
- compare against CPU reference metrics and acceptance thresholds

**Likely breakages**
- hidden coupling with surface tension/wall treatment
- startup seed mismatch masking real BC correctness

#### Step 20 — Profile and remove remaining fallback paths

**Modify**
- startup policy and any temporary mirrors or fallbacks

**Why**
- production acceptance requires a fully device-resident supported nozzle path

**Expected output**
- zero fallback hits
- zero boundary-path UVM migrations in steady state

**Verify**
- Nsight Systems with UVM tracing [R2]
- startup support report shows no fallback patches

**Likely breakages**
- latent host access in write/debug paths
- temporary mirrors still active in one stage

#### What not to do in Phase 6

The coding agent must not do the following:

1. **Do not** launch one kernel per patch.
2. **Do not** allocate inside `applyAlphaBoundaries`, `applyVelocityPreMomentum`, `applyPressureBoundaryState`, or `applyVelocityPostPressure`.
3. **Do not** leave CPU fallback enabled by default.
4. **Do not** parse generic arbitrary `Function2` subclasses.
5. **Do not** implement a second source of truth for boundary field values unless absolutely necessary for bring-up, and remove it before acceptance.
6. **Do not** silently clamp or alter sign conventions to “make the case run.”
7. **Do not** defer `fixedFluxPressure` correctness until later; it is required now.
8. **Do not** let debug instrumentation force a synchronization in production builds.

#### Rollback / fallback options

If a blocking integration issue appears, only the following temporary fallback paths are allowed:

1. **CPU fallback for one unsupported patch family**, explicitly enabled in dictionary and counted in telemetry.
2. **Boundary mirror array + scatter/gather**, only if real flat boundary spans cannot be exposed immediately; remove before acceptance.
3. **Host `setFields` for startup only**, only while device seeding is under validation.

Any other fallback is considered uncontrolled scope growth.

### Instrumentation and profiling hooks

#### 1. Mandatory NVTX v3 ranges

Use NVTX v3, not the deprecated NVTX v2. [R16]

Required range names:

- `bc.startup.validate`
- `bc.startup.manifestUpload`
- `bc.startup.seed`
- `bc.alpha`
- `bc.velocity.preMomentum`
- `bc.pressure.values`
- `bc.pressure.fixedFluxGrad`
- `bc.velocity.postPressure`

#### 2. Mandatory counters

Add device or stage counters for:

- `nBackflowFaces`
- `nAxisRegularizedFaces`
- `nNaNFaces`
- `nCpuFallbackHits`
- `nUnsupportedPatchHits` (startup only)

Counters should be copied to host only:
- at timestep end in debug mode, or
- on failure.

#### 3. Required Nsight Systems checks

For the Phase 6 acceptance run, inspect:

1. boundary stage launch count
2. host API synchronization around boundary stages
3. UVM CPU page faults and GPU page faults
4. stage overlap with solver kernels if multiple streams are used later
5. whether any boundary stage triggers implicit device-to-host migrations

Use the same general tracing stance recommended by SPUMA’s wiki, including CUDA and NVTX tracing and UVM page-fault visibility. [R2]

#### 4. Required Nsight Compute focus kernels

Only profile the top boundary kernels individually:

1. `applyPressureSwirlInletVelocityKernel`
2. `applyAlphaBoundaryKernel`
3. `computeFixedFluxPressureGradientKernel`
4. `seedFieldsKernel` (startup only)

Metrics to inspect:
- branch efficiency
- warp execution efficiency
- achieved occupancy
- global load/store efficiency
- L2 hit rate
- register pressure

Interpretation guidance:
- low occupancy is acceptable if launch count is low and branch divergence dominates
- branch efficiency matters more than raw DRAM bandwidth for these kernels

#### 5. Debug dump hooks

Add optional debug dumps for:

- per-patch integrated flux
- per-patch mean tangential velocity
- per-patch mean alpha
- sampled `snGradp`
- sampled seed-mask counts

Use pinned host staging buffers and one explicit copy. Do not dump full face arrays by default.

### Validation strategy

Validation must be staged. Do not skip directly to the full nozzle case.

#### 1. Unit tests

##### 1.1 Profile evaluation
- `constant`
- `radialPolynomial`
- `radialTable`
- `separableProfile`

Pass criteria:
- exact equality for `constant`
- relative error `< 1e-13` for polynomial on representative radii
- relative error `< 1e-12` for linear-table interpolation on simple cases

##### 1.2 Cylindrical basis construction
Check:
- `|eAxial| = 1`
- `|eR| = 1` when `r > eps`
- `|eTheta| = 1` when `r > eps`
- `dot(eAxial, eR)`, `dot(eAxial, eTheta)`, `dot(eR, eTheta)` all close to zero
- right-handedness: `eAxial x eR` aligned with `eTheta`

Pass criteria:
- absolute error `< 1e-12` for orthogonality in double precision

##### 1.3 Flux sign convention
Synthetic patch with prescribed `phi`:
- `phi < 0` must be treated as inflow
- `phi >= 0` must be treated as outflow

Pass criteria:
- exact branch selection for all tested faces

##### 1.4 Fixed-flux pressure gradient
Compare device formula against CPU reference implementation of the same formula and, where possible, against CPU `constrainPressure` output on a frozen field snapshot.

Pass criteria:
- max relative error `< 1e-12`

##### 1.5 Seed-shape inclusion
For each supported region type, compare cell-inclusion mask to a CPU implementation of the same geometry test.

Pass criteria:
- exact mask match on the same mesh for all supported shapes

#### 2. Component regression tests

##### 2.1 Swirl inlet patch snapshot test
Freeze:
- `phi`
- `rho`
- owner `U`
- time

Run:
- CPU fallback `gpuPressureSwirlInletVelocity`
- GPU boundary kernel

Compare:
- face `U_b`
- integrated normal flux
- mean tangential speed

Pass criteria:
- `L_inf(U_b)` relative error `< 1e-11`
- integrated normal flux relative error `< 1e-12`

##### 2.2 Ambient/open boundary snapshot tests
Run both supported standard open-boundary kinds against frozen fields.

Pass criteria:
- `L_inf(U_b)` relative error `< 1e-11`

##### 2.3 Hydrostatic pressure snapshot tests
For `prghPressure` and `prghTotalHydrostaticPressure`, compare face values against CPU reference.

Pass criteria:
- `L_inf(p_rgh_b)` relative error `< 1e-11`

##### 2.4 Alpha boundary snapshot tests
Check:
- inlet fixedValue
- wall zeroGradient
- open inletOutlet under mixed sign flux

Pass criteria:
- exact match for fixed value
- relative error `< 1e-12` otherwise
- all values remain within `[−1e-12, 1 + 1e-12]`

#### 3. Startup seeding regression

Run the same initialization using:

- Phase 6 device seeding grammar
- CPU reference implementation of the **same grammar**

Compare:
- seeded cell count per region
- resulting cellwise alpha/U/p_rgh fields

Pass criteria:
- exact equality for cellwise fields on the same mesh

Optional cross-check against host `setFields`:
- only if the case can be expressed within the constrained Phase 6 grammar.

#### 4. End-to-end reduced nozzle case (R1)

Required checks:
- timestep-to-timestep mass conservation
- pressure drop across nozzle
- inlet mass flow
- integrated liquid flow at outlet/ambient cut
- alpha boundedness
- no NaNs

Recommended pass/fail thresholds:
- mass-flow difference vs CPU reference: `< 0.5%`
- pressure-drop difference vs CPU reference: `< 0.5%`
- max alpha overshoot/undershoot magnitude: `< 1e-8`
- no solver divergence attributable to boundary path within first 100 timesteps

#### 5. End-to-end representative nozzle case (R0)

Required checks:
- discharge coefficient proxy
- spray half-angle proxy
- air-core startup/onset proxy
- transient stability through the reference startup window

Recommended pass/fail thresholds:
- discharge coefficient difference: `< 1%`
- spray half-angle difference: `< 2 degrees`
- air-core onset time difference: `< 5% of reference startup window`

These thresholds are local engineering recommendations, not sourced facts; the formal project gates are owned by the centralized acceptance manifest.

#### 6. Performance validation

##### Mandatory pass criteria
1. **Zero recurring boundary-path UVM migrations** in steady-state timestep traces. [R2]
2. **No CPU fallback hits** in the acceptance configuration.
3. Boundary launch count depends on stage family, not patch count.

##### Recommended performance targets
4. Boundary stages combined consume `< 10%` of timestep wall time on R1 after the first performance hardening pass.
5. Startup seeding consumes `< 5%` of total initialization time and never appears inside the timestep loop.
6. The custom swirl inlet kernel has branch efficiency `> 70%` on the reference inlet patch set.

These performance numbers are engineering targets, not sourced facts.

### Performance expectations

1. Do **not** expect spectacular standalone kernel throughput from Phase 6 kernels.
   - They are small, branchy, and metadata-heavy.
2. The performance win comes primarily from:
   - eliminating host round trips,
   - eliminating hidden UVM faults,
   - eliminating per-patch dispatch,
   - keeping launch count bounded.
3. The swirl inlet kernel will likely be **control-flow limited**, not memory-bandwidth-limited.
4. The fixed-flux pressure gradient kernel should be cheap and predictable.
5. Startup seeding should be one-time and negligible in steady-state profiling.

Interpretation rule:
- if boundary kernels are “fast” but Nsight Systems still shows UVM traffic or CPU synchronization around them, Phase 6 is not done.

### Common failure modes

1. **Wrong face-normal orientation assumption**
   - symptom: inlet flow goes outward or pressure correction diverges
2. **Wrong `phi` sign convention**
   - symptom: inlet/outlet switching inverted for open boundaries
3. **Swirl component changes flux**
   - symptom: integrated inlet mass flow does not match imposed boundary flux
4. **Axis singularity not regularized**
   - symptom: NaNs near the nozzle centerline
5. **Stale `snGradp`**
   - symptom: pressure solve instability or mismatch vs CPU reference
6. **Pressure value kernel using stale `U_b`**
   - symptom: wrong `prghTotalHydrostaticPressure` values
7. **Field-boundary flat index mismatch**
   - symptom: one patch overwrites another patch’s boundary values
8. **Unsupported function profile silently accepted**
   - symptom: wrong swirl profile with no startup error
9. **CPU fallback left enabled**
   - symptom: run “works” but Nsight shows host/device migrations
10. **Startup seed precedence bug**
    - symptom: region overrides apply in the wrong order
11. **Contact-angle wall treatment accidentally bypassed**
    - symptom: interface behavior changes even when boundary math looks correct
12. **Graph capture broken by debug hooks**
    - symptom: runtime failure only under graph-enabled runs

### Debugging playbook

#### When the run is numerically unstable

1. Disable graph capture if enabled.
2. Enable `gpuNozzleBC.debug 1`.
3. Dump:
   - inlet patch integrated `phi`
   - inlet patch mean `U_t`
   - wall `snGradp` samples
4. Compare one timestep against CPU reference snapshots.
5. If mismatch appears first in pressure:
   - inspect `snGradp`
   - confirm stage ordering
6. If mismatch appears first in momentum:
   - inspect `U_b` on swirl inlets and open boundaries
7. If mismatch appears first in alpha:
   - inspect alpha patch values before each subcycle

#### When performance is poor

1. Run Nsight Systems with UVM fault tracing as recommended by SPUMA wiki. [R2]
2. Confirm:
   - no host fallback
   - no allocations inside stage methods
   - no boundary-stage synchronization beyond necessary graph or stage boundaries
3. Inspect launch count:
   - if it grows with patch count, patch batching is broken
4. Inspect boundary traces around writes/debug dumps:
   - output may be reintroducing host access

#### When the swirl pattern is wrong

1. Dump sample face basis vectors:
   - `eAxial`, `eR`, `eTheta`
2. Check right-handedness and orthogonality.
3. Check radius values on the same faces.
4. Compare profile evaluation on those radii/times.
5. Confirm `orthogonalizeSwirl yes` is active.

#### When startup seeding is wrong

1. Dump seeded cell counts per region.
2. Compare device mask against CPU reference mask.
3. Check region precedence (`lastWins`).
4. Check field-name binding (`alpha.water` in the frozen nozzle bundle, not a legacy alias).

#### When graph capture fails

1. Search for allocations in stage methods.
2. Search for dictionary lookups in stage methods.
3. Search for debug dumps or host reductions in stage methods.
4. Search for patch-object access beyond prebound parameter views.

### Acceptance checklist

A phase is not accepted until every line below is checked off.

- [ ] Startup support report shows zero unsupported patches in the acceptance configuration.
- [ ] No CPU fallback hits in the acceptance configuration.
- [ ] `gpuPressureSwirlInletVelocity` parses and runs in CPU fallback mode for snapshot tests.
- [ ] Flat boundary spans are exposed by the device field layer.
- [ ] Boundary manifest is persistent and device-resident.
- [ ] Alpha boundary kernels pass component tests.
- [ ] Ambient/open velocity kernels pass component tests.
- [ ] Swirl inlet kernel passes component tests.
- [ ] `prghPressure` and `prghTotalHydrostaticPressure` kernels pass component tests.
- [ ] `fixedFluxPressure` gradient kernel matches CPU reference.
- [ ] Startup seeding matches CPU reference grammar exactly.
- [ ] Boundary stages are wired into the correct solver points.
- [ ] No hot-path allocations remain.
- [ ] No recurring UVM boundary migrations appear in steady-state Nsight traces.
- [ ] Reduced nozzle case (R1) passes numerical regression thresholds.
- [ ] Representative nozzle case (R0) passes agreed numerical regression thresholds.
- [ ] Stage NVTX ranges appear correctly in Nsight Systems.
- [ ] Graph-capture safety audit is complete.

### Future extensions deferred from this phase

1. Generic `Function1` / `Function2` device interpretation.
2. Contact-angle wall models if not already present.
3. MRF-aware pressure gradient correction.
4. Processor/cyclic/AMI/overset support.
5. Multi-GPU boundary exchange.
6. Geometry-from-surface startup seeding.
7. Full boundary mega-kernel fusion.
8. Persistent-kernel boundary execution.
9. Runtime adaptive profile updates from external controls.

#### Implementation tasks for coding agent

1. Add validation/reporting infrastructure.
2. Expose flat boundary spans in device field abstractions.
3. Build boundary manifest compiler and uploader.
4. Implement constrained profile system.
5. Implement custom `gpuPressureSwirlInletVelocity` patch class.
6. Implement alpha/open/swirl/pressure boundary kernels.
7. Integrate `snGradp` into pressure assembly.
8. Implement startup seeding grammar and kernel.
9. Add instrumentation and tests.
10. Remove all temporary fallback/mirror mechanisms before acceptance.

#### Do not start until

- field boundary spans are available or the exact plan for exposing them is approved
- the reference patch names, support-matrix-selected boundary kinds, and startup-seeding grammar are frozen
- the pressure-boundary-state / `snGradp` handoff for pressure assembly is identified
- the Phase 5 wall/interface dependency is resolved for the reference case, including an explicit yes/no on contact-angle for the accepted case family

#### Safe parallelization opportunities

1. **In parallel**
   - boundary manifest builder
   - profile parser/evaluator
   - startup seeding parser/kernel
2. **In parallel after flat boundary spans exist**
   - alpha boundary kernels
   - ambient/open velocity kernels
   - swirl inlet kernel
3. **Not safe to parallelize independently**
   - pressure gradient integration and pressure-assembly hook
   - final solver-stage wiring

#### Governance guardrails

1. Phase 6 consumes the frozen `support_matrix.md` subset for inlet/open/wall/startup semantics and may not broaden it locally.
2. End-to-end nozzle metric thresholds are owned by `acceptance_manifest.md`; local ranges in this phase are guidance only.
3. Any accepted AmgX production path in this branch must already satisfy the Phase 4 `DeviceDirect` requirement before it can participate in production claims.

#### Artifacts to produce

1. source code for the new Phase 6 modules
2. updated case dictionaries for R1 and R0
3. startup support report example
4. component-test results
5. end-to-end regression summary
6. Nsight Systems trace screenshots or exported reports
7. a short developer note documenting any temporary fallback used during bring-up



## 6. Validation and benchmarking framework

This section is global to the current deliverable but focused on Phase 6.

### 6.1 Validation ladder

The coding agent shall not advance to the next rung until the current rung passes.

1. **Rung A — Parser and manifest tests**
   - dictionary parsing
   - patch classification
   - manifest sizing and offsets
2. **Rung B — Pure math tests**
   - profile evaluation
   - cylindrical basis
   - geometry inclusion tests
3. **Rung C — Frozen-field BC snapshot tests**
   - U / p_rgh / alpha
   - `snGradp`
4. **Rung D — Startup seeding regression**
   - exact field match against CPU reference grammar
5. **Rung E — Reduced nozzle case (R1)**
   - 10 timesteps
   - 100 timesteps
   - full startup window
6. **Rung F — Representative nozzle case (R0)**
   - agreed validation window
   - performance trace
   - numerical metrics

### 6.2 Benchmark outputs required for every serious run

For each acceptance benchmark, collect:

- wall-clock time per timestep
- time breakdown by NVTX boundary stage
- kernel launch count by stage
- UVM migration stats
- integrated inlet fluxes
- integrated outlet/open-boundary fluxes
- alpha boundedness metrics
- pressure-drop metric
- one representative pressure-corrector iteration trace

### 6.3 Stop-and-benchmark gates

The coding agent must stop and benchmark at these exact points:

1. after Step 8 (first working swirl inlet kernel)
2. after Step 10 (`snGradp` wired into pressure assembly)
3. after Step 16 (first full seeded startup run)
4. after Step 20 (production hardening)

Do not postpone all benchmarking to the end.

## 7. Toolchain / environment specification

### 7.1 Required software baseline

- SPUMA target tree pinned to the reviewed commit from the master pin manifest
- toolchain lane pinned by the master pin manifest (current default freeze: CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`)
- Nsight Systems
- Nsight Compute
- Compute Sanitizer
- NVTX v3 headers/runtime [R15][R16]

### 7.2 GPU target assumptions

- NVIDIA GeForce RTX 5080
- Blackwell architecture
- CUDA capability 12.0
- 16 GB GDDR7 [R17][R18]

### 7.3 Build requirements

1. Build device code for the actual Blackwell target where possible.
2. Include PTX according to NVIDIA’s Blackwell compatibility guidance so JIT compatibility can be tested if needed. [R15]
3. Prefer explicit architecture coverage over “let the toolchain guess.”
4. Use NVTX v3 includes.

### 7.4 Debug build modes

Provide at least:

- `DebugCPUFallback`
- `DebugDeviceBoundary`
- `ReleaseDeviceBoundary`

`DebugDeviceBoundary` should enable:
- device asserts / `trap`
- sampled debug dumps
- extra counter checks

## 8. Module / file / ownership map

### 8.1 Deliverable file map

```text
src/gpuNozzleBC/
    BoundarySupportReport.H/.C
    gpuNozzleBCConfig.H/.C
    BoundaryRegistry.H/.C
    BoundaryManifest.H/.C
    BoundaryProfiles.H/.C
    BoundaryProfilesDevice.cu
    BoundaryExecutor.H/.C
    BoundaryExecutorKernels.cu
    PressureBoundaryState.H/.C
    StartupSeeder.H/.C
    StartupSeederKernels.cu
    gpuPressureSwirlInletVelocityFvPatchVectorField.H/.C
    gpuNozzleBCDebug.H/.C
    Make/files
    Make/options
```

### 8.2 File ownership expectations

| File group | Primary responsibility | Review owner |
|---|---|---|
| `BoundaryRegistry.*` | patch discovery, validation, manifest build | senior OpenFOAM engineer |
| `BoundaryProfiles.*` | profile parsing/evaluation correctness | numerics lead |
| `BoundaryExecutor*` | device execution correctness/performance | CUDA lead |
| `PressureBoundaryState.*` | pressure-coupling correctness | pressure-solver lead |
| `StartupSeeder*` | deterministic initialization correctness | CFD application owner |
| `gpuPressureSwirlInletVelocity*` | runtime-selection integration and CPU reference parity | OpenFOAM BC owner |

### 8.3 Areas where ownership must not be ambiguous

1. `snGradp` ownership
2. flat boundary storage ownership
3. startup seeding field-name mapping
4. patch classification rules

## 9. Coding-agent execution roadmap

This is the concrete build order for Phase 6.

### 9.1 Milestones

#### M0 — Preconditions verified
- freeze patch names and boundary dictionaries
- confirm flat boundary span strategy
- confirm pressure-assembly hook

**Stop here if any precondition is unresolved.**

#### M1 — Validation and manifest substrate
- support report
- patch classifier
- flat boundary mapping
- manifest upload

**Benchmark/verify**
- zero unsupported patches on R1 bring-up case

#### M2 — Profile and custom BC substrate
- constrained profile system
- `gpuPressureSwirlInletVelocity` patch class
- CPU snapshot reference path

**Benchmark/verify**
- snapshot tests pass on CPU path

#### M3 — Alpha and ambient/open device BCs
- alpha kernels
- standard open-boundary velocity kernels

**Parallelizable**
- yes, after M1 and M2

**Benchmark/verify**
- frozen-field snapshot tests

#### M4 — Swirl inlet device kernel
- custom nozzle inlet kernel
- basis regularization
- integrated-flux checks

**Benchmark/verify**
- annular inlet synthetic case
- inlet flux invariance test

#### M5 — Pressure boundary integration
- `prgh*` value kernels
- fixed-flux gradient kernel
- pressure assembly hook

**Do not parallelize with other integration work**
- this is the numerically riskiest coupling point

**Stop and benchmark here**
- compare `snGradp` to CPU reference
- run short R1 transient

#### M6 — Startup seeding
- grammar parser
- seed kernel
- post-seed refresh sequence

**Parallelizable**
- parser/kernel can be developed in parallel with M5, but solver integration waits for M5

**Benchmark/verify**
- exact mask/field regression

#### M7 — Solver-stage integration and hardening
- alpha/momentum/pressure stage calls
- runtime-param update path
- remove allocations
- graph-safety audit

**Stop and benchmark here**
- full R1 run
- Nsight Systems

#### M8 — Production acceptance pass
- remove temporary fallbacks
- finalize instrumentation
- R0 regression
- document support matrix

### 9.2 Dependency graph

```text
M0
 └─> M1
      ├─> M2
      │    └─> M4
      ├─> M3
      └─> M5
            └─> M7
M6 depends on M1 and solver field bindings, then feeds M7
M8 depends on M3 + M4 + M5 + M6 + M7
```

### 9.3 What should be prototyped before being productized

Prototype first:
- profile parser/evaluator
- swirl inlet kernel on a synthetic annular patch
- fixed-flux gradient kernel against frozen data
- seed-region inclusion kernel

Productize only after prototypes pass:
- full solver integration
- graph-safe runtime-param handling
- production acceptance runs

### 9.4 What should remain experimental

These should remain explicitly experimental after Phase 6:
- any mixed-precision variant of boundary kernels
- any mega-kernel fusion beyond one kernel per stage family
- any attempt to generalize the profile grammar to arbitrary function objects

### 9.5 Where to stop and benchmark before proceeding

Mandatory benchmark stops:
- after M4
- after M5
- after M7

If performance or correctness regresses at any stop, do not continue to the next milestone.

## 10. Imported Authorities and Residual Governance Notes

Phase 6 does not reopen support-scope decisions that are owned by the centralized support matrix, acceptance manifest, or master pin manifest. The implementation branch must import those authorities unchanged.

1. **Support-matrix scope is fixed.**
   - `support_matrix.md` already specifies the authoritative inlet `p_rgh` kind, the ambient/open `U` BC subset, the wall alpha/contact-angle scope, and the allowed startup-seeding grammar for the accepted case family.
2. **Pressure-backend production scope is fixed.**
   - If AmgX participates in an accepted production configuration, the Phase 4 `DeviceDirect` bridge—not `PinnedHost` staging—must back any no-field-scale-host-transfer claim.
3. **Formal acceptance thresholds are fixed centrally.**
   - `acceptance_manifest.md` owns the formal R1/R0 gates. The numerical values recorded locally in this document remain engineering guidance only unless the manifest is revised.

### Human review checklist

- [ ] Verify the implementation branch records its exact SPUMA commit in `manifest_refs.json`.
- [ ] Verify the consumed Phase 6 support subset matches `support_matrix.md`.
- [ ] Verify the patch naming convention and runtime type names are frozen before end-to-end wiring.
- [ ] Verify any accepted AmgX production lane in scope already has the Phase 4 `DeviceDirect` bridge.
- [ ] Verify the active R1/R0 gates match `acceptance_manifest.md`.

### Coding agent kickoff checklist

- [ ] Pin target repo/commit and inspect actual solver file paths.
- [ ] Verify exact runtime type names for supported BCs in the target tree.
- [ ] Locate or implement flat boundary spans in device field storage.
- [ ] Locate the pressure assembly hook for `snGradp`.
- [ ] Add `BoundarySupportReport` and startup validation first.
- [ ] Build component tests before end-to-end integration.
- [ ] Keep all stage methods allocation-free from the start.
- [ ] Benchmark at M4, M5, and M7.

### Highest risk implementation assumptions

1. The target SPUMA tree can expose flat boundary spans without invasive redesign.
2. The pressure assembly path can consume device-side `snGradp` without falling back to host patch objects.
3. The reference nozzle cases can be expressed with the constrained profile grammar.
4. Contact-angle remains out of milestone-1 scope per `support_matrix.md`, so Phase 6 does not depend on wall-wetting model expansion.
5. The reference startup seeding can be represented with the constrained canonical `gpuRuntime.startupSeed` grammar (`gpuStartupSeedDict` only as a compatibility shim if still accepted in this branch).

## References

- **[R1]** SPUMA paper / arXiv HTML: https://arxiv.org/html/2512.22215v1
- **[R2]** SPUMA GPU support wiki (supported solvers, unsupported-feature warning, profiling guidance): https://gitlab.hpc.cineca.it/exafoam/spuma/-/wikis/GPU-support/diff?version_id=ad2a385e44f2c01b7d1df44c5bc51d7996c95554
- **[R3]** OpenFOAM solver loop ordering (`foamMultiRun`, representative PIMPLE stage ordering): https://cpp.openfoam.org/v12/foamMultiRun_8C_source.html
- **[R4]** `incompressibleVoF` class reference: https://cpp.openfoam.org/v13/classFoam_1_1solvers_1_1incompressibleVoF.html
- **[R5]** `alphaPredictor.C` / MULES / alpha subcycling source: https://cpp.openfoam.org/v13/incompressibleMultiphaseVoF_2alphaPredictor_8C_source.html
- **[R6]** `fvPatchField` base-class source/API (`updateCoeffs`, `updated`, runtime patch-field semantics): https://cpp.openfoam.org/v12/fvPatchField_8H_source.html
- **[R7]** `pressureInletOutletVelocity` class reference: https://cpp.openfoam.org/v4/classFoam_1_1pressureInletOutletVelocityFvPatchVectorField.html
- **[R8]** `pressureDirectedInletOutletVelocity` class reference: https://cpp.openfoam.org/v13/classFoam_1_1pressureDirectedInletOutletVelocityFvPatchVectorField.html
- **[R9]** `swirlFlowRateInletVelocity` class reference: https://cpp.openfoam.org/v13/classFoam_1_1swirlFlowRateInletVelocityFvPatchVectorField.html
- **[R10]** `swirlInletVelocity` source/API reference: https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/finiteVolume/fields/fvPatchFields/derived/swirlInletVelocity/swirlInletVelocityFvPatchVectorField.H
- **[R11]** `PrghPressure` formula/source reference: https://cpp.openfoam.org/v12/PrghPressureFvPatchScalarField_8H_source.html
- **[R12]** `prghTotalHydrostaticPressure` formula/source reference: https://cpp.openfoam.org/v11/prghTotalHydrostaticPressureFvPatchScalarField_8H_source.html
- **[R13]** `fixedFluxPressure` implementation/source reference: https://cpp.openfoam.org/v12/fixedFluxPressureFvPatchScalarField_8C_source.html
- **[R14]** `constrainPressure` source reference: https://cpp.openfoam.org/v10/constrainPressure_8C_source.html
- **[R15]** NVIDIA Blackwell compatibility guide for CUDA 12.8: https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-compatibility-guide/index.html
- **[R16]** CUDA 12.8 features/release notes (Blackwell compiler support, conditional CUDA graphs, NVTX v3 deprecation note): https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/
- **[R17]** NVIDIA GeForce RTX 5080 official specs page: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/
- **[R18]** NVIDIA CUDA GPU compute capability table: https://developer.nvidia.com/cuda/gpus
