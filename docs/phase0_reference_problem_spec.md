## 1. Executive overview

Phase 0 is the control-point phase. Its job is to freeze a reproducible, reviewable, CPU-correct reference problem before any SPUMA, CUDA, AmgX, or custom-kernel work begins. That is mandatory here because the target runtime line and the current repo line are already materially different: SPUMA’s published validation target is OpenFOAM-v2412, SPUMA’s published feature state still excludes multiphase solvers, and its current published GPU-support notes do not list `incompressibleVoF` among the supported GPU solver set. Without a frozen CPU reference, later nozzle regressions would be impossible to attribute cleanly to version migration, solver-family drift, or GPU-port drift.

The repo is already concrete enough to support a rigorous Phase 0. Repo findings: it contains a real pressure-swirl builder (`scripts/cfd/build_pressure_swirl_case.py`), a real template (`cfd/templates/pressureSwirlNozzleVOF`), a real staged startup path (`vof_transient_preconditioned`), real startup conditioning via `setFieldsDict`, a real feature extractor (`scripts/cfd/extract_openfoam_case_features.py`), an existing environment probe (`scripts/cfd/openfoam_env_probe.py`), and an existing `damBreak_amgx` smoke asset showing `foamRun -solver incompressibleVoF` plus persistent LDU-to-CSR offload behavior in the log. Phase 0 must harden and freeze those actual interfaces, not invent a parallel benchmark harness.

The recommended architecture is a **dual-baseline freeze**:

- **Baseline A — current control**: the repo’s current CPU workflow on the present OpenFOAM-12-style line.
- **Baseline B — target migration line**: the SPUMA/v2412 CPU comparison line when it can execute the nozzle workflow without GPU-path interference. If that canonical line cannot execute the Phase 0 freeze after environment and documented pressure-backend adjustments, stock OpenFOAM-v2412 CPU is the contingency-only freeze fallback and must be labeled as such in reports and artifacts.

Phase 0 also freezes two cross-phase scope decisions that later phases already assume. First, the milestone-1 GPU semantic target is the algebraic `incompressibleVoF` / explicit MULES / PIMPLE family. `geometric_iso` / `interIsoFoam` may remain CPU-only shadow references in Phase 0, but they are not the canonical GPU target or the hard acceptance line. Second, Phase 0 freezes the full staged CPU workflow, but milestone-1 GPU acceptance starts at transient solve + device startup seeding; the steady precondition / `potentialFoam` path remains an allowed CPU pre-run utility unless a later explicit milestone promotes it.

That recommendation is based on sourced fact plus conservative engineering judgment. The cited OpenFOAM documentation shows that `foamRun`/modular-solver execution and the `incompressibleVoF` family still provide the correct algorithmic reference for a transient VOF/MULES/PIMPLE nozzle workflow, but the target SPUMA line is aligned to v2412 and multiphase is not yet a supported SPUMA feature. The safest path is therefore: freeze numerics on CPU first, then port to SPUMA GPU execution.

The recommended Phase 0 reference set is:

- **R0 — full nozzle production reference**: `57-28 @ 1000 psi`, full 360°, `vof_transient_preconditioned`, compact near-field plume, startup conditioning enabled, fully resolved numerics.
- **R1-core — Phase-5-friendly reduced reference**: reduced algebraic case using only the frozen generic BC/scheme subset and no Phase 6-only nozzle BC/startup dependency. This is the reduced case later generic-VOF phases consume before nozzle-specific GPU BC/startup work.
- **R1 — reduced nozzle development reference**: same nozzle family and solver family as R0, but reduced-domain / internal-only and lower resolution for fast iteration. Unlike `R1-core`, it retains the nozzle-specific startup/boundary behavior needed for later Phase 6 validation.
- **R2 — generic VOF verification reference**: the existing `cfd/smoke/damBreak_amgx` asset, with an allowed native-CPU pressure-backend fallback on Baseline B if the external solver stack is unavailable.

The exit product of Phase 0 is not “a case that runs.” It is a **reference contract**: frozen manifests, resolved execution metadata, mesh and patch fingerprints, stage provenance, feature metrics, tolerances, comparison rules, and a pass/fail report that later phases can treat as the definition of correctness.

---

## 2. Global architecture decisions

### Decision A — use dual baselines, not a single migration target

**Sourced fact:** SPUMA’s published benchmark baseline is OpenFOAM-v2412, and its published feature state still excludes multiphase solvers.

**Repo finding:** the current repo already runs real pressure-swirl workflows on an OpenFOAM-12-style environment.

**Engineering inference:** jumping directly from the current repo line to a modified SPUMA line would mix three deltas at once: OpenFOAM-line migration, runtime/backend drift, and GPU-port drift.

**Recommendation:** keep Baseline A and Baseline B alive simultaneously until Phase 0 sign-off is complete. Baseline B is the SPUMA/v2412 CPU comparison line when runnable; stock OpenFOAM-v2412 CPU is contingency-only freeze fallback and must be labeled as such.

---

### Decision B — freeze the **resolved** problem, not only the requested manifest

**Sourced fact:** `foamRun` is a modular front end that resolves the actual solver module at runtime, and `incompressibleVoF` uses explicit MULES logic with optional alpha sub-cycling.

**Repo finding:** the current builder can request `geometric_iso` but resolve to `interIsoFoam` only when available; otherwise it falls back to `incompressibleVoF`. It also hides important numerics behind profile names like `direct_slot_v2`.

**Engineering inference:** a CSV row or JSON row alone is not the reference problem. The reference problem is the fully resolved execution contract: requested solver family, resolved executable, resolved pressure backend, resolved numerics, resolved startup seed after any cap, resolved turbulence models, resolved stage sequence, and resolved OpenFOAM environment.

**Recommendation:** every frozen case must emit a machine-readable `case_meta.json` and `stage_plan.json` that fully materialize the resolved runtime behavior.

---

### Decision C — Phase 0 is single-rank CPU only

**Sourced fact:** SPUMA’s published material notes that unsupported paths can degrade into unwanted transfers rather than failing cleanly, and GPU-aware MPI has pointer-ownership constraints that are irrelevant to a correctness freeze.

**Engineering inference:** even though Phase 0 is CPU-only, decomposed or MPI runs would add another axis of nondeterminism and complicate later debugging.

**Recommendation:** no `decomposePar`, no MPI, no GPU offload, no managed memory, no AMGX performance experiments in Phase 0.

---

### Decision D — make solver-family mismatch explicit, never silent

**Sourced fact:** SPUMA’s current published GPU-supported solver list excludes `incompressibleVoF`, while the nozzle workflow’s algorithmic reference is the `incompressibleVoF`/VOF/MULES/PIMPLE family.

**Repo finding:** the builder silently resolves some requested VOF modes to a different executable.

**Engineering inference:** silent fallback poisons the reference set. A later “regression” could simply mean that one baseline ran a different solver family.

**Recommendation:** add a strict mode. If requested and resolved solver families differ, Phase 0 must either fail under `--strict-solver-family` or emit a split-baseline report that requires human sign-off.

---

### Decision E — R0 should be representative, not maximal

**Repo finding:** the repo handoff identifies `57-28 @ 1000 psi` as a strong sentinel nozzle with manufacturing truth available and current active angle/air-core work already focused there; it identifies `40-28` as a harder transfer case.

**Engineering inference:** the first hard-gating full reference should be the case most likely to rerun often and survive migration, not the hardest edge case.

**Recommendation:** make `57-28 @ 1000 psi` the hard-gating R0 case. Keep `40-28 @ 1000 psi` as an optional shadow reference in Phase 0 and a likely hard gate in a later phase.

---

### Decision F — Phase 0 acceptance is numerical and provenance-based, not performance-based

**Sourced fact:** the relevant solver family is transient, VOF-based, PIMPLE-coupled, and explicit in its limiter/sub-cycling behavior. Phase 0’s job is therefore to pin behavior before any attempt to change execution mechanics.

**Engineering inference:** wall-clock CPU numbers are useful as future comparison points, but they are not the gate for this phase.

**Recommendation:** Phase 0 hard gates are solver stability, mesh/patch identity, startup/provenance identity, and primary physics metrics. Runtime is recorded, not gated.

---

### Decision G — pressure backend is orthogonal in Phase 0 unless it blocks execution

**Repo finding:** current templates and smoke assets already assume AmgX-oriented pressure settings.

**Engineering inference:** a CPU-only freeze should not fail purely because a specific external pressure backend is unavailable on the target baseline, provided the case can be run with a documented native fallback and the backend difference is recorded.

**Recommendation:** the native SPUMA/OpenFOAM pressure solve is the required Phase 0 hard-gate baseline. Pressure-backend parity beyond that is preferred but not mandatory in Phase 0. If an external backend blocks execution or is unavailable, use the native CPU path, archive the full resolved `fvSolution`, and treat the external-backend mismatch as provenance, not as an automatic fail.

---

### Decision H — harden existing repo tooling; do not create a parallel builder/extractor stack

**Repo finding:** the repo already has the right authoritative components: builder, extractor, runner, probe, tests, and smoke assets.

**Engineering inference:** writing a second Phase 0-only case builder or extractor would create divergence immediately.

**Recommendation:** extend existing scripts in place, and add a new `reference_problem` orchestration package around them.

---

### Decision I — remove hard-coded OpenFOAM-12 environment assumptions from generated scripts and runners

**Repo finding:** the generated stage scripts and the current `run_manifest.py` path hard-code `. /opt/openfoam12/etc/bashrc`, which is acceptable for Baseline A but wrong for a v2412 target baseline.

**Engineering inference:** this is the single most dangerous hidden coupling in the current repo for Phase 0 migration.

**Recommendation:** generated scripts must become environment-neutral. The caller should source the target environment; scripts should not hard-code `/opt/openfoam12`.

---

## 3. Global assumptions and constraints

1. The current repo is the authoritative source for nozzle geometry generation, startup conditioning, run staging, and post-processing for this project.

2. The long-term execution target remains a single RTX 5080 workstation, but Phase 0 intentionally excludes GPU execution. Future-phase toolchain pins are owned by the centralized master pin manifest rather than this file. Default until superseded: CUDA 12.9.1 primary, CUDA 13.2 experimental, driver `>=595.45.04`, `sm_120` + PTX, NVTX3. The RTX 5080 is a Blackwell-generation GeForce part with 16 GB GDDR7, and NVIDIA’s Blackwell guidance still makes PTX/JIT compatibility validation relevant.

3. The cited OpenFOAM docs for `foamRun`, `incompressibleVoF`, and alpha sub-cycling are Foundation-line algorithmic references. The target SPUMA comparison line is OpenFOAM-v2412. Treat the docs as algorithmically relevant, not as proof of identical CLI or dictionary behavior on Baseline B.

4. Repo finding: the builder enforces an exact patch schema: `swirlInletA`, `swirlInletB`, `walls`, `exitPatch`, `farOutlet`, `farSide`. Phase 0 must preserve that schema exactly.

5. Repo finding: `vof_transient_preconditioned` is a real staged workflow, not wrapper sugar. It builds the mesh, runs `checkMesh`, creates a steady precondition stage, runs `potentialFoam`, runs the steady stage, copies selected fields back, applies `setFields`, and then runs the transient stage. Phase 0 must freeze that actual sequence.

6. Repo finding: startup conditioning is first-class. `startup_fill_extension_d` and `air_core_seed_radius_d` already shape `setFieldsDict`, and the seed radius is capped internally by geometry. Phase 0 must record both requested values and resolved physical values.

7. Repo finding: the extractor already emits `water_flow_cfd_gph`, `cd_cfd`, `cd_mass_cfd`, `air_core_area_ratio`, `sheet_thickness_ratio`, `swirl_number_proxy`, `spray_angle_cfd_deg`, `spray_angle_source`, `recirculation_fraction`, `solver_ok`, and `mass_imbalance_pct`.

8. Repo finding: angle extraction is source-typed. `spray_angle_source=geometric_a` is higher-confidence than `velocity_proxy`. Unlike sources must not be hard-compared.

9. Repo finding: template defaults differ materially from the active `direct_slot_v2` profile. The template’s default `nAlphaCorr`, `nAlphaSubCycles`, `nLimiterIter`, and PIMPLE counts are lighter than the profile applied by the builder. Phase 0 must freeze resolved numerics, not template defaults.

10. Repo finding: `build_pressure_swirl_case.py` already writes `case_meta.json`, `rheology_contract.json`, and `build_log.json`. Phase 0 should extend these artifacts, not replace them.

11. Manufacturing truth from `data/raw/manufacturing/sk_nozzle_water_data.csv` is a sanity reference, not the primary Phase 0 equivalence criterion. The primary equivalence criterion is Baseline A vs Baseline B consistency.

12. Full field outputs and transient logs should live under an artifact root outside version control. Only manifests, tolerance configs, tests, and summary reports should be committed.

---

## 4. Cross-cutting risks and mitigation

### Risk 1 — OpenFOAM-line drift masquerades as GPU-port drift
Mitigation: dual baselines, explicit environment probes, resolved solver/back-end metadata, and a compare report that distinguishes case intent from execution resolution.

### Risk 2 — hard-coded `/opt/openfoam12` path breaks Baseline B silently
Mitigation: remove environment sourcing from generated scripts and runners; require the Phase 0 orchestrator to source the probed environment explicitly.

### Risk 3 — solver-family mismatch hides behind `vof_solver_mode`
Mitigation: record both requested and resolved solver family, add `--strict-solver-family`, and treat mismatch as a review blocker.

### Risk 4 — AmgX / `libpetscFoam.so` availability differs between baselines
Mitigation: support a documented CPU fallback pressure backend for Phase 0, archive the resolved `fvSolution`, and keep backend mismatch out of hard numerical equivalence gates.

### Risk 5 — startup conditioning is under-specified
Mitigation: freeze startup fill and seed explicitly, record the resolved capped seed radius in meters, and archive the exact generated `setFieldsDict`.

### Risk 6 — angle metrics are compared across unlike extraction modes
Mitigation: every angle metric must carry a `source` tag. Hard gating is allowed only when both baselines report the same approved source.

### Risk 7 — mesh/patch drift is missed because `checkMesh` still passes
Mitigation: fingerprint `constant/polyMesh/*`, patch names, patch counts, and bounding box, not just `checkMesh` stdout.

### Risk 8 — the build path and the run path diverge
Mitigation: generate a machine-readable `stage_plan.json` from the builder and use it as the Phase 0 runner’s source of truth.

### Risk 9 — field comparison fails because output format changed, not numerics
Mitigation: apply a non-physics I/O normalization overlay for Phase 0: ASCII, no compression, fixed precision.

### Risk 10 — stale or conflicting external sources are treated as release contracts
Mitigation: treat the SPUMA arXiv paper and mutable wiki as orientation, then verify actual checked-out code and actual env probe results before acting on them.

---

## 5. Phase-by-phase implementation specification

Only **Phase 0** is expanded exhaustively in this run, per scope. Cross-phase sequencing, centralized definitions, support scope, toolchain pins, and later-phase acceptance policy belong in the master roadmap and companion machine-readable contracts produced alongside Phase 0; this file records only the local handoff conditions that later phases consume.

## Phase 0 — Freeze the reference problem

### Purpose

Freeze a reproducible, implementation-grade CPU reference contract for the pressure-swirl workflow before any SPUMA/CUDA work starts.

### Why this phase exists

The nozzle target is not a trivial linear-solve benchmark. Its algorithmic reference is the transient VOF/MULES/PIMPLE family, and the cited solver docs show explicit alpha prediction, momentum prediction, pressure correction, and surface-tension paths. At the same time, SPUMA’s published baseline is v2412 and its published feature state still excludes multiphase. That means correctness must be frozen before the runtime substrate changes.

Repo findings make this phase concrete rather than abstract: the repo already has a staged VOF precondition workflow, real nozzle geometry generation, real startup seeding, real feature extraction, and an existing environment probe. The highest-risk hidden coupling today is that generated scripts and the generic runner are hard-wired to `/opt/openfoam12/etc/bashrc`, which will break a v2412 migration even before any CFD numerics are compared. Phase 0 exists to remove those couplings and freeze the real problem definition.
Program scope is frozen here as well: milestone-1 GPU acceptance starts at transient solve + device startup seeding, while the steady precondition / `potentialFoam` path remains an allowed CPU pre-run utility unless a later explicit milestone promotes it into GPU acceptance.

### Entry criteria

1. Repo checkout is clean and tests pass in the Python environment for existing pure-Python paths.
2. A working Baseline A OpenFOAM environment is available.
3. A Baseline B environment is available or at least installable. Canonical comparison line: SPUMA/v2412 CPU. Contingency-only freeze fallback: stock OpenFOAM-v2412 CPU if the canonical line cannot execute after environment and documented backend adjustments.
4. The repo’s existing `build_pressure_swirl_case.py`, `extract_openfoam_case_features.py`, `openfoam_env_probe.py`, and `cfd/smoke/damBreak_amgx` asset are present.
5. Storage exists for Phase 0 artifacts outside version control.
6. Default human assumptions are accepted unless overridden:
   - R0 hard gate = `57-28 @ 1000 psi`
   - milestone-1 semantic target / hard acceptance line = algebraic `incompressibleVoF` / explicit MULES / PIMPLE
   - optional CPU-only shadow rows may retain `geometric_iso` / `interIsoFoam` for historical comparison, but they are outside hard milestone-1 acceptance
   - R0 startup defaults = `startup_fill_extension_d=0.0`, `air_core_seed_radius_d=1.0` until a human approves a different frozen row
   - Baseline B canonical comparison line = SPUMA/v2412 CPU; stock-v2412 CPU is contingency-only freeze fallback if the canonical line is blocked.

### Exit criteria

Phase 0 exits only when all of the following exist:

1. Dedicated Phase 0 reference manifests are committed, including the hard-gate algebraic line and any optional CPU-only geometric shadow rows.
2. Dedicated Phase 0 tolerance config is committed.
3. `reference_case_contract.md`, `support_matrix.md`, and `acceptance_manifest.md` are committed.
4. Baseline A environment probe artifact exists and is archived.
5. Baseline B environment probe artifact exists and is archived.
6. R2 is built, run, extracted, and validated on both baselines, or a documented backend/family exception exists.
7. `R1-core` is built, run, extracted, and validated on both baselines.
8. R1 is built, run, extracted, and validated on both baselines.
9. R0 is built, run, extracted, and validated on both baselines.
10. A compare report exists for:
   - Baseline A vs Baseline B
   - each baseline vs manufacturing sanity truth where applicable
11. Hard gates pass, or any non-passing item is explicitly classified as a human-approved split baseline with rationale.
12. Full artifact bundle exists:
   - manifests
   - probe outputs
   - `reference_case_contract.md`
   - `support_matrix.md`
   - `acceptance_manifest.md`
   - case metadata
   - stage plans
   - mesh/patch fingerprints
   - normalized config snapshots
   - field signatures
   - extracted metrics
   - compare report
   - logs
13. Human sign-off is recorded.

### Goals

1. Separate current-line drift from target-line drift.
2. Separate solver-family resolution from requested solver intent.
3. Freeze real nozzle startup conditioning, not simplified startup.
4. Freeze mesh and patch topology.
5. Freeze resolved numerics, not symbolic profiles.
6. Produce machine-readable artifacts that later GPU phases can consume automatically.
7. Establish a reduced developer case that is cheap enough to rerun routinely.
8. Establish a generic VOF verification case independent of nozzle-specific post-processing.
9. Ensure Phase 0 tooling is environment-neutral and portable across Baseline A and B.
10. Prevent silent fallback behavior.

### Non-goals

1. No GPU execution.
2. No CUDA, SPUMA kernel, AmgX performance, or graph work.
3. No multi-rank or MPI runs.
4. No solver retuning to “make Baseline B match.”
5. No change to numerical method beyond:
   - environment decoupling
   - output-format normalization
   - documented pressure-backend fallback when required to run
6. No geometric/physics redesign of the nozzle.
7. No attempt to declare manufacturing truth “solved.”
8. No attempt to choose final production GPU pressure backend.
9. No attempt to make R0 fast.

### Technical background

`foamRun` is the modular front end in the relevant OpenFOAM family, and `incompressibleVoF` is the correct algorithmic reference class for a two-incompressible-fluid, single-mixture-momentum VOF workflow. The documented class interface includes `alphaPredictor`, `momentumPredictor`, `pressureCorrector`, `surfaceTensionForce`, and cached state such as `rAU`. That is the solver family Phase 0 must preserve semantically even if the exact executable differs across baselines.

MULES is documented as an explicit multidimensional limiter, and the OpenFOAM VOF path explicitly exposes alpha sub-cycling. That matters for Phase 0 because it means correctness is path-dependent: startup conditioning, time-step controls, limiter counts, and alpha-correction counts all influence the reference behavior. Freezing only mesh and final flow rate would be insufficient.

SPUMA is the intended future runtime base, but its currently published feature state still excludes multiphase solvers and its current published supported solver list does not include `incompressibleVoF`. That makes Phase 0 a CPU freeze phase, not a GPU bring-up phase in disguise.

Repo findings add the implementation-critical details:

- `build_pressure_swirl_case.py` already applies resolved VOF stability overrides.
- `direct_slot_v2` currently resolves to:
  - `nAlphaCorr=6`
  - `nAlphaSubCycles=8`
  - `nLimiterIter=12`
  - `nOuterCorrectors=3`
  - `nCorrectors=5`
  - `nNonOrthogonalCorrectors=1`
  - `U` relaxation `0.5`
  - `p_rgh` relaxation `0.2`
  - interface compression `0.1`
- the template defaults are materially lighter than those values.
- `vof_transient_preconditioned` writes a steady precondition stage and a staged run script.
- the generated scripts currently hard-code `/opt/openfoam12`.
- `extract_openfoam_case_features.py` already returns a meaningful metric set and distinguishes `geometric_a` from `velocity_proxy`.

### Research findings relevant to this phase

1. **Sourced fact:** SPUMA’s published comparison target is OpenFOAM-v2412.  
2. **Sourced fact:** SPUMA’s published feature state still excludes multiphase solvers.  
3. **Sourced fact:** SPUMA’s current published GPU-support list excludes `incompressibleVoF`.  
4. **Sourced fact:** `incompressibleVoF` is the correct algorithmic reference family for a transient VOF/PIMPLE nozzle workflow.  
5. **Sourced fact:** MULES is explicit and alpha sub-cycling exists in the relevant VOF path.  
6. **Repo finding:** the current builder already writes `case_meta.json`, `rheology_contract.json`, and `build_log.json`.  
7. **Repo finding:** the current environment probe exists but is incomplete for Phase 0 migration needs; it does not currently record `foamRun`, `incompressibleVoF`, `setFields`, `potentialFoam`, or backend library availability.  
8. **Repo finding:** the current generated scripts and generic CPU runner hard-code `/opt/openfoam12/etc/bashrc`.  
9. **Repo finding:** the extractor already emits source-typed spray-angle metrics and continuity-derived imbalance metrics.  
10. **Repo finding:** the existing `damBreak_amgx` smoke logs show `foamRun -solver incompressibleVoF` and persistent “LDU matrix arrays to CSR, then values only” behavior, which makes it an excellent R2 smoke anchor.  
11. **Staleness note:** the SPUMA paper is an arXiv preprint and the GPU-support wiki is mutable. Treat both as orientation, then verify the actual checked-out code and actual environment probe before acting.  
12. **Conflict note:** the algorithmic docs and the target runtime line are not the same distribution line. Phase 0 must probe actual executable behavior rather than assuming CLI parity.

### Design decisions

#### D0.1 — Freeze **case intent** separately from **execution resolution**

**Repo finding:** requested VOF mode and resolved executable are already distinct in the builder.

**Engineering inference:** comparisons must distinguish “same nozzle physics intent” from “same exact solver executable/back-end.”

**Recommendation:** define two objects:
- `ReferenceCaseIntent` = nozzle/physics/startup/numerics intent
- `ExecutionResolution` = baseline environment, resolved executable, resolved backend, actual stage commands

This is the core Phase 0 data model.

---

#### D0.2 — R0 hard-gate default is a fully specified `57-28 @ 1000 psi` algebraic reference; geometric VOF remains shadow-only

**Repo finding:** manufacturing truth exists for `57-28 @ 1000 psi`, and the handoff identifies it as a current sentinel. Manufacturing row values are:
- orifice diameter `1.09 mm`
- slot count `6`
- nominal slot width `0.64 mm`
- angle A target `53 deg`
- angle B at 1000 psi `33 deg`
- water capacity `54.2 gph`

**Engineering inference:** this is the best first hard gate, but the hard-gate row must align with the milestone-1 canonical solver family rather than a historical geometric request mode.

**Recommendation:** create `cfd/manifests/reference_phase0/r0_57_28_1000_full360.json` as the hard-gate algebraic reference with these defaults. If historical comparison against `geometric_iso` / `interIsoFoam` is still useful, commit it separately as a CPU-only shadow row and keep it out of hard milestone-1 acceptance:

- `case_id = "phase0_r0_57_28_1000_full360_v1"`
- `insert_size_no = 57`
- `core_size_no = 28`
- `nozzle_key = "57-28"`
- `pressure_psi = 1000`
- `hydraulic_cfd_mode = "vof_transient_preconditioned"`
- `vof_solver_mode` = the builder's explicit algebraic `incompressibleVoF`-family mode name
- `geometry_mode = "sk_standard_shared_body_with_coeffs_v2"` unless a validated repo row exists; fallback `vendor_inferred_sk_v1` requires human sign-off
- `mesh_full_360 = 1`
- `hydraulic_domain_mode = "external_nearfield"`
- `mesh_resolution_scale = 4.5`
- `near_field_radius_d = 10`
- `near_field_length_d = 20`
- `steady_end_time_iter = 10`
- `steady_write_interval_iter = 5`
- `steady_turbulence_model = "kOmegaSST"`
- `vof_turbulence_model = "laminar"`
- `delta_t_s = 1e-8`
- `write_interval_s = 1e-8`
- `end_time_s = 1e-7`
- `max_co = 0.05`
- `max_alpha_co = 0.01`
- `n_alpha_corr = 6`
- `n_alpha_subcycles = 8`
- `n_alpha_limiter_iter = 12`
- `pimple_n_outer_correctors = 3`
- `pimple_n_correctors = 5`
- `pimple_n_nonorthogonal_correctors = 1`
- `equation_relaxation_u = 0.5`
- `equation_relaxation_p_rgh = 0.2`
- `alpha_interface_compression = 0.1`
- `rho_phi_u_scheme = "Gauss upwind"`
- `startup_fill_extension_d = 0.0`
- `air_core_seed_radius_d = 1.0`

Because the current archive quotes `geometric_iso` explicitly but does not quote the builder's exact symbolic manifest value for the algebraic path, the committed Phase 0 row must freeze that explicit algebraic mode name in the manifest and record `resolved_vof_solver_exec = incompressibleVoF` in `case_meta.json`. That is a naming-resolution detail, not an open solver-family decision.

This package freezes the default startup row at `startup_fill_extension_d = 0.0` and `air_core_seed_radius_d = 1.0`. Any future change requires an explicit revision to `reference_case_contract.md` plus regenerated compare artifacts; it is not an open Phase 0 decision.

---

#### D0.3 — R1 default is a reduced internal-only nozzle case, not a second full plume case

Phase 0 also freezes a separate `R1-core` reduced case for the validation ladder. `R1-core` must use the canonical algebraic solver family and only the frozen generic BC/scheme subset; it is the Phase-5-friendly reduced case consumed before nozzle-specific Phase 6 BC/startup work. `R1` remains the reduced nozzle case that retains the nozzle-specific startup and boundary behavior.

**Engineering inference:** the fast developer baseline should stress startup conditioning and internal air-core behavior, not spray-angle truth.

**Recommendation:** create `cfd/manifests/reference_phase0/r1_57_28_1000_internal.json` with:
- same nozzle and physical properties as R0
- same requested solver family and same resolved numerics
- `hydraulic_domain_mode = "internal_only"`
- `mesh_full_360 = 1`
- `mesh_resolution_scale = 2.0`
- `delta_t_s = 1e-8`
- `write_interval_s = 1e-8`
- `end_time_s = 2e-8`
- same steady precondition settings
- same startup fill/seed values

This makes R1 fast enough to use for routine migration checks while keeping the same startup and limiter logic.

`R1-core` is frozen explicitly in `reference_case_contract.md` and `support_matrix.md` and must not depend on Phase 6-only BC/startup support.

---

#### D0.4 — R2 default is the existing `damBreak_amgx` asset with an allowed CPU fallback

**Repo finding:** the smoke asset already exists and already exercises `foamRun -solver incompressibleVoF`.

**Engineering inference:** R2 should stay close to an already archived asset to reduce bring-up time.

**Recommendation:** use `cfd/smoke/damBreak_amgx` as R2 intent. If Baseline B cannot run the external pressure backend, clone it into `cfd/smoke/damBreak_reference_cpu` with only a documented pressure-backend substitution. Do not otherwise change mesh, physics, or solver family.

---

#### D0.5 — use JSON-row manifests for Phase 0, not only shared DOE CSVs

**Engineering inference:** the builder already accepts a single JSON manifest row directly, and Phase 0 needs explicit, fully materialized rows rather than inference from broader DOE files.

**Recommendation:** commit four canonical JSON rows plus one optional shadow row:
- `r0_57_28_1000_full360.json`
- `r1_core_phase5_friendly.json` (or equivalently named frozen reduced-core row)
- `r1_57_28_1000_internal.json`
- `r2_dambreak_reference.json`
- optional `r0_shadow_40_28_1000_full360.json`

---

#### D0.6 — extend existing `case_meta.json`; do not invent a parallel metadata format

**Repo finding:** `build_pressure_swirl_case.py` already emits `case_meta.json`.

**Engineering inference:** modifying that file is lower risk than creating a second authoritative metadata object.

**Recommendation:** extend `case_meta.json` to include all Phase 0 resolved fields listed below, and add `stage_plan.json` plus `reference_freeze_overlay.json`.

---

#### D0.7 — add a non-physics I/O normalization overlay for Phase 0

**Engineering inference:** deterministic field hashing and parsing are much easier if outputs are ASCII and uncompressed.

**Recommendation:** after build and before run, apply a Phase 0 overlay that sets:
- `writeFormat ascii`
- `writeCompression off`
- `writePrecision 12`
- `timePrecision 12`

This overlay must be recorded as provenance and must not change numerics.

---

#### D0.8 — remove hard-coded environment sourcing from generated scripts

**Repo finding:** generated scripts embed `/opt/openfoam12/etc/bashrc`.

**Engineering inference:** Baseline B will fail or silently run the wrong line if this stays in place.

**Recommendation:** generated scripts must become environment-neutral. The Phase 0 orchestrator owns environment sourcing.

---

#### D0.9 — add hard-gate vs review-gate classification to metrics

**Engineering inference:** some metrics are already robust (`water_flow_cfd_gph`, `cd_mass_cfd`, `mass_imbalance_pct`), while others depend on solver family or extractor source (`spray_angle_cfd_deg`).

**Recommendation:** define:
- **hard gates**: solver stability, patch schema, mesh fingerprint, startup provenance, flow, Cd, mass imbalance
- **review gates**: air-core ratio, swirl proxy, recirculation fraction, spray angle, pressure-drop proxy
- **informational**: raw walltime, RSS, raw field hashes across unlike baselines

---

#### D0.10 — pressure drop is advisory in Phase 0 unless its definition is human-approved

**Engineering inference:** current nozzle runs use `p_rgh`, and a physically meaningful nozzle pressure-drop metric may require hydrostatic interpretation and patch-mean conventions that are not yet frozen.

**Recommendation:** add a `delta_p_rgh_mean_pa` or equivalent patch-average proxy only as a review metric in Phase 0. Do not make it a hard exit criterion until a senior CFD reviewer approves the exact definition.

---

#### D0.11 — archive manufacturing truth as external context, not primary equivalence

**Repo finding:** the repo includes manufacturing water capacity and angle truth.

**Engineering inference:** the migration freeze should primarily preserve current computational behavior, not re-open full model calibration.

**Recommendation:** compare both baselines to manufacturing truth, but use that as advisory context. The hard equivalence criterion is Baseline A vs Baseline B.

### Alternatives considered

1. **Single-baseline freeze only.** Rejected. It cannot isolate migration drift.
2. **Freeze directly on SPUMA GPU.** Rejected. Multiphase is not yet a supported SPUMA feature.
3. **Use only R2 damBreak as the reference.** Rejected. It does not exercise nozzle startup, air-core, compact plume, or nozzle BC logic.
4. **Use only manufacturing truth as the reference.** Rejected. That would conflate calibration error with migration error.
5. **Require exact external pressure-backend parity on both baselines.** Rejected. That makes Phase 0 brittle for the wrong reason.
6. **Rebuild builder/extractor logic from scratch for Phase 0.** Rejected. High divergence risk.
7. **Use `40-28` as the only hard gate.** Rejected. Too aggressive for the first freeze.
8. **Keep current hard-coded `/opt/openfoam12` behavior and patch around it manually.** Rejected. Too error-prone.

### Interfaces and dependencies

#### Existing modules to extend

- `scripts/cfd/openfoam_env_probe.py`
- `scripts/cfd/build_pressure_swirl_case.py`
- `scripts/cfd/extract_openfoam_case_features.py`
- `scripts/cfd/run_manifest.py`
- existing tests under `tests/`

#### New Phase 0 modules to add

- `scripts/cfd/reference_problem/models.py`
- `scripts/cfd/reference_problem/overlays.py`
- `scripts/cfd/reference_problem/fingerprints.py`
- `scripts/cfd/reference_problem/field_signatures.py`
- `scripts/cfd/reference_problem/runner.py`
- `scripts/cfd/reference_problem/compare.py`
- `scripts/cfd/reference_problem/report.py`
- `scripts/cfd/freeze_reference_problem.py`
- `scripts/cfd/compare_reference_problem.py`

#### External dependencies

- Python environment already used by repo
- an OpenFOAM environment for Baseline A
- an OpenFOAM-v2412-compatible or SPUMA CPU environment for Baseline B
- shell utilities: `bash`, `checkMesh`, `foamRun`, `foamListTimes`, `setFields`, `potentialFoam`

#### CLI contract to implement

`freeze_reference_problem.py` must accept at least:

- `--baseline-name`
- `--openfoam-bashrc`
- `--case-spec` (JSON row path)
- `--template-dir`
- `--artifact-root`
- `--strict-solver-family {0|1}`
- `--allow-pressure-backend-fallback {0|1}`
- `--apply-reference-io-overlay {0|1}`
- `--mode {probe,build,run,extract,compare,freeze}`

`compare_reference_problem.py` must accept at least:

- `--lhs-artifact-root`
- `--rhs-artifact-root`
- `--tolerances-yaml`
- `--report-out`
- `--json-out`

### Data model / memory model

Phase 0 is host-only. There is no device memory model yet. The persistent state is filesystem-based and must be explicitly versioned.

#### Core objects

```python
@dataclass(frozen=True)
class BaselineEnv:
    name: str
    bashrc_path: Path | None
    openfoam_family: str
    openfoam_version_string: str
    commands: dict[str, str | None]          # foamRun, incompressibleVoF, interIsoFoam, setFields, checkMesh, potentialFoam, foamListTimes
    available_solvers: list[str]
    available_backends: list[str]            # e.g. amgx, petscFoam, native
    notes: list[str]

@dataclass(frozen=True)
class ReferenceCaseIntent:
    case_id: str
    case_kind: Literal["R0", "R1", "R2"]
    manifest_row: dict[str, Any]
    truth_reference: dict[str, float | str | None]   # manufacturing values when applicable
    tolerance_profile: str

@dataclass(frozen=True)
class ExecutionResolution:
    baseline_name: str
    requested_vof_solver_mode: str
    resolved_vof_solver_exec: str
    resolved_pressure_backend: str
    hydraulic_cfd_mode: str
    hydraulic_domain_mode: str
    stage_plan: list[dict[str, Any]]

@dataclass(frozen=True)
class BuildFingerprint:
    config_hashes: dict[str, str]
    mesh_hashes: dict[str, str]
    patch_schema: list[dict[str, Any]]
    mesh_counts: dict[str, int]
    bbox: dict[str, float]

@dataclass(frozen=True)
class FieldSignature:
    time_dir: str
    field_name: str
    file_sha256: str
    stats: dict[str, float | int | str]      # min/max/sum/L2/source-format

@dataclass(frozen=True)
class MetricRecord:
    name: str
    value: float | int | str
    unit: str
    source: str
    gate_class: Literal["hard", "review", "info"]

@dataclass(frozen=True)
class FrozenReferenceArtifact:
    env: BaselineEnv
    intent: ReferenceCaseIntent
    resolution: ExecutionResolution
    build_fp: BuildFingerprint
    field_sigs: list[FieldSignature]
    metrics: dict[str, MetricRecord]
    logs: dict[str, Path]
```

#### Mandatory `case_meta.json` additions

Extend the existing emitted file to include:

- `requested_vof_solver_mode`
- `resolved_vof_solver_exec`
- `resolved_pressure_backend`
- `openfoam_bashrc_used`
- `available_commands`
- `mesh_full_360`
- `mesh_resolution_scale`
- `hydraulic_domain_mode`
- `near_field_radius_d`
- `near_field_length_d`
- `steady_end_time_iter`
- `steady_write_interval_iter`
- `steady_turbulence_model`
- `vof_turbulence_model`
- `delta_t_s`
- `write_interval_s`
- `end_time_s`
- `max_co`
- `max_alpha_co`
- resolved direct-slot numerics
- `startup_fill_extension_d`
- `air_core_seed_radius_d_requested`
- `air_core_seed_radius_m_resolved`
- `air_core_seed_cap_applied`
- `fill_radius_m_resolved`
- `fill_z_start_m`
- `fill_z_stop_m`
- `DeltaP_Pa`
- `DeltaP_effective_Pa`
- `check_valve_loss_applied`

#### Mandatory `stage_plan.json`

This file is new. It must contain the environment-neutral ordered stages with arguments, for example:

```json
{
  "case_id": "phase0_r0_57_28_1000_full360_v1",
  "stages": [
    {"name": "buildMesh", "cmd": "bash system/buildMesh.sh"},
    {"name": "checkMesh_build", "cmd": "checkMesh"},
    {"name": "steady_prepare", "cmd": "copy polyMesh into steady_precondition_stage"},
    {"name": "steady_checkMesh", "cmd": "checkMesh", "cwd": "steady_precondition_stage"},
    {"name": "steady_potentialFoam", "cmd": "potentialFoam -writePhi", "cwd": "steady_precondition_stage"},
    {"name": "steady_run", "cmd": "foamRun", "cwd": "steady_precondition_stage"},
    {"name": "copy_precondition_fields", "cmd": "copy latest U k omega nut to 0/"},
    {"name": "setFields", "cmd": "setFields"},
    {"name": "transient_run", "cmd": "foamRun -solver incompressibleVoF"}
  ]
}
```

#### Memory model

- All Phase 0 objects live on host filesystem.
- Full case directories are ephemeral artifacts under an artifact root, not source-controlled.
- Only manifests, tolerance configs, tests, and summary reports are committed.

### Algorithms and control flow

#### Top-level freeze workflow

1. Probe environment.
2. Load case-intent JSON.
3. Build case through existing builder.
4. Apply reference I/O overlay.
5. Capture normalized config snapshot and build fingerprint.
6. Run case using environment-neutral stage runner.
7. Capture stage logs and walltimes.
8. Extract nozzle features or generic VOF features.
9. Compute selected field signatures.
10. Evaluate hard/review/info gates.
11. Archive artifacts.
12. Compare against the sibling baseline and emit report.

#### Case selection logic

- **R0** and **R1** use the pressure-swirl builder.
- **R2** uses the existing static smoke asset, optionally patched only for pressure backend.

#### Solver resolution logic

- `requested_vof_solver_mode` is frozen from manifest.
- `resolved_vof_solver_exec` is what the builder/environment actually resolves.
- Under `--strict-solver-family=1`, mismatch fails the run before execution.
- Under `--strict-solver-family=0`, mismatch is recorded and the case becomes split-baseline review-only.

#### Pressure backend logic

- Detect backend from `fvSolution`, loaded libraries, and environment probe.
- If unavailable, patch to a native CPU fallback and record the entire patched stanza as artifact provenance.
- Do not hide the fallback.

#### Field-signature logic

Selected fields:

- Steady precondition latest time:
  - `U`
  - `k` if present
  - `omega` if present
  - `nut` if present
- Transient latest time:
  - `alpha.water`
  - `U`
  - `p_rgh`
  - `rho` if present
  - `phi` if present

Selected statistics:

- scalar: min, max, sum, mean
- vector: min |U|, max |U|, L2(|U|)
- exact normalized file SHA256

#### Mesh/patch fingerprint logic

Compute and archive:

- SHA256 of:
  - `constant/polyMesh/points`
  - `faces`
  - `owner`
  - `neighbour`
  - `boundary`
- semantic patch list:
  - patch name
  - patch type
  - `nFaces`
  - `startFace`
- mesh counts:
  - cells, faces, points, internal faces, patches
- bounding box

### Required source changes

#### 1. `scripts/cfd/openfoam_env_probe.py`

Extend it to emit:

- exact bashrc used
- `foamRun`, `incompressibleVoF`, `interIsoFoam`, `interFoam`, `setFields`, `checkMesh`, `potentialFoam`, `foamListTimes`, `decomposePar`
- library/backend hints:
  - `libpetscFoam.so`
  - `amgx` config files or linked support if detectable
- `WM_PROJECT`, `WM_PROJECT_VERSION`, `WM_OPTIONS`
- notes on missing commands

Do **not** leave it limited to `interIsoFoam` and `interFoam`.

#### 2. `scripts/cfd/build_pressure_swirl_case.py`

Modify to:

- store `requested_vof_solver_mode`
- store resolved startup quantities
- emit `stage_plan.json`
- emit `resolved numerics` explicitly, not only profile names
- remove hard-coded environment sourcing from generated scripts
- optionally emit a minimal `reference_contract_build.json`

Do **not** create a second builder.

#### 3. `scripts/cfd/run_manifest.py`

Modify to:

- stop hard-coding `/opt/openfoam12/etc/bashrc`
- accept environment prefix or bashrc override
- expose a path to use the Phase 0 stage runner cleanly
- preserve current behavior for Baseline A where possible

#### 4. `scripts/cfd/extract_openfoam_case_features.py`

Modify to:

- add `--json-out`
- emit explicit `metric_source` metadata for each major metric
- emit time-window metadata already implicit in flow extraction
- emit angle-source metadata explicitly in JSON
- optionally add advisory `delta_p_rgh_mean_pa` if a stable definition is implemented
- avoid forcing CSV-only output

#### 5. New `scripts/cfd/reference_problem/*`

Add:

- manifest loading
- overlay patching
- mesh/patch fingerprinting
- field signature computation
- baseline compare engine
- report generation
- R2 generic VOF validator

#### 6. Tests

Add or extend tests for:

- env probe command detection
- no hard-coded openfoam12 path remains in generated scripts
- stage plan emission
- reference I/O overlay
- field parser / field signature hashing
- compare engine gate rules
- angle-source mismatch handling
- pressure-backend fallback tagging

### Proposed file layout and module boundaries

```text
cfd/
  manifests/
    reference_phase0/
      r0_57_28_1000_full360.json
      r1_core_phase5_friendly.json
      r1_57_28_1000_internal.json
      r2_dambreak_reference.json
      r0_shadow_40_28_1000_full360.json        # optional

configs/
  cfd/
    reference_phase0_tolerances.yaml

scripts/
  cfd/
    openfoam_env_probe.py                       # extend
    build_pressure_swirl_case.py                # extend
    extract_openfoam_case_features.py           # extend
    run_manifest.py                             # extend
    freeze_reference_problem.py                 # new CLI
    compare_reference_problem.py                # new CLI
    reference_problem/
      __init__.py
      models.py
      overlays.py
      fingerprints.py
      field_signatures.py
      runner.py
      compare.py
      report.py

docs/
  reference_problem_phase0.md                   # new

tests/
  test_cfd_reference_problem_phase0.py          # new
  test_cfd_openfoam_extract.py                  # extend
  test_cfd_sk_geometry.py                       # extend
  test_cfd_rheology_contract_and_probe.py       # extend
```

### Pseudocode

#### Pseudocode 1 — freeze orchestration

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class FreezeConfig:
    baseline_name: str
    bashrc_path: Path | None
    case_spec_path: Path
    template_dir: Path
    artifact_root: Path
    strict_solver_family: bool
    allow_pressure_backend_fallback: bool
    apply_reference_io_overlay: bool

def freeze_case(cfg: FreezeConfig) -> FrozenReferenceArtifact:
    # 1. Probe actual environment; this is the source of truth for commands.
    env = probe_environment(cfg.bashrc_path)
    write_json(cfg.artifact_root / "probe.json", env)

    # 2. Load explicit case intent.
    intent = load_case_intent(cfg.case_spec_path)

    # 3. Build case using existing authoritative builder.
    case_dir = cfg.artifact_root / "case"
    build_result = build_case_from_intent(intent, env, case_dir, cfg.template_dir)

    # 4. Apply non-physics overlay for deterministic comparison.
    if cfg.apply_reference_io_overlay:
        overlay_meta = apply_reference_io_overlay(case_dir)
        write_json(cfg.artifact_root / "reference_freeze_overlay.json", overlay_meta)

    # 5. Capture resolved metadata and check solver family policy.
    case_meta = read_json(case_dir / "case_meta.json")
    resolution = resolve_execution(case_meta, env)
    if cfg.strict_solver_family and solver_family_mismatch(intent, resolution):
        raise RuntimeError("Requested and resolved solver families differ")

    # 6. If backend unavailable, patch pressure backend only if explicitly allowed.
    if backend_unavailable(resolution, env):
        if not cfg.allow_pressure_backend_fallback:
            raise RuntimeError("Pressure backend unavailable and fallback disabled")
        patch_meta = patch_pressure_backend_to_cpu_native(case_dir, env)
        write_json(cfg.artifact_root / "pressure_backend_patch.json", patch_meta)
        case_meta = refresh_case_meta_with_backend(case_dir, case_meta, patch_meta)
        resolution = resolve_execution(case_meta, env)

    # 7. Fingerprint config and mesh before run.
    config_fp = fingerprint_case_configuration(case_dir)
    write_json(cfg.artifact_root / "build_fingerprint.json", config_fp.as_dict())

    # 8. Execute stage plan under the chosen environment.
    stage_plan = read_json(case_dir / "stage_plan.json")
    run_meta = run_stage_plan(
        case_dir=case_dir,
        env=env,
        stage_plan=stage_plan,
        logs_dir=cfg.artifact_root / "logs",
    )
    write_json(cfg.artifact_root / "run_meta.json", run_meta)

    # 9. Extract metrics.
    if intent.case_kind in {"R0", "R1"}:
        metrics = extract_nozzle_metrics(case_dir, solver_log=run_meta["solver_log"])
    else:
        metrics = extract_r2_vof_metrics(case_dir, solver_log=run_meta["solver_log"])
    write_json(cfg.artifact_root / "metrics.json", metrics)

    # 10. Compute selected field signatures.
    field_sigs = compute_field_signatures(case_dir, intent.case_kind)
    write_json(cfg.artifact_root / "field_signatures.json", field_sigs)

    # 11. Evaluate gates local to this baseline.
    local_verdict = evaluate_single_baseline_invariants(intent, metrics, field_sigs)
    write_json(cfg.artifact_root / "baseline_verdict.json", local_verdict)

    return FrozenReferenceArtifact(
        env=env,
        intent=intent,
        resolution=resolution,
        build_fp=config_fp,
        field_sigs=field_sigs,
        metrics=metrics,
        logs=run_meta["logs"],
    )
```

#### Pseudocode 2 — environment-neutral stage runner

```python
def run_stage_plan(case_dir: Path, env: BaselineEnv, stage_plan: dict[str, Any], logs_dir: Path) -> dict[str, Any]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stage_results: list[dict[str, Any]] = []

    for stage in stage_plan["stages"]:
        stage_name = stage["name"]
        cmd = stage["cmd"]
        cwd = case_dir / stage.get("cwd", ".")
        log_path = logs_dir / f"{stage_name}.log"

        # Runner owns environment sourcing; stage scripts do not.
        wrapped_cmd = wrap_with_bashrc(env.bashrc_path, f"cd {sh_quote(cwd)} && {cmd}")

        wall_s, rc = run_and_capture(
            command=wrapped_cmd,
            stdout_stderr_path=log_path,
            time_verbose_path=logs_dir / f"{stage_name}.time.txt",
        )

        stage_results.append(
            {
                "stage": stage_name,
                "cwd": str(cwd),
                "rc": rc,
                "wall_s": wall_s,
                "log": str(log_path),
            }
        )

        if rc != 0:
            raise RuntimeError(f"Stage {stage_name} failed")

        # Numerically important integrity checks.
        if stage_name == "copy_precondition_fields":
            assert_precondition_fields_present(case_dir, ["U", "k", "omega", "nut"])

        if stage_name == "setFields":
            assert_alpha_field_exists(case_dir / "0" / "alpha.water")

    solver_log = choose_primary_solver_log(stage_results)
    return {"stages": stage_results, "solver_log": solver_log, "logs": {r["stage"]: r["log"] for r in stage_results}}
```

#### Pseudocode 3 — cross-baseline comparison

```python
def compare_baselines(lhs: FrozenReferenceArtifact, rhs: FrozenReferenceArtifact, tol: ToleranceSet) -> ComparisonResult:
    findings = []
    hard_fail = False

    # 1. Exact provenance comparisons
    findings += compare_exact("patch_schema", lhs.build_fp.patch_schema, rhs.build_fp.patch_schema, hard=True)
    findings += compare_exact("requested_solver_mode", lhs.intent.manifest_row["vof_solver_mode"], rhs.intent.manifest_row["vof_solver_mode"], hard=True)

    # 2. Resolved execution differences are recorded, not always failed.
    findings += compare_exact("resolved_vof_solver_exec", lhs.resolution.resolved_vof_solver_exec, rhs.resolution.resolved_vof_solver_exec, hard=False)
    findings += compare_exact("resolved_pressure_backend", lhs.resolution.resolved_pressure_backend, rhs.resolution.resolved_pressure_backend, hard=False)

    # 3. Hard metric comparisons
    for metric_name in ("solver_ok", "mass_imbalance_pct", "water_flow_cfd_gph", "cd_mass_cfd"):
        findings += compare_metric(lhs.metrics[metric_name], rhs.metrics[metric_name], tol.rules[metric_name])

    # 4. Review-only metrics with source constraints
    for metric_name in ("spray_angle_cfd_deg", "air_core_area_ratio", "swirl_number_proxy", "recirculation_fraction"):
        rule = tol.rules[metric_name]
        if rule.requires_same_source:
            lhs_source = lhs.metrics[f"{metric_name}_source"].value if f"{metric_name}_source" in lhs.metrics else lhs.metrics.get("spray_angle_source", MetricRecord("", "", "", "", "info")).value
            rhs_source = rhs.metrics[f"{metric_name}_source"].value if f"{metric_name}_source" in rhs.metrics else rhs.metrics.get("spray_angle_source", MetricRecord("", "", "", "", "info")).value
            if lhs_source != rhs_source:
                findings.append(review_only(metric_name, f"source mismatch: {lhs_source} vs {rhs_source}"))
                continue
        findings += compare_metric(lhs.metrics[metric_name], rhs.metrics[metric_name], rule)

    for f in findings:
        if f.gate_class == "hard" and not f.passed:
            hard_fail = True

    return ComparisonResult(
        passed=not hard_fail,
        findings=findings,
        summary=make_summary(findings),
    )
```

### Step-by-step implementation guide

1. **Create canonical Phase 0 case-spec files.**  
   Modify: add `cfd/manifests/reference_phase0/*.json`.  
   Why: Phase 0 needs explicit, stable, fully materialized inputs.  
   Expected output: four JSON rows and one optional shadow row.  
   Verify: JSON schema check passes and each file contains all required resolved controls.  
   Likely breakages: missing geometry-mode or startup fields.

2. **Add a Phase 0 tolerance config and shared acceptance scaffold.**  
   Modify: add `configs/cfd/reference_phase0_tolerances.yaml` plus `acceptance_manifest.md`.  
   Why: the compare engine and later phase gates must be data-driven, not hard-coded or redefined per phase.  
   Expected output: YAML with case-kind-specific hard/review/info rules plus the centralized acceptance cases/fallback classes consumed later.  
   Verify: a unit test loads it and validates required rule names.  
   Likely breakages: inconsistent metric naming vs extractor output.

3. **Harden `openfoam_env_probe.py`.**  
   Modify: extend command/backend detection and expose `--bashrc` override.  
   Why: Baseline B cannot rely on current-path heuristics alone.  
   Expected output: richer `probe.json`.  
   Verify: probe run on Baseline A and B shows commands and family correctly.  
   Likely breakages: shells that expose `WM_PROJECT_VERSION` but not `foamVersion`.

4. **Remove hard-coded OpenFOAM-12 sourcing from generated scripts.**  
   Modify: `build_pressure_swirl_case.py` and `run_manifest.py`.  
   Why: this is a migration blocker.  
   Expected output: generated scripts that assume the caller already sourced the environment.  
   Verify: grep the repo and built cases for `/opt/openfoam12/etc/bashrc`; only tests/docs may contain it after change.  
   Likely breakages: existing tests expecting the old script text.

5. **Emit `stage_plan.json` from the builder.**  
   Modify: `build_pressure_swirl_case.py`.  
   Why: the Phase 0 runner needs an environment-neutral source of truth.  
   Expected output: one `stage_plan.json` per built case.  
   Verify: R0 build artifact contains ordered stages including steady precondition and transient run.  
   Likely breakages: command resolution for `interIsoFoam` vs `foamRun -solver incompressibleVoF`.

6. **Extend `case_meta.json` with resolved Phase 0 fields.**  
   Modify: `build_pressure_swirl_case.py`.  
   Why: current metadata is insufficient for strict equivalence.  
   Expected output: `case_meta.json` contains resolved numerics, startup values, effective delta-P, and resolved solver/back-end fields.  
   Verify: unit test reads the file and checks required keys.  
   Likely breakages: forgetting to record derived startup cap values.

7. **Add reference I/O normalization overlay.**  
   Modify: new `reference_problem/overlays.py`.  
   Why: stable field parsing and hashing require deterministic output format.  
   Expected output: `reference_freeze_overlay.json` and patched `controlDict`.  
   Verify: built case writes ASCII, uncompressed fields.  
   Likely breakages: accidental numerical changes if overlay touches runtime controls other than I/O format.

8. **Implement config and mesh fingerprinting.**  
   Modify: new `reference_problem/fingerprints.py`.  
   Why: `checkMesh` alone is not enough.  
   Expected output: `build_fingerprint.json`.  
   Verify: two same-baseline builds of R1 yield identical fingerprints.  
   Likely breakages: normalization bugs in file hashing.

9. **Implement selected field-signature logic.**  
   Modify: new `reference_problem/field_signatures.py`.  
   Why: later GPU phases need stable field-level anchors.  
   Expected output: `field_signatures.json`.  
   Verify: latest-time `alpha.water`, `U`, `p_rgh` signatures are emitted for R0/R1.  
   Likely breakages: OpenFOAM field parser assumptions about ASCII layout.

10. **Extend the extractor to emit JSON and source metadata cleanly.**  
    Modify: `extract_openfoam_case_features.py`.  
    Why: CSV-only output is too weak for automated gating.  
    Expected output: `metrics.json` plus optional CSV compatibility.  
    Verify: existing extractor tests still pass and new JSON tests pass.  
    Likely breakages: downstream scripts assuming CSV only.

11. **Add an R2-specific generic VOF validator.**  
    Modify: new `reference_problem/runner.py` or `compare.py`.  
    Why: nozzle-specific metrics do not apply to damBreak.  
    Expected output: `r2_metrics.json` with alpha-boundedness and conservation stats.  
    Verify: R2 computes `alpha_integral_change_pct`, `alpha_min`, `alpha_max`, `solver_ok`.  
    Likely breakages: field parser handling of damBreak alpha field.

12. **Implement the Phase 0 runner CLI.**  
    Modify: new `scripts/cfd/freeze_reference_problem.py`.  
    Why: existing generic orchestration does not produce Phase 0-grade provenance.  
    Expected output: `probe/build/run/extract/freeze` modes.  
    Verify: `--mode probe` and `--mode build` work independently before full run.  
    Likely breakages: environment wrapping and cwd handling.

13. **Freeze Baseline A R2 first.**  
    Modify: none beyond using new CLI.  
    Why: R2 is the fastest integration smoke.  
    Expected output: full artifact bundle for R2 Baseline A.  
    Verify: alpha bounds and mass conservation pass.  
    Likely breakages: backend assumptions in the smoke case.

14. **Freeze Baseline A `R1-core` next, then R1.**  
    Why: `R1-core` exercises the Phase-5-friendly reduced algebraic path before the nozzle-specific reduced case, and R1 then exercises the nozzle-specific builder and startup path cheaply.  
    Expected output: full artifact bundles for `R1-core` Baseline A and R1 Baseline A.  
    Verify: `R1-core` matches the frozen generic BC/scheme subset, R1 hard gates pass, and two repeated builds give identical fingerprints where applicable.  
    Likely breakages: reduced-case support classification, startup staging, and field-copy logic.

15. **Freeze Baseline A R0.**  
    Why: this becomes the control truth.  
    Expected output: full artifact bundle for R0 Baseline A.  
    Verify: hard gates pass and manufacturing sanity looks reasonable.  
    Likely breakages: compact plume geometry or solver-family availability.

16. **Stand up Baseline B probe and build-only validation.**  
    Why: catch environment/CLI drift before long runs.  
    Expected output: probe artifact and built but not yet executed R0/R1/R2 cases.  
    Verify: no hard-coded OF12 path remains, generated scripts run under Baseline B.  
    Likely breakages: `foamRun` packaging, solver availability, backend libraries.

17. **Freeze Baseline B R2, then `R1-core`, then R1, then R0.**  
    Why: shortest-to-longest progression catches drift quickly while preserving the `R2 -> R1-core -> R1 -> R0` validation ladder.  
    Expected output: full artifact bundles for all four.  
   Verify: compare engine produces a verdict without missing required fields.  
   Likely breakages: `interIsoFoam` availability, backend fallback, source-tag mismatch on angle.

18. **Generate compare reports and obtain human sign-off.**  
    Modify: new `compare_reference_problem.py` and `report.py`.  
    Why: later phases need a stable, reviewable contract.  
    Expected output: Markdown and JSON reports.  
    Verify: each hard gate is marked pass/fail/review with rationale.  
    Likely breakages: mismatched metric names or missing source tags.

### Instrumentation and profiling hooks

Phase 0 is CPU-only, so instrumentation is lightweight but rigorous:

- capture stdout/stderr per stage
- capture `/usr/bin/time -v` per stage
- archive `checkMesh` output
- archive `foamListTimes -latestTime`
- archive selected copied precondition fields provenance
- archive extraction stdout/stderr
- archive exact executed command strings

Mandatory stage names:

- `buildMesh`
- `checkMesh_build`
- `steady_prepare`
- `steady_checkMesh`
- `steady_potentialFoam`
- `steady_run`
- `copy_precondition_fields`
- `setFields`
- `transient_run`
- `extract_features`

These names are not arbitrary. Later GPU phases should preserve them as stage-plan / provenance identifiers. Formal NVTX parent-range names for acceptance are owned separately by `graph_capture_support_matrix.md` / `graph_capture_support_matrix.json`.

### Validation strategy

#### A. Unit validation

Must add tests for:

- env probe enrichment
- environment-neutral script generation
- stage plan emission
- startup seed cap recording
- I/O overlay patching
- field signature parser on representative scalar/vector fields
- compare-engine rule semantics
- angle-source mismatch behavior

#### B. Build determinism validation

For R1 on each baseline:

- build the same manifest twice
- require exact match for:
  - normalized config hashes
  - patch schema
  - mesh hashes
  - case metadata fields that do not encode timestamps

Pass/fail:
- pass if exact
- fail if mesh or patch fingerprint changes
- review if only absolute path strings differ

#### C. Run determinism validation

For R1 on each baseline, run twice and compare:

- `solver_ok` exact
- `mass_imbalance_pct` within `0.2` absolute percentage points
- `water_flow_cfd_gph` within `0.5%`
- `cd_mass_cfd` within `0.5%`
- `air_core_area_ratio` within `0.02` absolute

#### D. R2 validation

Hard gates:

- solver log contains no floating-point exception or crash marker
- `solver_ok = 1`
- `alpha_min >= -1e-6`
- `alpha_max <= 1 + 1e-6`
- `alpha_integral_change_pct <= 0.5%`

Review gates:

- field hash exact match within same-baseline rerun
- free-surface centroid drift small if implemented

#### E. R1 validation

Hard gates:

- `solver_ok = 1`
- `mass_imbalance_pct <= 1.0`
- patch schema exact match to R1 intent
- startup provenance exact
- precondition field-copy provenance exact
- `water_flow_cfd_gph` A-vs-B relative diff `<= 3%`
- `cd_mass_cfd` A-vs-B relative diff `<= 3%` or abs diff `<= 0.03`

Review gates:

- `air_core_area_ratio` abs diff `<= 0.05` or rel diff `<= 10%`
- `swirl_number_proxy` rel diff `<= 10%`
- `recirculation_fraction` abs diff `<= 0.05`

#### F. R0 validation

Hard gates:

- `solver_ok = 1`
- `mass_imbalance_pct <= 1.0`
- patch schema exact
- resolved solver family recorded
- startup provenance exact
- resolved direct-slot numerics exact
- `water_flow_cfd_gph` A-vs-B relative diff `<= 2%`
- `cd_mass_cfd` A-vs-B relative diff `<= 2%` or abs diff `<= 0.02`

Review gates:

- `spray_angle_cfd_deg` abs diff `<= 5 deg` only when both sides report `spray_angle_source = geometric_a`
- if one side is `velocity_proxy`, classify as review-only, not fail
- `air_core_area_ratio` abs diff `<= 0.05` or rel diff `<= 10%`
- `swirl_number_proxy` rel diff `<= 10%`
- `recirculation_fraction` abs diff `<= 0.05`
- advisory `delta_p_rgh_mean_pa` if implemented

#### G. Manufacturing sanity validation

For R0 Baseline A and B individually:

- compare `water_flow_cfd_gph` to manufacturing `54.2 gph`
- compare angle A only if `geometric_a`
- compare Cd if derived consistently

This is informational, not a hard migration gate.

### Performance expectations

Performance is not a Phase 0 gate. Still, record:

- total walltime
- stage walltimes
- peak RSS from `/usr/bin/time -v`

Recommended expectations:

- R1 should be cheap enough to rerun routinely. Target: under 25% of R0 walltime on the same baseline.
- R2 should complete fast enough to use as a preflight.
- R0 may remain expensive; that is acceptable.

Do not retune numerics to hit a walltime target in Phase 0.

### Common failure modes

1. Generated scripts still source `/opt/openfoam12` and Baseline B runs the wrong line.
2. `geometric_iso` requested but `interIsoFoam` unavailable, silently falling back.
3. `libpetscFoam.so` or external backend missing on Baseline B.
4. Builder emits correct case files but runner uses a different stage sequence.
5. `setFieldsDict` seed cap changes startup behavior but is not recorded.
6. `foamListTimes -latestTime` resolves unexpectedly in the steady stage and wrong fields are copied.
7. Patch schema changes while `checkMesh` still passes.
8. Output written in binary/compressed form and field signature parser fails.
9. Angle metric changes source from `geometric_a` to `velocity_proxy`.
10. Template defaults leak back in because resolved direct-slot numerics were not recorded.
11. `DeltaP_effective_Pa` differs due to check-valve-loss logic and is not frozen.
12. Pressure-drop proxy is compared as if it were physically absolute, although only `p_rgh` proxy was measured.

### Debugging playbook

1. **Case fails before solve.**  
   Check `probe.json`, generated scripts, and `stage_plan.json`. Confirm environment-neutral command wrapping.

2. **Case builds but Baseline B script fails immediately.**  
   Search artifact scripts and logs for `/opt/openfoam12`. If found, the environment decoupling change is incomplete.

3. **Resolved solver family differs unexpectedly.**  
   Compare:
   - requested `vof_solver_mode`
   - `available_solvers`
   - resolved executable in `case_meta.json`
   Decide whether this is strict fail or split-baseline review.

4. **R2 passes but R1/R0 fail.**  
   This usually means nozzle-specific builder or startup conditioning drift, not generic VOF failure. Inspect:
   - `setFieldsDict`
   - copied precondition fields
   - patch schema
   - resolved numerics

5. **Flow/Cd drift but mesh fingerprint matches.**  
   Inspect:
   - `DeltaP_effective_Pa`
   - `steady_turbulence_model`
   - `vof_turbulence_model`
   - startup seed/fill
   - pressure backend changes

6. **Angle drift only.**  
   First inspect `spray_angle_source`. If source mismatch exists, do not treat as hard fail. Then inspect compact plume domain settings.

7. **`air_core_area_ratio` drifts strongly.**  
   Inspect startup seed/fill, copied steady fields, and whether the transient started from the same latest steady time.

8. **Field parser fails.**  
   Confirm I/O overlay actually forced ASCII and no compression. Then inspect whether a field is uniform vs nonuniform or missing by design.

9. **Backend fallback applied unexpectedly.**  
   Inspect `pressure_backend_patch.json` and `probe.json`. Confirm whether Baseline B genuinely lacked the external backend.

### Acceptance checklist

Hard pass required:

- [ ] Phase 0 manifests committed
- [ ] tolerance config committed
- [ ] `reference_case_contract.md` committed
- [ ] `reference_case_contract.json` committed
- [ ] `support_matrix.md` committed
- [ ] `support_matrix.json` committed
- [ ] `acceptance_manifest.md` committed
- [ ] `acceptance_manifest.json` committed
- [ ] `graph_capture_support_matrix.md` committed
- [ ] `graph_capture_support_matrix.json` committed
- [ ] Baseline A probe archived
- [ ] Baseline B probe archived
- [ ] no hard-coded OF12 path remains in generated stage execution
- [ ] R2 hard gates pass on both baselines
- [ ] `R1-core` hard gates pass on both baselines
- [ ] R1 hard gates pass on both baselines
- [ ] R0 hard gates pass on both baselines
- [ ] compare report generated
- [ ] human sign-off recorded

Review allowed with sign-off:

- [ ] CPU-only geometric shadow reference or other documented solver-family split outside the hard acceptance line
- [ ] pressure-backend mismatch
- [ ] angle-source mismatch
- [ ] advisory pressure-drop proxy not yet approved
- [ ] optional `40-28` shadow case omitted for compute-budget reasons

### Future extensions deferred from this phase

1. SPUMA GPU bring-up on RTX 5080
2. device-resident field ownership
3. AmgX/native pressure path benchmarking
4. CUDA Graph packaging
5. alpha/MULES device port
6. device-resident surface tension and curvature
7. nozzle-specific GPU boundary kernels
8. Nsight Systems / Nsight Compute production profiling
9. custom CUDA kernels for limiter, curvature, and patch work

### Implementation tasks for coding agent

1. Create canonical Phase 0 JSON case specs.
2. Create Phase 0 tolerance YAML.
3. Extend environment probe.
4. Remove hard-coded OpenFOAM-12 environment sourcing.
5. Emit `stage_plan.json`.
6. Extend `case_meta.json` with resolved Phase 0 fields.
7. Add reference I/O overlay.
8. Add mesh/patch fingerprinting.
9. Add field-signature computation.
10. Add JSON output to extractor.
11. Add R2 generic VOF validator.
12. Add Phase 0 freeze CLI.
13. Add compare/report CLI.
14. Extend tests.
15. Freeze Baseline A R2/R1-core/R1/R0.
16. Freeze Baseline B R2/R1-core/R1/R0.
17. Generate signed-off Phase 0 report.

### Do not start until

- Baseline A environment path is known
- Baseline B environment path is known, and any contingency-only stock-v2412 fallback use is explicitly recorded if it becomes necessary
- R0 default nozzle selection (`57-28`) is accepted
- startup seed/fill default is frozen by `reference_case_contract.md`
- artifact storage location is available

### Safe parallelization opportunities

- env-probe hardening and compare-engine development can proceed in parallel
- overlay/fingerprint code and extractor JSON output can proceed in parallel
- R2 validation can proceed before nozzle R1/R0 runs
- tests for metadata/fingerprints can proceed before live OpenFOAM runs

### Requires human sign-off on

- whether any optional CPU-only geometric shadow row is retained in the phase bundle
- whether `40-28` becomes a hard Phase 0 shadow gate in a later milestone
- whether the advisory pressure-drop proxy is promoted after the initial freeze

### Artifacts to produce

Under `artifacts/reference_phase0/<baseline>/<case_id>/`:

- `probe.json`
- `case_meta.json`
- `rheology_contract.json`
- `build_log.json`
- `stage_plan.json`
- `reference_freeze_overlay.json`
- `build_fingerprint.json`
- `field_signatures.json`
- `metrics.json`
- `baseline_verdict.json`
- `pressure_backend_patch.json` if used
- `logs/*.log`
- `logs/*.time.txt`

Under `artifacts/reference_phase0/comparisons/`:

- `compare.json`
- `compare.md`

Committed cross-phase contract artifacts produced alongside Phase 0:

- `reference_case_contract.md`
- `support_matrix.md`
- `acceptance_manifest.md`
- `docs/gpu_solver_master_plan.md` or equivalent centralized roadmap owner
- `semantic_source_map.md` or equivalent early source-audit handoff before Phase 4–7 implementation PRs

---

### Later-phase handoff context (ownership lives outside this file)

Cross-phase sequencing, centralized definitions, toolchain pins, support scope, runtime schema, and performance thresholds belong in the master roadmap and companion machine-readable contracts produced alongside Phase 0 rather than in this phase file. Phase 0 hands off only the local contracts that later phases consume:

- Milestone-1 GPU acceptance starts at transient solve + device startup seeding; the steady precondition / `potentialFoam` path remains an allowed CPU pre-run utility unless later promoted.
- Phase 1 consumes the Baseline B comparison direction and the source/version/toolchain pin proposal scaffold, but the frozen pin manifest is owned outside this file.
- Phase 5 consumes the canonical algebraic `incompressibleVoF` / explicit MULES / PIMPLE freeze and the `R2 -> R1-core -> R1 -> R0` validation ladder.
- Phase 6 consumes the frozen patch schema, startup provenance, stage taxonomy, and support-matrix classifications for nozzle-specific BC/startup execution.
- Phase 7 starts after Phases 3–6 are correct and interim profiling identifies the remaining hotspots.
- Phase 8 is the final profiling/performance acceptance regime and consumes, rather than redefines, the centralized acceptance manifest.

---

## 6. Validation and benchmarking framework

The global framework must be the same across all later phases, but Phase 0 establishes it.

### Validation hierarchy

1. **Level V0 — provenance correctness**
   - environment probe valid
   - manifest explicit
   - stage plan archived
   - patch schema exact

2. **Level V1 — generic solver correctness**
   - R2 boundedness and conservation

3. **Level V2 — reduced core correctness**
   - `R1-core` reduced-case correctness using only the frozen generic BC/scheme subset

4. **Level V3 — reduced nozzle correctness**
   - R1 flow/Cd/startup stability

5. **Level V4 — full nozzle correctness**
   - R0 flow/Cd/air-core/angle review contract

6. **Level V5 — future GPU performance**
   - only evaluated after V0–V4 pass

### Benchmarking rule

Later GPU phases must compare against **Baseline B Phase 0** first, not against manufacturing truth only and not against Baseline A only. Baseline B is the intended migration line.

### Stage taxonomy

Use the same stage names in all later profiling:

- buildMesh
- checkMesh_build
- steady_prepare
- steady_checkMesh
- steady_potentialFoam
- steady_run
- copy_precondition_fields
- setFields
- transient_run
- extract_features

That naming continuity is valuable later for NVTX ranges and Nsight timelines.

### Metric taxonomy

Every metric must include:

- `name`
- `value`
- `unit`
- `source`
- `gate_class`
- `comparison_rule`

### Comparison rule taxonomy

Rules must support:

- exact match
- absolute tolerance
- relative tolerance
- `requires_same_source`
- `requires_same_solver_family`
- `not_applicable_for_case_kind`

### When to stop and review

- After Baseline A R0 freeze
- After Baseline B R2 smoke
- After Baseline B R0 compare

No later phase should start before the previous stop point is reviewed.

---

## 7. Toolchain / environment specification

### Phase 0 required environment

- Linux x86_64 workstation
- repo Python environment as already used by the project
- Baseline A OpenFOAM environment
- Baseline B SPUMA/v2412 CPU environment, or contingency-only stock OpenFOAM-v2412 CPU fallback if the canonical line cannot execute
- shell tools:
  - `bash`
  - `checkMesh`
  - `setFields`
  - `foamRun`
  - `potentialFoam`
  - `foamListTimes`

### Phase 0 environment policy

- Environment sourcing must be explicit and parameterized.
- No script may assume `/opt/openfoam12`.
- The environment probe output is part of the frozen artifact set.

### Future-phase reserved GPU toolchain

Future-phase GPU toolchain pins are owned by the master pin manifest, not by this phase file. Default proposal until superseded: RTX 5080 workstation, CUDA 12.9.1 primary lane, CUDA 13.2 experimental lane, driver `>=595.45.04`, `sm_120` + PTX, NVTX3, Nsight Systems, Nsight Compute, and Compute Sanitizer. NVIDIA’s Blackwell compatibility guidance still makes PTX inclusion and PTX/JIT validation part of the bring-up checklist.

### Bring-up guardrails for later phases

- Use SPUMA’s currently published supported solvers first for hardware/toolchain health checks.
- Do not begin nozzle multiphase GPU work until those health checks pass.

---

## 8. Module / file / ownership map

### Existing authoritative modules to extend

- `scripts/cfd/openfoam_env_probe.py`  
  Owner: coding agent implementation, senior engineer review.  
  Responsibility: environment discovery.

- `scripts/cfd/build_pressure_swirl_case.py`  
  Owner: coding agent implementation, mandatory senior engineer review.  
  Responsibility: authoritative case generation, stage plan emission, resolved metadata.

- `scripts/cfd/extract_openfoam_case_features.py`  
  Owner: coding agent implementation, mandatory senior engineer review.  
  Responsibility: authoritative nozzle metrics and source tagging.

- `scripts/cfd/run_manifest.py`  
  Owner: coding agent implementation, mandatory senior engineer review.  
  Responsibility: generic orchestration; must be made environment-neutral.

- `cfd/templates/pressureSwirlNozzleVOF/`  
  Owner: senior engineer review required for any physics change.  
  Responsibility: template source; Phase 0 should avoid physics edits here.

- `cfd/smoke/damBreak_amgx/`  
  Owner: coding agent may clone or patch only for backend fallback.  
  Responsibility: generic VOF smoke.

### New Phase 0 modules

- `scripts/cfd/reference_problem/models.py`  
  Data classes and schemas.

- `scripts/cfd/reference_problem/overlays.py`  
  I/O normalization overlay and backend fallback patch helpers.

- `scripts/cfd/reference_problem/fingerprints.py`  
  config / mesh / patch hashing and semantic fingerprints.

- `scripts/cfd/reference_problem/field_signatures.py`  
  selected field parsing and hashing.

- `scripts/cfd/reference_problem/runner.py`  
  environment-neutral stage execution.

- `scripts/cfd/reference_problem/compare.py`  
  hard/review/info gate engine.

- `scripts/cfd/reference_problem/report.py`  
  Markdown and JSON compare reports.

- `scripts/cfd/freeze_reference_problem.py`  
  top-level CLI for freezing one case on one baseline.

- `scripts/cfd/compare_reference_problem.py`  
  top-level CLI for comparing two frozen artifacts.

### New config and manifest assets

- `cfd/manifests/reference_phase0/*.json`
- `configs/cfd/reference_phase0_tolerances.yaml`
- `docs/reference_problem_phase0.md`

### Tests to extend

- `tests/test_cfd_sk_geometry.py`
- `tests/test_cfd_openfoam_extract.py`
- `tests/test_cfd_rheology_contract_and_probe.py`

### New tests

- `tests/test_cfd_reference_problem_phase0.py`

---

## 9. Coding-agent execution roadmap

### Milestone M0 — inventory and freeze intent
Outputs:
- canonical Phase 0 JSON case specs, including `R1-core`
- tolerance YAML
- `reference_case_contract.md`, `support_matrix.md`, and `acceptance_manifest.md`
- short design note in `docs/reference_problem_phase0.md`

Stop here and review:
- R0 nozzle
- startup seed/fill
- `R1-core` definition

### Milestone M1 — environment hardening
Dependencies:
- M0
Work:
- extend env probe
- remove hard-coded `/opt/openfoam12`
- add environment-neutral runner wrappers

Parallel work:
- compare engine skeleton
- overlay/fingerprint skeleton

### Milestone M2 — metadata and artifact hardening
Dependencies:
- M1
Work:
- `stage_plan.json`
- extended `case_meta.json`
- I/O overlay
- build fingerprinting
- field signature module
- extractor JSON output

Prototype before productizing:
- field parser on R2 fields first
- pressure-drop proxy remains experimental

### Milestone M3 — Baseline A freeze
Dependencies:
- M2
Work:
- freeze R2
- freeze R1
- freeze R0

Stop and review:
- Baseline A artifacts become the initial control truth

### Milestone M4 — Baseline B build-only bring-up
Dependencies:
- M3
Work:
- probe Baseline B
- build R2/R1/R0 without execution first
- validate no OF12 hard-coding remains

Stop and review:
- do not run long cases if build-only already shows drift

### Milestone M5 — Baseline B R2 smoke
Dependencies:
- M4
Work:
- run and validate R2
- verify backend fallback policy if needed

Stop and review:
- if R2 fails, do not proceed to nozzle cases

### Milestone M6 — Baseline B R1 nozzle freeze
Dependencies:
- M5
Work:
- run and validate R1
- compare against Baseline A R1

### Milestone M7 — Baseline B R0 nozzle freeze
Dependencies:
- M6
Work:
- run and validate R0
- compare against Baseline A R0

Stop and review:
- this is the final Phase 0 gate

### Milestone M8 — sign-off package
Dependencies:
- M7
Work:
- produce compare JSON/Markdown
- produce concise human review packet
- archive artifacts

### What can be done in parallel

- env probe hardening and compare engine
- overlay implementation and field signature parser
- test scaffolding and manifest creation

### What should be prototyped before productized

- field parser
- pressure-drop proxy
- backend fallback patching

### What should remain experimental after Phase 0

- pressure-drop proxy as hard gate
- `40-28` shadow case as mandatory
- exact field-hash cross-baseline equivalence

### Where to stop and benchmark before proceeding

- after M3
- after M5
- after M7

---

## 10. Residual non-blocking governance notes

1. Should `40-28 @ 1000 psi` be generated as an additional shadow/reference case in the same implementation window?

Phase 0 keeps the advisory `p_rgh` pressure-drop proxy review-only inside this phase. Any later hard pressure-drop acceptance gate is owned exclusively by `acceptance_manifest.md` / `acceptance_manifest.json`.

---

### Human review checklist

- [ ] `reference_case_contract.md` matches the frozen `R2 -> R1-core -> R1 -> R0` ladder.
- [ ] `support_matrix.md` keeps `R1-core` inside the generic Phase 5 subset.
- [ ] Phase 0 pressure-drop proxy remains review-only locally, and later hard-gate ownership is delegated to `acceptance_manifest.md` / `acceptance_manifest.json`.
- [ ] Optional `40-28` shadow-case policy is recorded explicitly.

### Coding agent kickoff checklist

- [ ] add Phase 0 manifests, including `R1-core`
- [ ] add tolerance YAML and align it with `acceptance_manifest.md`
- [ ] add `reference_case_contract.md` and `support_matrix.md`
- [ ] harden env probe
- [ ] remove hard-coded OF12 sourcing
- [ ] emit stage plan
- [ ] extend case metadata
- [ ] add overlay, fingerprints, field signatures
- [ ] add JSON extractor output
- [ ] add freeze/compare CLIs
- [ ] extend tests before live runs

### Highest risk implementation assumptions

1. Canonical Baseline B (SPUMA/v2412 CPU) can run the nozzle workflow on CPU with only environment and backend adjustments, or the contingency-only stock-v2412 freeze remains sufficient to preserve Phase 0 progress while being labeled as such.
2. The builder exposes an explicit algebraic manifest path that resolves to `incompressibleVoF` on both hard-gate baselines without silent fallback.
3. The startup seed/fill defaults chosen here are close enough to the intended current branch that Phase 0 will remain meaningful.
4. ASCII/uncompressed output normalization will not interfere with the existing workflow.
5. The field parser will be robust enough across both baselines for selected signature fields.
6. Pressure-backend fallback, if used, will not move primary nozzle metrics beyond the declared tolerances.
