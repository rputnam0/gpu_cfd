# Phase 0 — Reference problem freeze

Phase 0 owns `P0-01..P0-08` and freezes CPU reference truth for `R2`, `R1-core`, `R1`, and `R0` before downstream GPU implementation phases consume it. This section consumes Foundation contracts for pins, case-role resolution, and support vocabulary, and does not redefine runtime structure, bring-up policy, or GPU execution semantics.

## P0-01 Environment probe hardening

- Objective:
  - Harden environment probing so both Baseline A and Baseline B resolve to machine-readable manifests with no hard-coded `/opt/openfoam12` assumptions.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-02 Pin-manifest consumption and environment manifest emission`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `### Decision I — remove hard-coded OpenFOAM-12 environment assumptions from generated scripts and runners`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.8 — remove hard-coded environment sourcing from generated scripts`
    - `docs/phase0_reference_problem_spec.md` -> `#### 1. scripts/cfd/openfoam_env_probe.py`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M1 — environment hardening`
  - Backlog scope:
    - "Extend the existing OpenFOAM environment probe so both Baseline A and Baseline B resolve into machine-readable manifests, with no hard-coded `/opt/openfoam12` assumptions left in tooling."
  - Backlog done_when:
    - "The probe captures environment details for both baselines and generated tooling no longer embeds fixed OpenFOAM paths."
- Depends on (backlog IDs):
  - `FND-02`
- Prerequisites:
  - Foundation pin-resolution API and canonical environment artifact contract from `FND-02`.
  - Existing probe entrypoint and baseline selection inputs.
- Concrete task slices:
  1. Extend probe outputs to emit baseline-qualified environment records keyed by Baseline A and Baseline B.
  2. Route probe output naming through Foundation naming policy with `host_env.json` canonical and `env.json` compatibility alias only.
  3. Remove all fixed OpenFOAM path assumptions from generated scripts and orchestrator defaults.
  4. Add explicit diagnostics for missing baseline environment activation and unresolved toolkit lane.
- Artifacts/contracts introduced or consumed:
  - Consumed: `host_env.json` schema, `manifest_refs.json`, pin-resolution API from `FND-02`.
  - Introduced: baseline-resolved probe manifest entries used by runner wrappers.
- Validation:
  - Probe emits complete records for both baselines in one non-interactive invocation.
  - Generated tooling contains no hard-coded OpenFOAM installation path.
  - Alias handling keeps `host_env.json` canonical while accepting legacy `env.json` where needed.
- Done criteria:
  - Environment probing is baseline-neutral and machine-readable.
  - Phase 0 tooling uses probe output rather than embedded path assumptions.
- Exports to later PRs/phases:
  - Baseline-qualified environment probe payload consumed by `P0-02` and Phase 1 host/CUDA probes.

## P0-02 Environment-neutral runner wrappers

- Objective:
  - Add wrappers that source probed environment data explicitly and run staged workflows without hidden baseline coupling.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-02 Pin-manifest consumption and environment manifest emission`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.8 — remove hard-coded environment sourcing from generated scripts`
    - `docs/phase0_reference_problem_spec.md` -> `#### Pseudocode 2 — environment-neutral stage runner`
    - `docs/phase0_reference_problem_spec.md` -> `#### 3. scripts/cfd/run_manifest.py`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M1 — environment hardening`
  - Backlog scope:
    - "Add runner wrappers that source the probed environment explicitly and execute the staged workflow in a baseline-neutral way."
  - Backlog done_when:
    - "Generated case runners execute against both baselines without manual script edits or hidden environment coupling."
- Depends on (backlog IDs):
  - `P0-01`
- Prerequisites:
  - Baseline-qualified probe payload from `P0-01`.
  - Foundation pin-resolution contract from `FND-02`.
- Concrete task slices:
  1. Implement wrapper entrypoints that resolve baseline choice from probe artifacts and explicit manifest options.
  2. Standardize stage runner invocation so build, run, and extract stages share one environment resolution path.
  3. Prevent implicit shell inheritance by enforcing explicit source and validation checks before launch.
  4. Emit run-context provenance fields linking each stage execution to probe and pin manifests.
- Artifacts/contracts introduced or consumed:
  - Consumed: `host_env.json` baseline records, `manifest_refs.json`.
  - Introduced: baseline-neutral runner contract for downstream stage orchestration.
- Validation:
  - Same runner wrappers execute both Baseline A and Baseline B flows without script edits.
  - Missing baseline environment fails with deterministic pre-run diagnostics.
  - Stage logs include resolved baseline and manifest references.
- Done criteria:
  - Runner wrappers are baseline-neutral and explicit.
  - Hidden environment coupling is removed from staged workflow execution.
- Exports to later PRs/phases:
  - Environment-neutral stage runner contract consumed by `P0-03` orchestration and Baseline A/B freeze runs.

## P0-03 Case metadata and stage-plan emission

- Objective:
  - Extend orchestration outputs so every generated case carries complete machine-readable execution plan and resolved metadata.
- Exact citations:
  - Authority:
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/reference_case_contract.json` -> `frozen_cases`, `phase_gate_mapping`, `locked_defaults`
    - `docs/validation_ladder.md` -> `## Frozen Ladder`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-03 Reference-case and validation-ladder utilities`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.1 — Freeze case intent separately from execution resolution`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.5 — use JSON-row manifests for Phase 0, not only shared DOE CSVs`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.6 — extend existing case_meta.json; do not invent a parallel metadata format`
    - `docs/phase0_reference_problem_spec.md` -> `#### Mandatory case_meta.json additions`
    - `docs/phase0_reference_problem_spec.md` -> `#### Mandatory stage_plan.json`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M2 — metadata and artifact hardening`
  - Backlog scope:
    - "Extend the builder/orchestrator to emit `stage_plan.json`, richer `case_meta.json`, resolved numerics, startup-seed metadata, and provenance fields needed by later comparisons."
  - Backlog done_when:
    - "Every generated case carries a complete machine-readable execution plan and resolved metadata bundle."
- Depends on (backlog IDs):
  - `P0-02`
- Prerequisites:
  - Foundation case-role resolver and `case_meta.json`/`stage_plan.json` schemas from `FND-03`.
  - Baseline-neutral runner contract from `P0-02`.
- Concrete task slices:
  1. Populate `case_meta.json` with resolved case role, baseline context, numerics tuple, startup-seed settings, and provenance references.
  2. Emit `stage_plan.json` from authority-derived stage and gate selection inputs without redefining ladder roles locally.
  3. Validate emitted metadata bundle against Foundation schemas before stage execution.
  4. Attach manifest and probe pointers so later compare tooling can trace every output artifact to resolved run intent.
- Artifacts/contracts introduced or consumed:
  - Consumed: case-role resolver, `case_meta.json` schema, `stage_plan.json` schema from `FND-03`.
  - Introduced: enriched per-case metadata bundle used by normalization, extractor, and freeze steps.
- Validation:
  - Every Phase 0 case emits both files with required fields populated.
  - Case-role and phase-gate values match authority contracts exactly.
  - Metadata validation fails if any required provenance field is missing.
- Done criteria:
  - Complete, machine-readable execution plan and metadata bundle exists for all ladder cases.
  - Later compare workflows can run non-interactively from emitted metadata.
- Exports to later PRs/phases:
  - Canonical `case_meta.json` and `stage_plan.json` instances consumed by `P0-04`, `P0-05`, and downstream phase compare logic.

## P0-04 I/O normalization overlay

- Objective:
  - Implement non-physics output normalization so comparison artifacts remain stable across reruns.
- Exact citations:
  - Authority:
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-03 Reference-case and validation-ladder utilities`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.7 — add a non-physics I/O normalization overlay for Phase 0`
    - `docs/phase0_reference_problem_spec.md` -> `#### 3. scripts/cfd/run_manifest.py`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> C. Run determinism validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M2 — metadata and artifact hardening`
  - Backlog scope:
    - "Implement the non-physics output normalization layer for ASCII/uncompressed/fixed-precision output so comparison artifacts are stable across reruns."
  - Backlog done_when:
    - "Same-baseline reruns produce stable normalized outputs and do not fail due to incidental I/O formatting drift."
- Depends on (backlog IDs):
  - `P0-03`
- Prerequisites:
  - Emitted per-case metadata bundle from `P0-03`.
- Concrete task slices:
  1. Add normalization rules for representation-only drift: output format, compression, and numeric text precision policy.
  2. Apply normalization as a dedicated compare-prep stage without altering physical solver settings.
  3. Record normalization manifest fields so compare reports can prove representation parity.
  4. Integrate normalization checks into rerun determinism gates before reference freeze.
- Artifacts/contracts introduced or consumed:
  - Consumed: `case_meta.json` and `stage_plan.json` outputs.
  - Introduced: normalized output artifact set and normalization metadata stamp.
- Validation:
  - Repeated runs of same baseline and case produce stable normalized representations.
  - Drift alerts only fire on physics-relevant differences after normalization.
  - Normalization stage is auditable from emitted metadata.
- Done criteria:
  - Non-physics I/O drift no longer causes false compare failures.
  - Normalized artifacts are ready for freeze and comparison tooling.
- Exports to later PRs/phases:
  - Stable normalized artifact layer consumed by freeze and cross-baseline comparison steps.

## P0-05 Fingerprints, field signatures, and extractor JSON

- Objective:
  - Add non-interactive fingerprint and field-signature extraction outputs for automated comparison workflows.
- Exact citations:
  - Authority:
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/validation_ladder.md` -> `## Frozen Ladder`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-03 Reference-case and validation-ladder utilities`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `#### Field-signature logic`
    - `docs/phase0_reference_problem_spec.md` -> `#### Mesh/patch fingerprint logic`
    - `docs/phase0_reference_problem_spec.md` -> `#### 4. scripts/cfd/extract_openfoam_case_features.py`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> A. Unit validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M2 — metadata and artifact hardening`
  - Backlog scope:
    - "Add mesh/patch fingerprints, selected field-signature extraction, and JSON output from the feature extractor for automated comparison workflows."
  - Backlog done_when:
    - "R2/R1-core/R1/R0 cases can emit comparison-ready fingerprints and field signatures non-interactively."
- Depends on (backlog IDs):
  - `P0-03`
- Prerequisites:
  - Case-role-aware metadata from `P0-03`.
  - Phase 0 extractor integration points.
- Concrete task slices:
  1. Define mesh/patch fingerprint payload keyed by canonical case IDs and ladder roles.
  2. Implement selected field-signature extraction with consistent JSON schema across all four ladder rungs.
  3. Add extractor orchestration hook so signatures are emitted automatically in staged runs.
  4. Record extraction provenance links to normalized outputs and run metadata.
- Artifacts/contracts introduced or consumed:
  - Consumed: `case_meta.json`, `stage_plan.json`, normalized outputs.
  - Introduced: fingerprint JSON and field-signature JSON artifacts used in freeze and compare reports.
- Validation:
  - All ladder cases emit machine-readable fingerprints and signatures with no manual steps.
  - JSON schema remains stable across reruns and baseline variants.
  - Missing required fields or schema mismatch fails extraction gate.
- Done criteria:
  - Comparison-ready fingerprints and signatures are available for `R2`, `R1-core`, `R1`, and `R0`.
  - Outputs are deterministic enough for automated compare pipelines.
- Exports to later PRs/phases:
  - Fingerprint and signature artifact contracts consumed by Baseline A freeze and Baseline A/B comparison.

## P0-06 Baseline A reference freeze

- Objective:
  - Freeze Baseline A reference bundles for all ladder cases as initial control truth.
- Exact citations:
  - Authority:
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/reference_case_contract.md` -> `## Locked Defaults`
    - `docs/reference_case_contract.json` -> `frozen_cases`, `phase_gate_mapping`, `locked_defaults`
    - `docs/validation_ladder.md` -> `## Frozen Ladder`
    - `docs/validation_ladder.md` -> `## Usage Rule`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `### Decision A — use dual baselines, not a single migration target`
    - `docs/phase0_reference_problem_spec.md` -> `### Decision C — Phase 0 is single-rank CPU only`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.2 — R0 hard-gate default is a fully specified 57-28 @ 1000 psi algebraic reference`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.3 — R1 default is a reduced internal-only nozzle case`
    - `docs/phase0_reference_problem_spec.md` -> `#### D0.4 — R2 default is the existing damBreak_amgx asset with an allowed CPU fallback`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> D. R2 validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> E. R1 validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> F. R0 validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M3 — Baseline A freeze`
  - Backlog scope:
    - "Generate and freeze the Baseline A reference bundles for `R2`, `R1-core`, `R1`, and `R0`, including artifacts, metadata, and initial tolerance checks."
  - Backlog done_when:
    - "A complete Baseline A bundle exists for all ladder cases and becomes the initial control truth."
- Depends on (backlog IDs):
  - `P0-04`
  - `P0-05`
  - `FND-03`
- Prerequisites:
  - Normalized output layer from `P0-04`.
  - Fingerprint and signature outputs from `P0-05`.
  - Foundation case-role and ladder resolver from `FND-03`.
- Concrete task slices:
  1. Execute Baseline A freeze runs for all ladder roles resolved from authority contracts.
  2. Package per-case artifacts: metadata, normalized outputs, fingerprints, and signatures into frozen reference bundles.
  3. Apply initial tolerance and provenance checks defined for Phase 0 acceptance.
  4. Publish immutable Baseline A control bundle index for cross-baseline comparison consumption.
- Artifacts/contracts introduced or consumed:
  - Consumed: `case_meta.json`, `stage_plan.json`, normalized outputs, fingerprint and signature JSON.
  - Introduced: Baseline A frozen bundles and Baseline A control index.
- Validation:
  - All four ladder cases complete and freeze successfully under Baseline A.
  - Bundle contents match required artifact manifest for each case.
  - Initial tolerance and provenance checks pass or are clearly classified for review.
- Done criteria:
  - Baseline A control truth is complete for `R2`, `R1-core`, `R1`, and `R0`.
  - Freeze bundle is reproducible and referenced by compare workflows.
- Exports to later PRs/phases:
  - Baseline A frozen reference bundles and control index consumed by `P0-07` and `P0-08`.

## P0-07 Baseline B bring-up and R2 smoke

- Objective:
  - Bring up Baseline B in build-first mode, remove legacy path coupling, and complete `R2` smoke execution before nozzle-case freeze.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Resolved Frozen Source Tuple`
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/reference_case_contract.md` -> `## Phase-Gate Mapping`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `### Decision A — use dual baselines, not a single migration target`
    - `docs/phase0_reference_problem_spec.md` -> `### Decision I — remove hard-coded OpenFOAM-12 environment assumptions from generated scripts and runners`
    - `docs/phase0_reference_problem_spec.md` -> `### Risk 2 — hard-coded /opt/openfoam12 path breaks Baseline B silently`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M4 — Baseline B build-only bring-up`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M5 — Baseline B R2 smoke`
  - Backlog scope:
    - "Bring up Baseline B in build-only mode first, validate that no legacy OpenFOAM-12 coupling remains, then run the `R2` smoke case with documented backend policy."
  - Backlog done_when:
    - "Baseline B builds the frozen cases and `R2` executes cleanly enough to justify proceeding to nozzle cases."
- Depends on (backlog IDs):
  - `P0-06`
- Prerequisites:
  - Baseline A control truth from `P0-06`.
  - Environment-neutral wrappers from `P0-02`.
  - Pin-manifest resolution inputs from `P0-01` and Foundation contracts.
- Concrete task slices:
  1. Execute Baseline B build-only checks for frozen case set with explicit source tuple traceability.
  2. Validate environment coupling removal by proving runs depend on probe/manifests rather than fixed installation paths.
  3. Run `R2` smoke case and collect run health plus backend-policy evidence.
  4. Produce go/no-go decision artifact for progressing to nozzle-case freeze.
- Artifacts/contracts introduced or consumed:
  - Consumed: Baseline A control bundles, `host_env.json`, `manifest_refs.json`, case-role mappings.
  - Introduced: Baseline B build and `R2` smoke evidence packet.
- Validation:
  - Baseline B builds frozen case targets without legacy path coupling.
  - `R2` executes with required metadata and health checks.
  - Failure modes are categorized as blocker vs review before nozzle progression.
- Done criteria:
  - Baseline B is operational enough for nozzle-case freeze attempts.
  - Documented smoke evidence supports proceeding to `R1` and `R0`.
- Exports to later PRs/phases:
  - Baseline B bring-up and `R2` smoke packet consumed by `P0-08`.

## P0-08 Baseline B nozzle freeze and sign-off package

- Objective:
  - Freeze Baseline B `R1` and `R0`, compare to Baseline A, and produce the complete Phase 0 sign-off packet.
- Exact citations:
  - Authority:
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/reference_case_contract.json` -> `frozen_cases`, `phase_gate_mapping`, `locked_defaults`
    - `docs/validation_ladder.md` -> `## Frozen Ladder`
    - `docs/validation_ladder.md` -> `## Usage Rule`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase0_reference_problem_spec.md` -> `### Decision F — Phase 0 acceptance is numerical and provenance-based, not performance-based`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> E. R1 validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Validation strategy -> F. R0 validation`
    - `docs/phase0_reference_problem_spec.md` -> `### Acceptance checklist`
    - `docs/phase0_reference_problem_spec.md` -> `### Artifacts to produce`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M6 — Baseline B R1 nozzle freeze`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M7 — Baseline B R0 nozzle freeze`
    - `docs/phase0_reference_problem_spec.md` -> `### Milestone M8 — sign-off package`
  - Backlog scope:
    - "Run and freeze `R1` and `R0` on Baseline B, compare against Baseline A, and generate the Phase 0 sign-off packet with compare JSON and Markdown and archived artifacts."
  - Backlog done_when:
    - "Phase 0 ends with reviewed Baseline A/B compare reports and a frozen reference contract suitable for downstream implementation."
- Depends on (backlog IDs):
  - `P0-07`
- Prerequisites:
  - Baseline B bring-up and `R2` smoke evidence from `P0-07`.
  - Baseline A control bundles from `P0-06`.
  - Case-role mappings and ladder rules from Foundation contracts.
- Concrete task slices:
  1. Run and freeze Baseline B `R1` and `R0` using canonical case-role selection.
  2. Execute cross-baseline comparison workflow against Baseline A bundles using normalized outputs and signature artifacts.
  3. Generate sign-off artifacts: compare JSON, compare Markdown, archive index, and provenance manifests.
  4. Publish Phase 0 freeze contract package and handoff summary for downstream phases.
- Artifacts/contracts introduced or consumed:
  - Consumed: Baseline A control bundles, Baseline B run outputs, normalized compare inputs, fingerprint and signature JSON.
  - Introduced: Phase 0 sign-off packet, Baseline A/B compare reports, frozen reference contract package.
- Validation:
  - `R1` and `R0` Baseline B freezes complete and are fully traceable.
  - Compare reports include required numerical and provenance gates and review classifications.
  - Handoff package is complete for downstream consumption without additional case reinterpretation.
- Done criteria:
  - Reviewed Baseline A/B compare reports exist for nozzle cases.
  - Phase 0 delivers frozen reference contract suitable for later implementation phases.
- Exports to later PRs/phases:
  - Phase 0 sign-off packet and frozen reference contract consumed by Phase 4 pressure replay setup and downstream validation gates.

## imports_from_prev

- `host_env.json` canonical environment artifact contract from `FND-02`.
- `manifest_refs.json` and pin-resolution API from `FND-02`.
- Case-role resolver and canonical `case_meta.json` and `stage_plan.json` schemas from `FND-03`.
- Foundation seam rule that ladder roles, supported tuples, fallback policy, and toolchain lanes must not be restated in Phase 0.

## exports_to_next

- Baseline A frozen bundle set for `R2`, `R1-core`, `R1`, `R0`.
- Baseline B smoke and nozzle freeze evidence bundles.
- Baseline A/B compare JSON and compare Markdown reports.
- Phase 0 sign-off packet containing archived artifacts, provenance manifests, and freeze contract summary.
- Canonicalized per-case `case_meta.json` and `stage_plan.json` instances with resolved run intent.

## shared_terms

- `Baseline A`: control truth baseline used to create initial frozen reference bundles in Phase 0.
- `Baseline B`: target baseline brought up and compared against Baseline A in Phase 0.
- `reference freeze`: immutable bundle of case outputs, metadata, and compare-ready artifacts.
- `sign-off packet`: final Phase 0 archive containing compare outputs, provenance, and freeze contract summary.
- `case role`: canonical ladder role selected through Foundation case resolver, `R2`, `R1-core`, `R1`, and `R0`.
- `host_env.json`: canonical environment manifest artifact name from Foundation contracts.

## open_discontinuities

- `[tracked] env-artifact-alias-policy`: Foundation tracks `host_env.json` as canonical while `master_pin_manifest.md` still mentions `env.json`; Phase 0 will consume `host_env.json` and accept `env.json` only as compatibility alias where required (`docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## open_discontinuities`, impacted PR IDs: `P0-01`, `P0-02`, preferred reading: preserve `host_env.json` in Phase 0 outputs and logs).

## validation_checks

- All Phase 0 cards preserve backlog dependency edges: `P0-01 <- FND-02`, `P0-02 <- P0-01`, `P0-03 <- P0-02`, `P0-04 <- P0-03`, `P0-05 <- P0-03`, `P0-06 <- P0-04,P0-05,FND-03`, `P0-07 <- P0-06`, `P0-08 <- P0-07`.
- Every card cites authority docs, exact Phase 0 spec subsection anchors, backlog scope, and backlog done_when.
- Phase 0 consumes Foundation exports and does not redefine ladder roles, supported tuples, fallback policy, or toolchain lanes.
- Baseline A and Baseline B freeze ownership and compare report ownership remain explicit within Phase 0.
- Exported artifacts satisfy the Foundation -> Phase 0 seam gate and are ready for downstream pressure and validation consumption.
