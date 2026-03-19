# Phase 1 — Blackwell bring-up

Phase 1 owns `P1-01..P1-07` and is limited to workstation, toolchain, and profiler readiness plus smoke-case admissibility on the frozen SPUMA/v2412 line. It does not define Phase 2 residency policy, Phase 3 execution semantics, or any Phase 0 baseline-freeze behavior.

## P1-01 Host and CUDA discovery probes

- Objective:
  - Add reproducible host and CUDA probe outputs for the target workstation lane using Foundation pin-resolution contracts and canonical manifest naming.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/authority/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/authority/master_pin_manifest.md` -> `## Required Revalidation If This Manifest Changes`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (source/version/toolchain pin ownership and runtime normalization)
    - `docs/authority/support_matrix.md` -> `## Global Policy` (default fail-fast posture)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-02 Pin-manifest consumption and environment manifest emission`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 2 — Add environment discovery tooling`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 3 — Add native CUDA and managed-memory probe binary`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Tooling-side host data structures`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Artifacts to produce`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `## Required baseline`
  - Backlog scope:
    - "Add the Phase 1 host/CUDA discovery binaries and emit `host_env.json` and `cuda_probe.json` for the target workstation lane."
  - Backlog done_when:
    - "The workstation characteristics, driver state, compute capability, and CUDA availability are captured in reproducible machine-readable form."
- Depends on (backlog IDs):
  - `FND-02`
- Prerequisites:
  - Foundation `FND-02` pin-resolution API and canonical manifest schemas.
  - Frozen lane defaults from `master_pin_manifest.md`.
- Concrete task slices:
  1. Implement host discovery wrapper that resolves lane and tool versions only from Foundation pin ingestion, then emits canonical `host_env.json`.
  2. Implement CUDA probe binary and wrapper to emit `cuda_probe.json` with device identity, compute capability, native-kernel check, and managed-memory probe result.
  3. Enforce fail-fast startup behavior when required lane assumptions are violated, with explicit diagnostics instead of implicit fallback.
  4. Add compatibility alias handling so legacy `env.json` is read-only compatibility input and `host_env.json` remains canonical output.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `cuda_probe.json` artifact contract
    - host and CUDA probe execution report with deterministic keys and lane metadata
  - Consumed:
    - Foundation `host_env.json` and `manifest_refs.json` schema contracts from `FND-02`
    - frozen pin tuple from `master_pin_manifest.md`
- Validation:
  - Probe run emits both `host_env.json` and `cuda_probe.json` on the required lane.
  - Probe data confirms workstation GPU identity and required lane metadata without local pin restatement.
  - Invalid or missing lane prerequisites fail before build begins.
- Done criteria:
  - Reproducible machine-readable host/CUDA discovery artifacts exist for the target lane.
  - Canonical naming is `host_env.json`, with any `env.json` handling explicitly documented as compatibility-only.
- Exports to later PRs/phases:
  - `host_env.json` for Phase 1 acceptance and Phase 8 environment capture.
  - `cuda_probe.json` for Phase 1 acceptance and later profiling and acceptance bundles.

## P1-02 Blackwell build-system enablement

- Objective:
  - Produce the required primary-lane build with `sm_120` plus PTX, NVTX3, and frozen toolchain lane adherence through wrapperized, reproducible build orchestration.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/authority/master_pin_manifest.md` -> `## Resolved Frozen Source Tuple`
    - `docs/authority/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/authority/continuity_ledger.md` -> `# 1. Frozen global decisions` (toolchain ownership and NVTX3 requirement)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-02 Pin-manifest consumption and environment manifest emission`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 4 — Add Blackwell environment wrapper`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 5 — Add NVTX3 wrapper and audit old includes`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 7 — Add build wrapper and build modes`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `## Compiler / flag policy`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `## Environment variables`
  - Backlog scope:
    - "Patch or wrap the build so the primary lane emits `sm_120` plus PTX, adopts NVTX3, and respects the frozen toolchain lane."
  - Backlog done_when:
    - "A clean primary-lane build is produced with the required Blackwell targets and without relying on ad hoc local edits."
- Depends on (backlog IDs):
  - `P1-01`
- Prerequisites:
  - `host_env.json` and `cuda_probe.json` from `P1-01`.
  - Foundation pin-resolution contract from `FND-02`.
- Concrete task slices:
  1. Implement lane-aware build wrapper that resolves toolkit and driver expectations from Foundation and exports `have_cuda=true`, `NVARCH=120`, and NVTX3 toggles deterministically.
  2. Add NVTX3 include and wrapper migration checks so NVTX2-era include usage fails early.
  3. Build required lane artifacts in reproducible modes and capture build metadata and logs for acceptance evidence.
  4. Add hard checks to prevent ad hoc local edits from bypassing frozen lane selections.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - primary-lane build logs keyed by lane and mode
    - build metadata summary linked to `host_env.json`
  - Consumed:
    - `host_env.json`, `manifest_refs.json`, and pin-resolution API from `FND-02`
- Validation:
  - Required-lane build succeeds with `sm_120` target settings and PTX retention policy enabled.
  - NVTX3 path compiles cleanly and direct NVTX2 usage is rejected.
  - Build wrapper reports deterministic lane selection and source tuple references.
- Done criteria:
  - Reproducible primary-lane build artifacts are generated with required Blackwell target policy.
  - Build process is wrapperized and no longer relies on one-off local edits.
- Exports to later PRs/phases:
  - Blackwell-ready build artifacts for `P1-03` binary inspection and downstream smoke and profiling lanes.
  - Lane-stable build metadata consumed by Phase 4 dependency freeze context.

## P1-03 Fatbinary inspection and reporting

- Objective:
  - Verify native cubin and PTX composition for produced binaries and emit deterministic `fatbinary_report.json` before smoke execution.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults` (GPU target and PTX/JIT requirement)
    - `docs/authority/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-02 Pin-manifest consumption and environment manifest emission`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 8 — Inspect produced binaries for native target and PTX`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Acceptance checklist`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Artifacts to produce`
  - Backlog scope:
    - "Add binary inspection tooling that verifies the expected native cubin and PTX composition and writes `fatbinary_report.json`."
  - Backlog done_when:
    - "The build artifact proves `sm_120` coverage and PTX retention before smoke cases begin."
- Depends on (backlog IDs):
  - `P1-02`
- Prerequisites:
  - Required-lane build outputs from `P1-02`.
  - Frozen GPU target and PTX policy from pin manifest.
- Concrete task slices:
  1. Implement binary inspection wrapper that extracts cubin and PTX composition from built artifacts and validates expected target presence.
  2. Emit `fatbinary_report.json` with explicit native target coverage and PTX retention fields.
  3. Gate smoke execution on inspection success and fail immediately when native or PTX evidence is missing.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `fatbinary_report.json`
    - binary inspection logs and extracted composition evidence
  - Consumed:
    - build outputs from `P1-02`
    - target policy from pin manifest and Foundation pin-resolution contracts
- Validation:
  - Report confirms `sm_120` and PTX coverage in inspected binaries.
  - Smoke lane is blocked if report criteria are unmet.
- Done criteria:
  - Deterministic binary-inspection artifacts prove Blackwell and PTX coverage before smoke runs.
- Exports to later PRs/phases:
  - `fatbinary_report.json` for Phase 1 acceptance packet and later environment and profiling provenance.

## P1-04 Repo-local smoke-case pack and solver audit

- Objective:
  - Build the repo-local smoke-case bundle and enforce SPUMA-supported solver and runtime settings before profiling and sanitizer lanes proceed.
- Exact citations:
  - Authority:
    - `docs/authority/support_matrix.md` -> `## Global Policy`
    - `docs/authority/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/authority/support_matrix.md` -> `## FunctionObject Classification`
    - `docs/authority/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/authority/support_matrix.json` -> `global_policy`, `exact_audited_scheme_tuple`, `function_object_policy`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-03 Reference-case and validation-ladder utilities`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-04 Support-matrix scanner and fail-fast policy`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 9 — Add three repo-local smoke cases`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 10 — Audit fvSolution in smoke cases`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 11 — Run unprofiled smoke cases in the primary lane`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Validation strategy`
  - Backlog scope:
    - "Prepare the repo-local smoke cases and the `fvSolution` and runtime sanity checks that gate Phase 1 bring-up."
  - Backlog done_when:
    - "All required smoke cases run unprofiled, and unsupported solver settings are caught before deeper tooling passes."
- Depends on (backlog IDs):
  - `P1-02`
- Prerequisites:
  - Build outputs from `P1-02`.
  - Foundation support scanner and case-role utilities, `FND-03` and `FND-04`.
  - Canonical case metadata contract `case_meta.json` from Foundation, populated by Phase 0 freeze workflow when available.
- Concrete task slices:
  1. Create or maintain repo-local smoke-case pack with deterministic setup and no hidden external tutorial coupling.
  2. Wire support-scanner checks to fail-fast on unsupported solver, scheme, and functionObject settings before first timestep.
  3. Run unprofiled smoke execution and capture machine-readable case-level outcomes.
  4. Ensure smoke-case selection consumes canonical case metadata contract, not phase-local case-role restatements.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - smoke-case run result JSON set and audited-case manifest
    - solver and runtime audit report for smoke inputs
  - Consumed:
    - `support_matrix.json` and Foundation `FND-04` scanner contract
    - Foundation and Phase 0 case metadata contracts, `case_meta.json` and `stage_plan.json`, for consistent case identity
- Validation:
  - Required smoke cases run unprofiled with deterministic success criteria.
  - Unsupported settings are rejected pre-run with explicit reason codes.
  - No local redefinition of ladder roles, supported tuples, or fallback policy is introduced.
- Done criteria:
  - Smoke pack and solver audit gate the Phase 1 path before sanitizer and profiling.
  - Failure modes are deterministic and tied to support-matrix classifications.
- Exports to later PRs/phases:
  - Smoke-case admissibility evidence for `P1-05` and `P1-06`.
  - Audited runtime-setting baseline consumed by later phase bring-up references.

## P1-05 Compute Sanitizer memcheck lane

- Objective:
  - Add a reduced memcheck lane over the smallest supported smoke case so memory errors are caught before profiling acceptance work.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults` (Compute Sanitizer pin)
    - `docs/authority/support_matrix.md` -> `## Global Policy` (fail-fast and debug and production separation)
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-04 Support-matrix scanner and fail-fast policy`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 12 — Run Compute Sanitizer memcheck on the smallest case`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Compute Sanitizer hook`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Validation strategy`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Acceptance checklist`
  - Backlog scope:
    - "Add the reduced memcheck workflow for the smallest bring-up case so memory errors are caught before profiling work proceeds."
  - Backlog done_when:
    - "The smallest supported case passes memcheck with no actionable errors."
- Depends on (backlog IDs):
  - `P1-04`
- Prerequisites:
  - Passing unprofiled smoke-case run from `P1-04`.
  - Required build lane from `P1-02`.
- Concrete task slices:
  1. Implement memcheck wrapper script and stable log capture for the smallest supported smoke case.
  2. Define actionable-error classification and fail gate rules for Phase 1 acceptance.
  3. Export memcheck summary metadata to the acceptance bundle path.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - Compute Sanitizer memcheck logs and summarized verdict artifact
  - Consumed:
    - smoke-case pack and admissibility outputs from `P1-04`
    - pin-manifest sanitizer lane defaults
- Validation:
  - Smallest supported smoke case executes under memcheck.
  - No actionable memcheck errors remain unresolved for pass status.
- Done criteria:
  - Memcheck lane is repeatable and gating before profiling traces proceed.
- Exports to later PRs/phases:
  - Sanitizer evidence consumed by `P1-07` acceptance bundle and later profiling readiness checks.

## P1-06 Nsight Systems baseline and UVM diagnostic traces

- Objective:
  - Stand up baseline timeline and UVM-diagnostic profiling lanes with initial NVTX smoke instrumentation and trace artifact standardization.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults` (Nsight Systems and NVTX3 pins)
    - `docs/authority/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/authority/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
    - `docs/authority/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 13 — Run low-overhead Nsight Systems profile`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 14 — Run UVM-fault diagnostic Nsight Systems profile`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Instrumentation and profiling hooks`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Profiling checks`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Artifacts to produce`
  - Backlog scope:
    - "Create the basic Nsight Systems profiling lane, the UVM-focused diagnostic lane, and the initial NVTX smoke instrumentation used in Phase 1."
  - Backlog done_when:
    - "Both baseline and UVM traces show GPU kernels and visible NVTX ranges, and recurring unexplained UVM migration is eliminated or documented."
- Depends on (backlog IDs):
  - `P1-05`
- Prerequisites:
  - Passing memcheck lane from `P1-05`.
  - Foundation stage-registry contract, `FND-06`, for naming compatibility and no local stage-ID restatement.
- Concrete task slices:
  1. Implement two Nsight Systems capture modes: low-overhead baseline and UVM-focused diagnostic.
  2. Add initial NVTX smoke instrumentation and enforce range presence checks in captured traces.
  3. Emit trace artifacts and parser-ready summaries with clear mode labeling so diagnostic traces are not used as timing baselines.
  4. Add recurring-UVM migration triage output that is either clean or explicitly documented.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - baseline `.nsys-rep` traces and summaries
    - UVM diagnostic `.nsys-rep` traces and summaries
    - NVTX smoke range presence report
  - Consumed:
    - Foundation stage-registry compatibility contract, `FND-06`
    - smoke-case and memcheck outputs from `P1-04` and `P1-05`
- Validation:
  - Baseline and diagnostic traces both show GPU kernels and visible NVTX ranges.
  - UVM findings are either absent for steady-state smoke path or documented with explicit evidence.
  - Diagnostic lane is clearly separated from baseline timing mode.
- Done criteria:
  - Both profiling lanes are reproducible, artifact-complete, and acceptance-bundle ready.
- Exports to later PRs/phases:
  - Nsight trace artifacts and NVTX smoke evidence for `P1-07`.
  - Initial profiling substrate used by early Phase 8 instrumentation planning.

## P1-07 PTX-JIT proof and Phase 1 acceptance bundle

- Objective:
  - Run the PTX-JIT compatibility proof and produce the final machine-readable and human-readable Phase 1 acceptance bundle that freezes bring-up status.
- Exact citations:
  - Authority:
    - `docs/authority/master_pin_manifest.md` -> `## Frozen Defaults` (PTX/JIT mandate)
    - `docs/authority/master_pin_manifest.md` -> `## Required Revalidation If This Manifest Changes`
    - `docs/authority/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/tasks/01_foundation_authority_consumption.md` -> `## FND-05 Acceptance-manifest evaluator scaffold`
  - Phase spec or `docs/authority/README.md` authority-order note for `FND-*`:
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 15 — Run PTX-JIT compatibility test`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `#### Step 16 — Produce acceptance report and stop`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Acceptance checklist`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `### Artifacts to produce`
    - `docs/specs/phase1_blackwell_bringup_spec.md` -> `## Milestone M7 — Acceptance freeze`
  - Backlog scope:
    - "Run the PTX-JIT compatibility proof and generate the Phase 1 acceptance reports that freeze the bring-up result."
  - Backlog done_when:
    - "PTX-JIT succeeds on the workstation lane and `phase1_acceptance_report.json` and `.md` are complete and reviewable."
- Depends on (backlog IDs):
  - `P1-06`
- Prerequisites:
  - Binary inspection evidence from `P1-03`.
  - Smoke, memcheck, and profiling artifacts from `P1-04..P1-06`.
  - Foundation acceptance vocabulary and disposition framework from `FND-05`.
- Concrete task slices:
  1. Execute PTX-JIT compatibility run using the required compatibility mode and archive logs.
  2. Aggregate Phase 1 artifact set into deterministic acceptance inputs with explicit pass and fail disposition.
  3. Emit `phase1_acceptance_report.json`, `phase1_acceptance_report.md`, and `phase1_acceptance_bundle_index.json` tied to lane, source tuple, and workstation manifests.
  4. Freeze final bring-up result and publish downstream handoff packet for Phase 2+ consumers.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase1_acceptance_report.json`
    - `phase1_acceptance_report.md`
    - PTX-JIT test logs and acceptance bundle index
  - Consumed:
    - `host_env.json`, `cuda_probe.json`, `fatbinary_report.json`
    - smoke, sanitizer, and Nsight artifacts from earlier Phase 1 cards
    - Foundation acceptance evaluator terminology and disposition contract, `FND-05`
- Validation:
  - PTX-JIT run succeeds on the target workstation lane.
  - Acceptance bundle includes all mandatory artifacts and deterministic verdict fields.
  - Report status is reviewable and traceable back to frozen lane and source contracts.
- Done criteria:
  - PTX-JIT proof passes.
  - Phase 1 acceptance reports are complete, machine-readable, and ready for review and signoff.
- Exports to later PRs/phases:
  - Frozen Phase 1 acceptance bundle for `P4-01` dependency freeze and later profiling and governance consumers.
  - Confirmed workstation, toolchain, and profiler evidence baseline for Phases 2-8.

## imports_from_prev

- `FND-02` pin-resolution API plus canonical `host_env.json` and `manifest_refs.json` schema contracts.
- `FND-03` canonical case-role and case-metadata contracts, `case_meta.json` and `stage_plan.json`, used to avoid local case-role redefinition in smoke planning.
- `FND-04` support-scanner and fail-fast policy contracts for smoke-case admissibility.
- `FND-05` acceptance vocabulary and disposition scaffolding used by Phase 1 acceptance reporting.
- `FND-06` stage-registry compatibility contract used to keep NVTX and stage naming aligned with centralized governance.

## exports_to_next

- `host_env.json` as canonical lane and environment manifest for downstream evidence bundles.
- `cuda_probe.json` as workstation CUDA capability proof.
- `fatbinary_report.json` proving native target and PTX retention.
- Phase 1 smoke-case run logs and audited runtime-setting artifacts.
- Compute Sanitizer memcheck logs and verdict summary.
- Baseline and UVM diagnostic Nsight Systems trace artifacts.
- PTX-JIT logs and final `phase1_acceptance_report.json`, `phase1_acceptance_report.md`, and `phase1_acceptance_bundle_index.json`.

## shared_terms

- `primary lane`: required Phase 1 toolchain lane frozen by the master pin manifest.
- `experimental lane`: verification-only lane that does not redefine required-lane acceptance.
- `smoke-case admissibility`: pre-run validation that inputs remain inside support-matrix scope.
- `failFast`: startup rejection behavior for unsupported settings or environment mismatches.
- `canonical manifest naming`: `host_env.json` is the planning seam artifact and `env.json` is compatibility-only where needed.
- `PTX-JIT proof`: explicit compatibility test mode used to verify retained PTX functionality.

## open_discontinuities

- `[tracked] manifest-artifact-name-compatibility`: `master_pin_manifest.md` still mentions `env.json` while Foundation seam governance uses `host_env.json`; Phase 1 adopts `host_env.json` as canonical output and treats `env.json` as compatibility alias only (`docs/authority/master_pin_manifest.md` -> `## Consumption Rules`, `docs/tasks/01_foundation_authority_consumption.md` -> `## open_discontinuities`, impacted PR IDs: `P1-01`, `P1-07`, preferred reading: preserve canonical planning seam name and document compatibility mapping where required).
- `[tracked] phase0-case-metadata-availability`: Phase 1 smoke-case planning must consume canonical case metadata contracts and should consume populated Phase 0 case metadata once available, but the backlog dependency graph does not directly gate `P1-04` on Phase 0 completion (`docs/tasks/pr_inventory.md` -> `## Cross-Section Dependency Edges (Canonical)`, `docs/tasks/boundary_matrix.md` -> `### Foundation -> Phase 0 / Phase 1`, impacted PR IDs: `P1-04`, preferred reading: use Foundation `FND-03` case-role and schema contracts immediately and switch to populated Phase 0 metadata artifact as soon as it is published).

## validation_checks

- All `P1-*` cards preserve canonical dependency edges exactly: `P1-01 <- FND-02`, `P1-02 <- P1-01`, `P1-03 <- P1-02`, `P1-04 <- P1-02`, `P1-05 <- P1-04`, `P1-06 <- P1-05`, `P1-07 <- P1-06`.
- Every card cites authority docs plus exact Phase 1 spec subsections plus backlog `scope` and `done_when`.
- No card restates ladder roles, support tuples, fallback policy, or toolchain lanes locally; all are consumed from Foundation and authority contracts.
- `host_env.json` is treated as canonical seam artifact, with explicit compatibility-only handling for `env.json`.
- `cuda_probe.json` and `fatbinary_report.json` are explicit Phase 1 outputs and are carried into the Phase 1 acceptance bundle.
- Phase 1 remains scoped to platform, tooling, and profiling readiness and does not absorb Phase 0 freeze ownership, Phase 2 residency policy, or Phase 3 execution-model ownership.
