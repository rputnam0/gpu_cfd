# Foundation / authority consumption

Foundation owns `FND-01..FND-07` and provides the frozen contract layer consumed by all later phase sections. This section is intentionally limited to authority ingestion, normalization, and reusable evaluators; it does not implement CPU baseline behavior or Blackwell bring-up execution.

## FND-01 Authority ingestion scaffold

- Objective:
  - Create one authoritative loader/validator for all package-level authority docs and JSON companions so every later tool consumes the same typed decisions.
- Exact citations:
  - Authority:
    - `docs/continuity_ledger.md` -> `# 4. Central Package Authorities`
    - `docs/continuity_ledger.md` -> `# 5. Package Consumption Rule`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (runtime configuration normalization and ownership centralization)
    - `docs/reference_case_contract.json` (machine-readable case-role source)
    - `docs/support_matrix.json` (machine-readable support-scope source)
    - `docs/acceptance_manifest.json` (machine-readable acceptance source)
    - `docs/graph_capture_support_matrix.json` (machine-readable stage-policy source)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Add a shared loader/validator for the package authority docs and JSON companions so later phases consume typed program-wide decisions instead of hard-coded constants."
  - Backlog done_when:
    - "One command/library path loads the authority set, validates schema/versioning, and exposes a stable API used by downstream tools."
- Depends on (backlog IDs):
  - None.
- Prerequisites:
  - None.
- Concrete task slices:
  1. Define a single `AuthorityBundle` contract with typed submodels for pins, cases, ladder, support scope, acceptance tuples, graph stages, and semantic source map ownership.
  2. Implement one loader entrypoint that ingests the authority Markdown set plus JSON companions and validates schema-version presence for JSON artifacts.
  3. Add consistency checks that reject contradictory ownership, for example phase-local values that conflict with authority defaults.
  4. Publish one stable API and one CLI-style command path used by downstream planning and validation tools.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `AuthorityBundle` API contract
    - authority-load report schema for pass/fail diagnostics
  - Consumed:
    - all authority docs listed in `README_FIRST`
    - JSON companions named in `README_FIRST` consumption rules
- Validation:
  - Positive test: complete authority bundle loads with no missing required documents.
  - Negative tests: missing file, unknown schema version, and conflicting authority value each fail fast with explicit diagnostics.
  - Contract test: downstream caller resolves all authority categories from one API without local restatement.
- Done criteria:
  - Single shared loader path exists and is adopted by downstream foundation utilities.
  - Loader enforces schema/version checks and deterministic error reporting.
- Exports to later PRs/phases:
  - Shared authority-loading API used by `FND-02..FND-07`, then by Phases 0-8.

## FND-02 Pin-manifest consumption and environment manifest emission

- Objective:
  - Normalize toolchain/source/profiler pins into machine-readable manifests used by build, run, and profiling workflows.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/master_pin_manifest.md` -> `## Resolved Frozen Source Tuple`
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/master_pin_manifest.md` -> `## Required Revalidation If This Manifest Changes`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (source/version/toolchain pin ownership)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Implement ingestion of the master pin manifest and emit machine-readable environment manifests for builds, runs, and profiling artifacts."
  - Backlog done_when:
    - "Build/run tooling can resolve the frozen toolchain lane and emit `host_env` / manifest artifacts without duplicating pin logic."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Implement pin-manifest parser/view on top of `AuthorityBundle` for runtime base, toolkit lanes, driver floor, GPU target, and profiler versions.
  2. Define and emit canonical machine-readable artifacts for environment and tuple traceability: `host_env.json` and `manifest_refs.json`.
  3. Define compatibility alias policy so older artifact names map to canonical outputs without changing frozen pin semantics.
  4. Add one shared resolution function used by build, run, and profiling launchers instead of per-tool pin duplication.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `host_env.json` schema
    - `manifest_refs.json` schema
    - pin-resolution API contract
  - Consumed:
    - `master_pin_manifest.md` frozen tuple and lane definitions
    - `FND-01` authority loader
- Validation:
  - Manifest emission reproduces frozen lane values exactly.
  - Drift test: when a local override conflicts with frozen pins, resolution fails with explicit violation.
  - Integration test: build, run, and profiling utilities consume emitted manifests, not duplicated constants.
- Done criteria:
  - Pin ingestion is centralized and deterministic.
  - `host_env` and manifest-reference artifacts are emitted from one shared source and consumed by downstream tooling.
- Exports to later PRs/phases:
  - `host_env.json`
  - `manifest_refs.json`
  - pin-resolution API for Phase 0 and Phase 1 probes and artifact packagers.

## FND-03 Reference-case and validation-ladder utilities

- Objective:
  - Provide canonical case and ladder resolution helpers so every phase selects `R2 -> R1-core -> R1 -> R0` by role, never by ad hoc strings.
- Exact citations:
  - Authority:
    - `docs/reference_case_contract.md` -> `## Frozen Cases`
    - `docs/reference_case_contract.md` -> `## Phase-Gate Mapping`
    - `docs/reference_case_contract.md` -> `## Locked Defaults`
    - `docs/reference_case_contract.json` -> `frozen_cases`, `phase_gate_mapping`, `locked_defaults`
    - `docs/validation_ladder.md` -> `## Frozen Ladder`
    - `docs/validation_ladder.md` -> `## Usage Rule`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Create helpers that resolve canonical case IDs, ladder roles (`R2`, `R1-core`, `R1`, `R0`), and phase-gate selection."
  - Backlog done_when:
    - "Scripts and tests can select cases by ladder role and reject unknown or out-of-scope case IDs."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Implement case resolver APIs for role-based lookup and phase-gate mapping from `reference_case_contract.json`.
  2. Add validation utilities that reject unknown case IDs, renamed ladder rungs, or out-of-scope gate selections.
  3. Define canonical `case_meta.json` schema emitted from authority-owned case identifiers and roles.
  4. Define canonical `stage_plan.json` schema for ladder-aware phase gate selection metadata without phase-local ladder rewrites.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - case-role resolver API
    - `case_meta.json` schema
    - `stage_plan.json` schema for authority-aware gate selection metadata
  - Consumed:
    - reference-case contract Markdown and JSON
    - validation ladder alias contract
    - `FND-01` authority loader
- Validation:
  - Positive: each ladder role resolves to the frozen case ID.
  - Negative: unknown role or case and reordered ladder members fail validation.
  - Phase-gate mapping tests confirm role availability per frozen mapping.
- Done criteria:
  - Case and ladder lookup is centralized and role-based.
  - Unknown or out-of-scope case selections fail before execution starts.
- Exports to later PRs/phases:
  - `case_meta.json` and case-role resolver, required for the Foundation -> Phase 0 / Phase 1 seam.
  - `stage_plan.json` schema for downstream stage planning and reporting.

## FND-04 Support-matrix scanner and fail-fast policy

- Objective:
  - Centralize admissibility checks for schemes, BCs, functionObjects, turbulence/contact-angle scope, backend eligibility, and fallback policy.
- Exact citations:
  - Authority:
    - `docs/support_matrix.md` -> `## Global Policy`
    - `docs/support_matrix.md` -> `## Exact Audited Scheme Tuple`
    - `docs/support_matrix.md` -> `## FunctionObject Classification`
    - `docs/support_matrix.md` -> `## Phase 6 Nozzle-Specific Envelope`
    - `docs/support_matrix.md` -> `## Canonical Startup-Seed DSL`
    - `docs/support_matrix.md` -> `## Backend and Operational Policy`
    - `docs/support_matrix.json` -> `global_policy`, `exact_audited_scheme_tuple`, `function_object_policy`, `phase6_nozzle_specific_envelope`, `startup_seed_dsl`, `backend_operational_policy`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (support-matrix ownership and GPU-only operational contract)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Implement centralized support scanning for BCs, schemes, turbulence scope, function objects, backend eligibility, and fallback policy."
  - Backlog done_when:
    - "Unsupported tuples are rejected before the run starts, and debug-only fallback stays non-default and explicit."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Build support-scanner core that evaluates runtime and case configuration against `support_matrix.json` instead of phase-local rules.
  2. Implement fail-fast startup checks for unsupported schemes, BC tuples, turbulence/contact-angle scope violations, and disallowed functionObjects in production mode.
  3. Implement explicit mode labeling for debug-only fallback paths so no unsupported production run silently degrades.
  4. Emit deterministic support-scan reports with machine-readable reject reasons and authority citations.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - support-scanner API
    - support-scan report schema
    - fail-fast policy contract with `failFast` default and explicit debug-only fallback
  - Consumed:
    - `support_matrix.md` and `support_matrix.json`
    - `FND-01` authority loader
- Validation:
  - Supported tuple passes and generates a clean report.
  - Unsupported tuple classes fail before first timestep with explicit reasons.
  - Debug-only fallback test confirms opt-in requirement and non-production labeling.
- Done criteria:
  - Startup scanner rejects unsupported configurations pre-run.
  - Production path cannot silently use debug-only fallback.
- Exports to later PRs/phases:
  - support-scanner API and report schema for Phase 5 and Phase 6 gating and Phase 1 smoke-case admissibility checks.

## FND-05 Acceptance-manifest evaluator scaffold

- Objective:
  - Establish one reusable evaluator for tuple admission, gate checks, threshold classes, parity classes, and deterministic disposition.
- Exact citations:
  - Authority:
    - `docs/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/acceptance_manifest.md` -> `## Soft Gates`
    - `docs/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/acceptance_manifest.md` -> `## Production Defaults`
    - `docs/acceptance_manifest.json` -> `accepted_tuples`, `coverage_rules`, `nvtx_contract_defaults`, `hard_gates`, `soft_gates`, `threshold_classes`, `disposition_rules`, `production_defaults`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Build a reusable acceptance evaluator that understands tuple IDs, tolerance classes, parity classes, hard/soft gates, and waiver hooks."
  - Backlog done_when:
    - "Later phases can submit artifacts to one common evaluator and get deterministic pass/fail dispositions tied to manifest entries."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Implement tuple resolver keyed by accepted tuple IDs from `acceptance_manifest.json`.
  2. Implement evaluator core for hard and soft gate calculation and threshold-class dispatch.
  3. Define waiver-hook interface with explicit revision and tuple binding requirements.
  4. Emit deterministic acceptance-verdict artifact schema with tuple ID, gate results, thresholds used, and disposition reason.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - acceptance-evaluator API
    - acceptance-verdict artifact schema
    - waiver-hook interface contract
  - Consumed:
    - acceptance manifest Markdown and JSON
    - `FND-01` authority loader
- Validation:
  - Required tuple row resolves and evaluates deterministically.
  - Unknown tuple ID fails with explicit non-admitted disposition.
  - Hard-gate violation forces fail regardless of soft-gate status.
- Done criteria:
  - One common evaluator exists and is reusable by later phases.
  - Deterministic disposition output is tied to manifest tuple IDs and threshold classes.
- Exports to later PRs/phases:
  - acceptance-evaluator API and verdict schema for Phase 5 and Phase 8 formal acceptance runs.

## FND-06 Graph stage registry and graph-support-matrix loader

- Objective:
  - Centralize stage IDs, run-mode policy, and graph support metadata so execution, instrumentation, and acceptance share one stage taxonomy.
- Exact citations:
  - Authority:
    - `docs/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/graph_capture_support_matrix.json` -> `run_modes`, `stages`, `required_orchestration_ranges`, `global_capture_rules`
    - `docs/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/acceptance_manifest.json` -> `accepted_tuples[*].required_stage_ids`, `nvtx_contract_defaults.required_orchestration_ranges`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (`gpuRuntime` normalization and centralized graph policy ownership)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Create the canonical stage-ID registry and graph-policy loader used by execution, instrumentation, and acceptance code."
  - Backlog done_when:
    - "All graph-aware code resolves stage IDs from one shared source, and tuple-to-stage validation runs without local restatement."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Implement stage-registry loader from `graph_capture_support_matrix.json` and expose stage and range lookup API.
  2. Implement run-mode and fallback-policy resolver from the same source: `sync_debug`, `async_no_graph`, and `graph_fixed`.
  3. Add tuple-to-stage validator that checks acceptance tuple stage requirements against canonical stage IDs.
  4. Emit registry report artifact for downstream stage-plan and NVTX instrumentation checks.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - canonical stage registry API
    - graph-policy loader API
    - tuple-to-stage validation API
    - stage-registry report schema
  - Consumed:
    - graph-capture support matrix Markdown and JSON
    - acceptance manifest tuple-stage requirements
    - `FND-01` authority loader
- Validation:
  - Every accepted tuple required stage ID resolves to canonical stage registry entries.
  - Unknown stage IDs or local aliases fail validation.
  - Fallback mode resolution is deterministic and matches authority defaults.
- Done criteria:
  - One shared stage registry is used by graph-aware callers.
  - Tuple-to-stage checks run without local stage restatement.
- Exports to later PRs/phases:
  - stage registry and graph-policy APIs for Phase 3 execution model and Phase 8 instrumentation and acceptance.
  - canonical stage metadata input for `stage_plan.json` consumers.

## FND-07 Semantic source-audit helper

- Objective:
  - Ensure all later implementation phases patch the correct local SPUMA/v2412 semantic targets and avoid wrong upstream analog edits.
- Exact citations:
  - Authority:
    - `docs/semantic_source_map.md` -> `## Frozen Mapping`
    - `docs/semantic_source_map.md` -> `## Implementation Rule`
    - `docs/continuity_ledger.md` -> `# 5. Package Consumption Rule` (semantic patch targets cannot be redefined locally)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/README_FIRST.md` -> `## Authoritative docs`
    - `docs/README_FIRST.md` -> `## Consumption rules`
  - Backlog scope:
    - "Add tooling and templates that map semantic source references to the exact local SPUMA/v2412 patch targets before any solver modifications."
  - Backlog done_when:
    - "Each implementation phase can generate a reviewed source-audit note and avoid patching the wrong upstream analog."
- Depends on (backlog IDs):
  - `FND-01`
- Prerequisites:
  - `FND-01`
- Concrete task slices:
  1. Define source-audit template structure keyed by semantic contract surface and local target family.
  2. Implement helper that resolves each semantic surface to required local target family entries before patch planning.
  3. Add validation that rejects implementation plans missing a reviewed source-audit note for touched semantic surfaces.
  4. Standardize source-audit note output format for downstream phase sections.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - source-audit helper API
    - `source_audit_note.md` template contract
  - Consumed:
    - semantic source map authority
    - `FND-01` authority loader
- Validation:
  - Positive: each mapped semantic surface resolves to one local target family.
  - Negative: unknown semantic surface or missing source-audit note fails pre-implementation checks.
  - Reviewability: generated note includes semantic reference, local target family, and ownership scope.
- Done criteria:
  - Later phase planning can generate a reviewed, consistent source-audit note.
  - Wrong-target patch risk is reduced through required semantic mapping checks.
- Exports to later PRs/phases:
  - source-audit helper and template used by Phase 4-7 implementation planning and review.

## imports_from_prev

- None. Foundation is the first section and defines shared authority-consumption contracts.

## exports_to_next

- `AuthorityBundle` loader and validator API (`FND-01`).
- Pin-resolution and environment-manifest contracts: `host_env.json`, `manifest_refs.json` (`FND-02`).
- Case and ladder contracts: case-role resolver, `case_meta.json`, `stage_plan.json` schemas (`FND-03`).
- Support-scanner and fail-fast policy contracts (`FND-04`).
- Acceptance-evaluator and verdict schema (`FND-05`).
- Canonical stage-registry and graph-policy loader contracts (`FND-06`).
- Semantic source-audit helper and `source_audit_note.md` template (`FND-07`).
- Foundation seam artifacts explicitly required by downstream gates:
  - `case_meta.json`
  - `stage_plan.json`
  - `host_env.json`
  - `cuda_probe.json` produced by Phase 1 using Foundation schemas and contracts
  - `fatbinary_report.json` produced by Phase 1 using Foundation schemas and contracts

## shared_terms

- `authority layer`: the ordered set in `README_FIRST` that overrides conflicting phase-local wording.
- `JSON companion`: machine-readable authority source consumed by automation and validators.
- `case role`: one of `R2`, `R1-core`, `R1`, `R0` resolved from `reference_case_contract`.
- `failFast`: default unsupported-config behavior; reject before first timestep.
- `accepted tuple`: one admitted acceptance row keyed by tuple ID in `acceptance_manifest`.
- `stage ID`: canonical parent NVTX and stage taxonomy key from `graph_capture_support_matrix`.
- `semantic contract surface`: high-level solver responsibility mapped to local SPUMA/v2412 target families.

## open_discontinuities

- `[tracked] manifest-artifact-name-compatibility`: `master_pin_manifest.md` references `env.json` and `manifest_refs.json`, while the seam gate and Phase 1 planning use `host_env.json` plus probe and fatbinary reports; Foundation will standardize canonical names and document compatibility aliases in `FND-02` (`docs/master_pin_manifest.md` -> `## Consumption Rules`, `docs/planning/pr_expansion/boundary_matrix.md` -> `### Foundation -> Phase 0 / Phase 1`, impacted PR IDs: `FND-02`, `P0-01`, `P1-01`, preferred reading: keep `host_env.json` canonical for planning seams and allow `env.json` as compatibility alias only).

## validation_checks

- All seven `FND-*` cards preserve backlog dependency edges: `FND-01` root, then `FND-02..FND-07` depend on `FND-01`.
- Every card cites authority sources plus `README_FIRST` authority-order and consumption anchors.
- No Foundation card absorbs Phase 0 baseline-freeze implementation behavior.
- No Foundation card absorbs Phase 1 Blackwell bring-up execution behavior.
- Exported artifacts satisfy the Foundation -> Phase 0 / Phase 1 seam gate contract in `boundary_matrix.md`.
- Any unresolved naming or ownership ambiguity is recorded under `open_discontinuities` instead of widened scope.
