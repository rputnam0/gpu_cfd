# Phase 8 — Profiling and performance acceptance

Phase 8 pass A owns `P8-01..P8-08` and delivers the instrumentation and profiling substrate consumed by later acceptance and optimization phases. This section now also includes pass B `P8-09`, which consumes `P7-08` evidence and formalizes tuple-bound baseline locks plus CI/nightly scheduling policy.

## P8-01 NVTX wrapper library and build flags

- Objective:
  - Add permanent NVTX v3 wrapper infrastructure and build-time switches so profiling instrumentation is compiled consistently across supported solver modules.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (NVTX3 baseline and centralized runtime governance)
    - `docs/planning/pr_expansion/03_phase1_blackwell_bringup.md` -> `## P1-02 Blackwell build-system enablement`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.4 — Make NVTX v3 mandatory and design the range hierarchy before adding kernels`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 1 — Add NVTX v3 dependency and build flags`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 8. Add build-system changes`
  - Backlog scope:
    - "Add the permanent NVTX3 wrapper library, compile-time switches, and dependency wiring required by the profiling subsystem."
  - Backlog done_when:
    - "Profiling instrumentation can be compiled in/out cleanly and is available to all supported solver modules."
- Depends on (backlog IDs):
  - `P1-02`
- Prerequisites:
  - Phase 1 Blackwell build-lane enablement and frozen toolchain lane evidence.
  - Foundation stage-registry contract vocabulary from `FND-06`.
- Concrete task slices:
  1. Add `src/gpu/profiling` wrapper substrate and guarded NVTX include path with centralized macros and RAII scopes.
  2. Wire build flags and dependency checks so instrumentation toggles are explicit and deterministic.
  3. Add compile-path validation for enabled and disabled instrumentation modes across supported modules.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - NVTX wrapper API and compile-switch contract
    - build-time profiling capability manifest fields
  - Consumed:
    - Phase 1 build metadata and lane constraints
    - canonical stage naming contract from Foundation
- Validation:
  - Both instrumented and non-instrumented builds compile cleanly on required lane.
  - Wrapper API is reachable from supported solver modules without direct ad hoc NVTX calls.
- Done criteria:
  - NVTX wrapper and build switches are permanent and reusable.
  - Instrumentation availability is deterministic across supported modules.
- Exports to later PRs/phases:
  - Wrapper and build-switch substrate consumed by `P8-02`, `P8-03`, and `P8-04`.

## P8-02 Canonical profiling and acceptance config parser

- Objective:
  - Implement canonical runtime parsing for `gpuRuntime.profiling` and `gpuRuntime.acceptance`, with compatibility shims but no local shadow schemas.
- Exact citations:
  - Authority:
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (normalized `gpuRuntime.*` contract)
    - `docs/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-05 Acceptance-manifest evaluator scaffold`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.1 — Introduce six explicit profiling modes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.2 — Use NVTX v3 domains and categories with a fixed naming grammar`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 3 — Add canonical gpuRuntime.profiling / gpuRuntime.acceptance parsing`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 2. Runtime configuration interface`
  - Backlog scope:
    - "Implement the normalized `gpuRuntime.profiling` / `gpuRuntime.acceptance` parser and compatibility shims."
  - Backlog done_when:
    - "All profiling and acceptance behavior is runtime-configurable from the canonical tree with no local shadow schema."
- Depends on (backlog IDs):
  - `FND-05`
  - `FND-06`
  - `P8-01`
- Prerequisites:
  - NVTX wrapper and compile-switch substrate from `P8-01`.
  - Foundation acceptance and stage-registry loaders.
- Concrete task slices:
  1. Implement canonical profiling and acceptance parser aligned with `gpuRuntime.*` normalization.
  2. Add compatibility mapping for legacy keys into canonical schema without changing canonical source of truth.
  3. Bind parsed config to accepted mode set and canonical stage references.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - profiling and acceptance parser contract
    - normalized config validation report
  - Consumed:
    - acceptance manifest and stage-registry authority contracts
    - NVTX wrapper capabilities from `P8-01`
- Validation:
  - Valid configurations resolve into one canonical runtime config.
  - Unknown/invalid keys fail with deterministic diagnostics.
  - Parsed stage references map only to canonical stage IDs from Foundation contracts.
- Done criteria:
  - Runtime behavior is driven by canonical profiling/acceptance config tree.
  - No local shadow schemas remain in pass-A planning scope.
- Exports to later PRs/phases:
  - Canonical profiling config API consumed by `P8-03..P8-08` and pass-B `P8-09`.

## P8-03 Solver-stage instrumentation coverage

- Objective:
  - Instrument required solver stages with canonical NVTX hierarchy aligned to stage-registry contract.
- Exact citations:
  - Authority:
    - `docs/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/acceptance_manifest.json` -> `accepted_tuples[*].required_stage_ids`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
    - `docs/planning/pr_expansion/pr_inventory.md` -> `## Cross-Section Dependency Edges (Canonical)` (`P5-11 <- P8-03`)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.2 — Use NVTX v3 domains and categories with a fixed naming grammar`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 4 — Instrument top-level timestep and PIMPLE scopes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 5 — Instrument alpha path scopes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 6 — Instrument momentum, pressure, and surface-tension scopes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 7 — Instrument boundary and output staging scopes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Mandatory NVTX range inventory`
  - Backlog scope:
    - "Instrument the top-level timestep/PIMPLE/alpha/momentum/pressure/surface/boundary/output stages with the canonical NVTX naming hierarchy."
  - Backlog done_when:
    - "The required solver-stage ranges appear in traces for accepted tuples and are aligned with the canonical stage registry."
- Depends on (backlog IDs):
  - `P8-02`
  - `P3-05`
- Prerequisites:
  - Canonical profiling parser from `P8-02`.
  - Phase 3 canonical stage scaffolding.
- Concrete task slices:
  1. Apply NVTX instrumentation to required solver stage boundaries using canonical names only.
  2. Add range-presence validation hooks keyed to tuple-required stage IDs.
  3. Emit instrumentation coverage report used by downstream acceptance and seam checks.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - stage-instrumentation coverage report
    - tuple-to-range presence checks
  - Consumed:
    - canonical stage IDs and tuple-stage requirements from authority contracts
    - Phase 3 stage scaffolding
- Validation:
  - Required stage ranges appear in traces for accepted tuple classes.
  - Range names match canonical registry identifiers with no local aliases.
- Done criteria:
  - Stage instrumentation coverage is complete for pass-A required stages.
  - Coverage output is consumable by Phase 5 baseline-freeze prerequisites.
- Exports to later PRs/phases:
  - `P8-03` coverage artifact explicitly feeding `P5-11`.
  - Canonical stage-coverage evidence for `P8-05` and `P8-06`.

## P8-04 Graph lifecycle instrumentation

- Objective:
  - Instrument graph capture/build/replay/update lifecycle events without changing solver semantics.
- Exact citations:
  - Authority:
    - `docs/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-06 Graph stage registry and graph-support-matrix loader`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.6 — Use graph-level tracing by default; use node-level tracing only for graph debugging`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 8 — Add graph lifecycle markers`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 5. Graph instrumentation`
  - Backlog scope:
    - "Instrument graph capture/build/replay/update lifecycle events so graph behavior is visible in traces and parsers."
  - Backlog done_when:
    - "Graph-enabled runs emit the required graph lifecycle markers without changing solver semantics."
- Depends on (backlog IDs):
  - `P8-02`
  - `P3-07`
- Prerequisites:
  - Canonical profiling parser from `P8-02`.
  - Phase 3 graph fingerprint/cache/rebuild policy.
- Concrete task slices:
  1. Add lifecycle markers for graph build, replay, update, and rebuild decision points.
  2. Preserve graph-debug marker separation from baseline timing instrumentation.
  3. Emit lifecycle-marker report for parser and acceptance consumption.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - graph lifecycle marker set and report contract
  - Consumed:
    - canonical run modes and stage IDs from graph-support authority
    - Phase 3 graph policy interfaces
- Validation:
  - Graph-enabled traces contain required lifecycle markers in expected order.
  - Marker instrumentation does not alter execution semantics or fallback behavior.
- Done criteria:
  - Graph lifecycle visibility is present and parser-consumable.
- Exports to later PRs/phases:
  - Lifecycle instrumentation artifacts consumed by `P8-05` and pass-B baseline-lock analysis.

## P8-05 Nsight Systems capture scripts and artifact layout

- Objective:
  - Build reproducible `nsys` capture workflow, mode selection, environment manifests, and standardized artifact layout.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/acceptance_manifest.md` -> `## Production Defaults`
    - `docs/planning/pr_expansion/03_phase1_blackwell_bringup.md` -> `## exports_to_next`
    - `docs/planning/pr_expansion/pr_inventory.md` -> `## Cross-Section Dependency Edges (Canonical)` (`P7-01 <- P8-05`)
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.5 — Baseline Nsight Systems command uses --trace=cuda,nvtx,osrt and graph-level graph tracing`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.6 — --cuda-event-trace=false by default in baseline mode`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.12 — Record full environment metadata in every acceptance artifact`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 9 — Add baseline nsys wrapper script`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Required profiler command families`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 6. Baseline nsys wrapper`
  - Backlog scope:
    - "Create the Phase 8 `nsys` capture scripts, mode selection, environment manifests, and artifact directory layout."
  - Backlog done_when:
    - "A baseline `R1` timeline can be captured end-to-end in the standardized artifact layout with reproducible command lines."
- Depends on (backlog IDs):
  - `P8-03`
  - `P8-04`
- Prerequisites:
  - Solver-stage coverage from `P8-03`.
  - Graph lifecycle markers from `P8-04`.
  - Phase 1 environment evidence (`host_env.json`, `cuda_probe.json`, `fatbinary_report.json`).
- Concrete task slices:
  1. Implement baseline and diagnostic capture script entrypoints with explicit mode selection.
  2. Define standardized artifact directory schema and manifest files for capture provenance.
  3. Capture and attach required environment metadata for each profiling run.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `nsys` capture script suite
    - standardized profiling artifact layout and manifest schema
  - Consumed:
    - phase-1 environment manifests and probe artifacts
    - coverage and lifecycle markers from `P8-03` and `P8-04`
- Validation:
  - Baseline `R1` timeline capture succeeds in standardized layout.
  - Command lines and artifacts are reproducible across repeated captures.
  - Artifact manifests include required environment metadata fields.
- Done criteria:
  - `nsys` capture workflow is reproducible and standardized.
  - Capture outputs are ready for parser ingestion and downstream hotspot analysis.
- Exports to later PRs/phases:
  - `P8-05` artifact layout and capture scripts explicitly feeding `P7-01`.
  - Capture artifacts consumed by `P8-06`.

## P8-06 Stats export and profile-acceptance parser

- Objective:
  - Export `nsys` reports and map metrics to manifest-driven hard and soft gates for first formal `R1` profiling acceptance summary.
- Exact citations:
  - Authority:
    - `docs/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/acceptance_manifest.md` -> `## Soft Gates`
    - `docs/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/acceptance_manifest.md` -> `## Exact Threshold Classes`
    - `docs/acceptance_manifest.json` -> `hard_gates`, `soft_gates`, `threshold_classes`, `disposition_rules`
    - `docs/planning/pr_expansion/01_foundation_authority_consumption.md` -> `## FND-05 Acceptance-manifest evaluator scaffold`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.9 — Use nsys stats built-in reports as the primary parser surface`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.10 — Define pass/fail gates in tiers`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 10 — Add nsys export + report-generation script`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 11 — Implement parser and acceptance evaluator`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 7. nsys export/parser`
  - Backlog scope:
    - "Implement report export from `nsys` outputs and the acceptance parser that maps metrics back to manifest-driven hard/soft gates."
  - Backlog done_when:
    - "The branch can produce a first formal `R1` acceptance summary from captured profiling artifacts."
- Depends on (backlog IDs):
  - `P8-05`
  - `FND-05`
- Prerequisites:
  - Standardized capture artifacts from `P8-05`.
  - Foundation acceptance evaluator scaffold and vocabulary.
- Concrete task slices:
  1. Implement `nsys stats` export pipeline and normalized metric extraction.
  2. Map parsed metrics into acceptance-manifest hard and soft gate evaluation.
  3. Emit first formal `R1` profile-acceptance summary artifacts with deterministic disposition fields.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - profile-acceptance parser contract
    - formal `R1` profiling acceptance summary schema
  - Consumed:
    - `nsys` capture artifacts from `P8-05`
    - acceptance evaluator and threshold vocabulary from Foundation
- Validation:
  - Parser output includes gate-level pass/fail and disposition mapping.
  - Metric-to-gate mapping is deterministic for repeated runs of same artifact set.
- Done criteria:
  - First formal `R1` profiling acceptance summary can be produced from pass-A artifacts.
- Exports to later PRs/phases:
  - Acceptance parser and summary artifacts consumed by `P8-07`, `P8-08`, and pass-B baseline lock decisions.

## P8-07 Diagnostic profiling modes

- Objective:
  - Add `uvmAudit`, `syncAudit`, and `graphDebug` modes as diagnostics only, outside formal baseline timing mode.
- Exact citations:
  - Authority:
    - `docs/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/planning/pr_expansion/boundary_matrix.md` -> `### Phase 7 -> Phase 8`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.1 — Introduce six explicit profiling modes`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.7 — UVM page-fault tracing enabled only in uvmAudit mode`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.11 — Use CPU sampling/backtraces only in syncAudit`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 12 — Add UVM audit mode`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 13 — Add sync audit mode and expert analysis capture`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 3. UVM audit algorithm`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 4. Synchronization audit algorithm`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 5. Graph debug algorithm`
  - Backlog scope:
    - "Add the `uvmAudit`, `syncAudit`, and `graphDebug` modes, keeping them diagnostically useful but outside baseline production timing mode."
  - Backlog done_when:
    - "Reduced `R1` runs can separately audit unexpected UVM traffic, inner-loop synchronization, and graph structure with stable artifacts."
- Depends on (backlog IDs):
  - `P8-06`
- Prerequisites:
  - Profile-acceptance parser and summary outputs from `P8-06`.
- Concrete task slices:
  1. Implement diagnostic mode routing and guardrails that prevent contamination of baseline timing rows.
  2. Add mode-specific capture and report outputs for UVM, synchronization, and graph diagnostics.
  3. Add validation that diagnostic modes remain explicitly labeled and excluded from formal baseline outputs.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - diagnostic mode output schemas for `uvmAudit`, `syncAudit`, and `graphDebug`
  - Consumed:
    - pass-A capture and parser infrastructure from `P8-05` and `P8-06`
- Validation:
  - Reduced `R1` runs produce stable artifacts for each diagnostic mode.
  - Diagnostic outputs are clearly excluded from formal timing baseline artifacts.
- Done criteria:
  - Diagnostic modes are operational, separable, and non-contaminating.
- Exports to later PRs/phases:
  - Diagnostic artifacts consumed by `P8-08` and later pass-B readiness review.

## P8-08 Top-kernel NCU and sanitizer automation

- Objective:
  - Add top-five-kernel Nsight Compute and Compute Sanitizer automation without contaminating baseline timing runs.
- Exact citations:
  - Authority:
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/planning/pr_expansion/03_phase1_blackwell_bringup.md` -> `## P1-05 Compute Sanitizer memcheck lane`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.3 — Restrict Nsight Compute to the top five kernels per steady-state run`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### DD-8.11 — Use Compute Sanitizer in a staged order and only on reduced cases`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 14 — Add top-five-kernel selector and NCU wrapper`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 15 — Add sanitizer script and nightly target`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 8. Top-kernel selection and NCU invocation`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### 9. Sanitizer wrapper`
  - Backlog scope:
    - "Add the top-five-kernel Nsight Compute wrapper, section configuration, Compute Sanitizer wrapper, and nightly reduced-case automation."
  - Backlog done_when:
    - "The project can produce top-kernel reports and clean reduced-case sanitizer logs without contaminating baseline timing runs."
- Depends on (backlog IDs):
  - `P8-06`
  - `P8-07`
- Prerequisites:
  - Parser and gate mapping from `P8-06`.
  - Diagnostic mode routing from `P8-07`.
- Concrete task slices:
  1. Implement top-kernel selector from baseline captures and wrap Nsight Compute invocation with fixed section configuration.
  2. Add reduced-case Compute Sanitizer wrapper and stable artifact naming.
  3. Add nightly automation entrypoints for reduced-case NCU/sanitizer runs with explicit non-baseline labeling.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - top-kernel NCU report artifacts
    - reduced-case sanitizer automation artifacts
  - Consumed:
    - parser outputs and diagnostic mode outputs from `P8-06` and `P8-07`
    - phase-1 sanitizer-lane constraints and compatibility context
- Validation:
  - Top-five-kernel reports are produced reproducibly from captured runs.
  - Reduced-case sanitizer logs are clean or fail with deterministic diagnostics.
  - Outputs remain separated from baseline timing artifacts.
- Done criteria:
  - NCU and sanitizer automation is functional and repeatable for reduced-case lanes.
  - Baseline timing rows are not contaminated by deep-dive workflows.
- Exports to later PRs/phases:
  - Deep-dive artifact suite consumed by pass-B baseline-lock review and Phase 7/8 seam checks.

## P8-09 Baseline locks and CI/nightly integration

- Objective:
  - Run formal tuple-driven `R1` and `R0` acceptance, lock first approved baselines, and operationalize scheduled profiling/sanitizer jobs with explicit retention and lock-update policy.
- Exact citations:
  - Authority:
    - `docs/acceptance_manifest.md` -> `## Accepted Tuple Matrix`
    - `docs/acceptance_manifest.md` -> `## Coverage Rules`
    - `docs/acceptance_manifest.md` -> `## Tuple-Specific NVTX Contract`
    - `docs/acceptance_manifest.md` -> `## Hard Gates`
    - `docs/acceptance_manifest.md` -> `## Soft Gates`
    - `docs/acceptance_manifest.md` -> `## Disposition Rules`
    - `docs/acceptance_manifest.md` -> `## Production Defaults`
    - `docs/acceptance_manifest.json` -> `accepted_tuples`
    - `docs/acceptance_manifest.json` -> `accepted_tuples[*].required_stage_ids`
    - `docs/acceptance_manifest.json` -> `coverage_rules`
    - `docs/acceptance_manifest.json` -> `hard_gates`
    - `docs/acceptance_manifest.json` -> `soft_gates`
    - `docs/acceptance_manifest.json` -> `threshold_classes`
    - `docs/acceptance_manifest.json` -> `production_defaults`
    - `docs/graph_capture_support_matrix.md` -> `## Run Modes`
    - `docs/graph_capture_support_matrix.md` -> `## Canonical Stage IDs`
    - `docs/graph_capture_support_matrix.md` -> `## Global Capture Rules`
    - `docs/graph_capture_support_matrix.json` -> `run_modes`
    - `docs/graph_capture_support_matrix.json` -> `stages`
    - `docs/graph_capture_support_matrix.json` -> `required_orchestration_ranges`
    - `docs/continuity_ledger.md` -> `# 1. Frozen global decisions` (no silent fallback, canonical runtime ownership, native baseline policy)
    - `docs/master_pin_manifest.md` -> `## Frozen Defaults`
    - `docs/master_pin_manifest.md` -> `## Consumption Rules`
    - `docs/planning/pr_expansion/boundary_matrix.md` -> `### Phase 7 -> Phase 8`
    - `docs/planning/pr_expansion/09_phase7_custom_cuda_kernels.md` -> `## P7-08 Graph-safety cleanup, capture validation, and final regression package`
  - Phase spec or `README_FIRST` authority-order note for `FND-*`:
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 16 — Define default thresholds and baseline files`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 17 — Run R1 baseline and lock first baseline`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 18 — Run R0 acceptance and compare against R1`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 19 — Add CI/nightly gates`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `#### Step 20 — Write the profiling operator guide`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `### Milestone P8.7 — Production-case acceptance`
    - `docs/phase8_profiling_performance_acceptance_spec.md` -> `### Milestone P8.8 — CI/nightly integration`
  - Backlog scope:
    - "Run the formal `R1` / `R0` acceptance framework, lock the first approved baselines, and integrate the scheduled profiling/sanitizer jobs."
  - Backlog done_when:
    - "The first approved `R0` acceptance bundle exists, `R1` nightly / `R0` scheduled jobs are wired, and retention/lock policy is operational."
- Depends on (backlog IDs):
  - `P8-08`
  - `P7-08`
- Prerequisites:
  - `P8-08` parser + capture + diagnostic/deep-dive automation outputs are stable.
  - `P7-08` selected-backend decision and final Phase 7 regression package are complete and mapped to canonical stage IDs.
- Concrete task slices:
  1. Define and version lock inputs: `thresholds_default` values, tuple eligibility filters, baseline-lock schema, and lock-update workflow keyed by tuple ID.
  2. Run formal `R1` acceptance on required production-eligible rows (`P8_R1_NATIVE_ASYNC_BASELINE`, `P8_R1_NATIVE_GRAPH_BASELINE`) with hard/soft gate evaluation and required stage/range coverage checks.
  3. Run formal `R0` acceptance on required production-eligible row (`P8_R0_NATIVE_GRAPH_BASELINE`) using the same gate evaluator and tuple-stage contract checks.
  4. Lock first approved baseline bundle with immutable tuple-to-artifact mapping (artifact hash, toolchain metadata, threshold version, disposition).
  5. Wire scheduled jobs: nightly `R1` formal acceptance lane and scheduled `R0` acceptance lane; keep `uvmAudit`, `syncAudit`, and `graphDebug` out of formal timing rows.
  6. Publish retention and promotion policy with explicit artifact classes (formal lock bundles vs diagnostic artifacts) and lock-governance runbook.
- Artifacts/contracts introduced or consumed:
  - Introduced:
    - `phase8_baseline_lock_registry.json`
    - `phase8_r1_first_lock_bundle`
    - `phase8_r0_first_lock_bundle`
    - `phase8_ci_schedule_manifest.json`
    - `phase8_retention_policy.md`
    - `docs/profiling/README.md` operator runbook updates for lock and scheduling workflows
  - Consumed:
    - `P8-08` pass-A and deep-dive artifacts
    - `P7-08` selected custom-backend evidence package
    - acceptance and graph authority contracts from `acceptance_manifest(.json)` and `graph_capture_support_matrix(.json)`
- Validation:
  - Formal acceptance rows map to admitted tuple IDs only; unsupported tuple inference is rejected by policy.
  - Required stage IDs and orchestration ranges are present per tuple contract for every formal run.
  - Formal timing rows include only accepted execution modes (`async_no_graph`, `graph_fixed`) and exclude diagnostic modes.
  - `native` remains production baseline for lock bundles; any `amgx` row remains benchmark-only and cannot produce production locks.
  - Nightly/scheduled jobs emit artifacts that satisfy retention and lock-update policy schema.
- Done criteria:
  - First approved `R0` acceptance bundle is locked and references required `R1` baseline locks.
  - `R1` nightly and `R0` scheduled jobs are wired with deterministic tuple/stage contracts.
  - Retention and lock-policy workflow is operational and documented.
- Exports to later PRs/phases:
  - Locked baseline registry and first approved lock bundles consumed by post-milestone regression governance.
  - CI/nightly acceptance schedule and operator runbook consumed by release and maintenance workflows.

## imports_from_prev

- `FND-05` acceptance evaluator vocabulary and deterministic gate/disposition contracts.
- `FND-06` canonical stage registry and graph-support policy loader contracts.
- Phase 1 environment and build evidence artifacts: `host_env.json`, `cuda_probe.json`, `fatbinary_report.json`.
- Phase 2 production residency and visibility governance vocabulary, especially explicit staging boundaries and no post-warmup allocation churn constraints.
- `P7-08` selected-backend decision and final Phase 7 regression bundle mapped to tuple/stage IDs.
- `P8-08` top-kernel and sanitizer artifact set with stable naming and parser-consumable metadata.

## exports_to_next

- Pass-A profiling substrate complete for `P8-01..P8-08`.
- Canonical solver-stage instrumentation coverage artifact from `P8-03` explicitly feeding `P5-11`.
- Standardized `nsys` capture scripts and artifact layout from `P8-05` explicitly feeding `P7-01`.
- Parser-driven profiling acceptance summary and diagnostic mode outputs consumed by formal lock workflows.
- Baseline lock registry and first approved `R1`/`R0` lock bundles from `P8-09`.
- CI/nightly schedule manifest and retention/lock-governance runbook for ongoing acceptance operations.

## shared_terms

- `baseline mode`: formal timing mode for performance acceptance rows; excludes diagnostic-only instrumentation modes.
- `diagnostic modes`: `uvmAudit`, `syncAudit`, and `graphDebug`, used for debugging and audit, not baseline locks.
- `canonical stage IDs`: stage names owned by Foundation stage-registry contracts and consumed without local aliases.
- `pass A`: Phase 8 instrumentation and profiling substrate scope, `P8-01..P8-08`.
- `pass B`: Phase 8 baseline-lock and CI/nightly policy scope, `P8-09` after `P7-08`.
- `baseline lock registry`: immutable tuple-to-artifact mapping for approved acceptance bundles.

## open_discontinuities

- `[tracked] phase3-wave4-coupling`: `P8-03` depends on `P3-05` and `P8-04` depends on `P3-07`; pass-A signoff requires those Phase 3 contracts to be available and reviewed in the same wave (`docs/planning/pr_expansion/pr_inventory.md` -> `## Cross-Section Dependency Edges (Canonical)`, impacted PR IDs: `P8-03`, `P8-04`, preferred reading: treat as wave-level integration gate, not a scope expansion).
- `[tracked] ci-budget-retention-site-policy`: docs require operational retention policy for formal artifacts but leave concrete storage/runtime budgets to site policy; the Phase 8 workflow must enforce policy hooks without inventing local tuple thresholds. Impacted PR IDs: `P8-09`. Citations: `docs/phase8_profiling_performance_acceptance_spec.md` -> `6. CI runtime budget and artifact retention remain site-policy items`, `docs/phase8_profiling_performance_acceptance_spec.md` -> `### Milestone P8.8 — CI/nightly integration`, `docs/acceptance_manifest.md` -> `## Disposition Rules`. Preferred reading: tuple and gate logic stays authority-driven while budget numbers are set by deployment policy.

## validation_checks

- All pass-A cards preserve canonical dependency edges exactly: `P8-01 <- P1-02`, `P8-02 <- FND-05,FND-06,P8-01`, `P8-03 <- P8-02,P3-05`, `P8-04 <- P8-02,P3-07`, `P8-05 <- P8-03,P8-04`, `P8-06 <- P8-05,FND-05`, `P8-07 <- P8-06`, `P8-08 <- P8-06,P8-07`.
- `P8-09` is fully expanded with canonical dependencies `P8-08` and `P7-08`.
- Formal lock rows are tuple-driven and reference admitted `P8_R1_*` / `P8_R0_*` rows only.
- Every full pass-A card cites authority docs, exact Phase 8 spec subsections, backlog `scope`, and backlog `done_when`.
- `P8-09` cites authority docs/json, exact Phase 8 pass-B steps, backlog `scope`, and backlog `done_when`.
- Phase 8 pass A keeps canonical NVTX naming and does not restate tuple policy or stage IDs outside authority contracts.
- Pass-A seam exports are explicit: `P8-03` feeds `P5-11` and `P8-05` feeds `P7-01`.
- Formal baseline-lock thresholds and CI/nightly lock policy are owned by `P8-09` and mapped to accepted tuple/stage contracts.
