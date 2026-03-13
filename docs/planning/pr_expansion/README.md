# PR Expansion Workspace

This folder holds the durable planning docs used to expand the high-level PR backlog in [gpu_cfd_pr_backlog.json](/Users/rexputnam/Documents/projects/gpu_cfd/docs/PRs/gpu_cfd_pr_backlog.json) into section-owned, implementation-ready PR cards.

## Operating Model

- Treat [gpu_cfd_pr_backlog.json](/Users/rexputnam/Documents/projects/gpu_cfd/docs/PRs/gpu_cfd_pr_backlog.json) as the canonical worklist and dependency source.
- Use [gpu_cfd_pr_backlog.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/PRs/gpu_cfd_pr_backlog.md) as narrative context only.
- Every PR card must anchor to the same chain:
  1. authority doc(s) and JSON companions
  2. exact phase-spec subsection(s), or for `FND-*` cards the relevant [README_FIRST.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/README_FIRST.md) authority-order subsection
  3. backlog `scope`
  4. backlog `done_when`
- Section docs own only their assigned PR IDs and must not absorb neighboring work.
- Shared planning docs are coordinator-owned.

## Folder Contents

- [README.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/README.md): workspace overview, wave order, and ownership map
- [pr_inventory.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_inventory.md): exact section-to-PR inventory and dependency summary
- [pr_card_template.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_card_template.md): required card structure for every expanded PR
- [boundary_matrix.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/boundary_matrix.md): seam rules, handoff anchors, and review checkpoints
- [review_checklist.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/review_checklist.md): quality gates for section reviews and wave signoff
- [decision_notes.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/decision_notes.md): unresolved seam issues that cannot be settled from the docs
- Section docs:
  - [01_foundation_authority_consumption.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/01_foundation_authority_consumption.md)
  - [02_phase0_reference_problem_freeze.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/02_phase0_reference_problem_freeze.md)
  - [03_phase1_blackwell_bringup.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/03_phase1_blackwell_bringup.md)
  - [04_phase2_gpu_memory_model.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/04_phase2_gpu_memory_model.md)
  - [05_phase3_execution_model.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/05_phase3_execution_model.md)
  - [06_phase4_pressure_linear_algebra.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/06_phase4_pressure_linear_algebra.md)
  - [07_phase5_generic_vof_core.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/07_phase5_generic_vof_core.md)
  - [08_phase6_pressure_swirl_nozzle_bc_startup.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/08_phase6_pressure_swirl_nozzle_bc_startup.md)
  - [09_phase7_custom_cuda_kernels.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/09_phase7_custom_cuda_kernels.md)
  - [10_phase8_profiling_performance_acceptance.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/10_phase8_profiling_performance_acceptance.md)

## Wave Order

1. Wave 0: coordinator shared docs
2. Wave 1: Foundation
3. Wave 2: Phase 0 and Phase 1
4. Gate: Phase 0 signs off before correctness-sensitive GPU comparison language elsewhere
5. Wave 3: Phase 2
6. Wave 4: Phase 3, Phase 4, and Phase 8 pass A (`P8-01..P8-08`) drafted in parallel
7. Gate: Phase 8 pass A closes only after `P3-05` and `P3-07` dependencies are satisfied and `P8-05` is exported for downstream use
8. Wave 5: Phase 5
9. Wave 6: Phase 6
10. Wave 7: Phase 7
11. Wave 8: Phase 8 pass B (`P8-09` and any baseline-lock text gated on Phase 7 stability)

## Current Status

- Completed:
  - Wave 0 shared scaffold
  - Wave 1 Foundation
  - Wave 2 Phase 0 and Phase 1
  - Wave 3 Phase 2
  - Wave 4 Phase 3, Phase 4, and Phase 8 pass A
  - Wave 5 Phase 5
  - Wave 6 Phase 6
  - Wave 7 Phase 7
  - Wave 8 Phase 8 pass B (`P8-09`)
- Seam review status:
  - `Phase 2 -> Phase 3`: green with tracked `first-capture-stage-selection`
  - `Phase 4 -> Phase 5`: green with tracked `coupled-interface-phase4-scope` and `pressure-stage-range-name-drift-risk`
  - `Phase 8 pass A -> Phase 5 / Phase 7`: green and consumed by downstream sections
  - `Phase 5 -> Phase 6`: green with tracked seam items documented in section `open_discontinuities`
  - `Phase 6 -> Phase 7`: green with tracked `phase6_phase7_hotspot_scope_lock` and `contact_angle_conditionality_guard`
  - `Phase 7 -> Phase 8`: green with tracked `ci-budget-retention-site-policy`
- Workspace validation:
  - All numbered section docs `01..10` are populated.
  - Structural checks passed for every section doc.
  - Coverage check passed: `88` expected PR IDs, `88` found, `88` unique, no duplicates, no missing IDs.
- Active wave:
  - None. Drafting waves are complete; workspace is in coordinator maintenance/review state.

## Wave Exit Reviews

- Completed after Wave 1: review `Foundation -> Phase 0 / Phase 1`
- Completed after Wave 3: review `Phase 2 -> Phase 3`
- Completed after Wave 4: review `Phase 4 -> Phase 5` and verify Phase 8 pass-A exports (`P8-03`, `P8-05`)
- Completed after Wave 5: precheck `Phase 5 -> Phase 6` boundary packet completeness
- Completed after Wave 6: review `Phase 5 -> Phase 6`
- Completed after Wave 7: review `Phase 6 -> Phase 7` and `Phase 7 -> Phase 8`
- Completed after Wave 8: final workspace consistency review with [review_checklist.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/review_checklist.md)

## Ownership Map

- Coordinator-owned shared docs: [README.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/README.md), [pr_inventory.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_inventory.md), [pr_card_template.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_card_template.md), [boundary_matrix.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/boundary_matrix.md), [review_checklist.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/review_checklist.md), and [decision_notes.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/decision_notes.md)
- One section agent owns exactly one numbered section doc and the PR IDs listed for that section in [pr_inventory.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_inventory.md).
- The main thread acts as reviewer, integration owner, and final seam adjudicator.

## Review Rules

- Earlier phases define contracts; later phases consume them.
- Authority docs and JSON companions outrank phase prose when they conflict.
- Cross-section dependency edges in [pr_inventory.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/pr_inventory.md) are canonical and must be preserved in section cards.
- Unresolved seam issues go into [decision_notes.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/planning/pr_expansion/decision_notes.md) with blocked PR IDs and citations.
- Section docs must end with the exact blocks `imports_from_prev`, `exports_to_next`, `shared_terms`, `open_discontinuities`, and `validation_checks`.
