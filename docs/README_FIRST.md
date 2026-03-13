 # README_FIRST

This directory is the implementation handoff bundle. Start here before opening any phase spec or task brief.

## Authoritative docs

Read these in order:

1. `docs/authority/continuity_ledger.md`
2. `docs/authority/master_pin_manifest.md`
3. `docs/authority/reference_case_contract.md`
4. `docs/authority/validation_ladder.md`
5. `docs/authority/support_matrix.md`
6. `docs/authority/acceptance_manifest.md`
7. `docs/authority/graph_capture_support_matrix.md`
8. `docs/authority/semantic_source_map.md`
9. `docs/specs/phase0_reference_problem_spec.md` through `docs/specs/phase8_profiling_performance_acceptance_spec.md`

## Working folders

- `docs/authority/` contains the frozen authority bundle and JSON companions.
- `docs/specs/` contains the phase-by-phase implementation specs.
- `docs/backlog/gpu_cfd_pr_backlog.json` is the canonical PR backlog and dependency source.
- `docs/backlog/gpu_cfd_pr_backlog.md` is the human-readable overview of that backlog. It is narrative context, not the canonical worklist.
- `docs/tasks/pr_inventory.md` maps each PR ID to the planning doc that owns it.
- `docs/tasks/NN_*.md` files are the agent-facing implementation briefs for assigned PR ranges.
- `docs/tasks/boundary_matrix.md` records cross-phase seam rules and handoff gates.
- `docs/tasks/decision_notes.md` is only for seam issues that cannot be resolved from the authority docs and section docs.

## If you are assigned a PR

1. Read the authority docs above before changing implementation scope.
2. Find the PR ID in `docs/tasks/pr_inventory.md`.
3. Open the owning `docs/tasks/NN_*.md` file and treat it as the scoped task brief.
4. Use `docs/tasks/boundary_matrix.md` for cross-phase handoffs and `docs/tasks/decision_notes.md` only if the seam is genuinely unresolved by the docs.

## Consumption rules

- The authority docs above override any conflicting phase-local wording.
- The JSON companions are the machine-readable source of truth for case roles, support scope, acceptance tuples, and graph-stage policy.
- The phase specs consume the authority docs; they do not redefine them.
