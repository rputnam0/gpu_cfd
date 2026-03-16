# Review Checklist

Use this checklist for every section review and wave signoff.

## Section-Level Checks

- Every backlog PR ID for the section appears exactly once.
- Every PR card cites:
  - at least one authority doc or JSON companion
  - at least one exact phase-spec subsection, or for `FND-*` cards the relevant [authority/README.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/authority/README.md) authority-order subsection
  - the backlog `scope`
  - the backlog `done_when`
- Every PR card preserves the backlog dependency edges.
- Task slices are implementation-ready and bounded.
- Exported artifacts or contracts are named consistently with earlier sections and authority docs.
- The section ends with `imports_from_prev`, `exports_to_next`, `shared_terms`, `open_discontinuities`, and `validation_checks`.
- `open_discontinuities` items are tagged `blocking` or `tracked`.

## Wave Signoff Checks

- All earlier-wave blocking discontinuities are resolved before a dependent wave starts.
- Shared terms remain consistent across reviewed section docs.
- Boundary artifacts named in [boundary_matrix.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/tasks/boundary_matrix.md) exist in the upstream section before downstream cards consume them.
- Cross-section dependencies listed in [pr_inventory.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/tasks/pr_inventory.md) are satisfied before downstream cards are marked complete.
- No reviewed section widens support scope, solver-family choices, fallback behavior, or acceptance thresholds beyond the authority layer.
- Decision notes exist for every seam issue that could not be settled from the docs.

## Final Workspace Checks

- All 88 PR IDs from [gpu_cfd_pr_backlog.json](/Users/rexputnam/Documents/projects/gpu_cfd/docs/backlog/gpu_cfd_pr_backlog.json) are covered once and only once.
- Shared docs and section docs use the same artifact names, stage IDs, and boundary terms.
- Phase 8 baseline-lock text is clearly separated from instrumentation-pass text.
- Phase 8 pass-A dependencies (`P8-03 <- P3-05`, `P8-04 <- P3-07`) and downstream exports (`P5-11 <- P8-03`, `P7-01 <- P8-05`) are explicitly represented in section handoff blocks.
- No section card depends on a contract that is neither defined upstream nor tracked in [decision_notes.md](/Users/rexputnam/Documents/projects/gpu_cfd/docs/tasks/decision_notes.md).
