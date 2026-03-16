---
name: gpu-cfd-symphony
description: Use this skill when working a gpu_cfd Linear issue under Symphony orchestration.
---

# GPU CFD Symphony

Use this skill when working a `gpu_cfd` Linear issue under Symphony orchestration.

## Goal

Translate one Linear issue into one bounded implementation or rework run that respects this
repository's authority docs, backlog dependencies, and PR-card scope.

## Required opening sequence

1. Open `docs/README_FIRST.md`.
2. Open `docs/tasks/pr_inventory.md`.
3. Identify the PR ID from the Linear issue title or description.
4. Open the owning `docs/tasks/NN_*.md` file for that PR ID.
5. Find the exact matching `## <PR-ID>` card and treat it as the execution contract.

## Scope rules

- The repository docs are authoritative over informal issue wording when they differ.
- Work only the assigned PR ID.
- Preserve dependency edges from `docs/backlog/gpu_cfd_pr_backlog.json`.
- Do not start a blocked PR even if the Linear state is inconsistent.
- Do not widen support scope, fallback behavior, acceptance thresholds, or stage naming.

## Execution checklist

1. Extract the card's `Objective`, `Depends on`, `Concrete task slices`, `Validation`, and
   `Done criteria`.
2. Mirror that plan into a single persistent Linear workpad comment using worker-local Linear
   access.
   Prefer Symphony's injected `linear_graphql` tool when it is present. On the WSL worker host,
   the official Linear MCP is also configured and authenticated for individual Codex workers.
3. Confirm the branch state and reproduce the current behavior before editing.
4. Implement with TDD where practical, following `AGENTS.md`.
5. Keep the workpad current as plan, risks, validation, and review findings evolve.
6. Run the smallest direct validation first, then any broader checks required by the card.
7. When implementation or rework is ready for review, commit and push the branch, then run
   `python3 scripts/symphony/pr_handoff.py --workspace "$PWD"`.
8. If the handoff helper reports findings, inspect the latest artifact under
   `.codex/review_artifacts/`, fix the valid findings in the same run, rerun targeted validation,
   and rerun the handoff helper once.
9. When the handoff helper succeeds, it opens or updates the PR, enables GitHub auto-merge, moves
   the issue to `In Review`, and you should stop after you update the workpad with the PR URL.
10. `In Review` is a dormant automated-review queue. Do not wait or poll from the worker.
11. On a `Rework` run, start by using the GitHub CLI API to pull the latest review comments and
    review state for the current PR head. Do not rely on memory alone.
12. Treat valid actionable Devin feedback as mandatory fix work. Fix those findings first, rerun
    targeted validation, push, and rerun the handoff helper before returning to `In Review`.
    This workflow requires one Devin review round; after those actionable findings are resolved,
    GitHub may merge without waiting for a second Devin pass on the new head.
13. Use the `gh api` PR comment/review endpoints described in `AGENTS.md` when you need the full
    inline Devin feedback payload, including file path, line, diff hunk, and body.

## Handoff rules

- Record the PR URL in the Linear workpad comment.
- Treat worker-local Linear access as mandatory. If neither `linear_graphql` nor the configured
  Linear MCP is available, stop and leave a concise blocker note.
- Move the issue to `In Review` only through `scripts/symphony/pr_handoff.py` after validation is
  complete and the local Codex review gate has completed successfully for external review.
- Treat `In Review` as a dormant queue controlled by GitHub automation, not as a worker sleep loop.
- If blocked by missing auth, missing secrets, or missing external tools, leave a concise blocker
  note instead of guessing.
