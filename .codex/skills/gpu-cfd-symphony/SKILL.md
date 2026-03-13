# GPU CFD Symphony

Use this skill when working a `gpu_cfd` Linear issue under Symphony orchestration.

## Goal

Translate one Linear issue into one bounded implementation run that respects this repository's
authority docs, backlog dependencies, and PR-card scope.

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
2. Mirror that plan into a single persistent Linear workpad comment.
3. Confirm the branch state and reproduce the current behavior before editing.
4. Implement with TDD where practical, following `AGENTS.md`.
5. Keep the workpad current as plan, risks, and validation evolve.
6. Run the smallest direct validation first, then any broader checks required by the card.
7. Open or update the PR only when the card's validation and done criteria are satisfied.

## Handoff rules

- Attach the PR to the Linear issue.
- Move the issue to `In Review` only after validation is complete.
- If review requests changes, expect the issue to be moved back to `Todo` or `In Progress` for a
  new Symphony run.
- If blocked by missing auth, missing secrets, or missing external tools, leave a concise blocker
  note instead of guessing.
