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
7. Before opening or marking a PR ready for review, run `uv run python scripts/symphony/review_loop.py codex-review --base origin/main`, inspect the saved report in `.codex/review_artifacts/`, fix material findings, and rerun the review gate once.
8. Open or update the PR only when the card's validation and done criteria are satisfied.
9. Once the PR is in `In Review`, run `uv run python scripts/symphony/review_loop.py wait --reviewer devin-ai-integration[bot] --timeout-seconds 900` and use the result to drive the GitHub follow-up loop.
10. If Devin feedback is actionable, fix valid findings, rerun targeted validation, rerun the local Codex review gate, push, and wait for a fresh review on the new head.
11. Merge the PR only when the current head is clean according to the GitHub review loop and the PR is otherwise mergeable.

## Handoff rules

- Attach the PR to the Linear issue.
- Move the issue to `In Review` after validation is complete and the local Codex review loop is clean enough for external review.
- Keep `In Review` as an active automation state for Devin review polling and merge follow-through.
- If a fresh Devin review has not arrived yet, wait and keep concise notes rather than moving the issue backward.
- If blocked by missing auth, missing secrets, or missing external tools, leave a concise blocker
  note instead of guessing.
