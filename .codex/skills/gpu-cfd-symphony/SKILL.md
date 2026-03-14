---
name: gpu-cfd-symphony
description: Use this skill when working a gpu_cfd Linear issue under Symphony orchestration.
---

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
   Linear MCP on the worker host is required for state changes, comments, and review follow-up.
6. Record structured telemetry with `uv run python scripts/symphony/telemetry.py event ...` for issue start, blockers, PR open/update, review waiting, review findings, and merge.
7. Use the telemetry log for external blockers too, not just code defects, so operator follow-up is visible outside Linear.
8. Run the smallest direct validation first, then any broader checks required by the card.
9. Do not run the local Codex review gate inside the worker. The sanctioned host-side `hooks.after_run` path runs `scripts/symphony/after_run.py`, which executes the local Codex review on the worker host outside the Codex workspace sandbox.
10. Open or update the PR only when the card's validation and done criteria are satisfied, and only when the active issue state or prompt explicitly calls for it. If the run is finishing implementation without a linked PR, commit and push, then leave the host-side handoff to open or update the PR.
11. Once implementation or rework is ready for review, record validation evidence, emit a `review_requested` telemetry event, move the issue to `In Review`, and stop. Do not stay alive in a local sleep loop.
12. `In Review` is a dormant queue. Let the host-side handoff and GitHub/Linear integrations move the issue into `Rework` when fixes are needed or `Ready to Merge` when it is clear to land.
13. On a `Rework` run, start with the latest Devin-authored Linear comments when a PR exists, or the latest host-side local review artifact under `.codex/review_artifacts/` when no PR exists. Fix valid findings, rerun targeted validation, push, emit `review_requested`, and move back to `In Review`.
14. On a `Ready to Merge` run, confirm the linked PR is clean on the current head, merge it, and move the issue to `Done`.

## Handoff rules

- Record the PR URL in a Linear comment on the issue.
- Move the issue to `In Review` after validation is complete and the branch is ready for the host-side local Codex review gate plus external review.
- Treat `In Review` as a dormant queue controlled by Linear workflow transitions, not as a worker sleep loop.
- If a fresh Devin review has not arrived yet, leave concise notes and stop. Symphony should resume work only when the issue re-enters an active state like `Rework` or `Ready to Merge`.
- If blocked by missing auth, missing secrets, or missing external tools, leave a concise blocker
  note instead of guessing.
