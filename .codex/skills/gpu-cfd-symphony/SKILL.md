---
name: gpu-cfd-symphony
description: Use this skill when working a gpu_cfd Linear issue under Symphony orchestration.
---

# GPU CFD Symphony

Use this skill when working a `gpu_cfd` Linear issue under Symphony orchestration.

## Required opening sequence

1. Open `AGENTS.md`.
2. Identify the PR ID from the Linear issue title or description.
3. Open the owning `docs/tasks/NN_*.md` file for that PR ID.
4. Find the exact matching `## <PR-ID>` card and treat it as the execution contract.
5. Open only the supporting docs cited by that card or directly relevant review or boundary docs.

## Scope rules

- The repo docs are authoritative over informal issue wording when they differ.
- Work only the assigned PR ID.
- Preserve dependency edges from `docs/backlog/gpu_cfd_pr_backlog.json`.
- If the PR ID, owning task file, or card location is unclear, use `docs/tasks/pr_inventory.md` as the fallback map.
- Do not widen scope or absorb neighboring backlog items.
- Do not read the full docs corpus by default.

## Execution checklist

1. Extract the card's objective, depends-on list, validation, and done criteria.
2. After you have read `AGENTS.md`, this skill, the exact PR card, and the cited sections needed to understand the task, write or update the canonical Linear workpad before broader exploration or edits.
3. Use a non-interactive planning pass before edits. Record assumptions, boundaries, helper findings, and validation plans in the workpad.
4. Use native Codex sub-agents as bounded recursive research helpers when context discovery is the bottleneck.
   Trigger `docs_scout` before edits when the task spans docs plus code or cites multiple supporting docs.
   Trigger `codepath_scout` before edits when the change crosses multiple modules or test surfaces.
   Trigger `review_payload_scout` on `Rework`, on `Refresh Required`, or when the review payload is large.
   Use `gpt-5.4-mini` for those helpers, explicitly set that model if you spawn one directly, keep them narrow, and summarize their findings in the workpad before editing.
   The implementation worker profile explicitly enables multi-agent support and the project child-agent definitions.
   Do not delegate code edits, test authoring, branch management, Linear updates, PR handoff, or final technical judgment.
5. Confirm branch state and reproduce the current behavior before editing.
6. Implement with TDD where practical.
7. Keep the workpad current as plan, progress, decisions, validation, review findings, and future-agent notes evolve.
8. Run the smallest direct validation first, then broader checks required by the card.
9. If a fresh repro on the current branch head proves the issue is blocked by an external action outside worker authority or capability, and no repo-side critical-path change remains after targeted validation, record one canonical blocker packet in the workpad, move the issue to `Backlog`, and stop. Do not rerun the blocked command unless the branch head changed, the external host state changed, or a human explicitly requested another repro.
10. When implementation work is ready, commit and push the branch, then run `python3 "$GPU_CFD_CONTROL_REPO_ROOT/scripts/symphony/pr_handoff.py" --workspace "$PWD"`.
11. If remediation pass 1 or 2 reports findings, stay in the same implementation worker, fix the valid findings, rerun targeted validation, and rerun handoff.
12. The finite local-review cycle is 3 total passes: remediation pass 1, remediation pass 2, then one final local review pass.
13. If the third pass still reports findings, the helper creates one child `Backlog` issue per residual finding, opens or updates the PR, enables auto-merge, moves the parent issue to `In Review`, and returns `stop_worker=true`.
14. If the issue starts in `Rework`, pull the latest GitHub review comments and review state for the current PR head with `gh api`, record the actionable thread IDs and URLs in the workpad, fix the valid findings, rerun targeted validation, and rerun handoff.
15. If the issue starts in `Refresh Required`, merge the latest `origin/main`, rerun the smallest relevant validation, and rerun handoff. If that merge conflicts, move the issue to `Rework` and record that manual conflict resolution is required.
16. `Rework` is terminal with respect to local review. Do not reopen the local Codex review loop during `Rework`; after Devin findings are fixed, return directly to GitHub auto-merge.
17. Do not return a PR to `In Review` until the branch contains the latest `origin/main`, and do not open or update a conflicted PR.

## Handoff rules

- Record the PR URL in the Linear workpad.
- Treat worker-local Linear access as mandatory. If neither `linear_graphql` nor the configured Linear MCP is available, stop and leave a concise blocker note.
- Move the issue to `In Review` only through `scripts/symphony/pr_handoff.py` after validation is complete.
- Treat `In Review` as a dormant queue controlled by GitHub automation, not as a worker sleep loop.
