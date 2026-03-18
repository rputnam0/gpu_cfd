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

1. Open `AGENTS.md`.
2. Identify the PR ID from the Linear issue title or description.
3. Open the owning `docs/tasks/NN_*.md` file for that PR ID.
4. Find the exact matching `## <PR-ID>` card and treat it as the execution contract.
5. Open only the supporting docs cited by that card or directly relevant review/boundary docs.

## Scope rules

- The repository docs are authoritative over informal issue wording when they differ.
- Work only the assigned PR ID.
- Preserve dependency edges from `docs/backlog/gpu_cfd_pr_backlog.json`.
- If the PR ID, owning task file, or card location is unclear, use `docs/tasks/pr_inventory.md` as the fallback map.
- Do not start a blocked PR even if the Linear state is inconsistent.
- Do not widen support scope, fallback behavior, acceptance thresholds, or stage naming.
- Do not read the full docs corpus by default.

## Execution checklist

1. Extract the card's `Objective`, `Depends on`, `Concrete task slices`, `Validation`, and
   `Done criteria`.
2. After you have read `AGENTS.md`, this skill, the exact PR card, and the cited sections needed to
   understand the task, write or update the canonical Linear workpad comment before broader
   codebase exploration or edits, and use it as durable working memory for the run.
   Prefer Symphony's injected `linear_graphql` tool when it is present. On the WSL worker host,
   the official Linear MCP is also configured and authenticated for individual Codex workers.
3. Use a non-interactive planning pass before edits. Capture the current plan, assumptions,
   rationale, gotchas, and scope notes in the workpad; continue without asking humans unless there
   is a real authority conflict, missing auth/secret, or an unsafe destructive action not covered
   by repo policy.
4. Use native Codex sub-agents as bounded recursive research helpers when context discovery is the
   bottleneck. The implementation worker profile explicitly enables multi-agent support and the
   project child-agent definitions. Use `gpt-5.4-mini` for those helper agents.
   Good helper tasks: locate exact doc sections, trace code paths, summarize nearby tests or APIs,
   inspect review payloads, and compare adjacent implementations.
   Prefer the project helper agents in `.codex/agents/` when they fit: `docs_scout`,
   `codepath_scout`, and `review_payload_scout`.
   If you spawn a helper directly instead of using a project-defined agent, explicitly set that
   helper's model to `gpt-5.4-mini`.
   Keep each helper narrowly scoped and ask for short path/citation-focused findings.
   The main `gpt-5.4` worker remains the orchestrator: do not delegate code edits, test authoring,
   branch management, Linear updates, PR handoff, or final technical judgment to
   `gpt-5.4-mini`.
   Do not fetch full Linear details for already-done dependency issues unless the exact dependency
   contract is still unclear after reading the cited repo task card section. Prefer the repo task
   card and cited docs over loading blocker issue descriptions from Linear.
5. Confirm the branch state and reproduce the current behavior before editing.
6. Implement with TDD where practical, following `AGENTS.md`.
7. Keep the workpad current as plan, progress, decisions, rationale, validation, review findings,
   and future-agent notes evolve.
8. Run the smallest direct validation first, then any broader checks required by the card.
9. When implementation or rework is ready for review, commit and push the branch, then run
   `python3 scripts/symphony/pr_handoff.py --workspace "$PWD"`.
   Run that helper from the issue workspace directly. The dispatch wrapper and handoff helper
   refresh `origin` before review or PR automation. If the helper reports
   `branch_refresh_required`, stay in the same worker run, refresh the branch against the latest
   `origin/main`, rerun the smallest relevant validation, and rerun the handoff helper. Do not
   return a PR to `In Review` until the branch contains the latest `origin/main`, and do not
   open or update a conflicted PR. If the workspace still has unrelated dirty control-plane files,
   the helper creates its own clean committed review clone; do not invent a separate manual
   clean-clone workflow.
10. If the handoff helper reports findings on remediation pass 1 or 2, inspect the latest artifact
   under `.codex/review_artifacts/`, fix the valid findings in the same implementation run, rerun
   targeted validation, and rerun the handoff helper.
   Those first two remediation passes belong to the same implementation worker; stay in
   `In Progress` and keep going on the same branch.
   The local-review cycle is finite: remediation pass 1, remediation pass 2, then one final local
   review pass.
   If the third and final local review pass still reports findings, the helper creates one child
   `Backlog` issue per residual finding, opens or updates the PR, enables auto-merge, moves the
   parent issue to `In Review`, and returns `stop_worker=true`.
11. When the handoff helper succeeds cleanly, it opens or updates the PR, enables GitHub
    auto-merge, moves the issue to `In Review`, returns `stop_worker=true`, and you should stop
    after you update the workpad with the PR URL.
12. `In Review` is a dormant automated-review queue. Do not wait or poll from the worker.
13. On a `Rework` run, start by using the GitHub CLI API to pull the latest review comments and
    review state for the current PR head, and record the actionable thread IDs / URLs in the
    Linear workpad. Do not rely on memory alone.
14. Treat valid actionable Devin feedback as mandatory fix work. Fix those findings first, rerun
    targeted validation, push, and rerun the handoff helper before returning to `In Review`.
    Do not reopen the local Codex review loop during `Rework`; after Devin findings are fixed,
    the branch returns directly to GitHub auto-merge.
    If GitHub moved the issue to `Rework` because the PR became `BEHIND` or `DIRTY`, refresh the
    branch against the latest `origin/main`, rerun the smallest relevant validation, and then
    rerun the handoff helper to return directly to GitHub auto-merge.
    This workflow requires one Devin review round; after those actionable findings are resolved,
    GitHub may merge without waiting for a second Devin pass on the new head.
15. Use the `gh api` PR comment/review endpoints described in `AGENTS.md` when you need the full
    inline Devin feedback payload, including file path, line, diff hunk, and body.

## Handoff rules

- Record the PR URL in the Linear workpad comment.
- Treat worker-local Linear access as mandatory. If neither `linear_graphql` nor the configured
  Linear MCP is available, stop and leave a concise blocker note.
- Move the issue to `In Review` only through `scripts/symphony/pr_handoff.py` after validation is
  complete and the local Codex review gate has completed successfully for external review, or
  after a `Rework` pass has fixed actionable Devin findings and returned directly to auto-merge.
- Treat `In Review` as a dormant queue controlled by GitHub automation, not as a worker sleep loop.
- If blocked by missing auth, missing secrets, or missing external tools, leave a concise blocker
  note instead of guessing.
