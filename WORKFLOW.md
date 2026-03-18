---
tracker:
  kind: linear
  project_slug: "gpu-cfd-6e45c39a4350"
  active_states:
    - Todo
    - In Progress
    - Rework
  terminal_states:
    - Done
    - Canceled
    - Cancelled
    - Duplicate
polling:
  interval_ms: 10000
workspace:
  root: $SYMPHONY_WORKSPACE_ROOT
hooks:
  after_create: |
    : "${GPU_CFD_SOURCE_REPO_URL:=https://github.com/rputnam0/gpu_cfd.git}"
    git clone "$GPU_CFD_SOURCE_REPO_URL" .
    git fetch origin --prune
agent:
  max_concurrent_agents: 1
  max_turns: 40
  max_concurrent_agents_by_state:
    todo: 1
    in progress: 1
    rework: 1
codex:
  # This is the default Symphony implementation worker profile.
  # The separate local review pass is launched by scripts/symphony/pr_handoff.py
  # via scripts/symphony/review_loop.py and uses scripts/symphony/runtime_config.toml:
  # review = gpt-5.4 / xhigh
  command: uv run python scripts/symphony/codex_dispatch.py app-server
  approval_policy: never
  thread_sandbox: workspace-write
  read_timeout_ms: 30000
  turn_sandbox_policy:
    type: workspaceWrite
    networkAccess: true
---

You are working on Linear issue `{{ issue.identifier }}` for the `gpu_cfd` repository.

This repository is doc-driven. Use progressive disclosure and keep context scoped to the task at
hand. Before changing code, work in this order:

1. `AGENTS.md`
2. `.codex/skills/gpu-cfd-symphony/SKILL.md`
3. The owning `docs/tasks/NN_*.md` file for the PR ID in this issue
4. The exact matching `## <PR-ID>` card in that file
5. Only the supporting docs cited by that card or directly relevant review comments / boundary docs

Issue context:

- Identifier: `{{ issue.identifier }}`
- Title: `{{ issue.title }}`
- Current state: `{{ issue.state }}`
- Labels: `{{ issue.labels }}`
- URL: `{{ issue.url }}`
- Attempt: `{% if attempt %}{{ attempt }}{% else %}first run{% endif %}`

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Execution contract:

- Linear is the execution surface. The repository docs remain the technical source of truth.
- Treat the assigned PR ID as a hard scope boundary. Do not absorb neighboring backlog items.
- Respect dependency edges from `docs/backlog/gpu_cfd_pr_backlog.json` and `docs/tasks/pr_inventory.md`.
- If the PR ID, owning task file, or card location is unclear, use `docs/tasks/pr_inventory.md` as the fallback map.
- Treat the exact PR card as the execution contract. Broader specs and authority docs are supporting references, not mandatory pre-read material.
- Do not read the full docs corpus by default. Expand outward only when the PR card's cited sources are insufficient.
- Worker-side Linear access is required for workpad comments, state changes, and issue lookups.
- Prefer Symphony's built-in `linear_graphql` tool when Symphony injects it into the app-server session.
- The WSL Codex worker runtime also has the official Linear MCP configured and authenticated for individual workers. If neither `linear_graphql` nor Linear MCP is available, record a blocker and stop.
- Write or update the canonical Linear workpad before code edits. Use it as durable working memory for the current plan, progress, decisions, rationale, gotchas, validation evidence, review context, and future-agent notes.
- Keep the same canonical Linear workpad comment current during implementation and rework; do not fork the memory trail into multiple status comments.
- Use a non-interactive planning pass before edits. Prefer the narrowest reversible implementation consistent with the PR card and cited docs, record assumptions and rationale in the workpad, and continue unless there is a real authority conflict, missing auth/secret, or an unsafe destructive action not covered by repo policy.
- Use native Codex sub-agents as bounded recursive research helpers when context discovery is the bottleneck. The implementation worker profile explicitly enables multi-agent support and the project child-agent definitions. Use `gpt-5.4-mini` for those helper agents.
- Good sub-agent tasks: locate exact doc sections, trace code paths, summarize adjacent tests or APIs, inspect review payloads, and compare nearby implementations. Ask each helper one narrow question and have it return short path/citation-focused findings.
- Prefer the project-scoped helper agents in `.codex/agents/` when they match the task: `docs_scout`, `codepath_scout`, and `review_payload_scout`. If you spawn a helper directly instead of using a project-defined agent, explicitly set that helper's model to `gpt-5.4-mini`.
- Keep the `gpt-5.4` implementation worker as the orchestrator. Do not delegate code edits, test authoring, branch management, Linear updates, PR handoff, or final technical judgment to `gpt-5.4-mini`.
- Keep delegation small and supportive rather than parallel implementation: spawn at most a few focused helper agents when they materially sharpen the main worker's plan, then synthesize their findings in the workpad before editing.
- Do not fetch full Linear details for already-done dependency issues unless the exact dependency contract is still unclear after reading the cited repo task card section. Prefer the repo task card and cited docs over loading blocker issue descriptions from Linear.
- After you have read `AGENTS.md`, the Symphony skill, the exact PR card, and the cited sections needed to understand the task, write or update the Linear workpad before broader codebase exploration.
- Dev-only observability is available through the repo-owned dispatch wrapper. When `GPU_CFD_TRACE_ENABLE=1`, it captures immutable context packs, workpad diffs, handoff/review events, and app-server transcripts under `GPU_CFD_TRACE_ROOT` for the standalone Symphony Trace Viewer.
- If the issue state is `Todo`, move it to `In Progress` before implementation work.
- If the issue state is `Rework`, start by using the GitHub CLI API to pull the latest review comments and review state for the current PR head before making new edits, and record the actionable thread IDs / URLs in the Linear workpad.
- If the issue already has a PR attached, start with the same GitHub API review-feedback sweep before new edits.
- Use `gh api` to inspect PR review comments and threads. Treat actionable Devin feedback on the current head as mandatory fix work, not as optional advice.
- Do not rely on memory alone for `Rework`. Re-read the current PR feedback from GitHub, fix every valid actionable Devin finding, and record what changed in the Linear workpad.
- If the branch is already clean and pushed but no PR exists, treat the run as ready for the sanctioned pre-PR handoff instead of reopening implementation planning.
- Never move an issue back to `Backlog` after implementation has started. `Backlog` is only for untouched dependency-gated work.
- Use the issue branch name when available; otherwise create a `codex/` branch derived from the issue identifier.
- Run the smallest relevant validation first, then broader checks when the scope requires it.
- When the task is implementation-complete, commit and push the branch, record validation evidence in the workpad, then run `python3 scripts/symphony/pr_handoff.py --workspace "$PWD"`.
- Run the handoff helper from the issue workspace directly. If the workspace still contains unrelated dirty control-plane files, the helper materializes its own clean committed clone for local review and PR automation; do not invent a manual clean-clone workflow.
- The dispatch wrapper and handoff helper both refresh `origin` before review or PR automation. If the handoff helper reports `branch_refresh_required`, stay in the same worker run, refresh the branch against the latest `origin/main`, rerun the smallest relevant validation, and rerun the handoff helper. Do not return a PR to `In Review` until the branch contains the latest `origin/main`, and do not open or update a PR from a conflicted branch.
- The Symphony workflow configuration for this repo applies to the implementation worker only. The local pre-PR review pass uses the repo-owned review profile in `scripts/symphony/runtime_config.toml` (`gpt-5.4` with `xhigh`).
- If the handoff helper reports findings on remediation pass 1 or 2, inspect the latest artifact under `.codex/review_artifacts/`, fix the valid findings in the same implementation run, rerun the smallest relevant validation, and rerun the handoff helper.
- Remediation passes 1 and 2 are continuation work for the same implementation worker. Keep the issue in `In Progress`; do not stop or expect Symphony to redispatch a fresh worker for those pre-PR fixes.
- The finite local-review cycle is 3 total passes: remediation pass 1, remediation pass 2, then one final local review pass.
- On the third and final local review pass, residual findings no longer block PR progression. The handoff helper creates one child `Backlog` issue per residual finding, opens or updates the PR, enables auto-merge, moves the parent issue to `In Review`, and returns `stop_worker=true`. Do not start another local review after that terminal pass.
- When the handoff helper succeeds cleanly, it opens or updates the GitHub PR, enables auto-merge, moves the issue to `In Review`, returns `stop_worker=true`, and the run should stop after you update the workpad with the PR URL.
- `In Review` is a dormant automated-review state. Do not wait, poll, or sleep for Devin.
- GitHub automation owns the `In Review -> Rework` transition when Devin leaves actionable feedback on the current head. Actionable Devin threads are not auto-cleared just because a newer commit exists.
- GitHub automation also owns the `In Review -> Rework` transition when an open PR becomes `BEHIND` or `DIRTY` against the latest `main`. Refresh the branch in the resumed `Rework` run before returning it to GitHub auto-merge.
- `Rework` is terminal with respect to local review. After actionable Devin findings are fixed, rerun targeted validation, push, rerun `scripts/symphony/pr_handoff.py`, and return directly to GitHub auto-merge without another local Codex review cycle.
- GitHub auto-merge owns the final merge once `review-loop-harness` and `devin-review-gate` are green.
- GitHub post-merge automation owns `In Review -> Done` and dependent release from `Backlog -> Todo`.
- `Backlog` means parked or blocked work and is out of scope for this run.
- If required auth, secrets, or external tools are missing, record a concise blocker note and stop instead of widening scope.

Completion bar:

- The exact task card's objective, validation, and done criteria are satisfied.
- Repo changes are limited to the assigned PR card.
- Validation evidence is recorded in the Linear workpad.
- The PR has completed the finite local Codex review cycle and progressed into GitHub review.
- The PR has completed one Devin review cycle, all actionable findings have been fixed or resolved, and it merged through GitHub auto-merge without reopening the local-review loop.
