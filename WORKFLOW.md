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
  command: codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=medium --model gpt-5.4 app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
    networkAccess: true
---

You are working on Linear issue `{{ issue.identifier }}` for the `gpu_cfd` repository.

This repository is doc-driven. Before changing code, open these files in order:

1. `docs/README_FIRST.md`
2. `docs/tasks/pr_inventory.md`
3. The owning `docs/tasks/NN_*.md` file for the PR ID in this issue
4. `.codex/skills/gpu-cfd-symphony/SKILL.md`

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
- Worker-side Linear access is required for workpad comments, state changes, and issue lookups.
- Prefer Symphony's built-in `linear_graphql` tool when Symphony injects it into the app-server session.
- The WSL Codex worker runtime also has the official Linear MCP configured and authenticated for individual workers. If neither `linear_graphql` nor Linear MCP is available, record a blocker and stop.
- Keep one persistent Linear workpad comment or concise progress-note trail up to date during implementation and rework.
- If the issue state is `Todo`, move it to `In Progress` before implementation work.
- If the issue state is `Rework`, start by using the GitHub CLI API to pull the latest review comments and review state for the current PR head before making new edits.
- If the issue already has a PR attached, start with the same GitHub API review-feedback sweep before new edits.
- Use `gh api` to inspect PR review comments and threads. Treat actionable Devin feedback on the current head as mandatory fix work, not as optional advice.
- Do not rely on memory alone for `Rework`. Re-read the current PR feedback from GitHub, fix every valid actionable Devin finding, and record what changed in the Linear workpad.
- If the branch is already clean and pushed but no PR exists, treat the run as ready for the sanctioned pre-PR handoff instead of reopening implementation planning.
- Never move an issue back to `Backlog` after implementation has started. `Backlog` is only for untouched dependency-gated work.
- Use the issue branch name when available; otherwise create a `codex/` branch derived from the issue identifier.
- Run the smallest relevant validation first, then broader checks when the scope requires it.
- When the task is implementation-complete, commit and push the branch, record validation evidence in the workpad, then run `python3 scripts/symphony/pr_handoff.py --workspace "$PWD"`.
- The Symphony `codex` block above configures the implementation worker only. The local pre-PR review pass uses the repo-owned review profile in `scripts/symphony/runtime_config.toml` (`gpt-5.4` with `xhigh`).
- If the handoff helper reports findings, inspect the latest artifact under `.codex/review_artifacts/`, fix the valid findings in the same run, rerun the smallest relevant validation, and rerun the handoff helper once.
- When the handoff helper succeeds, it opens or updates the GitHub PR, enables auto-merge, moves the issue to `In Review`, and the run should stop after you update the workpad with the PR URL.
- `In Review` is a dormant automated-review state. Do not wait, poll, or sleep for Devin.
- GitHub automation owns the `In Review -> Rework` transition when Devin leaves actionable feedback on the current head.
- GitHub auto-merge owns the final merge once `review-loop-harness` and `devin-review-gate` are green.
- GitHub post-merge automation owns `In Review -> Done` and dependent release from `Backlog -> Todo`.
- `Backlog` means parked or blocked work and is out of scope for this run.
- If required auth, secrets, or external tools are missing, record a concise blocker note and stop instead of widening scope.

Completion bar:

- The exact task card's objective, validation, and done criteria are satisfied.
- Repo changes are limited to the assigned PR card.
- Validation evidence is recorded in the Linear workpad.
- The PR has passed one local Codex review loop.
- The PR has completed a fresh Devin review cycle on the current head and merged through GitHub auto-merge.
