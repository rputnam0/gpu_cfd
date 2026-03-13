---
tracker:
  kind: linear
  project_slug: "gpu-cfd-6e45c39a4350"
  active_states:
    - Todo
    - In Progress
    - Rework
    - Ready to Merge
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
    if [ -n "${GPU_CFD_BOOTSTRAP_REF:-}" ]; then
      git fetch origin "$GPU_CFD_BOOTSTRAP_REF"
      git checkout "$GPU_CFD_BOOTSTRAP_REF"
    fi
    git fetch origin --prune
  before_run: |
    git status --short --branch >/dev/null
agent:
  max_concurrent_agents: 1
  max_turns: 40
  max_concurrent_agents_by_state:
    todo: 1
    in progress: 1
    rework: 1
    ready to merge: 1
codex:
  command: "python3 scripts/symphony/codex_runner.py implementation app-server"
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
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
- Blockers: `{{ issue.blocked_by }}`

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
- Use the Linear MCP tools on the worker host for issue comments, state changes, and review follow-up. If Linear MCP is unavailable or not authenticated, leave a concise blocker note and stop.
- Keep one persistent Linear workpad comment or concise progress-note trail up to date during implementation and review follow-up.
- Record structured telemetry for important transitions with `uv run python scripts/symphony/telemetry.py event ...`; at minimum log issue start, blockers, PR open/update, `review_requested`, and merge.
- If the issue state is `Todo`, move it to `In Progress` before implementation work.
- If the issue state is `Rework`, start with a Linear comment sweep plus a GitHub PR review sweep before new edits.
- If the issue state is `Ready to Merge`, start by confirming the linked PR head is current and that required checks are green.
- If the issue already has a PR attached, start with a review-feedback sweep before new edits.
- Use the issue branch name when available; otherwise create a `codex/` branch derived from the issue identifier.
- Run the smallest relevant validation first, then broader checks when the scope requires it.
- Before opening or marking a PR ready for review, run `uv run python scripts/symphony/review_loop.py codex-review --issue {{ issue.identifier }} --base origin/main`, inspect the saved review artifact, fix material findings, and rerun the review gate once.
- When the task is implementation-complete, open or update the GitHub PR, record the PR URL in a Linear comment, emit a `review_requested` telemetry event, move the issue to `In Review`, and stop. Do not sit in a local sleep or polling loop.
- `In Review` is a dormant state for Symphony. The Linear and GitHub integrations should move the issue into `Rework` when changes are needed and `Ready to Merge` when the PR is clear to land.
- On a resumed `Rework` run, use the latest Devin-authored Linear comments as the primary review signal and GitHub review comments as detail when needed. Fix valid findings, rerun the smallest relevant validation, rerun the local Codex review gate, push, emit `review_requested`, and move the issue back to `In Review`.
- On a resumed `Ready to Merge` run, verify the linked PR is clean on the current head and branch protection is satisfied, merge with GitHub CLI, confirm the default branch contains the change, and then move the Linear issue to `Done`.
- `Backlog` means parked or blocked work and is out of scope for this run.
- If required auth, secrets, or external tools are missing, record a concise blocker note and stop instead of widening scope.

Completion bar:

- The exact task card's objective, validation, and done criteria are satisfied.
- Repo changes are limited to the assigned PR card.
- Validation evidence is recorded in the Linear workpad.
- The PR has passed one local Codex review loop and any valid Devin review findings surfaced through Linear on the current head are addressed.
- The PR is merged, not just opened.
