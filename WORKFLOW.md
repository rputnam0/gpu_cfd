---
tracker:
  kind: linear
  project_slug: "gpu-cfd-6e45c39a4350"
  active_states:
    - Todo
    - In Progress
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
  max_turns: 20
  max_concurrent_agents_by_state:
    todo: 1
    in progress: 1
codex:
  command: codex --config shell_environment_policy.inherit=all app-server
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
- Keep one persistent Linear workpad comment up to date if the runtime exposes `linear_graphql` or Linear MCP access.
- If the issue state is `Todo`, move it to `In Progress` before implementation work.
- If the issue already has a PR attached, start with a review-feedback sweep before new edits.
- Use the issue branch name when available; otherwise create a `codex/` branch derived from the issue identifier.
- Run the smallest relevant validation first, then broader checks when the scope requires it.
- When the task is complete, open or update a GitHub PR, attach the PR to the Linear issue, and move the issue to `In Review`.
- `Backlog` means parked or blocked work and is out of scope for this run.
- `In Review` is a human hold state in this repository. Do not modify issues parked there unless they are moved back to `Todo` or `In Progress`.
- If required auth, secrets, or external tools are missing, record a concise blocker note and stop instead of widening scope.

Completion bar:

- The exact task card's objective, validation, and done criteria are satisfied.
- Repo changes are limited to the assigned PR card.
- Validation evidence is recorded in the Linear workpad.
- The PR is ready for human review, not just local completion.
