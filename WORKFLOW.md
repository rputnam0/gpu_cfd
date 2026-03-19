---
tracker:
  kind: linear
  project_slug: "gpu-cfd-6e45c39a4350"
  active_states:
    - Todo
    - In Progress
    - Rework
    - Refresh Required
  terminal_states:
    - Done
    - Canceled
    - Cancelled
    - Duplicate
launch_filters:
  required_labels:
    - symphony-canary
workspace:
  root: $SYMPHONY_WORKSPACE_ROOT
hooks:
  after_create: |
    : "${GPU_CFD_SOURCE_REPO_URL:=https://github.com/rputnam0/gpu_cfd.git}"
    git clone "$GPU_CFD_SOURCE_REPO_URL" .
    git fetch origin --prune
control_plane_sync:
  source_ref: origin/main
  paths:
    - WORKFLOW.md
    - .codex/config.toml
    - .codex/skills/gpu-cfd-symphony/SKILL.md
    - .codex/agents/*.md
    - scripts/symphony/*.py
    - scripts/symphony/runtime_config.toml
refresh:
  linear_state: Refresh Required
agent:
  max_concurrent_agents: 1
  max_turns: 40
  max_concurrent_agents_by_state:
    todo: 1
    in progress: 1
    rework: 1
    refresh required: 1
codex:
  command: >
    uv run python "${GPU_CFD_CONTROL_REPO_ROOT}/scripts/symphony/codex_dispatch.py" app-server
  approval_policy: never
  thread_sandbox: workspace-write
  read_timeout_ms: 30000
  turn_sandbox_policy:
    type: workspaceWrite
    networkAccess: true
---

You are working on Linear issue `{{ issue.identifier }}` for the `gpu_cfd` repository.

Start in this order:

1. `AGENTS.md`
2. `.codex/skills/gpu-cfd-symphony/SKILL.md`
3. The owning `docs/tasks/NN_*.md` file for this PR ID
4. The exact `## <PR-ID>` card
5. Only the cited docs you actually need

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

Execution brief:

- Treat the exact PR card as the contract. Use `docs/tasks/pr_inventory.md` as the fallback map only when the PR ID or owning task file is unclear.
- Write or update the canonical Linear workpad before broader exploration, then keep it current as the durable memory trail for plan, helper findings, decisions, validation, and review context.
- `scripts/symphony/codex_dispatch.py` bootstraps the workspace from the canonical control repo before dispatch. `pr_handoff.py` repeats that sync before review and PR automation.
- Do not read the full docs corpus by default. Expand only when the cited docs are insufficient.
- Worker-local Linear access is required. If Linear is unavailable, record a blocker and stop.
- Use native Codex sub-agents as bounded recursive research helpers, not as implementation workers.
- Trigger `docs_scout` before edits when the task spans docs plus code or cites multiple supporting docs.
- Trigger `codepath_scout` before edits when the change crosses multiple modules or test surfaces.
- Trigger `review_payload_scout` on `Rework`, on `Refresh Required`, or when the review payload is large.
- Use `gpt-5.4-mini` for those helpers. Do not delegate code edits, tests, branch management, Linear updates, PR handoff, or final judgment.
- Summarize helper findings in the canonical workpad before editing.
- If the issue starts in `Todo`, move it to `In Progress` before implementation work.
- If the issue starts in `Rework`, pull the latest GitHub review state and actionable Devin threads first, record them in the workpad, fix the valid findings, and rerun handoff without reopening local review.
- If the issue starts in `Refresh Required`, merge the latest `origin/main`, run the smallest relevant validation, and rerun handoff. If the refresh merge conflicts, move the issue to `Rework` and record that manual conflict resolution is required.
- Use the issue branch when it exists; otherwise create a `codex/` branch derived from the issue identifier.
- Run the smallest relevant validation first, then broader checks only when the card requires them.
- When implementation work is ready for review, commit and push the branch, then run `python3 "$GPU_CFD_CONTROL_REPO_ROOT/scripts/symphony/pr_handoff.py" --workspace "$PWD"`.
- Remediation passes 1 and 2 stay on the same implementation worker in `In Progress`.
- The finite local-review cycle is 3 total passes: remediation pass 1, remediation pass 2, then one final local review pass.
- On the third pass, residual findings become child `Backlog` issues and the PR still moves to `In Review`. Those findings are promoted into a mandatory section cleanup sweep at the phase boundary.
- `In Review` is dormant. Do not poll, sleep, or wait in the worker.
- GitHub automation moves actionable review feedback to `Rework`.
- GitHub automation moves clean `BEHIND` or `DIRTY` PRs to `Refresh Required`.
- Do not return a PR to `In Review` until the branch contains the latest `origin/main`.
- Do not open or update a PR from a conflicted branch.
- GitHub post-merge automation owns `In Review -> Done`, section cleanup sweep creation, and downstream release from `Backlog -> Todo`.
- If auth, secrets, or required tools are missing, leave a concise blocker note and stop.
- If a fresh repro on the current branch head proves the issue is blocked by an external action outside worker authority or capability, and no repo-side critical-path change remains after targeted validation, record one canonical blocker packet in the workpad, move the issue to `Backlog`, and stop. Do not rerun the blocked command unless the branch head changed, the external host state changed, or a human explicitly requested another repro.
