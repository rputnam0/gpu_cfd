# Symphony Runbook

This runbook captures the V2 repository-side setup for running Symphony against the `GPU CFD`
Linear project.

## What now exists in-repo

- `WORKFLOW.md`: repo-owned Symphony workflow contract
- `scripts/symphony/runtime_config.toml`: repo-owned Codex defaults
- `scripts/symphony/review_loop.py`: local Codex review gate plus Devin review classifier
- `scripts/symphony/pr_handoff.py`: worker-owned PR handoff that opens or updates a PR, enables
  auto-merge, and moves the issue to `In Review`
- `scripts/symphony/devin_review_gate.py`: required GitHub status gate for the external Devin review
- `scripts/symphony/post_merge_bridge.py`: merged-PR bridge that moves issues to `Done` and releases
  newly unblocked dependents
- `scripts/symphony/linear_api.py`: small Linear GraphQL helper used by the bridges and handoff
- `scripts/symphony/apply_runtime_patch.sh`: applies the tracked Symphony runtime patch to the WSL
  Symphony checkout
- `scripts/symphony/patches/symphony-thread-resume-v2.patch`: tracked fork patch that adds exact
  Codex thread continuity across `In Review -> Rework`
- `scripts/symphony/setup_live_e2e_ssh.sh`: prepares a localhost SSH-worker config for the official
  Symphony live E2E
- `.codex/skills/gpu-cfd-symphony/SKILL.md`: project-specific execution skill
- `docs/ops/symphony-gpu-cfd.service`: user-systemd unit for the WSL runtime

## Supported board contract

This repository expects the Linear statuses below on the `Projects` team.

- `Backlog`: parked or dependency-blocked work; Symphony should not pick this up
- `Todo`: ready for Symphony to start
- `In Progress`: active implementation
- `In Review`: dormant automated Devin-review queue
- `Rework`: active review-followup work
- `Done`: terminal state

## Worker-side Linear access

Symphony's reference workflow allows the worker to talk to Linear through either a configured
Linear MCP server or the injected `linear_graphql` tool.

For this repository, individual workers on WSL are expected to have both:

- Symphony's injected `linear_graphql` tool inside the app-server session
- the official Linear MCP configured in `~/.codex/config.toml`

Treat missing worker-side Linear access as a blocker. Do not replace it with repo-side polling or
watcher logic.

## Runtime fork

V2 depends on a small tracked Symphony runtime patch so `Rework` resumes the exact Codex thread
instead of rebuilding context from repo-side continuity files.

Apply it to the WSL Symphony checkout before starting the service:

```bash
cd ~/projects/gpu_cfd
bash scripts/symphony/apply_runtime_patch.sh ~/projects/symphony
```

The patch is tracked in this repo at:

- `scripts/symphony/patches/symphony-thread-resume-v2.patch`

It adds:

- `thread/resume` support in the Symphony Codex app-server client
- persisted per-issue continuation metadata in the Symphony runtime logs area
- continuation-aware worker dispatch for dormant `In Review -> Rework` wakeups
- targeted upstream Elixir test coverage for the resume path

Review loop:

- Before a PR is opened or updated for review, the active worker run executes
  `scripts/symphony/pr_handoff.py` from the issue workspace.
- That helper runs one local Codex review pass. If findings remain, it returns them to the same run
  so the worker can fix them immediately.
- If the local review is clean, the helper opens or updates the PR, enables GitHub auto-merge, and
  moves the issue to `In Review`.
- `In Review` is dormant. No repo-owned watcher or worker sleep loop stays alive.
- The GitHub Actions workflow `.github/workflows/devin-review-gate.yml` sets the required
  `devin-review-gate` status for the PR review cycle.
- Actionable Devin feedback moves the issue to `Rework`.
- A `Rework` worker is expected to pull the latest Devin review comments directly from GitHub with
  `gh api` before editing, then fix every valid actionable finding.
- Once that first Devin review round has no remaining actionable feedback, `devin-review-gate`
  turns green and the PR can merge.
- GitHub auto-merge lands the PR only after both `review-loop-harness` and
  `devin-review-gate` are green and conversation resolution is satisfied.
- `.github/workflows/linear-post-merge.yml` moves the linked issue to `Done` and promotes newly
  unblocked direct dependents from `Backlog` to `Todo`.

## Remaining external prerequisites

These items still must exist on the worker host before Symphony can run end to end:

- `LINEAR_API_KEY` exported on the worker host
- `SYMPHONY_WORKSPACE_ROOT` exported on the worker host
- `GPU_CFD_SOURCE_REPO_URL` exported when you want to override the default clone URL
- Codex authenticated on the worker host via `~/.codex/auth.json`
- GitHub push and PR auth on the worker host (`gh auth login` or SSH push access)
- Symphony itself installed on the worker host
- Repo secret `REVIEW_BRIDGE_GH_TOKEN` configured with a GitHub token that can resolve PR review
  threads when stale or non-blocking Devin threads should be cleared automatically

Fresh Symphony workspaces clone from Git, so workflow changes must be available on a pushed branch
before the worker host can consume them.

## Codex CLI on the worker host

If `codex` is missing on the worker host, install the CLI and log in before starting Symphony.

```bash
npm install -g @openai/codex
codex login
codex --version
```

The checked-in workflow uses direct Codex CLI flags for the implementation worker profile:

- implementation app-server: `gpt-5.4` with `medium`
- local review gate: `gpt-5.4` with `xhigh`

The checked-in workflow keeps issue workspaces on the supported `workspaceWrite` sandbox and enables
network access for turn execution because the worker needs GitHub and Linear access during review
handoff and follow-up.

Verify the worker host can see the Linear MCP server:

```bash
codex mcp list
```

Expected result on WSL includes:

```text
linear  https://mcp.linear.app/mcp  enabled  OAuth
```

The WSL login shell must also expose `~/.npm-global/bin` so SSH worker shells can resolve the
Codex CLI during Symphony's live E2E and any future SSH-backed worker dispatch.

## GitHub-side enforcement

Protect `main` with:

- pull requests required before merging
- required status checks:
  - `review-loop-harness`
  - `devin-review-gate`
- required conversation resolution
- required approving review count: `0`

The required external-review gate is `devin-review-gate`, not a human approval.

## Recommended worker host

Run Symphony on the `wsl` workstation instead of the local Mac so issue workspaces live next to the
GPU environment.

## Official runtime entrypoint

For autonomous operation, run the official Symphony service on WSL through a user-systemd unit,
not a repo launcher script.

Install the tracked unit:

```bash
mkdir -p ~/.config/systemd/user
cp ~/projects/gpu_cfd/docs/ops/symphony-gpu-cfd.service ~/.config/systemd/user/symphony-gpu-cfd.service
systemctl --user daemon-reload
systemctl --user enable --now symphony-gpu-cfd.service
```

Useful commands:

```bash
systemctl --user status symphony-gpu-cfd.service
journalctl --user -u symphony-gpu-cfd.service -f
systemctl --user restart symphony-gpu-cfd.service
```

The service reads `~/projects/symphony/.env`, uses the checked-out `~/projects/symphony/elixir`
runtime, writes logs under `~/projects/symphony-logs/gpu_cfd`, serves the dashboard on
`$SYMPHONY_DASHBOARD_PORT` or `4040`, and points Symphony at `~/projects/gpu_cfd/WORKFLOW.md`.

## Dashboard access

Forward the WSL dashboard port to the Mac:

```bash
ssh -L 4040:127.0.0.1:4040 wsl
```

Then open `http://127.0.0.1:4040/`.

Useful API endpoints:

- `/api/v1/state`
- `/api/v1/<issue_identifier>`
- `/api/v1/refresh`

## Runtime visibility

Use these as the primary surfaces:

- Symphony dashboard and logs for orchestrator state
- Linear workpad comment for task history, validation evidence, and blockers
- GitHub checks for local harness plus Devin review gate status

## First-time launch on `wsl`

1. Clone or sync this repository on `wsl`.
2. Ensure `~/projects/symphony/.env` exports:

```bash
export LINEAR_API_KEY=...
export SYMPHONY_WORKSPACE_ROOT=~/projects/symphony-workspaces/gpu_cfd
export GPU_CFD_SOURCE_REPO_URL=https://github.com/rputnam0/gpu_cfd.git
export SYMPHONY_DASHBOARD_PORT=4040
```

3. Install Symphony reference implementation:

```bash
git clone https://github.com/openai/symphony ~/projects/symphony
cd ~/projects/symphony/elixir
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
```

4. Apply the tracked runtime patch:

```bash
cd ~/projects/gpu_cfd
bash scripts/symphony/apply_runtime_patch.sh ~/projects/symphony
```

5. Install and start the user-systemd unit shown above.

## Smoke test shape

The expected end-to-end flow is:

1. Linear issue moves `Todo -> In Progress`
2. Worker implements and passes local Codex review
3. `pr_handoff.py` opens or updates the PR, enables auto-merge, and moves the issue to `In Review`
4. `devin-review-gate` is `pending` until a fresh Devin pass lands on the current head
5. Actionable Devin feedback moves the issue to `Rework`
6. Resumed worker fixes the issue and returns it to `In Review`
7. Fresh clean Devin pass turns `devin-review-gate` green
8. GitHub auto-merges the PR
9. `linear-post-merge` moves the issue to `Done` and releases newly unblocked dependents

## Validation status

Validated on March 15, 2026:

- repo Python suite: `uv run python -m unittest discover -s tests`
- Symphony targeted Elixir coverage for the fork patch
- official Symphony live E2E: `make e2e`

The live E2E required these environment overrides on WSL:

- `SYMPHONY_LIVE_LINEAR_TEAM_KEY=PRO`
- `SYMPHONY_LIVE_SSH_WORKER_HOSTS=symphony-e2e-1,symphony-e2e-2`
- `SYMPHONY_SSH_CONFIG=$HOME/.ssh/symphony_live_e2e_config`

Prepare the SSH-worker side once on WSL:

```bash
cd ~/projects/gpu_cfd
bash scripts/symphony/setup_live_e2e_ssh.sh
```
