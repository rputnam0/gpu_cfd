# Symphony Runbook

This runbook captures the repository-side setup for running Symphony against the `GPU CFD` Linear
project.

## What now exists in-repo

- `WORKFLOW.md`: repo-owned Symphony workflow contract
- `scripts/symphony/runtime_config.toml`: repo-owned Codex defaults
- `scripts/symphony/codex_runner.py`: profile-aware Codex entrypoint
- `.codex/skills/gpu-cfd-symphony/SKILL.md`: project-specific execution skill
- `scripts/symphony/preflight.py`: preflight checker for repo and worker-host readiness
- `scripts/symphony/review_loop.py`: local Codex review gate plus GitHub review-state helper
- `scripts/symphony/github_linear_bridge.py`: GitHub-event bridge that moves linked Linear issues into `Rework` or `Ready to Merge`
- `scripts/symphony/telemetry.py`: structured event logger for launcher, blocker, and review-loop telemetry

## Supported board contract

This repository now expects the Linear statuses below to exist on the `Projects` team.

- `Backlog`: parked or dependency-blocked work; Symphony should not pick this up
- `Todo`: ready for Symphony to start
- `In Progress`: active implementation
- `In Review`: dormant external-review queue
- `Rework`: active review-followup work
- `Ready to Merge`: active merge-finalization work
- `Done`: terminal state

Review loop:

- Before a PR is opened or marked ready, the agent must run one local Codex review pass.
- After the PR enters `In Review`, the worker stops. No repo-owned watcher or local poll loop keeps running.
- The GitHub Actions workflow `.github/workflows/linear-review-bridge.yml` moves the linked issue into `Rework` when Devin finds issues and into `Ready to Merge` when the current head is clean and mergeable.
- `Rework` runs fix valid findings, revalidate, rerun the Codex review gate, and send the PR back to `In Review`.
- `Ready to Merge` runs perform the final merge and move the issue to `Done`.

## Remaining external prerequisites

These items still must exist on the worker host before Symphony can run end-to-end:

- `LINEAR_API_KEY` exported on the worker host
- `SYMPHONY_WORKSPACE_ROOT` exported on the worker host
- Codex authenticated on the worker host via `~/.codex/auth.json`
- Linear MCP configured in `~/.codex/config.toml` on the worker host and authenticated with `codex mcp login linear`
- GitHub push/PR auth on the worker host (`gh auth login` or SSH push access)
- Symphony itself installed on the worker host
- Recommended for the remote GPU workflow: `tmux` and `ts`

Fresh Symphony workspaces clone from Git, so any workflow changes must be available on a pushed
branch before the worker host can consume them. Leave `GPU_CFD_BOOTSTRAP_REF` unset for normal
default-branch runs. Set it only when you intentionally want issue workspaces to test an unmerged
branch.

## Codex CLI on the worker host

If `codex` is missing on the worker host, install the CLI and log in before starting Symphony.

```bash
npm install -g @openai/codex
codex login
codex --version
```

On this WSL host, `codex` is installed under `~/.npm-global/bin/codex`. Interactive shells see it,
but non-interactive login shells may not inherit that path automatically. The checked-in runtime
therefore uses `scripts/symphony/codex_runner.py`, which resolves the binary and applies the
repo-owned profile from `scripts/symphony/runtime_config.toml` so Symphony does not depend on
shell-specific `PATH` setup or on a human-maintained global Codex profile.

The sanctioned agent-side Linear path is Codex MCP, not repo-local API helper scripts. On WSL the
required config entry is:

```toml
[mcp_servers.linear]
url = "https://mcp.linear.app/mcp"
```

Verify it with:

```bash
~/.npm-global/bin/codex mcp list
```

The output should show `linear` as `enabled` and authenticated. If it shows `Not logged in`, run:

```bash
~/.npm-global/bin/codex mcp login linear
```

## Preflight

Run these checks from the repository root.

Repo-side assets:

```bash
uv run python scripts/symphony/preflight.py --mode repo
```

Worker-host runtime readiness:

```bash
uv run python scripts/symphony/preflight.py --mode runtime
```

The runtime preflight now checks both Codex auth and GitHub CLI auth because the automated review
loop needs to open PRs, read Linear and GitHub review state, and merge clean PRs without manual
handoffs.

## GitHub-side enforcement

For a stronger merge guarantee, this repository now includes a GitHub Actions workflow at
`.github/workflows/review-loop-harness.yml`. It runs the deterministic checks that should never be
skipped on a PR:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/symphony/preflight.py --mode repo`

Recommended GitHub setting:

- Protect `main`
- Require pull requests before merging
- Mark `review-loop-harness` as a required status check

That does not replace the local Codex review or the Devin review loop. It complements them with a
merge-time enforcement point that GitHub can actually block on.

This repository also includes `.github/workflows/linear-review-bridge.yml`, which reacts to PR and
review events, inspects the latest Devin review state on the current head, resolves the linked
Linear issue from the PR body/title/branch, and updates that issue into `Rework` or `Ready to Merge`
without keeping a repo-owned daemon alive.

## Recommended worker host

Run Symphony on the `wsl` workstation instead of the local Mac so issue workspaces live next to the
GPU environment and long-running jobs can use the existing `tmux` and `task-spooler` workflow.

## Required entrypoint

For autonomous operation, treat `./scripts/symphony/run_wsl_symphony.sh` as the only supported way
to start Symphony for this repository.

- It sources `~/projects/symphony/.env`
- It exports the runtime paths expected by the workflow
- It runs the worker-host preflight
- It emits launcher telemetry
- It starts Symphony with the required preview acknowledgement flag

Do not start the runtime with direct `bin/symphony` or ad hoc `mise exec -- ./bin/symphony`
commands. Those bypass the repo-owned safety and observability layer.

## First-time launch on `wsl`

1. Clone or sync this repository on `wsl`.
2. Export runtime variables:

```bash
export LINEAR_API_KEY=...
export SYMPHONY_WORKSPACE_ROOT=~/projects/symphony-workspaces/gpu_cfd
export GPU_CFD_SOURCE_REPO_URL=https://github.com/rputnam0/gpu_cfd.git
```

Add `GPU_CFD_BOOTSTRAP_REF=<branch-name>` only while testing a branch that has not yet been merged
to `main`.

Configure GitHub CLI as the git credential helper on WSL so HTTPS clones can still push branches and
open PRs:

```bash
gh auth setup-git
```

3. Run the runtime preflight:

```bash
cd ~/projects/gpu_cfd
uv run python scripts/symphony/preflight.py --mode runtime
```

4. Install Symphony reference implementation:

```bash
git clone https://github.com/openai/symphony ~/projects/symphony
cd ~/projects/symphony/elixir
mise trust
mise install
mise exec -- mix setup
mise exec -- mix build
```

5. Start Symphony in `tmux`:

```bash
tmux new -s symphony
cd ~/projects/gpu_cfd
./scripts/symphony/run_wsl_symphony.sh
```

The launcher script reads `~/projects/symphony/.env`, fills in sane defaults for
`SYMPHONY_WORKSPACE_ROOT`, `GPU_CFD_SOURCE_REPO_URL`, and `GPU_CFD_BOOTSTRAP_REF`, runs the
runtime preflight, and then starts Symphony from the checked-out `~/projects/symphony/elixir`
directory, and passes the required preview acknowledgement flag for the current Symphony reference
implementation.

## Telemetry and operator visibility

The launcher and agent workflow now emit structured telemetry under the Symphony logs root.

- Global event stream: `~/projects/symphony-logs/gpu_cfd/events.jsonl`
- Per-issue event streams: `~/projects/symphony-logs/gpu_cfd/issues/<ISSUE>.jsonl`
- Blocker-focused stream: `~/projects/symphony-logs/gpu_cfd/blockers.jsonl`

Recommended event types:

- `symphony_launcher_invoked`
- `issue_started`
- `blocker`
- `pr_opened`
- `review_requested`
- `review_feedback_detected`
- `merged`

Manual usage example:

```bash
cd ~/projects/gpu_cfd
uv run python scripts/symphony/telemetry.py event \
  --event-type blocker \
  --issue PRO-5 \
  --state "In Progress" \
  --message "Missing external credential for downstream system" \
  --detail subsystem=github \
  --detail severity=error
```

## Automated review loop

This repository now expects a two-stage automated review flow for every Symphony-driven PR.

1. Local review gate before PR submission:

```bash
cd ~/projects/gpu_cfd
uv run python scripts/symphony/review_loop.py codex-review --base origin/main
```

This stores Codex review artifacts under `.codex/review_artifacts/<branch>/`. The agent must inspect
the latest report, fix material findings, and rerun the gate once before sending the PR to GitHub
review. The local gate now has a hard timeout so a single slow review cannot stall Symphony forever.

2. Linear-driven review handoff after PR submission:

- record the PR URL in a Linear comment before moving it to `In Review`
- emit `review_requested` telemetry right before each review handoff
- stop the worker after moving to `In Review`
- rely on Linear workflow transitions to move the issue into `Rework` or `Ready to Merge`
- let `.github/workflows/linear-review-bridge.yml` perform the `In Review -> Rework` and
  `In Review -> Ready to Merge` transitions from GitHub events
- use GitHub review comments as detail lookup when working a `Rework` or `Ready to Merge` run
- merge only after the current head is clean and GitHub protection is green

## Recommended rollout

- Keep `agent.max_concurrent_agents` at `1` initially.
- Leave only `PRO-5` (`FND-01`) in `Todo` for the first unattended run.
- Keep all other issues in `Backlog` until their blockers are complete.
- Raise concurrency only after the first Foundation issues prove out the repo and review loop.

## Operator rules

- The repository docs remain the technical source of truth.
- Linear state decides whether Symphony runs an issue.
- Only move an issue to `Todo` when its blockers are truly resolved.
- Use `In Review` as the dormant review queue. The next work run should start only after Linear moves the issue into an active follow-up state such as `Rework` or `Ready to Merge`.
