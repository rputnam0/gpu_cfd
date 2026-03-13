# Symphony Runbook

This runbook captures the repository-side setup for running Symphony against the `GPU CFD` Linear
project.

## What now exists in-repo

- `WORKFLOW.md`: repo-owned Symphony workflow contract
- `.codex/skills/gpu-cfd-symphony/SKILL.md`: project-specific execution skill
- `scripts/symphony/preflight.py`: preflight checker for repo and worker-host readiness
- `scripts/symphony/review_loop.py`: local Codex review gate plus GitHub review polling helper
- `scripts/symphony/telemetry.py`: structured event logger for launcher, blocker, and review-loop telemetry

## Supported board contract

This repository can run correctly with the Linear statuses that already exist today.

- `Backlog`: parked or dependency-blocked work; Symphony should not pick this up
- `Todo`: ready for Symphony to start
- `In Progress`: active implementation or retry loop
- `In Review`: active PR review loop; Symphony waits for Devin review, fixes valid findings, and merges when clean
- `Done`: terminal state

Review loop:

- Before a PR is opened or marked ready, the agent must run one local Codex review pass.
- After the PR enters `In Review`, the agent waits for Devin GitHub review feedback.
- Valid findings are fixed on-branch, revalidated locally, rerun through the Codex review gate, and pushed.
- The agent waits for a fresh Devin review on the new head before merging.
- Merge completes the Linear issue and moves it to `Done`.

## Remaining external prerequisites

These items still must exist on the worker host before Symphony can run end-to-end:

- `LINEAR_API_KEY` exported on the worker host
- `SYMPHONY_WORKSPACE_ROOT` exported on the worker host
- Codex authenticated on the worker host via `~/.codex/auth.json`
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
but non-interactive login shells may not inherit that path automatically. The checked-in
`WORKFLOW.md` therefore uses the explicit `$HOME/.npm-global/bin/codex` path so Symphony does not
depend on shell-specific `PATH` setup.

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
loop needs to open PRs, poll review threads, and merge clean PRs without manual handoffs.

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
directory. It also passes the required preview acknowledgement flag for the current Symphony
reference implementation.

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
- `review_wait`
- `review_action_required`
- `review_clean`
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
review.

2. GitHub review polling after PR submission:

```bash
cd ~/projects/gpu_cfd
uv run python scripts/symphony/review_loop.py wait \
  --issue PRO-5 \
  --reviewer devin-ai-integration[bot] \
  --timeout-seconds 900
```

The helper classifies the PR head as one of:

- `pending_initial_review`: no Devin review has landed on this PR yet
- `pending_rereview`: previous Devin feedback exists, but only on an older head commit
- `action_required`: the current head still has actionable Devin review feedback
- `clean`: the current head has no actionable Devin review feedback

Recommended agent behavior:

- `action_required`: fix valid comments, rerun targeted validation, rerun the local Codex review gate, push
- `pending_*`: stay in `In Review`, leave a brief Linear workpad note, keep waiting
- `clean`: merge the PR and move the Linear issue to `Done`

## Recommended rollout

- Keep `agent.max_concurrent_agents` at `1` initially.
- Leave only `PRO-5` (`FND-01`) in `Todo` for the first unattended run.
- Keep all other issues in `Backlog` until their blockers are complete.
- Raise concurrency only after the first Foundation issues prove out the repo and review loop.

## Operator rules

- The repository docs remain the technical source of truth.
- Linear state decides whether Symphony runs an issue.
- Only move an issue to `Todo` when its blockers are truly resolved.
- Use `In Review` as the active review-and-merge loop, not a passive holding state.
