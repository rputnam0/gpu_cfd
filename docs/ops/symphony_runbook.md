# Symphony Runbook

This runbook captures the repository-side setup for running Symphony against the `GPU CFD` Linear
project.

## What now exists in-repo

- `WORKFLOW.md`: repo-owned Symphony workflow contract
- `.codex/skills/gpu-cfd-symphony/SKILL.md`: project-specific execution skill
- `scripts/symphony/preflight.py`: preflight checker for repo and worker-host readiness

## Supported board contract

This repository can run correctly with the Linear statuses that already exist today.

- `Backlog`: parked or dependency-blocked work; Symphony should not pick this up
- `Todo`: ready for Symphony to start
- `In Progress`: active implementation or retry loop
- `In Review`: human review hold state; Symphony stops touching the issue
- `Done`: terminal state

Review rework loop:

- If review requests changes, move the issue back to `Todo` or `In Progress`.
- Symphony will pick it up again on the next poll.

Optional future upgrade:

- Add `Rework` and `Merging` statuses in Linear if you want a richer automatic review/merge loop.
- The current repository setup does not require those extra states to begin execution safely.

## Remaining external prerequisites

These items still must exist on the worker host before Symphony can run end-to-end:

- `LINEAR_API_KEY` exported on the worker host
- `SYMPHONY_WORKSPACE_ROOT` exported on the worker host
- Codex authenticated on the worker host via `~/.codex/auth.json`
- GitHub push/PR auth on the worker host (`gh auth login` or SSH push access)
- Symphony itself installed on the worker host
- Recommended for the remote GPU workflow: `tmux` and `ts`

Fresh Symphony workspaces clone from Git, so this bootstrap setup must be available on a pushed
branch. Until these files are merged to the repo's default branch, set
`GPU_CFD_BOOTSTRAP_REF=codex/fix-review-findings` on the worker host so issue workspaces check out
the branch that contains the workflow bootstrap.

## Codex CLI on the worker host

If `codex` is missing on the worker host, install the CLI and log in before starting Symphony.

```bash
npm install -g @openai/codex
codex login
codex --version
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

## Recommended worker host

Run Symphony on the `wsl` workstation instead of the local Mac so issue workspaces live next to the
GPU environment and long-running jobs can use the existing `tmux` and `task-spooler` workflow.

## First-time launch on `wsl`

1. Clone or sync this repository on `wsl`.
2. Export runtime variables:

```bash
export LINEAR_API_KEY=...
export SYMPHONY_WORKSPACE_ROOT=~/projects/symphony-workspaces/gpu_cfd
export GPU_CFD_SOURCE_REPO_URL=git@github.com:rputnam0/gpu_cfd.git
export GPU_CFD_BOOTSTRAP_REF=codex/fix-review-findings
```

3. Run the runtime preflight:

```bash
cd /path/to/gpu_cfd
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
cd ~/projects/symphony/elixir
mise exec -- ./bin/symphony /home/rputn/projects/gpu_cfd/WORKFLOW.md --logs-root ~/projects/symphony-logs/gpu_cfd
```

## Recommended rollout

- Keep `agent.max_concurrent_agents` at `1` initially.
- Leave only `PRO-5` (`FND-01`) in `Todo` for the first unattended run.
- Keep all other issues in `Backlog` until their blockers are complete.
- Raise concurrency only after the first Foundation issues prove out the repo and review loop.

## Operator rules

- The repository docs remain the technical source of truth.
- Linear state decides whether Symphony runs an issue.
- Only move an issue to `Todo` when its blockers are truly resolved.
- Use `In Review` as the stop point for human inspection and merge approval.
