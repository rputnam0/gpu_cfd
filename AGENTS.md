# Repository Guidelines

## Project Structure & Module Organization
Describe where core modules live (e.g., `src/`, repo root) and the purpose of each major module. Identify where tests and test data live (e.g., `tests/`, `tests/data/`). Note build metadata location (e.g., `pyproject.toml`) and the lockfile (`uv.lock`).

## Build, Test, and Development Commands
- Use `uv run` for all commands. Do not use `pip`.
- `uv sync` installs dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python <entrypoint>.py` launches the app or primary workflow.
- `uv run python -m pytest` runs the full suite; narrow with `uv run python -m pytest <path>` when iterating.
- `uv run python -m black .` formats the codebase; run it before committing.

## Coding Style & Naming Conventions
Target Python 3.12 unless the project specifies otherwise. Use 4-space indentation and type hints on public functions. Prefer descriptive `snake_case` for functions and variables; keep constants uppercase. Use module-level loggers (`logger = logging.getLogger(__name__)`) for traceability. Favor docstrings describing intent and behavior over implementation trivia.

## Testing Guidelines (TDD Required)
Follow test-driven development.
- Red: write a failing test for the next small behavior change.
- Green: implement the smallest change to make the test pass.
- Refactor: improve structure without changing behavior, keeping tests green.
Best practices.
- Keep tests small and single-purpose.
- Prefer deterministic tests with explicit units, tolerances, and fixtures.
- Name tests by behavior, not implementation.
- Run the most relevant tests first, then the full suite before review.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (<=72 chars) and commit after each discrete logical block of work, aligned with your TDD steps. Create a new branch for every feature, bugfix, or experiment. Once changes are complete and tests pass, open a PR using the GitHub CLI (`gh`).

PR quality bar.
Title: Summarize the change clearly (e.g., "Fix: User authentication bug" rather than "Fixes").
Description: Explain the why (the problem being solved) and the what (the changes made), not just the how. Include screenshots, relevant ticket numbers, and testing instructions if applicable.

## Planning Documents and PRDs
This project has a written plan found `docs/`. The plan may be converted into PRs or an explicit checklist of to-dos.
Best practices.
- Keep the checklist in the repo and update it as work progresses.
- For each to-do, create a branch, implement the change, add/update tests, and commit in logical blocks.
- Mark items complete only when tests pass and the change is in a PR.
- Reference the checklist items in commit messages or PR descriptions to keep traceability.
- Leave clear notes for the next agent about what is done, what is in review, and what remains.

## PR Review Comments (GitHub CLI)
- To pull full inline review comments (including `diff_hunk`, `line`, `start_line`, `path`, and `body`), use:
  - `gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/:owner/:repo/pulls/<PR_NUMBER>/comments`
- To filter for a specific bot or reviewer (example: Devin bot), pipe through `jq`:
  - `gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/:owner/:repo/pulls/<PR_NUMBER>/comments | jq '[ .[] | select(.user.login == "devin-ai-integration[bot]") | { diff_hunk, line, start_line, body, path } ]'`

## Environment & Tooling
Use `uv` for environments and dependency management. Keep `pyproject.toml` and `uv.lock` in sync and committed. Do not commit local dev tooling or environments, including `.venv`, IDE state, caches, or tool binaries. Install dev tools locally via `uv` only.

## Remote GPU Workflow
Primary development may happen on a local Mac, but GPU-heavy work should run on the workstation reachable via `ssh wsl`.
- Prefer running long experiments, training jobs, large batch scripts, and other GPU-bound workflows on the workstation instead of the local laptop.
- Use SSH for one-off remote commands, for example: `ssh wsl 'cd /path/to/repo && uv run python <entrypoint>.py'`.
- Start a `tmux` session on the workstation before launching long-running jobs so work survives disconnects. Typical flow: `ssh wsl`, `tmux new -s gpu`, run commands, detach with `Ctrl-b` then `d`, and later reattach with `tmux attach -t gpu`.
- For Symphony orchestration on WSL, use the tracked user-systemd unit in [docs/ops/symphony-gpu-cfd.service](/Users/rexputnam/Documents/projects/gpu_cfd/docs/ops/symphony-gpu-cfd.service). Do not rely on ad hoc launcher scripts or manual long-lived shell sessions as the production control plane.
- For Symphony review handoffs, use Linear workflow states as the only control plane. `In Review` is dormant, and work resumes only when Linear moves the issue into an active state such as `Rework`. Do not add repo-side watcher daemons or local sleep/poll loops.
- Individual WSL Codex workers must retain worker-local Linear access. Symphony's injected `linear_graphql` tool is preferred when present, and the host Linear MCP server must also stay configured and logged in. Treat missing worker-side Linear access as a blocker rather than introducing repo-local polling fallbacks.
- Use `task-spooler` (`ts`) on the workstation to queue experiments instead of manually juggling multiple concurrent jobs. Typical flow: `ts uv run python <entrypoint>.py ...` and `ts` to inspect queue status.
- Keep large artifacts, caches, datasets, and intermediate outputs on the workstation when possible; only sync back the files needed for review or commit.
- If a dashboard or notebook is needed, prefer SSH port forwarding rather than exposing services directly on the network.
- When an agent starts a long remote run, it should leave behind a clear command history, log location, and any job or queue identifiers needed to resume or inspect the work later.

## Versioning and Release Notes
Define when to bump versions (e.g., bugfix, feature, breaking change) and where to update release notes or changelogs. Call out any required tagging or release workflow steps.

## Security and Secrets
Never commit secrets. Define how local environment variables are managed (e.g., `.env` with a `.env.example` template) and where secure storage is expected.

## Documentation Standards
Set minimum doc updates required with code changes (e.g., README updates, API docs, usage examples).

## Lint and Type Checks
Specify required lint and type checks (e.g., `ruff`, `mypy`, or project-specific tools) and when they must be run (local, CI, pre-commit).
