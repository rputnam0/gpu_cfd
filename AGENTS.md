# Repository Guidelines

## Repository Map
This repo is the doc-driven control plane for the `gpu_cfd` Symphony workflow. Keep context narrow:
start from the issue, the exact PR card, and the cited docs before opening broader material.

```text
.
├── AGENTS.md
├── WORKFLOW.md
├── docs/
│   ├── README.md
│   ├── authority/
│   ├── backlog/
│   ├── ops/
│   ├── specs/
│   └── tasks/
├── scripts/
│   ├── authority/
│   └── symphony/
├── tests/
└── dev/
```

### What each area is for
- `WORKFLOW.md`: implementation-worker contract injected by Symphony.
- `docs/README.md`: top-level knowledge-base index.
- `docs/authority/`: authority order, frozen bundle, and consumption rules.
- `docs/specs/`: phase specs; read only the cited sections you need.
- `docs/tasks/`: PR inventory, task files, templates, and review rules.
- `docs/backlog/gpu_cfd_pr_backlog.json`: canonical PR scope and dependency graph.
- `docs/ops/`: operator/runtime docs; open by default only for orchestration tasks.
- `scripts/authority/`: authority parsing and loader helpers.
- `scripts/symphony/`: dispatch, Linear, handoff, review, bridge, and trace helpers.
- `tests/`: Python `unittest` regression suite.
- `dev/`: orientation only, not source of truth.

### Default reading path
- Implementation or rework: `AGENTS.md` -> Linear issue/PR context -> exact task file and PR card -> cited docs only.
- Orchestration/runtime work: `AGENTS.md` -> `docs/ops/README.md` -> `docs/ops/symphony_runbook.md` -> exact code touched.
- Do not read the full docs corpus by default.
- If the PR ID, task file, or card location is unclear, use `docs/tasks/pr_inventory.md` as the fallback map.
- Update the canonical Linear workpad before edits and use it as working memory.

## Source of Truth Order
1. Exact PR card and the sources it cites
2. `docs/authority/`
3. Exact cited sections under `docs/specs/`
4. `docs/backlog/gpu_cfd_pr_backlog.json`
5. `docs/ops/` for orchestration/runtime rules only

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

## PR Review Comments (GitHub CLI)
- To pull full inline review comments (including `diff_hunk`, `line`, `start_line`, `path`, and `body`), use:
  - `gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/:owner/:repo/pulls/<PR_NUMBER>/comments`
- To filter for a specific bot or reviewer (example: Devin bot), pipe through `jq`:
  - `gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/:owner/:repo/pulls/<PR_NUMBER>/comments | jq '[ .[] | select(.user.login == "devin-ai-integration[bot]") | { diff_hunk, line, start_line, body, path } ]'`


## Symphony Guardrails
- `In Review` is dormant. Resume work only from active states such as `Todo`, `In Progress`, or `Rework`.
- Do not add repo-side watcher daemons, poll loops, or sleep-based control flow.
- Worker-local Linear access is mandatory. If missing, record a blocker and stop.
- Use the tracked WSL systemd unit for production orchestration, not ad hoc long-lived shells.
- Prefer WSL for GPU-heavy work; use `tmux` and `ts` for long remote runs.
- Leave behind enough notes, logs, and identifiers for another agent to resume the work safely.

## Documentation Standards
Set minimum doc updates required with code changes (e.g., README updates, API docs, usage examples).

## Lint and Type Checks
Specify required lint and type checks (e.g., `ruff`, `mypy`, or project-specific tools) and when they must be run (local, CI, pre-commit).
