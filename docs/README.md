# Docs Overview

This folder is organized into five layers:

1. `authority/` — frozen package-level decisions and their JSON companions
2. `specs/` — phase implementation specs that consume the authority bundle
3. `backlog/` — the canonical PR worklist plus the human-readable overview
4. `tasks/` — agent-facing task briefs, seam rules, and PR ownership mapping
5. `ops/` — operator-facing automation and runbook documents

This repo uses progressive disclosure. For implementation work, start in `AGENTS.md`, then use:

- `docs/tasks/pr_inventory.md` to resolve the assigned PR card
- `docs/tasks/README.md` for task-workspace rules
- `docs/authority/README.md` for authority order and consumption rules
- `docs/specs/README.md` for the spec map

Read only the exact supporting docs cited by the PR card unless the task requires broader discovery.
