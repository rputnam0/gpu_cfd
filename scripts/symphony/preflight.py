#!/usr/bin/env python3
"""Preflight checks for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

EXPECTED_PROJECT_SLUG = "gpu-cfd-6e45c39a4350"
EXPECTED_PR_COUNT = 88


@dataclass
class Check:
    level: str
    label: str
    detail: str


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def add_check(results: list[Check], level: str, label: str, detail: str) -> None:
    results.append(Check(level=level, label=label, detail=detail))


def parse_mcp_list(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Name"):
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "name": parts[0],
                "url": parts[1],
                "bearer_token_env_var": parts[2],
                "status": parts[3],
                "auth": parts[4],
            }
        )
    return rows


def linear_mcp_check(codex_binary: str) -> Check:
    completed = subprocess.run(
        [codex_binary, "mcp", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return Check("missing", "codex mcp", "failed to list MCP servers")

    rows = parse_mcp_list(completed.stdout)
    linear_row = next((row for row in rows if row["name"] == "linear"), None)
    if linear_row is None:
        return Check("missing", "mcp:linear", "not configured")
    if linear_row["status"] != "enabled":
        return Check(
            "missing", "mcp:linear", f"configured but status={linear_row['status']}"
        )
    if linear_row["auth"] == "Not logged in":
        return Check("missing", "mcp:linear", "configured but not logged in")
    return Check("ok", "mcp:linear", f"{linear_row['status']} / {linear_row['auth']}")


def run_repo_checks(root: pathlib.Path) -> list[Check]:
    results: list[Check] = []
    required_files = [
        root / "WORKFLOW.md",
        root / ".codex/skills/gpu-cfd-symphony/SKILL.md",
        root / "docs/README_FIRST.md",
        root / "docs/tasks/pr_inventory.md",
        root / "docs/backlog/gpu_cfd_pr_backlog.json",
        root / "docs/ops/symphony_runbook.md",
        root / ".github/workflows/review-loop-harness.yml",
        root / ".github/workflows/linear-review-bridge.yml",
        root / "scripts/symphony/runtime_config.toml",
        root / "scripts/symphony/codex_runner.py",
        root / "scripts/symphony/github_linear_bridge.py",
        root / "scripts/symphony/review_loop.py",
        root / "scripts/symphony/telemetry.py",
        root / "scripts/symphony/workspace_sync.py",
    ]

    for path in required_files:
        if path.exists():
            add_check(results, "ok", path.relative_to(root).as_posix(), "present")
        else:
            add_check(results, "missing", path.relative_to(root).as_posix(), "missing")

    workflow_path = root / "WORKFLOW.md"
    if workflow_path.exists():
        workflow_text = read_text(workflow_path)
        if EXPECTED_PROJECT_SLUG in workflow_text:
            add_check(results, "ok", "WORKFLOW project slug", EXPECTED_PROJECT_SLUG)
        else:
            add_check(
                results, "missing", "WORKFLOW project slug", EXPECTED_PROJECT_SLUG
            )

        if re.search(
            r"active_states:\s*\n\s*-\s*Todo\s*\n\s*-\s*In Progress\s*\n\s*-\s*Rework\s*\n\s*-\s*Ready to Merge",
            workflow_text,
        ):
            add_check(
                results,
                "ok",
                "WORKFLOW active states",
                "Todo/In Progress/Rework/Ready to Merge",
            )
        else:
            add_check(
                results,
                "warn",
                "WORKFLOW active states",
                "unexpected active state list",
            )

        if re.search(
            r"turn_sandbox_policy:\s*\n\s*type:\s*workspaceWrite\s*\n\s*networkAccess:\s*true",
            workflow_text,
        ):
            add_check(
                results,
                "ok",
                "WORKFLOW review network access",
                "workspaceWrite + networkAccess=true",
            )
        else:
            add_check(
                results,
                "missing",
                "WORKFLOW review network access",
                "expected workspaceWrite + networkAccess=true",
            )

    runtime_config_path = root / "scripts/symphony/runtime_config.toml"
    if runtime_config_path.exists():
        runtime_config_text = read_text(runtime_config_path)
        if re.search(
            r"\[codex\.review\][^\[]*extra_configs\s*=\s*\[[^\]]*shell_environment_policy\.inherit=all",
            runtime_config_text,
            re.DOTALL,
        ):
            add_check(
                results,
                "ok",
                "review profile shell env",
                "shell_environment_policy.inherit=all",
            )
        else:
            add_check(
                results,
                "missing",
                "review profile shell env",
                "expected shell_environment_policy.inherit=all",
            )

    backlog_path = root / "docs/backlog/gpu_cfd_pr_backlog.json"
    if backlog_path.exists():
        try:
            backlog = json.loads(read_text(backlog_path))
            pr_count = sum(len(section["prs"]) for section in backlog["sections"])
            if pr_count == EXPECTED_PR_COUNT:
                add_check(results, "ok", "Backlog PR count", str(pr_count))
            else:
                add_check(
                    results,
                    "missing",
                    "Backlog PR count",
                    f"expected {EXPECTED_PR_COUNT}, found {pr_count}",
                )
        except Exception as exc:  # pragma: no cover - defensive reporting
            add_check(results, "missing", "Backlog JSON parse", str(exc))

    return results


def run_runtime_checks() -> list[Check]:
    results: list[Check] = []

    required_bins = ["git", "gh"]
    optional_bins = ["ssh", "tmux", "ts", "elixir"]
    optional_candidates = {
        "mise": [
            shutil.which("mise"),
            str(pathlib.Path.home() / ".local" / "bin" / "mise"),
        ],
    }
    codex_candidates = [
        shutil.which("codex"),
        str(pathlib.Path.home() / ".npm-global" / "bin" / "codex"),
    ]

    for name in required_bins:
        if shutil.which(name):
            add_check(results, "ok", f"binary:{name}", "found")
        else:
            add_check(results, "missing", f"binary:{name}", "not on PATH")

    resolved_codex = next(
        (path for path in codex_candidates if path and pathlib.Path(path).exists()),
        None,
    )
    if resolved_codex:
        add_check(results, "ok", "binary:codex", resolved_codex)
        results.append(linear_mcp_check(resolved_codex))
    else:
        add_check(
            results,
            "missing",
            "binary:codex",
            "not on PATH and not found at ~/.npm-global/bin/codex",
        )

    for name in optional_bins:
        if shutil.which(name):
            add_check(results, "ok", f"binary:{name}", "found")
        else:
            add_check(results, "warn", f"binary:{name}", "not on PATH")

    for name, candidates in optional_candidates.items():
        resolved = next(
            (path for path in candidates if path and pathlib.Path(path).exists()), None
        )
        if resolved:
            add_check(results, "ok", f"binary:{name}", resolved)
        else:
            add_check(results, "warn", f"binary:{name}", "not on PATH")

    required_env = ["LINEAR_API_KEY", "SYMPHONY_WORKSPACE_ROOT"]
    for name in required_env:
        if os.environ.get(name):
            add_check(results, "ok", f"env:{name}", "set")
        else:
            add_check(results, "missing", f"env:{name}", "unset")

    source_repo = os.environ.get("GPU_CFD_SOURCE_REPO_URL")
    if source_repo:
        add_check(results, "ok", "env:GPU_CFD_SOURCE_REPO_URL", source_repo)
    else:
        add_check(
            results,
            "warn",
            "env:GPU_CFD_SOURCE_REPO_URL",
            "unset; WORKFLOW default will use https://github.com/rputnam0/gpu_cfd.git",
        )

    bootstrap_ref = os.environ.get("GPU_CFD_BOOTSTRAP_REF")
    if bootstrap_ref:
        add_check(results, "ok", "env:GPU_CFD_BOOTSTRAP_REF", bootstrap_ref)
    else:
        add_check(
            results,
            "ok",
            "env:GPU_CFD_BOOTSTRAP_REF",
            "unset; default-branch bootstrap",
        )

    codex_auth = pathlib.Path.home() / ".codex/auth.json"
    if codex_auth.exists():
        add_check(results, "ok", "codex auth", codex_auth.as_posix())
    else:
        add_check(results, "missing", "codex auth", codex_auth.as_posix())

    telemetry_root = pathlib.Path(
        os.environ.get("GPU_CFD_TELEMETRY_ROOT")
        or os.environ.get("SYMPHONY_LOGS_ROOT")
        or str(pathlib.Path.home() / "projects" / "symphony-logs" / "gpu_cfd")
    ).expanduser()
    add_check(results, "ok", "telemetry root", telemetry_root.as_posix())

    if shutil.which("gh"):
        gh_auth = subprocess.run(
            ["gh", "auth", "status", "--hostname", "github.com"],
            capture_output=True,
            text=True,
            check=False,
        )
        if gh_auth.returncode == 0:
            add_check(results, "ok", "gh auth", "github.com")
        else:
            add_check(results, "missing", "gh auth", "github.com")

    return results


def print_results(mode: str, results: list[Check]) -> int:
    print(f"Symphony preflight mode: {mode}")
    exit_code = 0
    for item in results:
        marker = {
            "ok": "[ok]",
            "warn": "[warn]",
            "missing": "[missing]",
        }[item.level]
        print(f"{marker} {item.label}: {item.detail}")
        if item.level == "missing":
            exit_code = 1
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("repo", "runtime"),
        default="repo",
        help="`repo` validates committed workflow files; `runtime` validates a host before launch.",
    )
    args = parser.parse_args()

    if args.mode == "repo":
        return print_results(args.mode, run_repo_checks(repo_root()))
    return print_results(args.mode, run_runtime_checks())


if __name__ == "__main__":
    sys.exit(main())
