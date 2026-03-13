#!/usr/bin/env python3
"""Preflight checks for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
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


def run_repo_checks(root: pathlib.Path) -> list[Check]:
    results: list[Check] = []
    required_files = [
        root / "WORKFLOW.md",
        root / ".codex/skills/gpu-cfd-symphony/SKILL.md",
        root / "docs/README_FIRST.md",
        root / "docs/tasks/pr_inventory.md",
        root / "docs/backlog/gpu_cfd_pr_backlog.json",
        root / "docs/ops/symphony_runbook.md",
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
            add_check(results, "missing", "WORKFLOW project slug", EXPECTED_PROJECT_SLUG)

        if re.search(r"active_states:\s*\n\s*-\s*Todo\s*\n\s*-\s*In Progress", workflow_text):
            add_check(results, "ok", "WORKFLOW active states", "Todo/In Progress")
        else:
            add_check(results, "warn", "WORKFLOW active states", "unexpected active state list")

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
    optional_bins = ["ssh", "tmux", "ts", "mise", "elixir"]
    codex_candidates = [
        shutil.which("codex"),
        str(pathlib.Path.home() / ".npm-global" / "bin" / "codex"),
    ]

    for name in required_bins:
        if shutil.which(name):
            add_check(results, "ok", f"binary:{name}", "found")
        else:
            add_check(results, "missing", f"binary:{name}", "not on PATH")

    resolved_codex = next((path for path in codex_candidates if path and pathlib.Path(path).exists()), None)
    if resolved_codex:
        add_check(results, "ok", "binary:codex", resolved_codex)
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
            "warn",
            "env:GPU_CFD_BOOTSTRAP_REF",
            "unset; use this when the Symphony bootstrap files live on a non-default branch",
        )

    codex_auth = pathlib.Path.home() / ".codex/auth.json"
    if codex_auth.exists():
        add_check(results, "ok", "codex auth", codex_auth.as_posix())
    else:
        add_check(results, "missing", "codex auth", codex_auth.as_posix())

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
