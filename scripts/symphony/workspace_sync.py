#!/usr/bin/env python3
"""Keep Symphony workspaces aligned with the latest base branch when safe."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


class WorkspaceSyncError(RuntimeError):
    """Raised when a Symphony workspace is in an unsafe state for execution."""


def run_git(
    workspace: pathlib.Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=check,
    )


def current_branch(workspace: pathlib.Path) -> str:
    completed = run_git(workspace, "branch", "--show-current")
    return completed.stdout.strip()


def worktree_clean(workspace: pathlib.Path) -> bool:
    completed = run_git(workspace, "status", "--short")
    return completed.stdout.strip() == ""


def origin_head_branch(workspace: pathlib.Path) -> str:
    completed = run_git(
        workspace,
        "symbolic-ref",
        "--quiet",
        "--short",
        "refs/remotes/origin/HEAD",
        check=False,
    )
    if completed.returncode == 0:
        remote_head = completed.stdout.strip()
        if remote_head.startswith("origin/"):
            return remote_head.removeprefix("origin/")
    branch = current_branch(workspace)
    return branch or "main"


def checkout_bootstrap_ref(workspace: pathlib.Path, bootstrap_ref: str) -> str:
    existing = run_git(
        workspace,
        "show-ref",
        "--verify",
        "--quiet",
        f"refs/heads/{bootstrap_ref}",
        check=False,
    )
    if existing.returncode == 0:
        run_git(workspace, "checkout", bootstrap_ref)
    else:
        run_git(
            workspace,
            "checkout",
            "-b",
            bootstrap_ref,
            "--track",
            f"origin/{bootstrap_ref}",
        )
    pull = run_git(workspace, "pull", "--ff-only", "origin", bootstrap_ref)
    summary = pull.stdout.strip() or pull.stderr.strip() or "already up to date"
    return f"synced `{bootstrap_ref}`: {summary}"


def sync_workspace(workspace: pathlib.Path, bootstrap_ref: str | None = None) -> str:
    workspace = workspace.resolve()
    branch = current_branch(workspace)
    base_branch = origin_head_branch(workspace)
    bootstrap_ref = bootstrap_ref or os.environ.get("GPU_CFD_BOOTSTRAP_REF") or None

    if not branch:
        raise WorkspaceSyncError(
            "workspace is on a detached HEAD; a named base branch or issue branch is required"
        )

    run_git(workspace, "fetch", "origin", "--prune")

    if bootstrap_ref:
        allowed_branches = {base_branch, "main", "master", bootstrap_ref}
        if branch not in allowed_branches:
            return (
                f"skipped: current branch `{branch or 'detached'}` is not compatible "
                f"with bootstrap ref `{bootstrap_ref}`"
            )
        if not worktree_clean(workspace):
            raise WorkspaceSyncError(
                f"base-like branch `{branch}` has local changes; refusing to run on ambiguous workspace state"
            )
        if branch != bootstrap_ref:
            return checkout_bootstrap_ref(workspace, bootstrap_ref)
        pull = run_git(workspace, "pull", "--ff-only", "origin", bootstrap_ref)
        summary = pull.stdout.strip() or pull.stderr.strip() or "already up to date"
        return f"synced `{bootstrap_ref}`: {summary}"

    if branch not in {base_branch, "main", "master"}:
        return f"skipped: current branch `{branch or 'detached'}` is not a clean base branch"

    if not worktree_clean(workspace):
        raise WorkspaceSyncError(
            f"base branch `{branch}` has local changes; refusing to run on ambiguous workspace state"
        )

    pull = run_git(workspace, "pull", "--ff-only", "origin", branch)
    summary = pull.stdout.strip() or pull.stderr.strip() or "already up to date"
    return f"synced `{branch}`: {summary}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace path to inspect and fast-forward when safe.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = sync_workspace(pathlib.Path(args.workspace))
    except WorkspaceSyncError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
