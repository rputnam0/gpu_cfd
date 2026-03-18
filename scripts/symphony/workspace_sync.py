#!/usr/bin/env python3
"""Workspace bootstrap helpers for Symphony control-plane sync."""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any


CONTROL_REPO_ROOT_ENV = "GPU_CFD_CONTROL_REPO_ROOT"
TRACE_FULL_CONTEXT_ENV = "GPU_CFD_TRACE_FULL_CONTEXT"
DEFAULT_CONTROL_PLANE_SOURCE_REF = "origin/main"
DEFAULT_REQUIRED_LAUNCH_LABELS = ("symphony-canary",)
DEFAULT_CONTROL_PLANE_PATH_PATTERNS = (
    "WORKFLOW.md",
    ".codex/config.toml",
    ".codex/skills/gpu-cfd-symphony/SKILL.md",
    ".codex/agents/*.md",
    "scripts/symphony/*.py",
    "scripts/symphony/runtime_config.toml",
)


class WorkspaceSyncError(RuntimeError):
    """Raised when a workspace cannot be synchronized safely."""


def resolve_control_repo_root(default_root: pathlib.Path) -> pathlib.Path:
    override = pathlib.Path(os.environ.get(CONTROL_REPO_ROOT_ENV, default_root.as_posix()))
    return override.expanduser().resolve()


def trace_full_context_enabled() -> bool:
    raw = os.environ.get(TRACE_FULL_CONTEXT_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_relative_path(root: pathlib.Path, path: pathlib.Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def expand_control_plane_paths(
    repo: pathlib.Path,
    *,
    patterns: tuple[str, ...] = DEFAULT_CONTROL_PLANE_PATH_PATTERNS,
) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        matches = sorted(
            path
            for path in repo.glob(pattern)
            if path.is_file()
        )
        if not matches and not any(char in pattern for char in "*?[]"):
            matches = [repo / pattern]
        for match in matches:
            if not match.exists():
                continue
            relative = normalize_relative_path(repo, match)
            if relative not in resolved:
                resolved.append(relative)
    return resolved


def run_git(
    root: pathlib.Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise WorkspaceSyncError(
            f"git {' '.join(args)} failed in {root}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def refresh_origin_refs(root: pathlib.Path) -> None:
    run_git(root, ["fetch", "origin", "--prune"])


def workspace_has_uncommitted_path(workspace: pathlib.Path, relative_path: str) -> bool:
    completed = run_git(
        workspace,
        ["status", "--short", "--", relative_path],
        check=False,
    )
    if completed.returncode != 0:
        raise WorkspaceSyncError(
            f"unable to inspect workspace dirtiness for {relative_path}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return bool(completed.stdout.strip())


def branch_owns_path(
    workspace: pathlib.Path,
    relative_path: str,
    *,
    source_ref: str = DEFAULT_CONTROL_PLANE_SOURCE_REF,
) -> bool:
    completed = run_git(
        workspace,
        ["diff", "--name-only", f"{source_ref}...HEAD", "--", relative_path],
        check=False,
    )
    if completed.returncode != 0:
        raise WorkspaceSyncError(
            f"unable to compare {relative_path} against {source_ref}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return bool(completed.stdout.strip())


def read_file_from_ref(
    repo: pathlib.Path,
    relative_path: str,
    *,
    source_ref: str = DEFAULT_CONTROL_PLANE_SOURCE_REF,
) -> str:
    completed = run_git(
        repo,
        ["show", f"{source_ref}:{relative_path}"],
        check=False,
    )
    if completed.returncode != 0:
        raise WorkspaceSyncError(
            f"unable to read {relative_path} from {source_ref}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def sync_control_plane(
    repo: pathlib.Path,
    workspace: pathlib.Path,
    *,
    source_ref: str = DEFAULT_CONTROL_PLANE_SOURCE_REF,
    patterns: tuple[str, ...] = DEFAULT_CONTROL_PLANE_PATH_PATTERNS,
) -> dict[str, Any]:
    refresh_origin_refs(repo)
    if workspace.resolve() != repo.resolve():
        refresh_origin_refs(workspace)

    synced: list[str] = []
    branch_owned: list[str] = []
    blocked_dirty: list[str] = []
    sync_candidates: list[str] = []

    for relative_path in expand_control_plane_paths(repo, patterns=patterns):
        if workspace_has_uncommitted_path(workspace, relative_path):
            blocked_dirty.append(relative_path)
            continue
        if branch_owns_path(workspace, relative_path, source_ref=source_ref):
            branch_owned.append(relative_path)
            continue
        sync_candidates.append(relative_path)

    if blocked_dirty:
        raise WorkspaceSyncError(
            "refusing to overwrite dirty control-plane files in the issue workspace: "
            + ", ".join(blocked_dirty)
        )

    for relative_path in sync_candidates:
        target_path = workspace / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            read_file_from_ref(repo, relative_path, source_ref=source_ref),
            encoding="utf-8",
        )
        synced.append(relative_path)

    result: dict[str, Any] = {
        "workspace": workspace.as_posix(),
        "control_repo": repo.as_posix(),
        "source_ref": source_ref,
        "synced_paths": synced,
        "branch_owned_paths": branch_owned,
        "blocked_dirty_paths": blocked_dirty,
        "status": "blocked" if blocked_dirty else "ok",
    }
    return result
