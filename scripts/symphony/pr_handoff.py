#!/usr/bin/env python3
"""Worker-owned Symphony PR handoff for local Codex review and PR opening."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

try:
    from scripts.symphony import github_linear_bridge, review_loop, telemetry
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import github_linear_bridge, review_loop, telemetry


IN_REVIEW_STATE = "In Review"
NO_FINDINGS_MARKERS = (
    "no material findings remain",
    "no material issues remain",
    "no material findings were identified",
    "no findings",
)
FINDING_MARKER_PATTERN = re.compile(r"\[P[0-3]\]")


@dataclass(frozen=True)
class PullRequestRef:
    number: int
    url: str
    is_draft: bool


@dataclass(frozen=True)
class ReviewResult:
    status: str
    message: str
    manifest: dict[str, Any] | None


class HandoffError(RuntimeError):
    """Raised when the PR handoff cannot complete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Issue workspace path. Defaults to the current directory.",
    )
    return parser.parse_args()


def control_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def workspace_issue_identifier(workspace: pathlib.Path) -> str:
    return workspace.name


def run_checked(
    command: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HandoffError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def current_branch(workspace: pathlib.Path) -> str:
    return run_checked(
        ["git", "branch", "--show-current"],
        cwd=workspace,
    ).stdout.strip()


def worktree_is_clean(workspace: pathlib.Path) -> bool:
    return not run_checked(
        ["git", "status", "--short"],
        cwd=workspace,
    ).stdout.strip()


def branch_exists_on_origin(workspace: pathlib.Path, branch: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def find_existing_pr(workspace: pathlib.Path, branch: str) -> PullRequestRef | None:
    completed = run_checked(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--json",
            "number,url,isDraft",
        ],
        cwd=workspace,
    )
    payload = json.loads(completed.stdout)
    if not payload:
        return None
    first = payload[0]
    return PullRequestRef(
        number=int(first["number"]),
        url=str(first["url"]),
        is_draft=bool(first["isDraft"]),
    )


def review_message_is_clean(message: str) -> bool:
    normalized = message.lower()
    if FINDING_MARKER_PATTERN.search(message):
        return False
    return any(marker in normalized for marker in NO_FINDINGS_MARKERS)


def review_message_has_findings(message: str) -> bool:
    if not message.strip():
        return False
    if review_message_is_clean(message):
        return False
    return True


def artifact_dir_for_branch(workspace: pathlib.Path, branch: str) -> pathlib.Path:
    return (
        workspace
        / ".codex"
        / "review_artifacts"
        / review_loop.sanitize_path_component(branch)
    )


def load_manifest_message(workspace: pathlib.Path, branch: str) -> tuple[dict[str, Any], str]:
    target_dir = artifact_dir_for_branch(workspace, branch)
    manifest_path = target_dir / "latest.json"
    if not manifest_path.exists():
        raise HandoffError(f"review manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    message_rel = manifest.get("message_path")
    message_path = workspace / str(message_rel)
    message = ""
    if message_path.exists():
        message = message_path.read_text(encoding="utf-8").strip()
    if not message:
        jsonl_rel = manifest.get("jsonl_path")
        jsonl_path = workspace / str(jsonl_rel)
        if jsonl_path.exists():
            message = review_loop.extract_last_agent_message(
                jsonl_path.read_text(encoding="utf-8")
            ).strip()
    return manifest, message


def run_host_review(workspace: pathlib.Path, issue_identifier: str) -> ReviewResult:
    env = os.environ.copy()
    env["GPU_CFD_REVIEW_REPO_ROOT"] = workspace.as_posix()
    command = [
        sys.executable,
        str(control_repo_root() / "scripts" / "symphony" / "review_loop.py"),
        "codex-review",
        "--issue",
        issue_identifier,
        "--base",
        "origin/main",
    ]
    completed = subprocess.run(
        command,
        cwd=control_repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    branch = current_branch(workspace)
    manifest, message = load_manifest_message(workspace, branch)
    if completed.returncode != 0 and not message:
        return ReviewResult(status="unavailable", message="", manifest=manifest)
    if review_message_has_findings(message):
        return ReviewResult(status="findings", message=message, manifest=manifest)
    if review_message_is_clean(message):
        return ReviewResult(status="clean", message=message, manifest=manifest)
    return ReviewResult(status="unavailable", message=message, manifest=manifest)


def build_pr_title(issue_identifier: str, issue_title: str) -> str:
    return f"[codex] {issue_identifier} {issue_title}"


def build_pr_body(issue_identifier: str, issue_title: str) -> str:
    return "\n".join(
        [
            "## Summary",
            f"- Implement {issue_identifier}: {issue_title}",
            "",
            f"Closes {issue_identifier}",
        ]
    )


def ensure_pr(
    workspace: pathlib.Path,
    issue_identifier: str,
    issue_title: str,
    branch: str,
) -> PullRequestRef:
    existing = find_existing_pr(workspace, branch)
    title = build_pr_title(issue_identifier, issue_title)
    body = build_pr_body(issue_identifier, issue_title)
    if existing is not None:
        run_checked(
            [
                "gh",
                "pr",
                "edit",
                str(existing.number),
                "--title",
                title,
                "--body",
                body,
            ],
            cwd=workspace,
        )
        if existing.is_draft:
            run_checked(
                ["gh", "pr", "ready", str(existing.number)],
                cwd=workspace,
            )
        return PullRequestRef(existing.number, existing.url, False)
    completed = run_checked(
        [
            "gh",
            "pr",
            "create",
            "--base",
            "main",
            "--head",
            branch,
            "--title",
            title,
            "--body",
            body,
        ],
        cwd=workspace,
    )
    url = completed.stdout.strip().splitlines()[-1]
    if not url:
        raise HandoffError("gh pr create did not return a PR URL")
    pr = find_existing_pr(workspace, branch)
    if pr is None:
        raise HandoffError("failed to resolve PR after creation")
    return PullRequestRef(pr.number, url, False)


def emit_handoff_telemetry(
    *,
    event_type: str,
    message: str,
    issue: str,
    details: dict[str, Any],
    workspace: pathlib.Path,
    pr: int | None = None,
    state: str | None = None,
) -> None:
    event = telemetry.build_event(
        event_type=event_type,
        message=message,
        issue=issue,
        pr=pr,
        state=state,
        details=details,
        cwd=workspace,
        repo_root=workspace,
    )
    telemetry.write_event(telemetry.default_telemetry_root(), event)


def result_payload(
    *,
    issue: str,
    branch: str,
    review_result: ReviewResult,
    pr: PullRequestRef | None = None,
) -> dict[str, Any]:
    manifest = review_result.manifest or {}
    payload: dict[str, Any] = {
        "issue": issue,
        "branch": branch,
        "status": review_result.status,
        "message_path": manifest.get("message_path"),
        "jsonl_path": manifest.get("jsonl_path"),
        "stderr_path": manifest.get("stderr_path"),
    }
    if pr is not None:
        payload["pr"] = {"number": pr.number, "url": pr.url, "is_draft": pr.is_draft}
    return payload


def main() -> int:
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    issue_identifier = workspace_issue_identifier(workspace)
    issue = github_linear_bridge.fetch_linear_issue(issue_identifier)
    issue_title = issue.get("title") or issue_identifier

    branch = current_branch(workspace)
    if branch in {"", "main"}:
        raise HandoffError("refusing PR handoff from the default branch")
    if not worktree_is_clean(workspace):
        raise HandoffError("workspace is dirty; commit or stash before PR handoff")
    if not branch_exists_on_origin(workspace, branch):
        raise HandoffError("branch is not pushed to origin; push before PR handoff")

    review_result = run_host_review(workspace, issue_identifier)
    emit_handoff_telemetry(
        event_type="host_local_review_completed",
        message=f"Host-side local Codex review completed with status={review_result.status}",
        issue=issue_identifier,
        details={
            "branch": branch,
            "status": review_result.status,
            "manifest": json.dumps(review_result.manifest or {}, sort_keys=True),
        },
        workspace=workspace,
    )

    if review_result.status == "findings":
        print(
            json.dumps(
                result_payload(
                    issue=issue_identifier,
                    branch=branch,
                    review_result=review_result,
                ),
                sort_keys=True,
            )
        )
        return 0

    if review_result.status != "clean":
        emit_handoff_telemetry(
            event_type="host_local_review_blocked",
            message="Host-side local Codex review did not produce a usable result",
            issue=issue_identifier,
            details={"branch": branch},
            workspace=workspace,
        )
        raise HandoffError("host-side local Codex review did not produce a usable result")

    pr = ensure_pr(workspace, issue_identifier, issue_title, branch)
    emit_handoff_telemetry(
        event_type="pr_opened",
        message="Opened or updated PR after a clean local Codex review loop",
        issue=issue_identifier,
        pr=pr.number,
        state=IN_REVIEW_STATE,
        details={"branch": branch, "url": pr.url},
        workspace=workspace,
    )
    github_linear_bridge.update_linear_issue_state(issue_identifier, IN_REVIEW_STATE)
    emit_handoff_telemetry(
        event_type="review_requested",
        message=f"Local Codex review is clean and PR #{pr.number} is ready for external review",
        issue=issue_identifier,
        pr=pr.number,
        state=IN_REVIEW_STATE,
        details={
            "branch": branch,
            "url": pr.url,
            "review_artifact": review_result.manifest.get("jsonl_path")
            if review_result.manifest
            else None,
        },
        workspace=workspace,
    )
    print(
        json.dumps(
            result_payload(
                issue=issue_identifier,
                branch=branch,
                review_result=ReviewResult(
                    status="in_review",
                    message=review_result.message,
                    manifest=review_result.manifest,
                ),
                pr=pr,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HandoffError, github_linear_bridge.review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
