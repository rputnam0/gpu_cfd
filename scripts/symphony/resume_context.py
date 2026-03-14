#!/usr/bin/env python3
"""Generate a deterministic resume brief for a Symphony issue workspace."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

try:
    from scripts.symphony import review_loop, telemetry
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import review_loop, telemetry


NO_FINDINGS_MARKERS = (
    "no material findings remain",
    "no material issues remain",
    "no material findings were identified",
    "no findings",
)
FINDING_MARKER_PATTERN = re.compile(r"\[P[0-3]\]")


@dataclass(frozen=True)
class PullRequestSnapshot:
    number: int
    url: str
    title: str
    state: str
    is_draft: bool
    review_decision: str | None


@dataclass(frozen=True)
class ResumeSnapshot:
    generated_at: str
    issue: str
    workspace: str
    branch: str
    head_commit: str | None
    base_ref: str
    commits_ahead: list[str]
    changed_files: list[str]
    pull_request: PullRequestSnapshot | None
    review_status: str | None
    review_message: str | None
    review_artifact: str | None
    recent_events: list[dict[str, Any]]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=".",
        help="Issue workspace path. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output path. Defaults to .codex/symphony/resume_context.md in the workspace.",
    )
    return parser.parse_args()


def run_command(
    command: list[str], *, cwd: pathlib.Path, check: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def git_stdout(workspace: pathlib.Path, *args: str) -> str | None:
    completed = run_command(["git", *args], cwd=workspace)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def current_branch(workspace: pathlib.Path) -> str:
    return git_stdout(workspace, "branch", "--show-current") or "detached"


def head_commit(workspace: pathlib.Path) -> str | None:
    return git_stdout(workspace, "rev-parse", "HEAD")


def default_base_ref(workspace: pathlib.Path) -> str:
    resolved = git_stdout(
        workspace,
        "symbolic-ref",
        "--quiet",
        "--short",
        "refs/remotes/origin/HEAD",
    )
    if resolved and resolved.startswith("origin/"):
        return resolved
    return "origin/main"


def commits_ahead(workspace: pathlib.Path, base_ref: str, limit: int = 8) -> list[str]:
    completed = run_command(
        [
            "git",
            "log",
            f"{base_ref}..HEAD",
            f"--max-count={limit}",
            "--format=%h %s",
        ],
        cwd=workspace,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def changed_files(workspace: pathlib.Path, base_ref: str, limit: int = 20) -> list[str]:
    completed = run_command(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=workspace,
    )
    if completed.returncode != 0:
        return []
    files = [line for line in completed.stdout.splitlines() if line.strip()]
    return files[:limit]


def find_pull_request(workspace: pathlib.Path, branch: str) -> PullRequestSnapshot | None:
    completed = run_command(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--json",
            "number,url,title,state,isDraft,reviewDecision",
        ],
        cwd=workspace,
    )
    if completed.returncode != 0:
        return None
    payload = json.loads(completed.stdout or "[]")
    if not payload:
        return None
    first = payload[0]
    return PullRequestSnapshot(
        number=int(first["number"]),
        url=str(first["url"]),
        title=str(first.get("title") or ""),
        state=str(first.get("state") or "UNKNOWN"),
        is_draft=bool(first.get("isDraft")),
        review_decision=first.get("reviewDecision"),
    )


def latest_review_result(
    workspace: pathlib.Path, branch: str
) -> tuple[str | None, str | None, str | None]:
    target_dir = (
        workspace
        / ".codex"
        / "review_artifacts"
        / review_loop.sanitize_path_component(branch)
    )
    manifest_path = target_dir / "latest.json"
    if not manifest_path.exists():
        return None, None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    message = ""
    message_rel = manifest.get("message_path")
    if message_rel:
        message_path = workspace / str(message_rel)
        if message_path.exists():
            message = message_path.read_text(encoding="utf-8").strip()
    artifact_path = str(manifest.get("jsonl_path") or "")
    normalized = message.lower()
    is_clean = bool(message) and not FINDING_MARKER_PATTERN.search(message) and any(
        marker in normalized for marker in NO_FINDINGS_MARKERS
    )
    if message and not is_clean:
        status = "findings"
    elif is_clean:
        status = "clean"
    else:
        status = None
    return status, (message or None), (artifact_path or None)


def load_recent_events(issue: str, limit: int = 5) -> list[dict[str, Any]]:
    issue_log = telemetry.default_telemetry_root() / "issues" / f"{issue}.jsonl"
    if not issue_log.exists():
        return []
    lines = [line for line in issue_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    recent: list[dict[str, Any]] = []
    for raw in lines[-limit:]:
        try:
            recent.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return recent


def collect_snapshot(workspace: pathlib.Path) -> ResumeSnapshot:
    workspace = workspace.resolve()
    issue = workspace.name
    branch = current_branch(workspace)
    base_ref = default_base_ref(workspace)
    warnings: list[str] = []
    pr = find_pull_request(workspace, branch)
    review_status, review_message, review_artifact = latest_review_result(workspace, branch)
    if pr is None:
        warnings.append("No PR is currently attached to this branch.")
    if review_status is None:
        warnings.append("No local review artifact is available for this branch yet.")
    return ResumeSnapshot(
        generated_at=dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z"),
        issue=issue,
        workspace=workspace.as_posix(),
        branch=branch,
        head_commit=head_commit(workspace),
        base_ref=base_ref,
        commits_ahead=commits_ahead(workspace, base_ref),
        changed_files=changed_files(workspace, base_ref),
        pull_request=pr,
        review_status=review_status,
        review_message=review_message,
        review_artifact=review_artifact,
        recent_events=load_recent_events(issue),
        warnings=warnings,
    )


def render_resume_context(snapshot: ResumeSnapshot) -> str:
    lines = [
        "# Worker Resume Context",
        "",
        f"- Generated at: `{snapshot.generated_at}`",
        f"- Issue workspace: `{snapshot.issue}`",
        f"- Workspace path: `{snapshot.workspace}`",
        f"- Branch: `{snapshot.branch}`",
        f"- Head commit: `{snapshot.head_commit or 'unknown'}`",
        f"- Base ref: `{snapshot.base_ref}`",
    ]
    if snapshot.pull_request is not None:
        pr = snapshot.pull_request
        lines.extend(
            [
                f"- PR: `#{pr.number}` `{pr.title}`",
                f"- PR URL: {pr.url}",
                f"- PR state: `{pr.state}` draft=`{str(pr.is_draft).lower()}` reviewDecision=`{pr.review_decision or ''}`",
            ]
        )
    else:
        lines.append("- PR: none attached to this branch")

    lines.extend(["", "## Commits Ahead of Base"])
    if snapshot.commits_ahead:
        lines.extend(f"- {commit}" for commit in snapshot.commits_ahead)
    else:
        lines.append("- No commits ahead of the base ref.")

    lines.extend(["", "## Changed Files"])
    if snapshot.changed_files:
        lines.extend(f"- `{path}`" for path in snapshot.changed_files)
    else:
        lines.append("- No changed files relative to the base ref.")

    lines.extend(["", "## Latest Local Review"])
    if snapshot.review_status is None:
        lines.append("- No local review artifact is available.")
    else:
        lines.append(f"- Status: `{snapshot.review_status}`")
        if snapshot.review_artifact:
            lines.append(f"- Artifact: `{snapshot.review_artifact}`")
        if snapshot.review_message:
            lines.extend(
                [
                    "",
                    "```text",
                    snapshot.review_message[:4000],
                    "```",
                ]
            )

    lines.extend(["", "## Recent Telemetry"])
    if snapshot.recent_events:
        for event in snapshot.recent_events:
            timestamp = event.get("timestamp", "")
            event_type = event.get("event_type", "")
            message = event.get("message", "")
            lines.append(f"- `{timestamp}` `{event_type}` {message}".rstrip())
    else:
        lines.append("- No recent telemetry events recorded for this issue.")

    if snapshot.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in snapshot.warnings)

    lines.extend(
        [
            "",
            "## Use On Resume",
            "- Read this file before new edits on `Rework` or `Ready to Merge` runs.",
            "- Treat it as the continuity brief for this issue branch and workspace.",
        ]
    )
    return "\n".join(lines) + "\n"


def output_path(workspace: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve()
    return workspace / ".codex" / "symphony" / "resume_context.md"


def write_resume_context(path: pathlib.Path, content: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    snapshot = collect_snapshot(workspace)
    content = render_resume_context(snapshot)
    path = write_resume_context(output_path(workspace, args.output), content)
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
