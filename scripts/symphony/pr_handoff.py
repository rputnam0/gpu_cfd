#!/usr/bin/env python3
"""Worker-owned Symphony PR handoff for local Codex review and PR automation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

try:
    from scripts.symphony import devin_review_gate, linear_api, review_loop, trace
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import devin_review_gate, linear_api, review_loop, trace


IN_REVIEW_STATE = "In Review"
REWORK_STATE = "Rework"
LOCAL_REMEDIATION_STATE = "In Progress"
LOCAL_REVIEW_PARKED_STATE = "Ready to Merge"
LOCAL_REVIEW_REMEDIATION_PASSES = 2
FINAL_LOCAL_REVIEW_PASS = LOCAL_REVIEW_REMEDIATION_PASSES + 1
NO_FINDINGS_MARKERS = (
    "no material findings remain",
    "no material issues remain",
    "no material findings were identified",
    "no findings",
)
FINDING_MARKER_PATTERN = re.compile(r"\[P[0-3]\]")
FINDING_LINE_PATTERN = re.compile(r"^- \[(?P<priority>P[0-3])\]\s+(?P<summary>.+?)\s*$")
FINDING_LOCATION_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$"
)


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


@dataclass(frozen=True)
class ResidualFinding:
    priority_label: str
    title: str
    body: str
    source_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None


class HandoffError(RuntimeError):
    """Raised when the PR handoff cannot complete."""


def local_review_state_path(workspace: pathlib.Path) -> pathlib.Path:
    return workspace / ".codex" / "symphony" / "local_review_state.json"


def _load_local_review_state(workspace: pathlib.Path) -> dict[str, Any]:
    path = local_review_state_path(workspace)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def record_local_review_round(
    workspace: pathlib.Path,
    branch: str,
) -> dict[str, Any]:
    path = local_review_state_path(workspace)
    existing = _load_local_review_state(workspace)
    previous_branch = str(existing.get("branch") or "")
    previous_count = int(existing.get("count") or 0) if previous_branch == branch else 0
    payload = {
        "branch": branch,
        "count": previous_count + 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def reset_local_review_rounds(workspace: pathlib.Path, branch: str) -> None:
    path = local_review_state_path(workspace)
    if not path.exists():
        return
    existing = _load_local_review_state(workspace)
    existing_branch = str(existing.get("branch") or "")
    if existing_branch not in {"", branch}:
        return
    path.unlink(missing_ok=True)


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


def run_checked(command: list[str], *, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
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
    return run_checked(["git", "branch", "--show-current"], cwd=workspace).stdout.strip()


def current_commit(workspace: pathlib.Path) -> str:
    return run_checked(["git", "rev-parse", "HEAD"], cwd=workspace).stdout.strip()


def worktree_is_clean(workspace: pathlib.Path) -> bool:
    return not run_checked(["git", "status", "--short"], cwd=workspace).stdout.strip()


def origin_remote_url(workspace: pathlib.Path) -> str:
    return run_checked(["git", "remote", "get-url", "origin"], cwd=workspace).stdout.strip()


def branch_exists_on_origin(workspace: pathlib.Path, branch: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", "origin", branch],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def ensure_branch_pushed(workspace: pathlib.Path, branch: str) -> None:
    if branch_exists_on_origin(workspace, branch):
        run_checked(["git", "push", "origin", branch], cwd=workspace)
    else:
        run_checked(["git", "push", "-u", "origin", branch], cwd=workspace)


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


def parse_finding_location(raw_value: str | None) -> tuple[str | None, int | None, int | None]:
    if not raw_value:
        return None, None, None
    match = FINDING_LOCATION_PATTERN.fullmatch(raw_value.strip())
    if match is None:
        return None, None, None
    start_line = int(match.group("start"))
    end_line = int(match.group("end")) if match.group("end") else start_line
    return match.group("path"), start_line, end_line


def parse_review_findings(message: str) -> list[ResidualFinding]:
    findings: list[ResidualFinding] = []
    current_priority: str | None = None
    current_title: str | None = None
    current_path: str | None = None
    current_start_line: int | None = None
    current_end_line: int | None = None
    body_lines: list[str] = []

    def finalize_current() -> None:
        nonlocal current_priority, current_title, current_path, current_start_line, current_end_line, body_lines
        if current_priority is None or current_title is None:
            return
        body = "\n".join(line.rstrip() for line in body_lines).strip() or current_title
        findings.append(
            ResidualFinding(
                priority_label=current_priority,
                title=current_title,
                body=body,
                source_path=current_path,
                start_line=current_start_line,
                end_line=current_end_line,
            )
        )
        current_priority = None
        current_title = None
        current_path = None
        current_start_line = None
        current_end_line = None
        body_lines = []

    for raw_line in message.splitlines():
        line = raw_line.rstrip()
        match = FINDING_LINE_PATTERN.match(line.strip())
        if match is None:
            if current_priority is not None:
                body_lines.append(line)
            continue

        finalize_current()
        current_priority = match.group("priority")
        summary = match.group("summary").strip()
        current_title = summary
        current_path = None
        current_start_line = None
        current_end_line = None
        for separator in (" — ", " - "):
            if separator not in summary:
                continue
            possible_title, possible_location = summary.rsplit(separator, 1)
            source_path, start_line, end_line = parse_finding_location(possible_location)
            if source_path is None:
                continue
            current_title = possible_title.strip()
            current_path = source_path
            current_start_line = start_line
            current_end_line = end_line
            break

    finalize_current()
    return findings


def local_review_phase(round_count: int) -> str:
    if round_count >= FINAL_LOCAL_REVIEW_PASS:
        return "final_local_review_pass"
    return f"local_review_remediation_pass_{round_count}"


def residual_followup_priority(priority_label: str) -> tuple[int, list[str]]:
    normalized = priority_label.strip().upper()
    if normalized == "P0":
        return 1, ["Bug"]
    if normalized in {"P1", "P2"}:
        return 2, ["Bug"]
    return 3, ["Improvement"]


def review_artifact_path(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return ".codex/review_artifacts/"
    for key in ("message_path", "jsonl_path"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    return ".codex/review_artifacts/"


def ensure_residual_followup_issues(
    *,
    issue_identifier: str,
    issue_title: str,
    findings: list[ResidualFinding],
    review_message: str,
    pr_url: str,
    branch: str,
    commit_sha: str,
    artifact_path: str,
) -> list[dict[str, Any]]:
    if findings:
        created: list[dict[str, Any]] = []
        for finding in findings:
            priority, label_names = residual_followup_priority(finding.priority_label)
            issue = linear_api.ensure_residual_followup_issue(
                parent_identifier=issue_identifier,
                parent_title=issue_title,
                finding_title=finding.title,
                finding_body=finding.body,
                priority=priority,
                priority_label=finding.priority_label,
                pr_url=pr_url,
                branch=branch,
                commit=commit_sha,
                artifact_path=artifact_path,
                source_path=finding.source_path,
                start_line=finding.start_line,
                end_line=finding.end_line,
                label_names=label_names,
            )
            created.append(
                {
                    "action": issue.get("action"),
                    "identifier": issue.get("identifier"),
                    "url": issue.get("url"),
                    "priority_label": finding.priority_label,
                    "title": finding.title,
                    "source_path": finding.source_path,
                    "start_line": finding.start_line,
                    "end_line": finding.end_line,
                }
            )
        return created

    fallback = linear_api.ensure_residual_followup_issue(
        parent_identifier=issue_identifier,
        parent_title=issue_title,
        finding_title="Residual local-review summary",
        finding_body=review_message.strip() or "Residual local-review findings remained after the final local-review pass.",
        priority=2,
        priority_label="P2",
        pr_url=pr_url,
        branch=branch,
        commit=commit_sha,
        artifact_path=artifact_path,
        label_names=["Bug"],
    )
    return [
        {
            "action": fallback.get("action"),
            "identifier": fallback.get("identifier"),
            "url": fallback.get("url"),
            "priority_label": "P2",
            "title": "Residual local-review summary",
            "source_path": None,
            "start_line": None,
            "end_line": None,
        }
    ]


def resolve_actionable_devin_threads(
    pr_number: int,
    reviewers: list[str] | None = None,
) -> list[str]:
    summary = review_loop.fetch_pr_summary(
        None,
        pr_number,
        reviewers or list(devin_review_gate.DEFAULT_REVIEWERS),
    )
    resolved_thread_ids: list[str] = []
    for thread in summary.actionable_threads:
        thread_id = thread.get("id")
        if not thread_id:
            continue
        devin_review_gate.resolve_review_thread(str(thread_id))
        resolved_thread_ids.append(str(thread_id))
    return resolved_thread_ids


def artifact_dir_for_branch(workspace: pathlib.Path, branch: str) -> pathlib.Path:
    return (
        workspace
        / ".codex"
        / "review_artifacts"
        / review_loop.sanitize_path_component(branch)
    )


def persist_review_artifacts(
    source_workspace: pathlib.Path,
    destination_workspace: pathlib.Path,
    branch: str,
) -> str | None:
    if source_workspace == destination_workspace:
        return None
    source_dir = artifact_dir_for_branch(source_workspace, branch)
    if not source_dir.exists():
        return None
    destination_dir = artifact_dir_for_branch(destination_workspace, branch)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
    return (
        "Copied the latest local review artifacts from the clean review clone into "
        f"`{destination_dir.relative_to(destination_workspace).as_posix()}`."
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


@contextlib.contextmanager
def prepared_review_workspace(
    workspace: pathlib.Path,
) -> Any:
    if worktree_is_clean(workspace):
        yield workspace, None
        return

    branch = current_branch(workspace)
    commit = current_commit(workspace)
    origin_url = origin_remote_url(workspace)
    with tempfile.TemporaryDirectory(prefix=f"{workspace.name.lower()}clone_") as temp_dir:
        clone_root = pathlib.Path(temp_dir) / workspace.name
        run_checked(
            [
                "git",
                "clone",
                "--no-local",
                "--branch",
                branch,
                workspace.as_posix(),
                clone_root.as_posix(),
            ],
            cwd=control_repo_root(),
        )
        run_checked(["git", "remote", "set-url", "origin", origin_url], cwd=clone_root)
        if current_commit(clone_root) != commit:
            raise HandoffError(
                "clean review clone does not match the workspace HEAD commit"
            )
        yield clone_root, (
            f"Used clean committed clone at {clone_root} for local review and PR handoff "
            "because the issue workspace had unrelated dirty changes."
        )


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
    if completed.returncode != 0:
        if review_message_has_findings(message):
            return ReviewResult(status="findings", message=message, manifest=manifest)
        return ReviewResult(status="unavailable", message=message, manifest=manifest)
    if review_message_has_findings(message):
        return ReviewResult(status="findings", message=message, manifest=manifest)
    if review_message_is_clean(message):
        return ReviewResult(
            status="local_review_complete", message=message, manifest=manifest
        )
    return ReviewResult(status="unavailable", message=message, manifest=manifest)


def build_pr_title(issue_identifier: str, issue_title: str) -> str:
    return f"[codex] {issue_identifier} {issue_title}"


def build_pr_body(issue_identifier: str, issue_title: str) -> str:
    return "\n".join(
        [
            linear_api.render_issue_link_marker(issue_identifier),
            "",
            "## Summary",
            f"- Implement {issue_identifier}: {issue_title}",
            "",
            f"Closes {issue_identifier}",
        ]
    )


def sync_workpad(
    *,
    issue_identifier: str,
    issue_title: str,
    current_status: str,
    task_summary: list[str] | None = None,
    execution_plan: list[str] | None = None,
    scoped_sources: list[str] | None = None,
    decisions_and_rationale: list[str] | None = None,
    validation: list[str] | None = None,
    risks_blockers: list[str] | None = None,
    review_handoff_notes: list[str] | None = None,
) -> dict[str, Any]:
    existing_comment = linear_api.find_workpad_comment(issue_identifier)
    merged_body = linear_api.merge_workpad_body(
        existing_comment.get("body") if existing_comment else None,
        issue_identifier=issue_identifier,
        issue_title=issue_title,
        current_status=current_status,
        task_summary=task_summary,
        execution_plan=execution_plan,
        scoped_sources=scoped_sources,
        decisions_and_rationale=decisions_and_rationale,
        validation=validation,
        risks_blockers=risks_blockers,
        review_handoff_notes=review_handoff_notes,
    )
    return linear_api.upsert_workpad_comment(issue_identifier, merged_body)


def sync_workpad_best_effort(**kwargs: Any) -> str | None:
    try:
        sync_workpad(**kwargs)
    except Exception as exc:
        return f"workpad sync failed: {exc}"
    return None


def park_issue_for_local_review(
    *,
    issue_identifier: str,
    issue_title: str,
    current_status: str,
    validation: list[str],
    review_handoff_notes: list[str],
    risks_blockers: list[str] | None = None,
) -> dict[str, Any]:
    linear_update = linear_api.update_issue_state(
        issue_identifier,
        LOCAL_REVIEW_PARKED_STATE,
    )
    workpad_warning = sync_workpad_best_effort(
        issue_identifier=issue_identifier,
        issue_title=issue_title,
        current_status=current_status,
        validation=validation,
        risks_blockers=risks_blockers,
        review_handoff_notes=review_handoff_notes,
    )
    return {
        "linear_update": {
            "previous_state": linear_update["previous_state"],
            "current_state": linear_update["current_state"],
            "changed": linear_update["changed"],
        },
        "workpad_warning": workpad_warning,
    }


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
            run_checked(["gh", "pr", "ready", str(existing.number)], cwd=workspace)
        return PullRequestRef(existing.number, existing.url, False)

    completed = run_checked(
        [
            "gh",
            "pr",
            "create",
            "--title",
            title,
            "--body",
            body,
            "--base",
            "main",
            "--head",
            branch,
        ],
        cwd=workspace,
    )
    pr_url = completed.stdout.strip().splitlines()[-1].strip()
    view_completed = run_checked(
        ["gh", "pr", "view", pr_url, "--json", "number,url,isDraft"],
        cwd=workspace,
    )
    payload = json.loads(view_completed.stdout)
    return PullRequestRef(
        number=int(payload["number"]),
        url=str(payload["url"]),
        is_draft=bool(payload["isDraft"]),
    )


def enable_auto_merge(workspace: pathlib.Path, pr_number: int) -> None:
    run_checked(
        ["gh", "pr", "merge", str(pr_number), "--auto", "--merge"],
        cwd=workspace,
    )


def main() -> int:
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    issue_identifier = workspace_issue_identifier(workspace)
    issue = linear_api.fetch_issue(issue_identifier)
    issue_title = issue.get("title") or issue_identifier
    issue_state = str(issue.get("state", {}).get("name") or "")
    workspace_branch = current_branch(workspace)

    if issue_state == IN_REVIEW_STATE:
        existing_pr = find_existing_pr(workspace, workspace_branch)
        result: dict[str, Any] = {
            "issue_identifier": issue_identifier,
            "issue_title": issue_title,
            "branch": workspace_branch,
            "handoff_status": "already_in_review",
            "review_cycle_phase": "in_review_terminal",
            "stop_worker": True,
            "continue_same_worker": False,
            "review": {
                "status": "skipped",
                "message": "The issue is already in In Review; do not rerun local review or PR handoff outside Rework.",
                "manifest": None,
            },
        }
        if existing_pr is not None:
            result["pull_request"] = {
                "number": existing_pr.number,
                "url": existing_pr.url,
                "auto_merge_enabled": True,
            }
        workpad_warning = sync_workpad_best_effort(
            issue_identifier=issue_identifier,
            issue_title=issue_title,
            current_status="in_review",
            validation=[
                "The branch is already in the GitHub / Devin review stage.",
            ],
            review_handoff_notes=[
                "The implementation run reached `In Review`; stop this worker instead of rerunning local review.",
            ],
        )
        if workpad_warning:
            result["workpad_sync_warning"] = workpad_warning
            print(workpad_warning, file=sys.stderr)
        print(json.dumps(result, indent=2))
        return 0

    is_rework_run = issue_state == REWORK_STATE

    with prepared_review_workspace(workspace) as (
        review_workspace,
        review_workspace_note,
    ):
        review_result = run_host_review(review_workspace, issue_identifier)
        branch = current_branch(review_workspace)
        artifact_copy_note = persist_review_artifacts(
            review_workspace,
            workspace,
            branch,
        )
        result: dict[str, Any] = {
            "issue_identifier": issue_identifier,
            "issue_title": issue_title,
            "branch": branch,
            "review": {
                "status": review_result.status,
                "message": review_result.message,
                "manifest": review_result.manifest,
            },
        }
        if review_workspace_note:
            result["review_workspace_note"] = review_workspace_note
        if artifact_copy_note:
            result["review_artifact_note"] = artifact_copy_note

        if review_result.status == "findings":
            round_info = record_local_review_round(workspace, branch)
            round_count = int(round_info["count"])
            phase = local_review_phase(round_count)
            if round_count < FINAL_LOCAL_REVIEW_PASS:
                review_handoff_notes = [
                    (
                        f"Local review round {round_count}/"
                        f"{FINAL_LOCAL_REVIEW_PASS} reported findings. "
                        "Stay in the same implementation run, inspect the latest "
                        "artifact under `.codex/review_artifacts/`, fix the valid "
                        "findings, rerun targeted validation, and rerun the handoff helper."
                    ),
                    (
                        "The issue remains in `In Progress` so the same implementation "
                        "worker owns local-review remediation on this branch."
                    ),
                    f"Review cycle phase: {phase}",
                    "Stop worker: false",
                ]
                if review_workspace_note:
                    review_handoff_notes.append(review_workspace_note)
                if artifact_copy_note:
                    review_handoff_notes.append(artifact_copy_note)
                linear_update = linear_api.update_issue_state(
                    issue_identifier,
                    LOCAL_REMEDIATION_STATE,
                )
                result["handoff_status"] = "local_review_findings"
                result["next_action"] = (
                    "inspect_latest_local_review_artifact_and_continue_same_run"
                )
                result["continue_same_worker"] = True
                result["stop_worker"] = False
                result["review_cycle_phase"] = phase
                result["local_review_round"] = round_count
                result["local_review_round_cap"] = FINAL_LOCAL_REVIEW_PASS
                result["linear_update"] = {
                    "previous_state": linear_update["previous_state"],
                    "current_state": linear_update["current_state"],
                    "changed": linear_update["changed"],
                }
                workpad_warning = sync_workpad_best_effort(
                    issue_identifier=issue_identifier,
                    issue_title=issue_title,
                    current_status="local_review_findings",
                    validation=[
                        (
                            f"Local Codex review round {round_count}/"
                            f"{FINAL_LOCAL_REVIEW_PASS} reported findings; "
                            "the same implementation worker must continue remediation "
                            "from the latest artifact before rerunning handoff."
                        ),
                    ],
                    review_handoff_notes=review_handoff_notes,
                )
                if workpad_warning:
                    result["workpad_sync_warning"] = workpad_warning
                    print(workpad_warning, file=sys.stderr)
                if trace.is_enabled():
                    run_manifest = trace.ensure_run(
                        issue_id=issue_identifier,
                        run_kind="implementation",
                        branch=branch,
                    )
                    artifact = trace.capture_json_artifact(
                        issue_id=issue_identifier,
                        run_id=run_manifest["run_id"],
                        artifact_type="pr_handoff_result",
                        label="PR Handoff Result",
                        payload=result,
                        filename="pr_handoff_result.json",
                    )
                    trace.capture_event(
                        issue_id=issue_identifier,
                        run_id=run_manifest["run_id"],
                        actor="Symphony",
                        stage="pr_handoff_findings",
                        summary="Local review findings remain; same worker should continue remediation",
                        decision="continue_same_worker",
                        decision_rationale=(
                            "Pre-PR local review findings are a continuation signal for "
                            "the active worker, not a terminal handoff state."
                        ),
                        artifact_refs=[artifact["artifact_id"]],
                        metadata={
                            "local_review_round": round_count,
                            "local_review_round_cap": FINAL_LOCAL_REVIEW_PASS,
                            "active_state": LOCAL_REMEDIATION_STATE,
                        },
                    )
                print(json.dumps(result, indent=2))
                return 0

            if branch in {"", "main"}:
                raise HandoffError("refusing PR handoff from the default branch")
            ensure_branch_pushed(review_workspace, branch)
            commit_sha = current_commit(review_workspace)
            pr_ref = ensure_pr(review_workspace, issue_identifier, issue_title, branch)
            enable_auto_merge(review_workspace, pr_ref.number)
            residual_followups = ensure_residual_followup_issues(
                issue_identifier=issue_identifier,
                issue_title=issue_title,
                findings=parse_review_findings(review_result.message),
                review_message=review_result.message,
                pr_url=pr_ref.url,
                branch=branch,
                commit_sha=commit_sha,
                artifact_path=review_artifact_path(review_result.manifest),
            )
            resolved_thread_ids = (
                resolve_actionable_devin_threads(pr_ref.number) if is_rework_run else []
            )
            linear_update = linear_api.update_issue_state(issue_identifier, IN_REVIEW_STATE)
            reset_local_review_rounds(workspace, branch)
            result["handoff_status"] = "local_review_cap_escalated"
            result["review_cycle_phase"] = phase
            result["stop_worker"] = True
            result["continue_same_worker"] = False
            result["local_review_round"] = round_count
            result["local_review_round_cap"] = FINAL_LOCAL_REVIEW_PASS
            result["residual_followups"] = residual_followups
            result["resolved_devin_threads"] = resolved_thread_ids
            result["pull_request"] = {
                "number": pr_ref.number,
                "url": pr_ref.url,
                "auto_merge_enabled": True,
            }
            result["linear_update"] = {
                "previous_state": linear_update["previous_state"],
                "current_state": linear_update["current_state"],
                "changed": linear_update["changed"],
            }
            residual_followup_notes = [
                (
                    "Residual local-review findings were parked as child Backlog follow-up "
                    "issues so the PR could progress to Devin review."
                )
            ]
            for followup in residual_followups:
                identifier = str(followup.get("identifier") or "").strip()
                url = str(followup.get("url") or "").strip()
                title = str(followup.get("title") or "").strip()
                if identifier and url:
                    residual_followup_notes.append(
                        f"Residual follow-up: {identifier} ({title}) - {url}"
                    )
            review_handoff_notes = [
                (
                    f"Final local review pass ({FINAL_LOCAL_REVIEW_PASS}/"
                    f"{FINAL_LOCAL_REVIEW_PASS}) reported residual findings. The PR was "
                    "opened or updated, auto-merge was enabled, and the branch progressed "
                    "to Devin review without another local remediation loop."
                ),
                f"PR opened or updated: {pr_ref.url}",
                (
                    f"Linear moved to {linear_update['current_state']} from "
                    f"{linear_update['previous_state']}."
                ),
                f"Review cycle phase: {phase}",
                "Stop worker: true",
                *residual_followup_notes,
            ]
            if resolved_thread_ids:
                review_handoff_notes.append(
                    "Resolved actionable Devin review threads for this Rework pass: "
                    + ", ".join(resolved_thread_ids)
                )
            if review_workspace_note:
                review_handoff_notes.append(review_workspace_note)
            if artifact_copy_note:
                review_handoff_notes.append(artifact_copy_note)
            workpad_warning = sync_workpad_best_effort(
                issue_identifier=issue_identifier,
                issue_title=issue_title,
                current_status="in_review",
                validation=[
                    (
                        f"Final local review pass ({FINAL_LOCAL_REVIEW_PASS}/"
                        f"{FINAL_LOCAL_REVIEW_PASS}) completed with residual findings; "
                        "the branch was escalated to GitHub / Devin review."
                    ),
                    "Latest local review findings remain available under `.codex/review_artifacts/`.",
                    "GitHub auto-merge enabled for the current pull request.",
                ],
                risks_blockers=[
                    "Residual local-review findings were deferred into Backlog follow-up issues.",
                ],
                review_handoff_notes=review_handoff_notes,
            )
            if workpad_warning:
                result["workpad_sync_warning"] = workpad_warning
                print(workpad_warning, file=sys.stderr)
            if trace.is_enabled():
                run_manifest = trace.ensure_run(
                    issue_id=issue_identifier,
                    run_kind="implementation",
                    branch=branch,
                    pr_number=pr_ref.number,
                )
                artifact = trace.capture_json_artifact(
                    issue_id=issue_identifier,
                    run_id=run_manifest["run_id"],
                    artifact_type="pr_handoff_result",
                    label="PR Handoff Result",
                    payload=result,
                    filename="pr_handoff_result.json",
                )
                trace.capture_event(
                    issue_id=issue_identifier,
                    run_id=run_manifest["run_id"],
                    actor="GitHub",
                    stage="pr_handoff_escalated",
                    summary="PR opened for external review after the final local review pass",
                    decision="in_review",
                    decision_rationale=(
                        "The final local review pass completed, residual findings were "
                        "parked as follow-up work, and the branch was escalated into the "
                        "external Devin review stage."
                    ),
                    artifact_refs=[artifact["artifact_id"]],
                    metadata={
                        "pr_url": pr_ref.url,
                        "pr_number": pr_ref.number,
                        "local_review_round_cap": FINAL_LOCAL_REVIEW_PASS,
                    },
                )
            print(json.dumps(result, indent=2))
            return 0
        if review_result.status != "local_review_complete":
            review_handoff_notes = [
                (
                    f"Issue parked in {LOCAL_REVIEW_PARKED_STATE} to prevent Symphony "
                    "redispatch until the local review blocker is resolved."
                ),
                "Review cycle phase: local_review_blocked",
                "Stop worker: true",
            ]
            if review_workspace_note:
                review_handoff_notes.append(review_workspace_note)
            parking_result = park_issue_for_local_review(
                issue_identifier=issue_identifier,
                issue_title=issue_title,
                current_status="local_review_blocked",
                validation=[
                    "Local Codex review did not produce a clean result.",
                ],
                risks_blockers=[
                    "Local review was unavailable or inconclusive. Resolve the review blocker before rerunning handoff.",
                ],
                review_handoff_notes=review_handoff_notes,
            )
            result["handoff_status"] = "local_review_unavailable"
            result["review_cycle_phase"] = "local_review_blocked"
            result["stop_worker"] = True
            result["continue_same_worker"] = False
            result["linear_update"] = parking_result["linear_update"]
            workpad_warning = parking_result["workpad_warning"]
            if workpad_warning:
                result["workpad_sync_warning"] = workpad_warning
                print(workpad_warning, file=sys.stderr)
            print(json.dumps(result, indent=2))
            return 1

        reset_local_review_rounds(workspace, branch)
        if branch in {"", "main"}:
            raise HandoffError("refusing PR handoff from the default branch")
        ensure_branch_pushed(review_workspace, branch)
        pr_ref = ensure_pr(review_workspace, issue_identifier, issue_title, branch)
        enable_auto_merge(review_workspace, pr_ref.number)
        resolved_thread_ids = (
            resolve_actionable_devin_threads(pr_ref.number) if is_rework_run else []
        )
        linear_update = linear_api.update_issue_state(issue_identifier, IN_REVIEW_STATE)

        result["branch"] = branch
        result["handoff_status"] = "local_review_complete"
        result["review_cycle_phase"] = "local_review_complete"
        result["stop_worker"] = True
        result["continue_same_worker"] = False
        result["residual_followups"] = []
        result["resolved_devin_threads"] = resolved_thread_ids
        result["pull_request"] = {
            "number": pr_ref.number,
            "url": pr_ref.url,
            "auto_merge_enabled": True,
        }
        result["linear_update"] = {
            "previous_state": linear_update["previous_state"],
            "current_state": linear_update["current_state"],
            "changed": linear_update["changed"],
        }
        review_handoff_notes = [
            f"PR opened or updated: {pr_ref.url}",
            f"Linear moved to {linear_update['current_state']} from {linear_update['previous_state']}.",
            "Review cycle phase: local_review_complete",
            "Stop worker: true",
        ]
        if resolved_thread_ids:
            review_handoff_notes.append(
                "Resolved actionable Devin review threads for this Rework pass: "
                + ", ".join(resolved_thread_ids)
            )
        if review_workspace_note:
            review_handoff_notes.append(review_workspace_note)
        if artifact_copy_note:
            review_handoff_notes.append(artifact_copy_note)
        workpad_warning = sync_workpad_best_effort(
            issue_identifier=issue_identifier,
            issue_title=issue_title,
            current_status="in_review",
            validation=[
                "Local Codex review completed with no material findings remaining.",
                "GitHub auto-merge enabled for the current pull request.",
            ],
            review_handoff_notes=review_handoff_notes,
        )
        if workpad_warning:
            result["workpad_sync_warning"] = workpad_warning
            print(workpad_warning, file=sys.stderr)
        if trace.is_enabled():
            run_manifest = trace.ensure_run(
                issue_id=issue_identifier,
                run_kind="implementation",
                branch=branch,
                pr_number=pr_ref.number,
            )
            artifact = trace.capture_json_artifact(
                issue_id=issue_identifier,
                run_id=run_manifest["run_id"],
                artifact_type="pr_handoff_result",
                label="PR Handoff Result",
                payload=result,
                filename="pr_handoff_result.json",
            )
            trace.capture_event(
                issue_id=issue_identifier,
                run_id=run_manifest["run_id"],
                actor="GitHub",
                stage="pr_handoff_complete",
                summary="PR opened or updated and Linear moved to In Review",
                decision="in_review",
                decision_rationale="Local review completed cleanly and GitHub auto-merge was enabled",
                artifact_refs=[artifact["artifact_id"]],
                metadata={"pr_url": pr_ref.url, "pr_number": pr_ref.number},
            )
        print(json.dumps(result, indent=2))
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (HandoffError, review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
