#!/usr/bin/env python3
"""Trace-aware wrapper for Symphony's Codex implementation worker."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
from typing import Any

try:
    from scripts.symphony import linear_api, runtime_config, trace
    from scripts.authority import load_authority_bundle, validate_source_audit_note
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import linear_api, runtime_config, trace
    from scripts.authority import load_authority_bundle, validate_source_audit_note


TASK_CARD_ID_PATTERN = re.compile(r"\b(?:FND-\d+|P\d+-\d+)\b", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?P<path>[^)]+)\)")
INLINE_REPO_PATH_PATTERN = re.compile(
    r"`(?P<path>(?:AGENTS|WORKFLOW)\.md|(?:docs|scripts|\.codex)/[^`]+)`"
)
ISSUE_SNAPSHOT_PATH_ENV = "GPU_CFD_TRACE_ISSUE_SNAPSHOT_PATH"
WORKPAD_SNAPSHOT_PATH_ENV = "GPU_CFD_TRACE_WORKPAD_SNAPSHOT_PATH"
SOURCE_AUDIT_EXEMPT_TASK_IDS = {"FND-07", "P5-01", "P7-01"}
SOURCE_AUDIT_PHASE_REQUIREMENTS = {
    "P5": {
        "note_filename": "phase5_symbol_reconciliation.md",
        "required_surfaces": [
            "alphaPredictor",
            "pressureCorrector",
            "interfaceProperties",
            "momentum stage",
        ],
    },
    "P7": {
        "note_filename": "phase7_source_audit.md",
        "required_surfaces": [
            "alphaPredictor",
            "pressureCorrector",
            "interfaceProperties",
            "momentum stage",
            "Nozzle BC runtime selection",
            "Profiling/instrumentation touch points",
        ],
    },
}


class DispatchError(RuntimeError):
    """Raised when the implementation worker cannot be launched safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "codex_args",
        nargs="*",
        help="Arguments forwarded to the implementation Codex profile. Defaults to app-server.",
    )
    return parser.parse_args()


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def workspace_root() -> pathlib.Path:
    return pathlib.Path.cwd().resolve()


def issue_identifier_from_workspace(workspace: pathlib.Path) -> str:
    return workspace.name


def current_branch(root: pathlib.Path) -> str | None:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    branch = completed.stdout.strip()
    return branch or None


def current_commit(root: pathlib.Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit or None


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_labels(labels_payload: Any) -> list[str]:
    if isinstance(labels_payload, dict):
        nodes = labels_payload.get("nodes", [])
        return [
            str(node.get("name")).strip()
            for node in nodes
            if isinstance(node, dict) and str(node.get("name", "")).strip()
        ]
    if isinstance(labels_payload, list):
        return [str(item).strip() for item in labels_payload if str(item).strip()]
    if isinstance(labels_payload, str) and labels_payload.strip():
        return [labels_payload.strip()]
    return []


def issue_state_name(issue_payload: dict[str, Any]) -> str:
    state = issue_payload.get("state")
    if isinstance(state, dict):
        return str(state.get("name") or "")
    return str(state or "")


def strip_front_matter(document_text: str) -> str:
    if not document_text.startswith("---\n"):
        return document_text
    closing_index = document_text.find("\n---\n", 4)
    if closing_index == -1:
        return document_text
    return document_text[closing_index + len("\n---\n") :].lstrip("\n")


def render_workflow_prompt(
    workflow_text: str,
    issue_payload: dict[str, Any],
    *,
    attempt: str | None = None,
) -> str:
    workflow_prompt_text = strip_front_matter(workflow_text)
    labels = ", ".join(normalize_labels(issue_payload.get("labels"))) or "none"
    replacements = {
        "{{ issue.identifier }}": str(issue_payload.get("identifier") or ""),
        "{{ issue.title }}": str(issue_payload.get("title") or ""),
        "{{ issue.state }}": issue_state_name(issue_payload),
        "{{ issue.labels }}": labels,
        "{{ issue.url }}": str(issue_payload.get("url") or ""),
        "{{ attempt }}": attempt or "first run",
    }
    rendered = workflow_prompt_text
    rendered = rendered.replace(
        "- Attempt: `{% if attempt %}{{ attempt }}{% else %}first run{% endif %}`",
        f"- Attempt: `{attempt or 'first run'}`",
    )
    description = str(issue_payload.get("description") or "").strip()
    description_block = (
        "Description:\n"
        "{% if issue.description %}\n"
        "{{ issue.description }}\n"
        "{% else %}\n"
        "No description provided.\n"
        "{% endif %}"
    )
    rendered = rendered.replace(
        description_block,
        f"Description:\n{description or 'No description provided.'}",
    )
    for raw, value in replacements.items():
        rendered = rendered.replace(raw, value)
    return rendered


def extract_task_card_candidates(*texts: str) -> list[str]:
    candidates: list[str] = []
    for text in texts:
        for match in TASK_CARD_ID_PATTERN.finditer(text or ""):
            candidate = match.group(0).upper()
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def parse_pr_inventory(repo: pathlib.Path) -> dict[str, pathlib.Path]:
    inventory_text = read_text(repo / "docs" / "tasks" / "pr_inventory.md")
    mapping: dict[str, pathlib.Path] = {}
    current_owner: pathlib.Path | None = None
    for raw_line in inventory_text.splitlines():
        owner_match = re.search(r"File owner: \[([^\]]+)\]", raw_line)
        if owner_match:
            current_owner = repo / "docs" / "tasks" / owner_match.group(1)
            continue
        pr_match = re.search(r"- `([A-Z0-9-]+)` ", raw_line)
        if pr_match and current_owner is not None:
            mapping[pr_match.group(1).upper()] = current_owner
    return mapping


def extract_card_markdown(task_text: str, pr_id: str) -> str:
    lines = task_text.splitlines()
    target_header = f"## {pr_id.upper()}"
    collecting = False
    collected: list[str] = []
    for line in lines:
        if line.startswith(target_header):
            collecting = True
        elif collecting and line.startswith("## "):
            break
        if collecting:
            collected.append(line)
    return "\n".join(collected).strip() + ("\n" if collected else "")


def resolve_repo_path(repo: pathlib.Path, raw_path: str) -> pathlib.Path | None:
    candidate = pathlib.Path(raw_path)
    if candidate.is_absolute():
        try:
            candidate.resolve().relative_to(repo.resolve())
        except ValueError:
            return None
        return candidate.resolve()
    resolved = (repo / raw_path).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        return None
    return resolved


def extract_repo_paths(repo: pathlib.Path, markdown_text: str) -> list[str]:
    resolved_paths: list[str] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(markdown_text):
        resolved = resolve_repo_path(repo, match.group("path"))
        if resolved and resolved.exists():
            value = str(resolved)
            if value not in resolved_paths:
                resolved_paths.append(value)
    for match in INLINE_REPO_PATH_PATTERN.finditer(markdown_text):
        resolved = resolve_repo_path(repo, match.group("path"))
        if resolved and resolved.exists():
            value = str(resolved)
            if value not in resolved_paths:
                resolved_paths.append(value)
    return resolved_paths


def resolve_pr_context(
    repo: pathlib.Path,
    issue_payload: dict[str, Any],
) -> dict[str, Any] | None:
    inventory = parse_pr_inventory(repo)
    candidate_texts = [
        str(issue_payload.get("title") or ""),
        str(issue_payload.get("description") or ""),
        str(issue_payload.get("branchName") or ""),
    ]
    comments_payload = issue_payload.get("comments", {})
    if isinstance(comments_payload, dict):
        for comment in comments_payload.get("nodes", []):
            if isinstance(comment, dict):
                candidate_texts.append(str(comment.get("body") or ""))

    for candidate in extract_task_card_candidates(*candidate_texts):
        task_file = inventory.get(candidate)
        if task_file is None or not task_file.exists():
            continue
        task_text = read_text(task_file)
        card_markdown = extract_card_markdown(task_text, candidate)
        if not card_markdown.strip():
            continue
        cited_paths = extract_repo_paths(repo, card_markdown)
        return {
            "pr_id": candidate,
            "task_file": task_file.as_posix(),
            "card_markdown": card_markdown,
            "cited_paths": cited_paths,
        }
    return None


def fetch_issue_snapshot(issue_identifier: str) -> dict[str, Any]:
    snapshot_override = os.environ.get(ISSUE_SNAPSHOT_PATH_ENV)
    if snapshot_override:
        payload = json.loads(
            pathlib.Path(snapshot_override).expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
        payload.setdefault("identifier", issue_identifier)
        return payload
    try:
        snapshot: dict[str, Any] = {
            "identifier": issue_identifier,
            **linear_api.fetch_issue(issue_identifier),
        }
    except Exception as exc:
        raise DispatchError(
            f"Failed to load Linear issue {issue_identifier}: {exc}"
        ) from exc
    try:
        comments = linear_api.list_issue_comments(issue_identifier)
    except Exception as exc:
        raise DispatchError(
            f"Failed to load Linear comments for {issue_identifier}: {exc}"
        ) from exc
    if comments:
        snapshot["comments"] = {"nodes": comments}
    return snapshot


def find_workpad_snapshot(issue_identifier: str) -> dict[str, Any] | None:
    snapshot_override = os.environ.get(WORKPAD_SNAPSHOT_PATH_ENV)
    if snapshot_override:
        return json.loads(
            pathlib.Path(snapshot_override).expanduser().resolve().read_text(
                encoding="utf-8"
            )
        )
    try:
        return linear_api.find_workpad_comment(issue_identifier)
    except Exception:
        return None


def build_dispatch_context(
    *,
    repo: pathlib.Path,
    workspace: pathlib.Path,
    issue_payload: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    workflow_text = read_text(repo / "WORKFLOW.md")
    rendered_prompt = render_workflow_prompt(
        workflow_text,
        issue_payload,
        attempt=os.environ.get("SYMPHONY_ATTEMPT"),
    )
    pr_context = resolve_pr_context(repo, issue_payload)
    workpad_comment = find_workpad_snapshot(str(issue_payload.get("identifier") or ""))
    resume_context_path = workspace / ".codex" / "symphony" / "resume_context.md"

    context_manifest = {
        "issue_id": issue_payload.get("identifier"),
        "issue_title": issue_payload.get("title"),
        "issue_state": issue_state_name(issue_payload),
        "issue_url": issue_payload.get("url"),
        "labels": normalize_labels(issue_payload.get("labels")),
        "workpad_comment_id": workpad_comment.get("id") if workpad_comment else None,
        "pr_context": {
            "pr_id": pr_context["pr_id"],
            "task_file": pr_context["task_file"],
            "cited_paths": pr_context["cited_paths"],
        }
        if pr_context
        else None,
        "resume_context_present": resume_context_path.exists(),
        "command": trace.redact_command_args(command),
    }

    artifacts: list[dict[str, Any]] = [
        {
            "type": "rendered_prompt",
            "label": "Rendered Worker Prompt",
            "content_type": "text/markdown",
            "filename": "rendered_prompt.md",
            "content": rendered_prompt,
        },
        {
            "type": "context_pack",
            "label": "Dispatch Context Pack",
            "filename": "dispatch_context_pack.json",
            "payload": context_manifest,
        },
        {
            "type": "issue_payload",
            "label": "Linear Issue Payload",
            "filename": "linear_issue_payload.json",
            "payload": issue_payload,
        },
        {
            "type": "agents_md",
            "label": "AGENTS.md Snapshot",
            "content_type": "text/markdown",
            "filename": "AGENTS.md",
            "content": read_text(repo / "AGENTS.md"),
        },
        {
            "type": "skill_snapshot",
            "label": "GPU CFD Symphony Skill",
            "content_type": "text/markdown",
            "filename": "gpu-cfd-symphony-SKILL.md",
            "content": read_text(repo / ".codex" / "skills" / "gpu-cfd-symphony" / "SKILL.md"),
        },
        {
            "type": "workflow_template",
            "label": "Workflow Template",
            "content_type": "text/markdown",
            "filename": "WORKFLOW.md",
            "content": workflow_text,
        },
        {
            "type": "pr_inventory",
            "label": "PR Inventory Snapshot",
            "content_type": "text/markdown",
            "filename": "pr_inventory.md",
            "content": read_text(repo / "docs" / "tasks" / "pr_inventory.md"),
        },
    ]
    if workpad_comment:
        artifacts.append(
            {
                "type": "workpad_snapshot",
                "label": "Current Linear Workpad",
                "content_type": "text/markdown",
                "filename": "linear_workpad.md",
                "content": str(workpad_comment.get("body") or ""),
            }
        )
    if resume_context_path.exists():
        artifacts.append(
            {
                "type": "resume_context",
                "label": "Resume Context",
                "content_type": "text/markdown",
                "filename": "resume_context.md",
                "content": read_text(resume_context_path),
            }
        )
    if pr_context:
        task_file_path = pathlib.Path(pr_context["task_file"])
        artifacts.extend(
            [
                {
                    "type": "task_file",
                    "label": "Owning Task File",
                    "content_type": "text/markdown",
                    "filename": task_file_path.name,
                    "content": read_text(task_file_path),
                },
                {
                    "type": "pr_card",
                    "label": f"PR Card {pr_context['pr_id']}",
                    "content_type": "text/markdown",
                    "filename": f"{pr_context['pr_id']}_card.md",
                    "content": pr_context["card_markdown"],
                },
            ]
        )
        for cited_path in pr_context["cited_paths"]:
            cited_file = pathlib.Path(cited_path)
            if not cited_file.exists():
                continue
            artifacts.append(
                {
                    "type": "cited_doc",
                    "label": f"Cited Doc: {cited_file.name}",
                    "content_type": "text/markdown",
                    "filename": cited_file.name,
                    "content": read_text(cited_file),
                    "metadata": {"source_path": cited_file.as_posix()},
                }
            )
    return {
        "rendered_prompt": rendered_prompt,
        "context_manifest": context_manifest,
        "artifacts": artifacts,
        "pr_context": pr_context,
    }


def task_requires_source_audit_gate(pr_context: dict[str, Any] | None) -> bool:
    if pr_context is None:
        return False
    pr_id = str(pr_context.get("pr_id") or "").upper()
    if pr_id in SOURCE_AUDIT_EXEMPT_TASK_IDS:
        return False
    phase_match = re.match(r"^P(?P<phase>\d+)-", pr_id)
    if not phase_match:
        return False
    phase_prefix = f"P{phase_match.group('phase')}"
    if phase_prefix not in SOURCE_AUDIT_PHASE_REQUIREMENTS:
        return False
    index_match = re.match(r"^P\d+-(?P<index>\d+)$", pr_id)
    if not index_match:
        return False
    return int(index_match.group("index")) >= 2


def resolve_source_audit_requirement(pr_context: dict[str, Any]) -> dict[str, Any]:
    pr_id = str(pr_context.get("pr_id") or "").upper()
    phase_match = re.match(r"^P(?P<phase>\d+)-", pr_id)
    if not phase_match:
        raise DispatchError(f"cannot infer source-audit requirement for task {pr_id}")
    phase_prefix = f"P{phase_match.group('phase')}"
    requirement = SOURCE_AUDIT_PHASE_REQUIREMENTS.get(phase_prefix)
    if requirement is None:
        raise DispatchError(f"no source-audit requirement configured for task {pr_id}")
    return requirement


def find_source_audit_note(repo: pathlib.Path, note_filename: str) -> pathlib.Path:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            note_filename,
            f":(glob)**/{note_filename}",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DispatchError(
            f"failed to resolve checked-in source-audit note {note_filename}: {completed.stderr.strip()}"
        )
    matches = [
        (repo / line.strip()).resolve()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if not matches:
        raise DispatchError(
            f"required reviewed source-audit note {note_filename} was not found in tracked files"
        )
    if len(matches) > 1:
        raise DispatchError(
            f"required reviewed source-audit note {note_filename} is ambiguous in tracked files: "
            + ", ".join(path.as_posix() for path in matches)
        )
    return matches[0]


def enforce_source_audit_gate(
    repo: pathlib.Path,
    pr_context: dict[str, Any] | None,
) -> None:
    if not task_requires_source_audit_gate(pr_context):
        return
    assert pr_context is not None
    requirement = resolve_source_audit_requirement(pr_context)
    note_path = find_source_audit_note(repo, str(requirement["note_filename"]))
    bundle = load_authority_bundle(repo)
    validate_source_audit_note(
        bundle,
        note_text=read_text(note_path),
        touched_surfaces=list(requirement["required_surfaces"]),
    )


def _stream_copy(
    source,
    targets: list[Any],
    transcript_path: pathlib.Path | None = None,
    *,
    close_targets: bool = False,
) -> None:
    transcript_handle = None
    if transcript_path is not None:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_handle = transcript_path.open("wb")
    try:
        while True:
            chunk_reader = getattr(source, "read1", None)
            if callable(chunk_reader):
                chunk = chunk_reader(8192)
            else:
                chunk = source.read(8192)
            if not chunk:
                break
            for target in targets:
                try:
                    target.write(chunk)
                    target.flush()
                except (BrokenPipeError, OSError):
                    return
            if transcript_handle is not None:
                transcript_handle.write(chunk)
                transcript_handle.flush()
    finally:
        if transcript_handle is not None:
            transcript_handle.close()
        if close_targets:
            for target in targets:
                try:
                    target.close()
                except OSError:
                    continue


def launch_codex_proxy(
    *,
    command: list[str],
    child_env: dict[str, str],
    transcript_dir: pathlib.Path | None,
) -> tuple[int, dict[str, pathlib.Path]]:
    process = subprocess.Popen(
        command,
        cwd=workspace_root(),
        env=child_env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    transcript_paths: dict[str, pathlib.Path] = {}
    if transcript_dir is not None:
        transcript_paths = {
            "stdin": transcript_dir / "app_server.stdin.log",
            "stdout": transcript_dir / "app_server.stdout.log",
            "stderr": transcript_dir / "app_server.stderr.log",
        }

    stdin_thread = threading.Thread(
        target=_stream_copy,
        args=(
            sys.stdin.buffer,
            [process.stdin] if process.stdin is not None else [],
            transcript_paths.get("stdin"),
        ),
        kwargs={"close_targets": True},
        daemon=True,
    )
    stdout_thread = threading.Thread(
        target=_stream_copy,
        args=(
            process.stdout,
            [sys.stdout.buffer] if process.stdout is not None else [],
            transcript_paths.get("stdout"),
        ),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_copy,
        args=(
            process.stderr,
            [sys.stderr.buffer] if process.stderr is not None else [],
            transcript_paths.get("stderr"),
        ),
        daemon=True,
    )
    stdin_thread.start()
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return return_code, transcript_paths


def capture_dispatch_bundle(
    *,
    repo: pathlib.Path,
    workspace: pathlib.Path,
    issue_payload: dict[str, Any],
    run_manifest: dict[str, Any],
    command: list[str],
) -> list[str]:
    bundle = build_dispatch_context(
        repo=repo,
        workspace=workspace,
        issue_payload=issue_payload,
        command=command,
    )
    artifact_ids: list[str] = []
    for artifact in bundle["artifacts"]:
        if "payload" in artifact:
            recorded = trace.capture_json_artifact(
                issue_id=run_manifest["issue_id"],
                run_id=run_manifest["run_id"],
                artifact_type=artifact["type"],
                label=artifact["label"],
                payload=artifact["payload"],
                filename=artifact["filename"],
                metadata=artifact.get("metadata"),
            )
        else:
            recorded = trace.capture_text_artifact(
                issue_id=run_manifest["issue_id"],
                run_id=run_manifest["run_id"],
                artifact_type=artifact["type"],
                label=artifact["label"],
                content=artifact["content"],
                content_type=artifact.get("content_type", "text/plain"),
                filename=artifact["filename"],
                metadata=artifact.get("metadata"),
            )
        artifact_ids.append(recorded["artifact_id"])

    trace.capture_event(
        issue_id=run_manifest["issue_id"],
        run_id=run_manifest["run_id"],
        actor="Symphony",
        stage="dispatch",
        summary="Dispatched implementation worker with a frozen context pack",
        decision=run_manifest["run_kind"],
        decision_rationale="Worker context captured before Codex app-server startup",
        artifact_refs=artifact_ids,
    )
    return artifact_ids


def main() -> int:
    args = parse_args()
    repo = repo_root()
    workspace = workspace_root()
    issue_identifier = issue_identifier_from_workspace(workspace)
    branch = current_branch(workspace)
    tracing_enabled = trace.is_enabled()
    issue_payload = fetch_issue_snapshot(issue_identifier)
    issue_payload.setdefault("identifier", issue_identifier)
    state_start: str | None = None
    run_kind = "implementation"
    state_start = issue_state_name(issue_payload) or None
    run_kind = "rework" if state_start == "Rework" else "implementation"
    enforce_source_audit_gate(
        repo,
        resolve_pr_context(repo, issue_payload),
    )

    codex_args = args.codex_args or ["app-server"]
    command = runtime_config.build_codex_command("implementation", codex_args)

    run_manifest: dict[str, Any] | None = None
    transcript_dir: pathlib.Path | None = None
    commit_before = current_commit(workspace)
    if tracing_enabled:
        env_metadata = {
            "workspace_path": workspace.as_posix(),
            "repo_root": repo.as_posix(),
            "model": runtime_config.load_codex_profile("implementation").model,
            "reasoning_effort": runtime_config.load_codex_profile("implementation").reasoning_effort,
            "sandbox_mode": "workspaceWrite",
            "trace_mode": os.environ.get(trace.TRACE_MODE_ENV) or trace.DEFAULT_TRACE_MODE,
            "issue_id": issue_identifier,
            "run_kind": run_kind,
            "branch": branch,
        }
        latest = trace.latest_run_summary(None, issue_identifier)
        run_manifest = trace.create_run(
            issue_id=issue_identifier,
            run_kind=run_kind,
            parent_run_id=latest["run_id"] if latest else None,
            branch=branch,
            state_start=state_start,
            env_metadata=env_metadata,
        )
        capture_dispatch_bundle(
            repo=repo,
            workspace=workspace,
            issue_payload=issue_payload,
            run_manifest=run_manifest,
            command=command,
        )
        transcript_dir = trace.run_dir(
            trace.resolve_trace_root(),
            issue_identifier,
            run_manifest["run_id"],
        ) / "transcripts"

    child_env = os.environ.copy()
    if run_manifest is not None:
        child_env.update(
            {
                trace.TRACE_ENABLE_ENV: "1",
                trace.TRACE_ROOT_ENV: str(trace.resolve_trace_root()),
                trace.TRACE_MODE_ENV: os.environ.get(trace.TRACE_MODE_ENV, trace.DEFAULT_TRACE_MODE),
                trace.TRACE_RUN_ENV: run_manifest["run_id"],
                trace.TRACE_ISSUE_ENV: issue_identifier,
                trace.TRACE_KIND_ENV: run_kind,
            }
        )
        if branch:
            child_env[trace.TRACE_BRANCH_ENV] = branch
        if run_manifest.get("pr_number") is not None:
            child_env[trace.TRACE_PR_ENV] = str(run_manifest["pr_number"])

    return_code, transcript_paths = launch_codex_proxy(
        command=command,
        child_env=child_env,
        transcript_dir=transcript_dir,
    )

    if run_manifest is not None:
        transcript_artifacts: list[str] = []
        for stream_name, transcript_path in transcript_paths.items():
            if not transcript_path.exists():
                continue
            recorded = trace.capture_file_artifact(
                issue_id=issue_identifier,
                run_id=run_manifest["run_id"],
                artifact_type=f"app_server_{stream_name}",
                label=f"Codex app-server {stream_name.upper()} transcript",
                source_path=transcript_path,
                filename=f"app_server.{stream_name}.log",
            )
            transcript_artifacts.append(recorded["artifact_id"])

        try:
            final_issue = fetch_issue_snapshot(issue_identifier)
            final_state = issue_state_name(final_issue) or state_start
        except Exception as exc:
            print(
                f"trace finalization: could not fetch final issue state: {exc}",
                file=sys.stderr,
            )
            final_state = state_start
        commit_after = current_commit(workspace)
        try:
            trace.capture_event(
                issue_id=issue_identifier,
                run_id=run_manifest["run_id"],
                actor="Codex worker",
                stage="worker_exit",
                summary="Codex app-server process exited",
                decision=f"returncode={return_code}",
                decision_rationale="Worker process finished and final state was captured",
                artifact_refs=transcript_artifacts,
                metadata={
                    "returncode": return_code,
                    "commit_before": commit_before,
                    "commit_after": commit_after,
                    "final_state": final_state,
                },
            )
            trace.finalize_run(
                issue_id=issue_identifier,
                run_id=run_manifest["run_id"],
                state_end=final_state,
                metadata={
                    "returncode": return_code,
                    "commit_before": commit_before,
                    "commit_after": commit_after,
                },
            )
        except Exception as exc:
            print(f"trace finalization: could not record trace data: {exc}", file=sys.stderr)
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DispatchError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
