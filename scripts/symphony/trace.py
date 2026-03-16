#!/usr/bin/env python3
"""Dev-only trace bundle collector for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import pathlib
import re
import shutil
import uuid
from typing import Any

TRACE_VERSION = 1
TRACE_ENABLE_ENV = "GPU_CFD_TRACE_ENABLE"
TRACE_ROOT_ENV = "GPU_CFD_TRACE_ROOT"
TRACE_MODE_ENV = "GPU_CFD_TRACE_MODE"
TRACE_RUN_ENV = "GPU_CFD_TRACE_RUN_ID"
TRACE_ISSUE_ENV = "GPU_CFD_TRACE_ISSUE"
TRACE_KIND_ENV = "GPU_CFD_TRACE_RUN_KIND"
TRACE_PR_ENV = "GPU_CFD_TRACE_PR_NUMBER"
TRACE_BRANCH_ENV = "GPU_CFD_TRACE_BRANCH"

DEFAULT_TRACE_MODE = "standard"
DEFAULT_TRACE_ROOT = (
    pathlib.Path.home() / "projects" / "symphony-traces" / "gpu_cfd"
)
SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|api[-_]?key|authorization|cookie|session)",
    re.IGNORECASE,
)
TRACE_TRUE_VALUES = {"1", "true", "yes", "on"}
ENV_METADATA_ALLOWLIST = {
    "workspace_path",
    "repo_root",
    "model",
    "reasoning_effort",
    "sandbox_mode",
    "trace_mode",
    "issue_id",
    "run_kind",
    "pr_number",
    "branch",
}
CLI_SECRET_FLAGS = {
    "--api-key",
    "--auth-token",
    "--password",
    "--secret",
    "--token",
}


def utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRACE_TRUE_VALUES


def is_enabled() -> bool:
    return truthy(os.environ.get(TRACE_ENABLE_ENV))


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def default_trace_root() -> pathlib.Path:
    raw_root = os.environ.get(TRACE_ROOT_ENV)
    if raw_root:
        return pathlib.Path(raw_root).expanduser().resolve()
    return DEFAULT_TRACE_ROOT


def resolve_trace_root(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or default_trace_root()).expanduser().resolve()


def issue_dir(root: pathlib.Path, issue_id: str) -> pathlib.Path:
    return root / "issues" / sanitize_component(issue_id)


def run_dir(root: pathlib.Path, issue_id: str, run_id: str) -> pathlib.Path:
    return issue_dir(root, issue_id) / "runs" / sanitize_component(run_id)


def run_manifest_path(root: pathlib.Path, issue_id: str, run_id: str) -> pathlib.Path:
    return run_dir(root, issue_id, run_id) / "run.json"


def ensure_directory(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_id(prefix: str) -> str:
    return f"{sanitize_component(prefix)}-{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def sanitize_env_metadata(env_metadata: dict[str, Any] | None) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in (env_metadata or {}).items():
        if key not in ENV_METADATA_ALLOWLIST:
            continue
        sanitized[key] = value
    return sanitized


def redact_command_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            redacted.append("[REDACTED]")
            skip_next = False
            continue
        lowered = arg.lower()
        if lowered in CLI_SECRET_FLAGS:
            redacted.append(arg)
            skip_next = True
            continue
        if "=" in arg:
            key, value = arg.split("=", 1)
            if SECRET_KEY_PATTERN.search(key):
                redacted.append(f"{key}=[REDACTED]")
                continue
            if key.startswith("-") and SECRET_KEY_PATTERN.search(key):
                redacted.append(f"{key}=[REDACTED]")
                continue
        if SECRET_KEY_PATTERN.search(arg) and arg.startswith("--"):
            redacted.append(f"{arg}=[REDACTED]")
            skip_next = True
            continue
        redacted.append(arg)
    return redacted


def relative_to_root(root: pathlib.Path, path: pathlib.Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def load_run_manifest(
    root: pathlib.Path | None,
    issue_id: str,
    run_id: str,
) -> dict[str, Any]:
    manifest_path = run_manifest_path(resolve_trace_root(root), issue_id, run_id)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_run_manifest(
    root: pathlib.Path | None,
    issue_id: str,
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    manifest_path = run_manifest_path(resolved_root, issue_id, run_id)
    ensure_directory(manifest_path.parent)
    manifest["updated_at"] = utc_now()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def latest_run_summary(root: pathlib.Path | None, issue_id: str) -> dict[str, Any] | None:
    issue_manifest_path = issue_dir(resolve_trace_root(root), issue_id) / "issue.json"
    if not issue_manifest_path.exists():
        return None
    payload = json.loads(issue_manifest_path.read_text(encoding="utf-8"))
    runs = payload.get("runs", [])
    return runs[-1] if runs else None


def current_trace_context() -> dict[str, str | None]:
    return {
        "issue_id": os.environ.get(TRACE_ISSUE_ENV),
        "run_id": os.environ.get(TRACE_RUN_ENV),
        "run_kind": os.environ.get(TRACE_KIND_ENV),
        "branch": os.environ.get(TRACE_BRANCH_ENV),
        "pr_number": os.environ.get(TRACE_PR_ENV),
        "trace_mode": os.environ.get(TRACE_MODE_ENV) or DEFAULT_TRACE_MODE,
    }


def create_run(
    *,
    issue_id: str,
    run_id: str | None = None,
    run_kind: str,
    parent_run_id: str | None = None,
    branch: str | None = None,
    pr_number: int | None = None,
    state_start: str | None = None,
    env_metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    resolved_run_id = run_id or generate_id(run_kind)
    manifest_path = run_manifest_path(resolved_root, issue_id, resolved_run_id)
    if manifest_path.exists():
        return load_run_manifest(resolved_root, issue_id, resolved_run_id)

    manifest = {
        "trace_version": TRACE_VERSION,
        "issue_id": issue_id,
        "run_id": resolved_run_id,
        "run_kind": run_kind,
        "parent_run_id": parent_run_id,
        "branch": branch,
        "pr_number": pr_number,
        "state_start": state_start,
        "state_end": None,
        "trace_mode": os.environ.get(TRACE_MODE_ENV) or DEFAULT_TRACE_MODE,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finalized_at": None,
        "env_metadata": sanitize_env_metadata(env_metadata),
        "events": [],
        "artifacts": [],
        "diffs": [],
    }
    save_run_manifest(resolved_root, issue_id, resolved_run_id, manifest)
    build_index(resolved_root)
    return manifest


def ensure_run(
    *,
    issue_id: str,
    run_kind: str,
    branch: str | None = None,
    pr_number: int | None = None,
    state_start: str | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    context = current_trace_context()
    run_id = context["run_id"]
    if run_id and context["issue_id"] == issue_id:
        return load_run_manifest(root, issue_id, run_id)
    latest = latest_run_summary(root, issue_id)
    return create_run(
        issue_id=issue_id,
        run_kind=run_kind,
        parent_run_id=latest["run_id"] if latest else None,
        branch=branch,
        pr_number=pr_number,
        state_start=state_start,
        root=root,
    )


def _update_run_list(
    manifest: dict[str, Any],
    key: str,
    entry: dict[str, Any],
    unique_key: str,
) -> None:
    existing = manifest.get(key, [])
    existing = [item for item in existing if item.get(unique_key) != entry.get(unique_key)]
    existing.append(entry)
    existing.sort(key=lambda item: item.get("timestamp") or item.get("created_at") or "")
    manifest[key] = existing


def capture_event(
    *,
    issue_id: str,
    run_id: str,
    actor: str,
    stage: str,
    summary: str,
    decision: str | None = None,
    decision_rationale: str | None = None,
    artifact_refs: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    manifest = load_run_manifest(root, issue_id, run_id)
    event = {
        "event_id": generate_id("event"),
        "actor": actor,
        "stage": stage,
        "timestamp": utc_now(),
        "summary": summary,
        "decision": decision,
        "decision_rationale": decision_rationale,
        "artifact_refs": artifact_refs or [],
        "metadata": metadata or {},
    }
    _update_run_list(manifest, "events", event, "event_id")
    save_run_manifest(root, issue_id, run_id, manifest)
    build_index(root)
    return event


def _artifact_target_path(
    *,
    root: pathlib.Path,
    issue_id: str,
    run_id: str,
    artifact_id: str,
    filename: str,
) -> pathlib.Path:
    safe_name = sanitize_component(pathlib.Path(filename).stem)
    suffix = pathlib.Path(filename).suffix or ".txt"
    return ensure_directory(run_dir(root, issue_id, run_id) / "artifacts") / (
        f"{artifact_id}__{safe_name}{suffix}"
    )


def _register_artifact(
    *,
    issue_id: str,
    run_id: str,
    artifact: dict[str, Any],
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    manifest = load_run_manifest(root, issue_id, run_id)
    _update_run_list(manifest, "artifacts", artifact, "artifact_id")
    save_run_manifest(root, issue_id, run_id, manifest)
    build_index(root)
    return artifact


def capture_text_artifact(
    *,
    issue_id: str,
    run_id: str,
    artifact_type: str,
    label: str,
    content: str,
    content_type: str = "text/plain",
    filename: str | None = None,
    event_id: str | None = None,
    redaction_status: str = "none",
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    artifact_id = generate_id("artifact")
    target = _artifact_target_path(
        root=resolved_root,
        issue_id=issue_id,
        run_id=run_id,
        artifact_id=artifact_id,
        filename=filename or f"{artifact_type}.txt",
    )
    target.write_text(content, encoding="utf-8")
    artifact = {
        "artifact_id": artifact_id,
        "type": artifact_type,
        "label": label,
        "path": relative_to_root(resolved_root, target),
        "content_type": content_type,
        "originating_event_id": event_id,
        "redaction_status": redaction_status,
        "metadata": metadata or {},
        "size_bytes": target.stat().st_size,
        "created_at": utc_now(),
    }
    return _register_artifact(issue_id=issue_id, run_id=run_id, artifact=artifact, root=resolved_root)


def capture_json_artifact(
    *,
    issue_id: str,
    run_id: str,
    artifact_type: str,
    label: str,
    payload: Any,
    filename: str | None = None,
    event_id: str | None = None,
    redaction_status: str = "none",
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    content = json.dumps(payload, indent=2) + "\n"
    return capture_text_artifact(
        issue_id=issue_id,
        run_id=run_id,
        artifact_type=artifact_type,
        label=label,
        content=content,
        content_type="application/json",
        filename=filename or f"{artifact_type}.json",
        event_id=event_id,
        redaction_status=redaction_status,
        metadata=metadata,
        root=root,
    )


def capture_file_artifact(
    *,
    issue_id: str,
    run_id: str,
    artifact_type: str,
    label: str,
    source_path: pathlib.Path,
    filename: str | None = None,
    event_id: str | None = None,
    redaction_status: str = "none",
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    artifact_id = generate_id("artifact")
    source = source_path.resolve()
    target = _artifact_target_path(
        root=resolved_root,
        issue_id=issue_id,
        run_id=run_id,
        artifact_id=artifact_id,
        filename=filename or source.name,
    )
    shutil.copyfile(source, target)
    artifact = {
        "artifact_id": artifact_id,
        "type": artifact_type,
        "label": label,
        "path": relative_to_root(resolved_root, target),
        "content_type": _guess_content_type(target),
        "originating_event_id": event_id,
        "redaction_status": redaction_status,
        "metadata": {
            **(metadata or {}),
            "source_path": str(source),
        },
        "size_bytes": target.stat().st_size,
        "created_at": utc_now(),
    }
    return _register_artifact(issue_id=issue_id, run_id=run_id, artifact=artifact, root=resolved_root)


def _guess_content_type(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    if suffix in {".jsonl", ".log", ".txt"}:
        return "text/plain"
    return "application/octet-stream"


def capture_diff(
    *,
    issue_id: str,
    run_id: str,
    label: str,
    before: str,
    after: str,
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    diff_id = generate_id("diff")
    unified_diff = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    payload = {
        "diff_id": diff_id,
        "label": label,
        "before": before,
        "after": after,
        "unified_diff": unified_diff,
        "metadata": metadata or {},
        "created_at": utc_now(),
    }
    target = ensure_directory(run_dir(resolved_root, issue_id, run_id) / "diffs") / f"{diff_id}.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    manifest = load_run_manifest(resolved_root, issue_id, run_id)
    diff_summary = {
        "diff_id": diff_id,
        "label": label,
        "path": relative_to_root(resolved_root, target),
        "created_at": payload["created_at"],
        "metadata": metadata or {},
    }
    _update_run_list(manifest, "diffs", diff_summary, "diff_id")
    save_run_manifest(resolved_root, issue_id, run_id, manifest)
    build_index(resolved_root)
    return payload


def capture_workpad_revision(
    *,
    issue_id: str,
    run_id: str,
    previous_body: str | None,
    current_body: str,
    action: str,
    comment_id: str | None,
    comment_url: str | None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    previous_artifact = None
    if previous_body:
        previous_artifact = capture_text_artifact(
            issue_id=issue_id,
            run_id=run_id,
            artifact_type="workpad_previous",
            label="Previous Workpad",
            content=previous_body,
            content_type="text/markdown",
            filename="workpad_previous.md",
            root=root,
        )
    current_artifact = capture_text_artifact(
        issue_id=issue_id,
        run_id=run_id,
        artifact_type="workpad_current",
        label="Current Workpad",
        content=current_body,
        content_type="text/markdown",
        filename="workpad_current.md",
        root=root,
    )
    diff = capture_diff(
        issue_id=issue_id,
        run_id=run_id,
        label="Workpad Revision Diff",
        before=previous_body or "",
        after=current_body,
        metadata={"comment_id": comment_id, "comment_url": comment_url, "action": action},
        root=root,
    )
    artifact_refs = [current_artifact["artifact_id"]]
    if previous_artifact is not None:
        artifact_refs.insert(0, previous_artifact["artifact_id"])
    event = capture_event(
        issue_id=issue_id,
        run_id=run_id,
        actor="Linear",
        stage="workpad_updated",
        summary=f"Canonical workpad comment {action}",
        decision=action,
        decision_rationale="Linear workpad updated to preserve task memory and operator visibility",
        artifact_refs=artifact_refs,
        metadata={"comment_id": comment_id, "comment_url": comment_url, "diff_id": diff["diff_id"]},
        root=root,
    )
    return {
        "event": event,
        "previous_artifact": previous_artifact,
        "current_artifact": current_artifact,
        "diff": diff,
    }


def finalize_run(
    *,
    issue_id: str,
    run_id: str,
    state_end: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: pathlib.Path | None = None,
) -> dict[str, Any]:
    manifest = load_run_manifest(root, issue_id, run_id)
    manifest["state_end"] = state_end or manifest.get("state_end")
    manifest["finalized_at"] = utc_now()
    if metadata:
        manifest.setdefault("final_metadata", {}).update(metadata)
    saved = save_run_manifest(root, issue_id, run_id, manifest)
    build_index(root)
    return saved


def build_index(root: pathlib.Path | None = None) -> dict[str, Any]:
    resolved_root = resolve_trace_root(root)
    ensure_directory(resolved_root / "issues")
    issues: list[dict[str, Any]] = []
    runs_index: dict[str, dict[str, Any]] = {}
    artifacts_index: dict[str, dict[str, Any]] = {}
    diffs_index: dict[str, dict[str, Any]] = {}

    for issue_path in sorted((resolved_root / "issues").iterdir() if (resolved_root / "issues").exists() else []):
        if not issue_path.is_dir():
            continue
        run_summaries: list[dict[str, Any]] = []
        run_root = issue_path / "runs"
        for run_path in sorted(run_root.iterdir() if run_root.exists() else []):
            manifest_path = run_path / "run.json"
            if not manifest_path.exists():
                continue
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_summary = {
                "run_id": run_manifest["run_id"],
                "run_kind": run_manifest.get("run_kind"),
                "parent_run_id": run_manifest.get("parent_run_id"),
                "branch": run_manifest.get("branch"),
                "pr_number": run_manifest.get("pr_number"),
                "state_start": run_manifest.get("state_start"),
                "state_end": run_manifest.get("state_end"),
                "started_at": run_manifest.get("started_at"),
                "updated_at": run_manifest.get("updated_at"),
                "finalized_at": run_manifest.get("finalized_at"),
                "event_count": len(run_manifest.get("events", [])),
                "artifact_count": len(run_manifest.get("artifacts", [])),
                "path": relative_to_root(resolved_root, manifest_path),
            }
            run_summaries.append(run_summary)
            runs_index[run_manifest["run_id"]] = {
                "issue_id": run_manifest["issue_id"],
                **run_summary,
            }
            for artifact in run_manifest.get("artifacts", []):
                artifacts_index[artifact["artifact_id"]] = {
                    "issue_id": run_manifest["issue_id"],
                    "run_id": run_manifest["run_id"],
                    **artifact,
                }
            for diff in run_manifest.get("diffs", []):
                diffs_index[diff["diff_id"]] = {
                    "issue_id": run_manifest["issue_id"],
                    "run_id": run_manifest["run_id"],
                    **diff,
                }
        run_summaries.sort(key=lambda item: item.get("started_at") or "")
        latest = run_summaries[-1] if run_summaries else {}
        issue_payload = {
            "issue_id": issue_path.name,
            "latest_state": latest.get("state_end") or latest.get("state_start"),
            "latest_branch": latest.get("branch"),
            "latest_pr_number": latest.get("pr_number"),
            "updated_at": latest.get("updated_at"),
            "runs": run_summaries,
            "lineage": [run["run_id"] for run in run_summaries],
        }
        (issue_path / "issue.json").write_text(
            json.dumps(issue_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        issues.append(issue_payload)

    issues.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    index = {
        "trace_version": TRACE_VERSION,
        "generated_at": utc_now(),
        "issues": issues,
        "runs": runs_index,
        "artifacts": artifacts_index,
        "diffs": diffs_index,
    }
    (resolved_root / "trace_index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def parse_json_argument(raw_value: str | None) -> Any:
    if not raw_value:
        return None
    candidate = pathlib.Path(raw_value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def capture_dispatch_cli(args: argparse.Namespace) -> int:
    env_metadata = parse_json_argument(args.env_json) or {}
    manifest = create_run(
        issue_id=args.issue,
        run_id=args.run_id,
        run_kind=args.run_kind,
        parent_run_id=args.parent_run_id,
        branch=args.branch,
        pr_number=args.pr_number,
        state_start=args.state_start,
        env_metadata=env_metadata,
        root=pathlib.Path(args.root).expanduser() if args.root else None,
    )
    if args.summary:
        capture_event(
            issue_id=args.issue,
            run_id=manifest["run_id"],
            actor="Symphony",
            stage="dispatch",
            summary=args.summary,
            root=pathlib.Path(args.root).expanduser() if args.root else None,
        )
    print(json.dumps(manifest, indent=2))
    return 0


def capture_event_cli(args: argparse.Namespace) -> int:
    event = capture_event(
        issue_id=args.issue,
        run_id=args.run_id,
        actor=args.actor,
        stage=args.stage,
        summary=args.summary,
        decision=args.decision,
        decision_rationale=args.decision_rationale,
        artifact_refs=parse_json_argument(args.artifact_refs) or [],
        metadata=parse_json_argument(args.metadata) or {},
        root=pathlib.Path(args.root).expanduser() if args.root else None,
    )
    print(json.dumps(event, indent=2))
    return 0


def capture_artifact_cli(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root).expanduser() if args.root else None
    if args.source:
        artifact = capture_file_artifact(
            issue_id=args.issue,
            run_id=args.run_id,
            artifact_type=args.type,
            label=args.label,
            source_path=pathlib.Path(args.source),
            filename=args.filename,
            root=root,
        )
    elif args.json_value:
        artifact = capture_json_artifact(
            issue_id=args.issue,
            run_id=args.run_id,
            artifact_type=args.type,
            label=args.label,
            payload=parse_json_argument(args.json_value),
            filename=args.filename,
            root=root,
        )
    else:
        artifact = capture_text_artifact(
            issue_id=args.issue,
            run_id=args.run_id,
            artifact_type=args.type,
            label=args.label,
            content=args.text or "",
            content_type=args.content_type,
            filename=args.filename,
            root=root,
        )
    print(json.dumps(artifact, indent=2))
    return 0


def finalize_run_cli(args: argparse.Namespace) -> int:
    manifest = finalize_run(
        issue_id=args.issue,
        run_id=args.run_id,
        state_end=args.state_end,
        metadata=parse_json_argument(args.metadata) or {},
        root=pathlib.Path(args.root).expanduser() if args.root else None,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def build_index_cli(args: argparse.Namespace) -> int:
    index = build_index(pathlib.Path(args.root).expanduser() if args.root else None)
    print(json.dumps(index, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dispatch_parser = subparsers.add_parser("capture-dispatch")
    dispatch_parser.add_argument("--issue", required=True)
    dispatch_parser.add_argument("--run-id")
    dispatch_parser.add_argument("--run-kind", required=True)
    dispatch_parser.add_argument("--parent-run-id")
    dispatch_parser.add_argument("--branch")
    dispatch_parser.add_argument("--pr-number", type=int)
    dispatch_parser.add_argument("--state-start")
    dispatch_parser.add_argument("--summary")
    dispatch_parser.add_argument("--env-json")
    dispatch_parser.add_argument("--root")

    event_parser = subparsers.add_parser("capture-event")
    event_parser.add_argument("--issue", required=True)
    event_parser.add_argument("--run-id", required=True)
    event_parser.add_argument("--actor", required=True)
    event_parser.add_argument("--stage", required=True)
    event_parser.add_argument("--summary", required=True)
    event_parser.add_argument("--decision")
    event_parser.add_argument("--decision-rationale")
    event_parser.add_argument("--artifact-refs")
    event_parser.add_argument("--metadata")
    event_parser.add_argument("--root")

    artifact_parser = subparsers.add_parser("capture-artifact")
    artifact_parser.add_argument("--issue", required=True)
    artifact_parser.add_argument("--run-id", required=True)
    artifact_parser.add_argument("--type", required=True)
    artifact_parser.add_argument("--label", required=True)
    artifact_parser.add_argument("--text")
    artifact_parser.add_argument("--json-value")
    artifact_parser.add_argument("--source")
    artifact_parser.add_argument("--content-type", default="text/plain")
    artifact_parser.add_argument("--filename")
    artifact_parser.add_argument("--root")

    finalize_parser = subparsers.add_parser("finalize-run")
    finalize_parser.add_argument("--issue", required=True)
    finalize_parser.add_argument("--run-id", required=True)
    finalize_parser.add_argument("--state-end")
    finalize_parser.add_argument("--metadata")
    finalize_parser.add_argument("--root")

    index_parser = subparsers.add_parser("build-index")
    index_parser.add_argument("--root")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "capture-dispatch":
        return capture_dispatch_cli(args)
    if args.command == "capture-event":
        return capture_event_cli(args)
    if args.command == "capture-artifact":
        return capture_artifact_cli(args)
    if args.command == "finalize-run":
        return finalize_run_cli(args)
    if args.command == "build-index":
        return build_index_cli(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
