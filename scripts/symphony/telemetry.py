#!/usr/bin/env python3
"""Structured telemetry helpers for Symphony operations in gpu_cfd."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import pathlib
import socket
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    event_parser = subparsers.add_parser("event", help="Append a telemetry event.")
    event_parser.add_argument("--event-type", required=True, help="Structured event type label.")
    event_parser.add_argument("--message", default="", help="Short human-readable event summary.")
    event_parser.add_argument("--issue", help="Linear issue identifier, if known.")
    event_parser.add_argument("--pr", type=int, help="GitHub pull request number, if known.")
    event_parser.add_argument("--state", help="Linear or workflow state associated with the event.")
    event_parser.add_argument("--branch", help="Git branch associated with the event.")
    event_parser.add_argument("--commit", help="Git commit associated with the event.")
    event_parser.add_argument("--root", help="Telemetry root. Defaults to the configured logs root.")
    event_parser.add_argument(
        "--detail",
        action="append",
        default=[],
        help="Extra structured fields in key=value form. Repeat as needed.",
    )

    return parser.parse_args()


def default_telemetry_root() -> pathlib.Path:
    raw_root = (
        os.environ.get("GPU_CFD_TELEMETRY_ROOT")
        or os.environ.get("SYMPHONY_LOGS_ROOT")
        or "~/projects/symphony-logs/gpu_cfd"
    )
    return pathlib.Path(raw_root).expanduser()


def parse_details(raw_pairs: list[str]) -> dict[str, str]:
    details: dict[str, str] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value detail, got: {pair}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"detail key must not be empty: {pair}")
        details[key] = value
    return details


def detect_git_value(args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def build_event(
    *,
    event_type: str,
    message: str,
    issue: str | None = None,
    pr: int | None = None,
    state: str | None = None,
    branch: str | None = None,
    commit: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z")
    resolved_branch = branch or detect_git_value(["branch", "--show-current"])
    resolved_commit = commit or detect_git_value(["rev-parse", "HEAD"])
    resolved_repo_root = detect_git_value(["rev-parse", "--show-toplevel"])
    event = {
        "timestamp": timestamp,
        "event_type": event_type,
        "message": message,
        "issue": issue,
        "pr": pr,
        "state": state,
        "branch": resolved_branch,
        "commit": resolved_commit,
        "repo_root": resolved_repo_root,
        "cwd": pathlib.Path.cwd().as_posix(),
        "hostname": socket.gethostname(),
        "username": getpass.getuser(),
        "details": details or {},
    }
    return event


def append_jsonl(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_event(root: pathlib.Path, event: dict[str, Any]) -> dict[str, pathlib.Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "events": root / "events.jsonl",
    }
    append_jsonl(paths["events"], event)

    issue = event.get("issue")
    if issue:
        paths["issue"] = root / "issues" / f"{issue}.jsonl"
        append_jsonl(paths["issue"], event)

    if "block" in str(event.get("event_type", "")).lower():
        paths["blockers"] = root / "blockers.jsonl"
        append_jsonl(paths["blockers"], event)

    return paths


def event_command(args: argparse.Namespace) -> int:
    details = parse_details(args.detail)
    event = build_event(
        event_type=args.event_type,
        message=args.message,
        issue=args.issue,
        pr=args.pr,
        state=args.state,
        branch=args.branch,
        commit=args.commit,
        details=details,
    )
    root = pathlib.Path(args.root).expanduser() if args.root else default_telemetry_root()
    paths = write_event(root, event)
    payload = {
        "root": root.as_posix(),
        "paths": {name: path.as_posix() for name, path in paths.items()},
        "event": event,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "event":
        return event_command(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
