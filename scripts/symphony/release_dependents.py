#!/usr/bin/env python3
"""Promote newly unblocked Linear dependents from Backlog to Todo."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

try:
    from scripts.symphony import github_linear_bridge, telemetry
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import github_linear_bridge, telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True, help="Merged Linear issue identifier.")
    parser.add_argument(
        "--workspace",
        default=".",
        help="Repo workspace used for telemetry context. Defaults to the current directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = pathlib.Path(args.workspace).resolve()
    released = github_linear_bridge.release_unblocked_dependents(args.issue)
    event = telemetry.build_event(
        event_type="dependency_release",
        message=f"Released {len(released)} dependent issue(s) after merge",
        issue=args.issue,
        state="Done",
        cwd=workspace,
        repo_root=workspace,
        details={"released": [entry["identifier"] for entry in released]},
    )
    telemetry.write_event(telemetry.default_telemetry_root(), event)
    print(json.dumps({"issue": args.issue, "released": released}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, github_linear_bridge.review_loop.CommandError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
