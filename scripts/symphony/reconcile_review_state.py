#!/usr/bin/env python3
"""Reconcile open Symphony-managed PRs with the finite review loop contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict
from typing import Any

try:
    from scripts.symphony import devin_review_gate, linear_api, review_loop
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import devin_review_gate, linear_api, review_loop


PR_FIELDS = ",".join(
    [
        "number",
        "title",
        "body",
        "headRefName",
        "headRefOid",
        "url",
        "state",
        "isDraft",
        "mergeStateStatus",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form.")
    parser.add_argument(
        "--reviewer",
        action="append",
        dest="reviewers",
        help="Reviewer login to include. Repeat for multiple reviewers.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reconciliation instead of reporting the planned actions.",
    )
    return parser.parse_args()


def list_open_prs(repo: str | None) -> list[devin_review_gate.PullRequestSnapshot]:
    resolved_repo = repo or "/".join(review_loop.infer_repo())
    response = review_loop.require_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            resolved_repo,
            "--state",
            "open",
            "--json",
            PR_FIELDS,
        ],
        cwd=review_loop.repo_root(),
    )
    payload = json.loads(response.stdout or "[]")
    snapshots: list[devin_review_gate.PullRequestSnapshot] = []
    for item in payload:
        snapshots.append(
            devin_review_gate.PullRequestSnapshot(
                number=int(item["number"]),
                title=str(item.get("title") or ""),
                body=str(item.get("body") or ""),
                head_ref_name=str(item.get("headRefName") or ""),
                head_oid=str(item.get("headRefOid") or ""),
                url=str(item.get("url") or ""),
                state=str(item.get("state") or "UNKNOWN"),
                is_draft=bool(item.get("isDraft")),
                merge_state_status=(
                    str(item.get("mergeStateStatus"))
                    if item.get("mergeStateStatus") is not None
                    else None
                ),
            )
        )
    return snapshots


def enable_auto_merge_best_effort(repo: str, pr_number: int) -> dict[str, Any]:
    completed = review_loop.run_command(
        [
            "gh",
            "pr",
            "merge",
            str(pr_number),
            "--repo",
            repo,
            "--auto",
            "--merge",
        ],
        cwd=review_loop.repo_root(),
    )
    if completed.returncode == 0:
        return {"applied": True, "status": "enabled"}
    combined_output = "\n".join([completed.stdout, completed.stderr]).lower()
    if "already enabled" in combined_output or "auto-merge is already enabled" in combined_output:
        return {"applied": False, "status": "already_enabled"}
    raise review_loop.CommandError(
        f"command failed: gh pr merge {pr_number} --auto --merge\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def reconcile_snapshot(
    snapshot: devin_review_gate.PullRequestSnapshot,
    *,
    repo: str,
    reviewers: list[str],
    apply_changes: bool,
) -> dict[str, Any]:
    issue_identifier = devin_review_gate.select_issue_identifier(snapshot)
    result: dict[str, Any] = {
        "pull_request": asdict(snapshot),
        "issue_identifier": issue_identifier,
    }
    if issue_identifier is None:
        result["status"] = "skipped"
        result["reason"] = "No linked Linear issue marker or identifier found."
        return result

    summary = review_loop.fetch_pr_summary(repo, snapshot.number, reviewers)
    decision = devin_review_gate.determine_gate_decision(snapshot, summary, issue_identifier)
    issue = linear_api.fetch_issue(issue_identifier)
    current_state = str(issue.get("state", {}).get("name") or "")
    result["review_summary"] = {
        "review_state": summary.review_state,
        "actionable_reviews": len(summary.actionable_reviews),
        "actionable_threads": len(summary.actionable_threads),
    }
    result["decision"] = asdict(decision)
    result["current_linear_state"] = current_state

    workpad_result = {"applied": False, "suppressed_comments": 0}
    if apply_changes:
        suppressed = linear_api.suppress_legacy_workpad_comments(issue_identifier)
        workpad_result = {
            "applied": True,
            "suppressed_comments": len(suppressed),
        }
    result["workpad_migration"] = workpad_result

    if decision.review_state == "action_required":
        linear_action = {"applied": False, "target_state": devin_review_gate.DEFAULT_REWORK_STATE}
        if apply_changes:
            update_result = linear_api.update_issue_state(
                issue_identifier,
                devin_review_gate.DEFAULT_REWORK_STATE,
            )
            linear_action = {
                "applied": bool(update_result["changed"]),
                "previous_state": update_result["previous_state"],
                "current_state": update_result["current_state"],
                "target_state": devin_review_gate.DEFAULT_REWORK_STATE,
            }
            workpad_warning = devin_review_gate.sync_followup_workpad_best_effort(
                issue_identifier,
                summary,
                target_state=devin_review_gate.DEFAULT_REWORK_STATE,
            )
            if workpad_warning:
                result["workpad_sync_warning"] = workpad_warning
        result["linear_action"] = linear_action
        result["status"] = "rework_required"
        return result

    if decision.review_state == "branch_refresh_required":
        linear_action = {
            "applied": False,
            "target_state": devin_review_gate.DEFAULT_REFRESH_REQUIRED_STATE,
        }
        if apply_changes:
            update_result = linear_api.update_issue_state(
                issue_identifier,
                devin_review_gate.DEFAULT_REFRESH_REQUIRED_STATE,
            )
            linear_action = {
                "applied": bool(update_result["changed"]),
                "previous_state": update_result["previous_state"],
                "current_state": update_result["current_state"],
                "target_state": devin_review_gate.DEFAULT_REFRESH_REQUIRED_STATE,
            }
            workpad_warning = devin_review_gate.sync_followup_workpad_best_effort(
                issue_identifier,
                summary,
                target_state=devin_review_gate.DEFAULT_REFRESH_REQUIRED_STATE,
            )
            if workpad_warning:
                result["workpad_sync_warning"] = workpad_warning
        result["linear_action"] = linear_action
        result["status"] = "refresh_required"
        return result

    auto_merge_result = {"applied": False, "status": "skipped"}
    if current_state == "In Review":
        if apply_changes:
            auto_merge_result = enable_auto_merge_best_effort(repo, snapshot.number)
        else:
            auto_merge_result = {"applied": False, "status": "would_enable"}
    result["auto_merge"] = auto_merge_result
    result["status"] = "review_complete"
    return result


def main() -> int:
    args = parse_args()
    repo = args.repo or "/".join(review_loop.infer_repo())
    reviewers = args.reviewers or list(devin_review_gate.DEFAULT_REVIEWERS)

    results = [
        reconcile_snapshot(
            snapshot,
            repo=repo,
            reviewers=reviewers,
            apply_changes=args.apply,
        )
        for snapshot in list_open_prs(repo)
    ]
    print(json.dumps({"repo": repo, "apply": args.apply, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
