#!/usr/bin/env python3
"""Drive the required Devin review gate and wake Rework from GitHub events."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any

try:
    from scripts.symphony import linear_api, review_loop, trace
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import linear_api, review_loop, trace


CHECK_CONTEXT = "devin-review-gate"
DEFAULT_REVIEWERS = ("devin-ai-integration[bot]",)
DEFAULT_REWORK_STATE = "Rework"
DEFAULT_REFRESH_REQUIRED_STATE = "Refresh Required"
REVIEW_BRIDGE_TOKEN_SECRET = "REVIEW_BRIDGE_GH_TOKEN"
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
RESOLVE_REVIEW_THREAD_MUTATION = """
mutation($threadId: ID!) {
  resolveReviewThread(input: { threadId: $threadId }) {
    thread {
      id
      isResolved
    }
  }
}
"""


@dataclass(frozen=True)
class PullRequestSnapshot:
    number: int
    title: str
    body: str
    head_ref_name: str
    head_oid: str
    url: str
    state: str
    is_draft: bool
    merge_state_status: str | None


@dataclass(frozen=True)
class GateDecision:
    issue_identifier: str | None
    status_state: str
    description: str
    review_state: str
    target_state: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form.")
    parser.add_argument("--pr", type=int, help="Pull request number.")
    parser.add_argument(
        "--issue",
        help="Override the Linear issue identifier instead of inferring it from the PR.",
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        dest="reviewers",
        help="Reviewer login to include. Repeat for multiple reviewers.",
    )
    return parser.parse_args()


def fetch_pr_snapshot(repo: str | None, pr_number: int | None) -> PullRequestSnapshot:
    resolved_repo = repo or "/".join(review_loop.infer_repo())
    resolved_pr = pr_number if pr_number is not None else review_loop.infer_pr_number()
    response = review_loop.require_command(
        [
            "gh",
            "pr",
            "view",
            str(resolved_pr),
            "--repo",
            resolved_repo,
            "--json",
            PR_FIELDS,
        ],
        cwd=review_loop.repo_root(),
    )
    payload = json.loads(response.stdout)
    return PullRequestSnapshot(
        number=int(payload["number"]),
        title=payload.get("title") or "",
        body=payload.get("body") or "",
        head_ref_name=payload.get("headRefName") or "",
        head_oid=payload.get("headRefOid") or "",
        url=payload.get("url") or "",
        state=payload.get("state") or "UNKNOWN",
        is_draft=bool(payload.get("isDraft")),
        merge_state_status=payload.get("mergeStateStatus"),
    )


def select_issue_identifier(
    snapshot: PullRequestSnapshot, override: str | None = None
) -> str | None:
    if override:
        return override.strip() or None
    explicit_identifier = linear_api.extract_linked_issue_identifier(
        snapshot.body,
        snapshot.title,
        snapshot.head_ref_name,
    )
    if explicit_identifier:
        return explicit_identifier
    closing_identifier = linear_api.extract_closing_issue_identifier(
        snapshot.body,
        snapshot.title,
    )
    if closing_identifier:
        return closing_identifier
    return linear_api.extract_branch_issue_identifier(snapshot.head_ref_name)


def is_thread_stale_for_current_head(
    thread: dict[str, Any], latest_commit_at_raw: str | None
) -> bool:
    latest_commit_at = review_loop.parse_timestamp(latest_commit_at_raw)
    if latest_commit_at is None:
        return False
    comment_timestamps = [
        review_loop.parse_timestamp(comment.get("created_at"))
        for comment in thread.get("comments", [])
        if comment.get("created_at")
    ]
    comment_timestamps = [timestamp for timestamp in comment_timestamps if timestamp]
    if not comment_timestamps:
        return False
    return max(comment_timestamps) < latest_commit_at


def thread_has_actionable_feedback(thread: dict[str, Any]) -> bool:
    return any(
        review_loop.is_actionable_thread_comment(str(comment.get("body") or ""))
        for comment in thread.get("comments", [])
    )


def collect_resolvable_thread_ids(summary: review_loop.ReviewSummary) -> list[str]:
    actionable_thread_ids = {
        str(thread["id"])
        for thread in summary.actionable_threads
        if thread.get("id") is not None
    }
    resolvable: list[str] = []
    for thread in summary.observed_threads:
        thread_id = thread.get("id")
        if thread_id is None:
            continue
        if thread.get("is_resolved"):
            continue
        if thread.get("is_outdated"):
            resolvable.append(str(thread_id))
            continue
        if str(thread_id) in actionable_thread_ids:
            continue
        if thread_has_actionable_feedback(thread):
            continue
        resolvable.append(str(thread_id))
    return resolvable


def resolve_review_thread(thread_id: str) -> None:
    command = [
        "gh",
        "api",
        "graphql",
        "-F",
        f"threadId={thread_id}",
        "-f",
        f"query={RESOLVE_REVIEW_THREAD_MUTATION}",
    ]
    try:
        review_loop.require_command(command, cwd=review_loop.repo_root())
    except review_loop.CommandError as exc:
        message = str(exc)
        if "Resource not accessible by integration" in message:
            raise ValueError(
                "Resolving Devin review threads from GitHub Actions requires "
                f"the repo secret {REVIEW_BRIDGE_TOKEN_SECRET} to contain a "
                "GitHub token with pull request write access."
            ) from exc
        raise


def determine_gate_decision(
    snapshot: PullRequestSnapshot,
    summary: review_loop.ReviewSummary,
    issue_identifier: str | None,
) -> GateDecision:
    if summary.review_state == "action_required":
        description = "Devin requested changes on the current head"
        if (summary.merge_state_status or "").strip() in review_loop.REFRESH_REQUIRED_MERGE_STATES:
            description += " and the PR branch also requires a refresh against current main"
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="failure",
            description=description,
            review_state=summary.review_state,
            target_state=DEFAULT_REWORK_STATE,
        )
    if summary.review_state == "branch_refresh_required":
        merge_state = (summary.merge_state_status or "").strip() or "BEHIND"
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="failure",
            description=(
                f"PR branch is {merge_state.lower()} current main and must be refreshed before merge"
            ),
            review_state=summary.review_state,
            target_state=DEFAULT_REFRESH_REQUIRED_STATE,
        )
    if summary.review_state == "review_complete":
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="success",
            description="Devin review is complete and no actionable feedback remains",
            review_state=summary.review_state,
            target_state=None,
        )
    if snapshot.state != "OPEN" or snapshot.is_draft:
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="pending",
            description="Waiting for an open ready PR head to review",
            review_state=summary.review_state,
            target_state=None,
        )
    description = "Waiting for initial Devin review on the current head"
    return GateDecision(
        issue_identifier=issue_identifier,
        status_state="pending",
        description=description,
        review_state=summary.review_state,
        target_state=None,
    )


def build_actionable_feedback_notes(summary: review_loop.ReviewSummary) -> list[str]:
    notes: list[str] = []
    for thread in summary.actionable_threads:
        thread_id = str(thread.get("id") or "").strip()
        path = str(thread.get("path") or "").strip()
        latest_comment = (thread.get("comments") or [{}])[-1]
        url = str(latest_comment.get("url") or "").strip()
        body = str(latest_comment.get("body") or "").strip()
        summary_line = body.splitlines()[-1].strip() if body else "Actionable Devin thread"
        line_bits: list[str] = []
        if thread_id:
            line_bits.append(f"thread `{thread_id}`")
        if path:
            line_bits.append(f"path `{path}`")
        if url:
            line_bits.append(url)
        prefix = "Actionable Devin thread"
        if line_bits:
            prefix += ": " + " | ".join(line_bits)
        notes.append(f"{prefix} - {summary_line}")
    for review in summary.actionable_reviews:
        author = str(review.get("author") or "").strip()
        state = str(review.get("state") or "").strip()
        url = str(review.get("url") or "").strip()
        parts = [part for part in [author, state, url] if part]
        notes.append("Actionable Devin review: " + " | ".join(parts))
    return notes


def build_branch_refresh_notes(summary: review_loop.ReviewSummary) -> list[str]:
    merge_state = (summary.merge_state_status or "").strip() or "BEHIND"
    return [
        (
            f"Branch refresh required: the PR is `{merge_state}` against the current "
            "base branch. Refresh against the latest `origin/main`, rerun the smallest "
            "relevant validation, push, and rerun `scripts/symphony/pr_handoff.py`."
        )
    ]


def sync_followup_workpad_best_effort(
    issue_identifier: str,
    summary: review_loop.ReviewSummary,
    *,
    target_state: str | None,
) -> str | None:
    try:
        issue = linear_api.fetch_issue(issue_identifier)
        issue_title = str(issue.get("title") or issue_identifier)
        existing = linear_api.find_workpad_comment(issue_identifier)
        validation: list[str] = []
        current_status = "rework"
        if summary.actionable_reviews or summary.actionable_threads:
            validation.append(
                "GitHub review bridge moved the issue to `Rework` because actionable Devin feedback remains on the current PR head."
            )
        if target_state == DEFAULT_REFRESH_REQUIRED_STATE:
            current_status = "refresh_required"
        if (summary.merge_state_status or "").strip() in review_loop.REFRESH_REQUIRED_MERGE_STATES:
            validation.append(
                "GitHub review bridge moved the issue to `Refresh Required` because the PR branch must be refreshed against the latest `main` before merge."
            )
        if not validation:
            validation.append(
                "GitHub review bridge moved the issue to `Rework` for follow-up action on the current PR head."
            )
        review_handoff_notes = build_actionable_feedback_notes(summary)
        review_handoff_notes.extend(build_branch_refresh_notes(summary))
        merged = linear_api.merge_workpad_body(
            existing.get("body") if existing else None,
            issue_identifier=issue_identifier,
            issue_title=issue_title,
            current_status=current_status,
            validation=validation,
            review_handoff_notes=review_handoff_notes,
        )
        linear_api.upsert_workpad_comment(issue_identifier, merged)
    except Exception as exc:
        return f"workpad sync failed: {exc}"
    return None


def process_pull_request(
    repo: str,
    pr_number: int,
    *,
    issue_override: str | None = None,
    reviewers: list[str] | None = None,
) -> dict[str, Any]:
    selected_reviewers = reviewers or list(DEFAULT_REVIEWERS)
    snapshot = fetch_pr_snapshot(repo, pr_number)
    summary = review_loop.fetch_pr_summary(repo, snapshot.number, selected_reviewers)
    issue_identifier = select_issue_identifier(snapshot, issue_override)

    resolved_thread_ids: list[str] = []
    for thread_id in collect_resolvable_thread_ids(summary):
        resolve_review_thread(thread_id)
        resolved_thread_ids.append(thread_id)

    if resolved_thread_ids:
        snapshot = fetch_pr_snapshot(repo, snapshot.number)
        summary = review_loop.fetch_pr_summary(repo, snapshot.number, selected_reviewers)

    decision = determine_gate_decision(snapshot, summary, issue_identifier)
    set_commit_status(
        repo,
        snapshot.head_oid,
        decision.status_state,
        decision.description,
        target_url=snapshot.url,
    )

    linear_update: dict[str, Any] = {"applied": False}
    if decision.target_state and decision.issue_identifier:
        update_result = linear_api.update_issue_state(
            decision.issue_identifier, decision.target_state
        )
        linear_update = {
            "applied": bool(update_result["changed"]),
            "previous_state": update_result["previous_state"],
            "current_state": update_result["current_state"],
            "issue_identifier": decision.issue_identifier,
        }
        workpad_warning = sync_followup_workpad_best_effort(
            decision.issue_identifier,
            summary,
            target_state=decision.target_state,
        )
    else:
        workpad_warning = None

    result = {
        "pull_request": asdict(snapshot),
        "review_summary": {
            "pr_number": summary.pr_number,
            "review_state": summary.review_state,
            "review_decision": summary.review_decision,
            "merge_state_status": summary.merge_state_status,
            "actionable_reviews": len(summary.actionable_reviews),
            "actionable_threads": len(summary.actionable_threads),
            "stale_reviews": len(summary.stale_reviews),
        },
        "decision": asdict(decision),
        "resolved_threads": resolved_thread_ids,
        "linear_update": linear_update,
        "status_context": CHECK_CONTEXT,
    }
    if workpad_warning:
        result["workpad_sync_warning"] = workpad_warning
    if issue_identifier and trace.is_enabled():
        run_manifest = trace.ensure_run(
            issue_id=issue_identifier,
            run_kind="review_bridge",
            branch=snapshot.head_ref_name,
            pr_number=snapshot.number,
        )
        artifact = trace.capture_json_artifact(
            issue_id=issue_identifier,
            run_id=run_manifest["run_id"],
            artifact_type="review_bridge_result",
            label="Devin Review Gate Result",
            payload=result,
            filename="devin_review_gate_result.json",
        )
        trace.capture_event(
            issue_id=issue_identifier,
            run_id=run_manifest["run_id"],
            actor="Bridge",
            stage="devin_review_gate",
            summary="Processed GitHub + Devin review state for the current PR head",
            decision=decision.review_state,
            decision_rationale=decision.description,
            artifact_refs=[artifact["artifact_id"]],
            metadata={"target_state": decision.target_state},
        )
        trace.finalize_run(
            issue_id=issue_identifier,
            run_id=run_manifest["run_id"],
            state_end=linear_update.get("current_state") or decision.target_state,
            metadata={"status_context": CHECK_CONTEXT},
        )
    return result


def set_commit_status(
    repo: str,
    sha: str,
    state: str,
    description: str,
    *,
    target_url: str | None = None,
) -> None:
    command = [
        "gh",
        "api",
        f"repos/{repo}/statuses/{sha}",
        "-f",
        f"state={state}",
        "-f",
        f"context={CHECK_CONTEXT}",
        "-f",
        f"description={description}",
    ]
    if target_url:
        command.extend(["-f", f"target_url={target_url}"])
    review_loop.require_command(command, cwd=review_loop.repo_root())


def main() -> int:
    args = parse_args()
    repo = args.repo or "/".join(review_loop.infer_repo())
    reviewers = args.reviewers or list(DEFAULT_REVIEWERS)
    result = process_pull_request(
        repo,
        args.pr if args.pr is not None else review_loop.infer_pr_number(),
        issue_override=args.issue,
        reviewers=reviewers,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
