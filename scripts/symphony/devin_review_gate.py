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
    from scripts.symphony import linear_api, review_loop
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import linear_api, review_loop


CHECK_CONTEXT = "devin-review-gate"
DEFAULT_REVIEWERS = ("devin-ai-integration[bot]",)
DEFAULT_REWORK_STATE = "Rework"
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
    )


def select_issue_identifier(
    snapshot: PullRequestSnapshot, override: str | None = None
) -> str | None:
    if override:
        return override.strip() or None
    identifiers = linear_api.extract_issue_identifiers(
        snapshot.body,
        snapshot.title,
        snapshot.head_ref_name,
    )
    return identifiers[0] if identifiers else None


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
        if thread.get("is_resolved") or thread.get("is_outdated"):
            continue
        if str(thread_id) in actionable_thread_ids and not is_thread_stale_for_current_head(
            thread, summary.latest_commit_at
        ):
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
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="failure",
            description="Devin requested changes on the current head",
            review_state=summary.review_state,
            target_state=DEFAULT_REWORK_STATE,
        )
    if summary.review_state == "clean":
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="success",
            description="Fresh Devin review is clean on the current head",
            review_state=summary.review_state,
            target_state=None,
        )
    if summary.review_state == "pending_rereview":
        return GateDecision(
            issue_identifier=issue_identifier,
            status_state="success",
            description="Initial Devin review is resolved; no actionable feedback remains",
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
    snapshot = fetch_pr_snapshot(repo, args.pr)
    summary = review_loop.fetch_pr_summary(repo, snapshot.number, reviewers)
    issue_identifier = select_issue_identifier(snapshot, args.issue)

    resolved_thread_ids: list[str] = []
    for thread_id in collect_resolvable_thread_ids(summary):
        resolve_review_thread(thread_id)
        resolved_thread_ids.append(thread_id)

    if resolved_thread_ids:
        snapshot = fetch_pr_snapshot(repo, snapshot.number)
        summary = review_loop.fetch_pr_summary(repo, snapshot.number, reviewers)

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

    result = {
        "pull_request": asdict(snapshot),
        "review_summary": {
            "pr_number": summary.pr_number,
            "review_state": summary.review_state,
            "review_decision": summary.review_decision,
            "actionable_reviews": len(summary.actionable_reviews),
            "actionable_threads": len(summary.actionable_threads),
            "stale_reviews": len(summary.stale_reviews),
        },
        "decision": asdict(decision),
        "resolved_threads": resolved_thread_ids,
        "linear_update": linear_update,
        "status_context": CHECK_CONTEXT,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
