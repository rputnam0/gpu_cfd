#!/usr/bin/env python3
"""Sync Devin review outcomes from GitHub events back into Linear states."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

try:
    from scripts.symphony import review_loop
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import review_loop


DEFAULT_REVIEWERS = ("devin-ai-integration[bot]",)
DEFAULT_REWORK_STATE = "Rework"
DEFAULT_READY_TO_MERGE_STATE = "Ready to Merge"
READY_MERGEABLE_STATUSES = {"CLEAN", "HAS_HOOKS"}
ISSUE_IDENTIFIER_PATTERN = re.compile(r"\bPRO-\d+\b")
LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
PR_FIELDS = ",".join(
    [
        "number",
        "title",
        "body",
        "headRefName",
        "url",
        "state",
        "isDraft",
        "mergeable",
        "mergeStateStatus",
        "reviewDecision",
    ]
)
LINEAR_ISSUE_QUERY = """
query($id: String!) {
  issue(id: $id) {
    id
    identifier
    title
    state {
      id
      name
    }
    team {
      key
      states {
        nodes {
          id
          name
        }
      }
    }
  }
}
"""
LINEAR_ISSUE_UPDATE_MUTATION = """
mutation($id: String!, $stateId: String!) {
  issueUpdate(id: $id, input: { stateId: $stateId }) {
    success
    issue {
      id
      identifier
      state {
        id
        name
      }
    }
  }
}
"""
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


@dataclass
class PullRequestSnapshot:
    number: int
    title: str
    body: str
    head_ref_name: str
    url: str
    state: str
    is_draft: bool
    mergeable: str | None
    merge_state_status: str | None
    review_decision: str | None


@dataclass
class BridgeDecision:
    issue_identifier: str | None
    target_state: str | None
    reason: str
    review_state: str


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate and print the decision without updating Linear.",
    )
    return parser.parse_args()


def extract_issue_identifiers(*texts: str) -> list[str]:
    identifiers: list[str] = []
    for text in texts:
        for match in ISSUE_IDENTIFIER_PATTERN.findall(text or ""):
            if match not in identifiers:
                identifiers.append(match)
    return identifiers


def select_issue_identifier(
    snapshot: PullRequestSnapshot, override: str | None = None
) -> str | None:
    if override:
        return override.strip() or None
    identifiers = extract_issue_identifiers(
        snapshot.body,
        snapshot.title,
        snapshot.head_ref_name,
    )
    return identifiers[0] if identifiers else None


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
        url=payload.get("url") or "",
        state=payload.get("state") or "UNKNOWN",
        is_draft=bool(payload.get("isDraft")),
        mergeable=payload.get("mergeable"),
        merge_state_status=payload.get("mergeStateStatus"),
        review_decision=payload.get("reviewDecision"),
    )


def determine_bridge_decision(
    snapshot: PullRequestSnapshot,
    summary: review_loop.ReviewSummary,
    issue_identifier: str | None,
) -> BridgeDecision:
    if not issue_identifier:
        return BridgeDecision(
            issue_identifier=None,
            target_state=None,
            reason="no_linked_linear_issue",
            review_state=summary.review_state,
        )
    if summary.review_state == "action_required":
        return BridgeDecision(
            issue_identifier=issue_identifier,
            target_state=DEFAULT_REWORK_STATE,
            reason="devin_actionable_feedback",
            review_state=summary.review_state,
        )
    if (
        summary.review_state == "clean"
        and snapshot.state == "OPEN"
        and not snapshot.is_draft
        and snapshot.mergeable == "MERGEABLE"
        and snapshot.merge_state_status in READY_MERGEABLE_STATUSES
    ):
        return BridgeDecision(
            issue_identifier=issue_identifier,
            target_state=DEFAULT_READY_TO_MERGE_STATE,
            reason="clean_head_is_mergeable",
            review_state=summary.review_state,
        )
    return BridgeDecision(
        issue_identifier=issue_identifier,
        target_state=None,
        reason="no_state_transition",
        review_state=summary.review_state,
    )


def linear_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    token = os.environ.get("LINEAR_API_KEY")
    if not token:
        raise ValueError("LINEAR_API_KEY is required")
    request = urllib.request.Request(
        LINEAR_GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(
            f"Linear GraphQL request failed with {exc.code}: {detail}"
        ) from exc
    if payload.get("errors"):
        raise ValueError(f"Linear GraphQL error: {payload['errors']}")
    return payload["data"]


def fetch_linear_issue(issue_identifier: str) -> dict[str, Any]:
    data = linear_graphql(LINEAR_ISSUE_QUERY, {"id": issue_identifier})
    issue = data.get("issue")
    if issue is None:
        raise ValueError(f"Linear issue not found: {issue_identifier}")
    return issue


def resolve_linear_state_id(issue: dict[str, Any], target_state_name: str) -> str:
    states = issue.get("team", {}).get("states", {}).get("nodes", [])
    for state in states:
        if state.get("name") == target_state_name:
            return str(state["id"])
    raise ValueError(
        f"state {target_state_name!r} not found on team {issue.get('team', {}).get('key', '')}"
    )


def update_linear_issue_state(
    issue_identifier: str, target_state_name: str
) -> dict[str, Any]:
    issue = fetch_linear_issue(issue_identifier)
    current_state = issue.get("state", {}).get("name")
    if current_state == target_state_name:
        return {
            "changed": False,
            "issue": issue,
            "previous_state": current_state,
            "current_state": current_state,
        }
    state_id = resolve_linear_state_id(issue, target_state_name)
    mutation_data = linear_graphql(
        LINEAR_ISSUE_UPDATE_MUTATION,
        {"id": issue["id"], "stateId": state_id},
    )
    updated_issue = mutation_data["issueUpdate"]["issue"]
    return {
        "changed": True,
        "issue": updated_issue,
        "previous_state": current_state,
        "current_state": updated_issue.get("state", {}).get("name"),
    }


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
        if str(thread_id) in actionable_thread_ids:
            continue
        resolvable.append(str(thread_id))
    return resolvable


def resolve_review_thread(thread_id: str) -> None:
    review_loop.require_command(
        [
            "gh",
            "api",
            "graphql",
            "-F",
            f"threadId={thread_id}",
            "-f",
            f"query={RESOLVE_REVIEW_THREAD_MUTATION}",
        ],
        cwd=review_loop.repo_root(),
    )


def main() -> int:
    args = parse_args()
    reviewers = args.reviewers or list(DEFAULT_REVIEWERS)
    snapshot = fetch_pr_snapshot(args.repo, args.pr)
    summary = review_loop.fetch_pr_summary(args.repo, snapshot.number, reviewers)
    issue_identifier = select_issue_identifier(snapshot, args.issue)
    decision = determine_bridge_decision(snapshot, summary, issue_identifier)

    result: dict[str, Any] = {
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
    }

    resolved_thread_ids: list[str] = []
    if not args.dry_run and summary.review_state == "clean":
        resolvable_thread_ids = collect_resolvable_thread_ids(summary)
        for thread_id in resolvable_thread_ids:
            resolve_review_thread(thread_id)
            resolved_thread_ids.append(thread_id)
        if resolved_thread_ids:
            snapshot = fetch_pr_snapshot(args.repo, args.pr)
            summary = review_loop.fetch_pr_summary(
                args.repo, snapshot.number, reviewers
            )
            decision = determine_bridge_decision(snapshot, summary, issue_identifier)
            result["pull_request"] = asdict(snapshot)
            result["review_summary"] = {
                "pr_number": summary.pr_number,
                "review_state": summary.review_state,
                "review_decision": summary.review_decision,
                "actionable_reviews": len(summary.actionable_reviews),
                "actionable_threads": len(summary.actionable_threads),
                "stale_reviews": len(summary.stale_reviews),
            }
            result["decision"] = asdict(decision)
    result["resolved_threads"] = resolved_thread_ids

    if args.dry_run or not decision.target_state:
        result["linear_update"] = {"applied": False}
        print(json.dumps(result, indent=2))
        return 0

    linear_update = update_linear_issue_state(
        decision.issue_identifier or "",
        decision.target_state,
    )
    result["linear_update"] = {
        "applied": bool(linear_update["changed"]),
        "previous_state": linear_update["previous_state"],
        "current_state": linear_update["current_state"],
        "issue_identifier": issue_identifier,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
