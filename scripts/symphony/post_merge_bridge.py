#!/usr/bin/env python3
"""Move merged PRs to Done in Linear and release newly unblocked dependents."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

try:
    from scripts.symphony import devin_review_gate, linear_api, review_loop, trace
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import devin_review_gate, linear_api, review_loop, trace


DONE_STATE = "Done"
IN_REVIEW_STATE = "In Review"
PR_FIELDS = ",".join(
    [
        "number",
        "title",
        "body",
        "headRefName",
        "url",
        "state",
        "mergedAt",
    ]
)
OPEN_PR_FIELDS = ",".join(["number"])
RECONCILE_MAX_ATTEMPTS = 3
RECONCILE_RETRY_DELAY_SECONDS = 15.0


@dataclass(frozen=True)
class PullRequestSnapshot:
    number: int
    title: str
    body: str
    head_ref_name: str
    url: str
    state: str
    merged_at: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form.")
    parser.add_argument("--pr", type=int, help="Pull request number.")
    parser.add_argument(
        "--issue",
        help="Override the Linear issue identifier instead of inferring it from the PR.",
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
        url=payload.get("url") or "",
        state=payload.get("state") or "UNKNOWN",
        merged_at=payload.get("mergedAt"),
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
    identifiers = linear_api.extract_issue_identifiers(
        snapshot.body,
        snapshot.title,
        snapshot.head_ref_name,
    )
    return identifiers[0] if identifiers else None


def list_open_pull_request_numbers(repo: str) -> list[int]:
    response = review_loop.require_command(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            OPEN_PR_FIELDS,
        ],
        cwd=review_loop.repo_root(),
    )
    payload = json.loads(response.stdout)
    return [int(item["number"]) for item in payload]


def collect_in_review_pull_request_candidates(
    repo: str,
    *,
    exclude_pr_number: int | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for pr_number in list_open_pull_request_numbers(repo):
        if exclude_pr_number is not None and pr_number == exclude_pr_number:
            continue
        snapshot = devin_review_gate.fetch_pr_snapshot(repo, pr_number)
        issue_identifier = devin_review_gate.select_issue_identifier(snapshot)
        if not issue_identifier:
            continue
        issue = linear_api.fetch_issue(issue_identifier)
        current_state = str((issue.get("state") or {}).get("name") or "")
        if current_state != IN_REVIEW_STATE:
            continue
        candidates.append(
            {
                "pr_number": pr_number,
                "issue_identifier": issue_identifier,
                "merge_state_status": (snapshot.merge_state_status or "").strip(),
            }
        )
    return candidates


def reconcile_open_review_pull_requests(
    repo: str,
    *,
    exclude_pr_number: int | None = None,
    max_attempts: int = RECONCILE_MAX_ATTEMPTS,
    retry_delay_seconds: float = RECONCILE_RETRY_DELAY_SECONDS,
) -> list[dict[str, object]]:
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        reconciled: list[dict[str, object]] = []
        candidates = collect_in_review_pull_request_candidates(
            repo,
            exclude_pr_number=exclude_pr_number,
        )
        if not candidates:
            return []
        refresh_candidates = [
            candidate
            for candidate in candidates
            if candidate["merge_state_status"] in review_loop.REFRESH_REQUIRED_MERGE_STATES
        ]
        for candidate in refresh_candidates:
            result = devin_review_gate.process_pull_request(
                repo,
                int(candidate["pr_number"]),
                issue_override=str(candidate["issue_identifier"]),
                reviewers=list(devin_review_gate.DEFAULT_REVIEWERS),
            )
            reconciled.append(
                {
                    "pr_number": int(candidate["pr_number"]),
                    "issue_identifier": str(candidate["issue_identifier"]),
                    "merge_state_status": candidate["merge_state_status"],
                    "decision": result["decision"],
                    "linear_update": result["linear_update"],
                }
            )
        if reconciled or attempt == attempts - 1:
            return reconciled
        time.sleep(max(0.0, retry_delay_seconds))
    return []


def main() -> int:
    args = parse_args()
    repo = args.repo or "/".join(review_loop.infer_repo())
    snapshot = fetch_pr_snapshot(args.repo, args.pr)
    issue_identifier = select_issue_identifier(snapshot, args.issue)
    if snapshot.state != "MERGED" or not snapshot.merged_at:
        raise ValueError(f"pull request #{snapshot.number} is not merged")
    reconciled_open_reviews = reconcile_open_review_pull_requests(
        repo,
        exclude_pr_number=snapshot.number,
    )
    if not issue_identifier:
        result = {
            "pull_request": asdict(snapshot),
            "issue_identifier": None,
            "skipped": True,
            "reason": "no linked Linear issue could be inferred from the pull request",
            "reconciled_open_reviews": reconciled_open_reviews,
        }
        print(json.dumps(result, indent=2))
        return 0

    done_update = linear_api.update_issue_state(issue_identifier, DONE_STATE)
    released_dependents = linear_api.release_direct_unblocked_dependents(issue_identifier)

    result = {
        "pull_request": asdict(snapshot),
        "issue_identifier": issue_identifier,
        "done_update": {
            "changed": done_update["changed"],
            "previous_state": done_update["previous_state"],
            "current_state": done_update["current_state"],
        },
        "released_dependents": released_dependents,
        "reconciled_open_reviews": reconciled_open_reviews,
    }
    if trace.is_enabled():
        trace_issue = issue_identifier or f"UNLINKED-PR-{snapshot.number}"
        run_manifest = trace.ensure_run(
            issue_id=trace_issue,
            run_kind="post_merge_bridge",
            branch=snapshot.head_ref_name,
            pr_number=snapshot.number,
        )
        artifact = trace.capture_json_artifact(
            issue_id=trace_issue,
            run_id=run_manifest["run_id"],
            artifact_type="post_merge_result",
            label="Post-merge Bridge Result",
            payload=result,
            filename="post_merge_bridge_result.json",
        )
        trace.capture_event(
            issue_id=trace_issue,
            run_id=run_manifest["run_id"],
            actor="Bridge",
            stage="post_merge",
            summary="Processed post-merge Linear transition and dependent release",
            decision=DONE_STATE,
            decision_rationale="Merged PR completed the issue lifecycle and released newly unblocked dependents",
            artifact_refs=[artifact["artifact_id"]],
        )
        trace.finalize_run(
            issue_id=trace_issue,
            run_id=run_manifest["run_id"],
            state_end=DONE_STATE if issue_identifier else None,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (review_loop.CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
