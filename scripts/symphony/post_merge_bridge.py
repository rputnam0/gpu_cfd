#!/usr/bin/env python3
"""Move merged PRs to Done in Linear and release newly unblocked dependents."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass

try:
    from scripts.symphony import linear_api, review_loop, trace
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import linear_api, review_loop, trace


DONE_STATE = "Done"
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


def main() -> int:
    args = parse_args()
    snapshot = fetch_pr_snapshot(args.repo, args.pr)
    issue_identifier = select_issue_identifier(snapshot, args.issue)
    if snapshot.state != "MERGED" or not snapshot.merged_at:
        raise ValueError(f"pull request #{snapshot.number} is not merged")
    if not issue_identifier:
        result = {
            "pull_request": asdict(snapshot),
            "issue_identifier": None,
            "skipped": True,
            "reason": "no linked Linear issue could be inferred from the pull request",
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
