#!/usr/bin/env python3
"""Codex and GitHub review helpers for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

try:
    from scripts.symphony import runtime_config, telemetry
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.symphony import runtime_config, telemetry


DEFAULT_BASE_BRANCH = "origin/main"
DEFAULT_REVIEWERS = ("devin-ai-integration[bot]",)
DEFAULT_CODEX_REVIEW_TIMEOUT_SECONDS = 300
DEFAULT_REVIEW_PROMPT = (
    "Review this branch for correctness, regressions, missing tests, scope drift, "
    "and missing validation evidence. Focus on concrete bugs, behavioral risks, "
    "missing tests, and workflow violations. If no material findings remain, say so explicitly."
)
ACTIONABLE_SUMMARY_PATTERNS = (
    re.compile(
        r"\bfound\s+(?P<count>\d+)\s+(?:new\s+)?potential issues?\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?P<count>\d+)\s+(?:new\s+)?potential issues?\s+found\b", re.IGNORECASE
    ),
)

GRAPHQL_QUERY = """
query($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      number
      url
      state
      isDraft
      reviewDecision
      headRefOid
      commits(last: 1) {
        nodes {
          commit {
            oid
            committedDate
          }
        }
      }
      reviews(last: 100) {
        nodes {
          author {
            login
          }
          state
          submittedAt
          body
          url
        }
      }
      reviewThreads(first: 100) {
        nodes {
          isResolved
          isOutdated
          path
          comments(first: 100) {
            nodes {
              author {
                login
              }
              body
              createdAt
              url
              line
              originalLine
            }
          }
        }
      }
    }
  }
}
"""


@dataclass
class ReviewSummary:
    pr_number: int
    pr_url: str
    pr_state: str
    review_state: str
    review_decision: str | None
    head_oid: str | None
    latest_commit_at: str | None
    reviewers: list[str]
    actionable_reviews: list[dict[str, Any]]
    actionable_threads: list[dict[str, Any]]
    stale_reviews: list[dict[str, Any]]
    observed_reviews: list[dict[str, Any]]
    observed_threads: list[dict[str, Any]]


class CommandError(RuntimeError):
    """Raised when a required shell command fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    codex_review_parser = subparsers.add_parser(
        "codex-review",
        help="Run a non-interactive Codex review and store artifacts locally.",
    )
    codex_review_parser.add_argument("--base", default=DEFAULT_BASE_BRANCH)
    codex_review_parser.add_argument(
        "--issue", help="Linear issue identifier for telemetry."
    )
    codex_review_parser.add_argument(
        "--artifact-dir",
        default=".codex/review_artifacts",
        help="Directory where Codex review artifacts should be written.",
    )
    codex_review_parser.add_argument(
        "--prompt",
        default=DEFAULT_REVIEW_PROMPT,
        help="Custom instructions for the Codex review pass.",
    )
    codex_review_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_CODEX_REVIEW_TIMEOUT_SECONDS,
        help="Maximum runtime for the local Codex review gate.",
    )

    review_parser = subparsers.add_parser(
        "status",
        help="Inspect GitHub PR review feedback for the current head.",
    )
    review_parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form.")
    review_parser.add_argument(
        "--pr", type=int, help="Pull request number. Defaults to current branch PR."
    )
    review_parser.add_argument("--issue", help="Linear issue identifier for telemetry.")
    review_parser.add_argument(
        "--reviewer",
        action="append",
        dest="reviewers",
        help="Reviewer login to include. Repeat for multiple reviewers.",
    )

    return parser.parse_args()


def run_command(
    command: list[str],
    *,
    cwd: pathlib.Path | None = None,
    stdin: str | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", errors="replace")
        )
        stderr = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode("utf-8", errors="replace")
        )
        stderr = (
            stderr + "\n" if stderr else ""
        ) + f"Timed out after {timeout_seconds} seconds."
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def require_command(
    command: list[str],
    *,
    cwd: pathlib.Path | None = None,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = run_command(command, cwd=cwd, stdin=stdin)
    if completed.returncode != 0:
        raise CommandError(
            f"command failed: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_remote(remote_url: str) -> tuple[str, str]:
    cleaned = remote_url.strip()
    match = re.search(
        r"github\.com[:/](?P<owner>[^/]+)/(?P<name>[^/.]+?)(?:\.git)?$", cleaned
    )
    if not match:
        raise ValueError(f"could not parse GitHub remote URL: {remote_url}")
    return match.group("owner"), match.group("name")


def infer_repo() -> tuple[str, str]:
    remote = require_command(
        ["git", "remote", "get-url", "origin"], cwd=repo_root()
    ).stdout.strip()
    return parse_remote(remote)


def infer_pr_number() -> int:
    response = require_command(
        ["gh", "pr", "view", "--json", "number"],
        cwd=repo_root(),
    )
    payload = json.loads(response.stdout)
    return int(payload["number"])


def parse_timestamp(raw_value: str | None) -> dt.datetime | None:
    if not raw_value:
        return None
    normalized = raw_value.replace("Z", "+00:00")
    return dt.datetime.fromisoformat(normalized)


def expand_reviewer_aliases(reviewers: list[str]) -> set[str]:
    expanded: set[str] = set()
    for reviewer in reviewers:
        normalized = reviewer.strip()
        if not normalized:
            continue
        expanded.add(normalized)
        if normalized.endswith("[bot]"):
            expanded.add(normalized.removesuffix("[bot]"))
        else:
            expanded.add(f"{normalized}[bot]")
    return expanded


def is_actionable_review_body(body: str) -> bool:
    normalized = re.sub(r"\s+", " ", body).strip()
    if not normalized:
        return False
    for pattern in ACTIONABLE_SUMMARY_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return int(match.group("count")) > 0
    return False


def evaluate_review_state(
    pull_request: dict[str, Any], reviewer_logins: set[str]
) -> ReviewSummary:
    commit_nodes = pull_request.get("commits", {}).get("nodes") or [{}]
    latest_commit = commit_nodes[0].get("commit", {})
    latest_commit_at = parse_timestamp(latest_commit.get("committedDate"))
    latest_commit_at_raw = latest_commit.get("committedDate")

    observed_reviews: list[dict[str, Any]] = []
    observed_threads: list[dict[str, Any]] = []
    actionable_reviews: list[dict[str, Any]] = []
    actionable_threads: list[dict[str, Any]] = []
    stale_reviews: list[dict[str, Any]] = []
    fresh_threads: list[dict[str, Any]] = []
    latest_fresh_reviews_by_author: dict[str, dict[str, Any]] = {}

    for review in pull_request.get("reviews", {}).get("nodes", []):
        author_login = ((review.get("author") or {}).get("login") or "").strip()
        if author_login not in reviewer_logins:
            continue
        review_summary = {
            "author": author_login,
            "state": review.get("state"),
            "submitted_at": review.get("submittedAt"),
            "body": (review.get("body") or "").strip(),
            "url": review.get("url"),
        }
        observed_reviews.append(review_summary)
        submitted_at = parse_timestamp(review.get("submittedAt"))
        if latest_commit_at and submitted_at and submitted_at < latest_commit_at:
            stale_reviews.append(review_summary)
            continue
        latest_review = latest_fresh_reviews_by_author.get(author_login)
        latest_review_at = (
            parse_timestamp(latest_review["submitted_at"]) if latest_review else None
        )
        if latest_review is None or (
            submitted_at is not None
            and (latest_review_at is None or submitted_at >= latest_review_at)
        ):
            latest_fresh_reviews_by_author[author_login] = review_summary

    for review_summary in latest_fresh_reviews_by_author.values():
        if review_summary["state"] == "CHANGES_REQUESTED":
            actionable_reviews.append(review_summary)
        elif review_summary["state"] == "COMMENTED" and is_actionable_review_body(
            review_summary["body"]
        ):
            actionable_reviews.append(review_summary)

    for thread in pull_request.get("reviewThreads", {}).get("nodes", []):
        target_comments = []
        for comment in thread.get("comments", {}).get("nodes", []):
            author_login = ((comment.get("author") or {}).get("login") or "").strip()
            if author_login in reviewer_logins:
                target_comments.append(
                    {
                        "author": author_login,
                        "body": (comment.get("body") or "").strip(),
                        "created_at": comment.get("createdAt"),
                        "line": comment.get("line"),
                        "original_line": comment.get("originalLine"),
                        "url": comment.get("url"),
                    }
                )
        if not target_comments:
            continue
        thread_summary = {
            "path": thread.get("path"),
            "is_resolved": bool(thread.get("isResolved")),
            "is_outdated": bool(thread.get("isOutdated")),
            "comments": target_comments,
        }
        observed_threads.append(thread_summary)
        comment_timestamps = [
            parse_timestamp(comment["created_at"])
            for comment in target_comments
            if comment["created_at"]
        ]
        latest_thread_comment_at = max(comment_timestamps, default=None)
        if latest_commit_at is None or (
            latest_thread_comment_at is not None
            and latest_thread_comment_at >= latest_commit_at
        ):
            fresh_threads.append(thread_summary)
        if not thread_summary["is_resolved"] and not thread_summary["is_outdated"]:
            actionable_threads.append(thread_summary)

    if pull_request.get("state") != "OPEN":
        review_state = pull_request.get("state", "UNKNOWN").lower()
    elif actionable_reviews or actionable_threads:
        review_state = "action_required"
    else:
        fresh_reviews = [
            review for review in observed_reviews if review not in stale_reviews
        ]
        if fresh_reviews or fresh_threads:
            review_state = "clean"
        elif observed_reviews or observed_threads:
            review_state = "pending_rereview"
        else:
            review_state = "pending_initial_review"

    return ReviewSummary(
        pr_number=int(pull_request["number"]),
        pr_url=pull_request["url"],
        pr_state=pull_request["state"],
        review_state=review_state,
        review_decision=pull_request.get("reviewDecision"),
        head_oid=pull_request.get("headRefOid"),
        latest_commit_at=latest_commit_at_raw,
        reviewers=sorted(reviewer_logins),
        actionable_reviews=actionable_reviews,
        actionable_threads=actionable_threads,
        stale_reviews=stale_reviews,
        observed_reviews=observed_reviews,
        observed_threads=observed_threads,
    )


def fetch_pr_summary(
    repo: str | None, pr_number: int | None, reviewer_logins: list[str]
) -> ReviewSummary:
    owner, name = parse_repo_argument(repo) if repo else infer_repo()
    number = pr_number if pr_number is not None else infer_pr_number()
    response = require_command(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={number}",
            "-f",
            f"query={GRAPHQL_QUERY}",
        ],
        cwd=repo_root(),
    )
    payload = json.loads(response.stdout)
    pull_request = payload["data"]["repository"]["pullRequest"]
    return evaluate_review_state(pull_request, expand_reviewer_aliases(reviewer_logins))


def parse_repo_argument(repo: str) -> tuple[str, str]:
    parts = repo.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected repo in OWNER/REPO form, got: {repo}")
    return parts[0], parts[1]


def current_branch() -> str:
    return require_command(
        ["git", "branch", "--show-current"],
        cwd=repo_root(),
    ).stdout.strip()


def current_commit() -> str:
    return require_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
    ).stdout.strip()


def sanitize_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def emit_review_telemetry(
    *,
    event_type: str,
    message: str,
    issue: str | None = None,
    pr: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    event = telemetry.build_event(
        event_type=event_type,
        message=message,
        issue=issue,
        pr=pr,
        details=details or {},
    )
    telemetry.write_event(telemetry.default_telemetry_root(), event)


def run_codex_review(
    base_branch: str,
    artifact_dir: str,
    prompt: str,
    issue: str | None,
    timeout_seconds: int,
) -> int:
    root = repo_root()
    branch = current_branch()
    commit_sha = current_commit()
    timestamp = utc_timestamp()
    safe_branch = sanitize_path_component(branch)
    target_dir = root / artifact_dir / safe_branch
    target_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = target_dir / f"{timestamp}-codex-review.jsonl"
    message_path = target_dir / f"{timestamp}-codex-review.md"
    stderr_path = target_dir / f"{timestamp}-codex-review.stderr.txt"
    manifest_path = target_dir / "latest.json"

    uses_generic_exec = bool(prompt.strip() and prompt != DEFAULT_REVIEW_PROMPT)
    if uses_generic_exec:
        command = [
            *runtime_config.build_codex_command(
                "review",
                [
                    "exec",
                    "--json",
                    "-o",
                    str(message_path),
                    "-",
                ],
            ),
        ]
        stdin = f"Review the current branch against {base_branch}. {prompt}".strip()
    else:
        command = [
            *runtime_config.build_codex_command(
                "review",
                [
                    "exec",
                    "review",
                    "--base",
                    base_branch,
                    "--json",
                    "-o",
                    str(message_path),
                ],
            ),
        ]
        stdin = None
    completed = run_command(
        command, cwd=root, stdin=stdin, timeout_seconds=timeout_seconds
    )

    jsonl_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    manifest = {
        "timestamp": timestamp,
        "branch": branch,
        "commit": commit_sha,
        "base_branch": base_branch,
        "prompt": prompt,
        "profile": "review",
        "review_driver": "generic_exec" if uses_generic_exec else "native_review",
        "timeout_seconds": timeout_seconds,
        "command": command,
        "returncode": completed.returncode,
        "jsonl_path": jsonl_path.relative_to(root).as_posix(),
        "message_path": message_path.relative_to(root).as_posix(),
        "stderr_path": stderr_path.relative_to(root).as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    emit_review_telemetry(
        event_type="local_codex_review_completed",
        message="Local Codex review gate completed",
        issue=issue,
        details={
            "base_branch": base_branch,
            "branch": branch,
            "commit": commit_sha,
            "returncode": str(completed.returncode),
            "timeout_seconds": str(timeout_seconds),
            "message_path": manifest["message_path"],
            "jsonl_path": manifest["jsonl_path"],
        },
    )
    print(json.dumps(manifest, indent=2))
    return completed.returncode


def status_command(args: argparse.Namespace) -> int:
    reviewers = args.reviewers or list(DEFAULT_REVIEWERS)
    summary = fetch_pr_summary(args.repo, args.pr, reviewers)
    emit_review_telemetry(
        event_type=f"github_review_{summary.review_state}",
        message=f"GitHub review status: {summary.review_state}",
        issue=args.issue,
        pr=summary.pr_number,
        details={
            "review_decision": summary.review_decision or "",
            "head_oid": summary.head_oid or "",
            "reviewers": ",".join(summary.reviewers),
        },
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.review_state in {"clean", "merged"} else 1


def main() -> int:
    args = parse_args()
    if args.command == "codex-review":
        return run_codex_review(
            args.base,
            args.artifact_dir,
            args.prompt,
            args.issue,
            args.timeout_seconds,
        )
    if args.command == "status":
        return status_command(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
