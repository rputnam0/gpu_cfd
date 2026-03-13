#!/usr/bin/env python3
"""Codex and GitHub review helpers for the gpu_cfd Symphony workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_BASE_BRANCH = "origin/main"
DEFAULT_REVIEWERS = ("devin-ai-integration[bot]",)
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_REVIEW_PROMPT = (
    "Review this branch for correctness, regressions, missing tests, scope drift, "
    "and missing validation evidence. Focus on concrete bugs, behavioral risks, "
    "missing tests, and workflow violations. If no material findings remain, say so explicitly."
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
        "--artifact-dir",
        default=".codex/review_artifacts",
        help="Directory where Codex review artifacts should be written.",
    )
    codex_review_parser.add_argument(
        "--prompt",
        default=DEFAULT_REVIEW_PROMPT,
        help="Custom instructions for the Codex review pass.",
    )

    for subcommand in ("status", "wait"):
        review_parser = subparsers.add_parser(
            subcommand,
            help="Inspect or wait for GitHub PR review feedback.",
        )
        review_parser.add_argument("--repo", help="GitHub repo in OWNER/REPO form.")
        review_parser.add_argument("--pr", type=int, help="Pull request number. Defaults to current branch PR.")
        review_parser.add_argument(
            "--reviewer",
            action="append",
            dest="reviewers",
            help="Reviewer login to wait for. Repeat for multiple reviewers.",
        )
        if subcommand == "wait":
            review_parser.add_argument(
                "--timeout-seconds",
                type=int,
                default=DEFAULT_TIMEOUT_SECONDS,
                help="Maximum time to wait for a fresh review on the current PR head.",
            )
            review_parser.add_argument(
                "--poll-interval-seconds",
                type=int,
                default=DEFAULT_POLL_INTERVAL_SECONDS,
                help="Polling interval while waiting for review feedback.",
            )

    return parser.parse_args()


def run_command(command: list[str], *, cwd: pathlib.Path | None = None, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed


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
    match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<name>[^/.]+?)(?:\.git)?$", cleaned)
    if not match:
        raise ValueError(f"could not parse GitHub remote URL: {remote_url}")
    return match.group("owner"), match.group("name")


def infer_repo() -> tuple[str, str]:
    remote = require_command(["git", "remote", "get-url", "origin"], cwd=repo_root()).stdout.strip()
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


def evaluate_review_state(pull_request: dict[str, Any], reviewer_logins: set[str]) -> ReviewSummary:
    latest_commit = (
        pull_request.get("commits", {})
        .get("nodes", [{}])[0]
        .get("commit", {})
    )
    latest_commit_at = parse_timestamp(latest_commit.get("committedDate"))
    latest_commit_at_raw = latest_commit.get("committedDate")

    observed_reviews: list[dict[str, Any]] = []
    observed_threads: list[dict[str, Any]] = []
    actionable_reviews: list[dict[str, Any]] = []
    actionable_threads: list[dict[str, Any]] = []
    stale_reviews: list[dict[str, Any]] = []

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
        if review_summary["state"] in {"CHANGES_REQUESTED", "COMMENTED"} and review_summary["body"]:
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
        if not thread_summary["is_resolved"] and not thread_summary["is_outdated"]:
            actionable_threads.append(thread_summary)

    if pull_request.get("state") != "OPEN":
        review_state = pull_request.get("state", "UNKNOWN").lower()
    elif actionable_reviews or actionable_threads:
        review_state = "action_required"
    elif observed_reviews or observed_threads:
        fresh_reviews = [
            review
            for review in observed_reviews
            if review not in stale_reviews
        ]
        if fresh_reviews:
            review_state = "clean"
        else:
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


def fetch_pr_summary(repo: str | None, pr_number: int | None, reviewer_logins: list[str]) -> ReviewSummary:
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
    return evaluate_review_state(pull_request, set(reviewer_logins))


def parse_repo_argument(repo: str) -> tuple[str, str]:
    parts = repo.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected repo in OWNER/REPO form, got: {repo}")
    return parts[0], parts[1]


def resolve_codex_binary() -> str:
    candidates = [
        shutil.which("codex"),
        str(pathlib.Path.home() / ".npm-global" / "bin" / "codex"),
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    raise FileNotFoundError("could not find codex on PATH or at ~/.npm-global/bin/codex")


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


def run_codex_review(base_branch: str, artifact_dir: str, prompt: str) -> int:
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

    command = [
        resolve_codex_binary(),
        "exec",
        "review",
        "--base",
        base_branch,
        "--json",
        "-o",
        str(message_path),
        "-",
    ]
    completed = run_command(command, cwd=root, stdin=prompt)

    jsonl_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    manifest = {
        "timestamp": timestamp,
        "branch": branch,
        "commit": commit_sha,
        "base_branch": base_branch,
        "prompt": prompt,
        "command": command,
        "returncode": completed.returncode,
        "jsonl_path": jsonl_path.relative_to(root).as_posix(),
        "message_path": message_path.relative_to(root).as_posix(),
        "stderr_path": stderr_path.relative_to(root).as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return completed.returncode


def status_command(args: argparse.Namespace) -> int:
    reviewers = args.reviewers or list(DEFAULT_REVIEWERS)
    summary = fetch_pr_summary(args.repo, args.pr, reviewers)
    print(json.dumps(asdict(summary), indent=2))
    return 0 if summary.review_state in {"clean", "merged"} else 1


def wait_command(args: argparse.Namespace) -> int:
    reviewers = args.reviewers or list(DEFAULT_REVIEWERS)
    deadline = time.monotonic() + args.timeout_seconds
    last_summary: ReviewSummary | None = None
    while True:
        summary = fetch_pr_summary(args.repo, args.pr, reviewers)
        last_summary = summary
        if summary.review_state in {"action_required", "clean", "merged", "closed"}:
            print(json.dumps(asdict(summary), indent=2))
            return 0 if summary.review_state in {"clean", "merged"} else 1
        if time.monotonic() >= deadline:
            payload = asdict(summary)
            payload["timed_out"] = True
            print(json.dumps(payload, indent=2))
            return 1
        time.sleep(args.poll_interval_seconds)


def main() -> int:
    args = parse_args()
    if args.command == "codex-review":
        return run_codex_review(args.base, args.artifact_dir, args.prompt)
    if args.command == "status":
        return status_command(args)
    if args.command == "wait":
        return wait_command(args)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CommandError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
