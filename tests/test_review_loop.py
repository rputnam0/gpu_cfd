from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
from unittest import mock
import unittest

from scripts.symphony import review_loop


class ParseRemoteTests(unittest.TestCase):
    def test_parses_https_remote(self) -> None:
        self.assertEqual(
            review_loop.parse_remote("https://github.com/rputnam0/gpu_cfd.git"),
            ("rputnam0", "gpu_cfd"),
        )

    def test_parses_ssh_remote(self) -> None:
        self.assertEqual(
            review_loop.parse_remote("git@github.com:rputnam0/gpu_cfd.git"),
            ("rputnam0", "gpu_cfd"),
        )

    def test_extracts_last_agent_message_from_jsonl(self) -> None:
        jsonl_text = "\n".join(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"second"}}',
            ]
        )

        self.assertEqual(review_loop.extract_last_agent_message(jsonl_text), "second")


class EvaluateReviewStateTests(unittest.TestCase):
    def make_pull_request(self) -> dict:
        return {
            "number": 17,
            "url": "https://github.com/rputnam0/gpu_cfd/pull/17",
            "state": "OPEN",
            "reviewDecision": None,
            "headRefOid": "abc123",
            "commits": {
                "nodes": [
                    {
                        "commit": {
                            "oid": "abc123",
                            "committedDate": "2026-03-13T20:00:00Z",
                        }
                    }
                ]
            },
            "reviews": {"nodes": []},
            "reviewThreads": {"nodes": []},
        }

    def test_pending_initial_review_when_no_target_feedback_exists(self) -> None:
        summary = review_loop.evaluate_review_state(
            self.make_pull_request(),
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "pending_initial_review")

    def test_handles_empty_commit_nodes_without_crashing(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["commits"]["nodes"] = []

        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )

        self.assertEqual(summary.review_state, "pending_initial_review")

    def test_normalizes_devin_aliases(self) -> None:
        self.assertEqual(
            review_loop.expand_reviewer_aliases(["devin-ai-integration[bot]"]),
            {"devin-ai-integration[bot]", "devin-ai-integration"},
        )

    def test_action_required_for_unresolved_target_thread(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviewThreads"]["nodes"].append(
            {
                "isResolved": False,
                "isOutdated": False,
                "path": "src/core.py",
                "comments": {
                    "nodes": [
                        {
                            "author": {"login": "devin-ai-integration[bot]"},
                            "body": (
                                "<!-- devin-review-comment "
                                '{"id": "BUG_pr-review-job_0001"} -->\n\n'
                                "This branch misses a required guard."
                            ),
                            "createdAt": "2026-03-13T20:01:00Z",
                            "url": "https://example.com/thread",
                            "line": 42,
                            "originalLine": 42,
                        }
                    ]
                },
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "action_required")
        self.assertEqual(len(summary.actionable_threads), 1)

    def test_clean_for_unresolved_analysis_thread(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviewThreads"]["nodes"].append(
            {
                "isResolved": False,
                "isOutdated": False,
                "path": "src/core.py",
                "comments": {
                    "nodes": [
                        {
                            "author": {"login": "devin-ai-integration[bot]"},
                            "body": (
                                "<!-- devin-review-comment "
                                '{"id": "ANALYSIS_pr-review-job_0001"} -->\n\n'
                                "This is informational."
                            ),
                            "createdAt": "2026-03-13T20:01:00Z",
                            "url": "https://example.com/thread",
                            "line": 42,
                            "originalLine": 42,
                        }
                    ]
                },
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "review_complete")
        self.assertEqual(len(summary.actionable_threads), 0)

    def test_action_required_for_fresh_changes_requested_without_body(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].append(
            {
                "author": {"login": "devin-ai-integration"},
                "state": "CHANGES_REQUESTED",
                "submittedAt": "2026-03-13T20:05:00Z",
                "body": "",
                "url": "https://example.com/review",
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            review_loop.expand_reviewer_aliases(["devin-ai-integration[bot]"]),
        )
        self.assertEqual(summary.review_state, "action_required")
        self.assertEqual(len(summary.actionable_reviews), 1)

    def test_clean_for_positive_devin_summary_review_without_bug_threads(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].append(
            {
                "author": {"login": "devin-ai-integration[bot]"},
                "state": "COMMENTED",
                "submittedAt": "2026-03-13T20:05:00Z",
                "body": "**Devin Review** found 4 new potential issues.",
                "url": "https://example.com/review",
            }
        )

        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )

        self.assertEqual(summary.review_state, "review_complete")
        self.assertEqual(len(summary.actionable_reviews), 0)

    def test_clean_for_zero_issue_devin_summary_review(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].append(
            {
                "author": {"login": "devin-ai-integration[bot]"},
                "state": "COMMENTED",
                "submittedAt": "2026-03-13T20:05:00Z",
                "body": "**Devin Review** found 0 new potential issues.",
                "url": "https://example.com/review",
            }
        )

        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )

        self.assertEqual(summary.review_state, "review_complete")
        self.assertEqual(len(summary.actionable_reviews), 0)

    def test_review_complete_when_only_stale_review_exists(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].append(
            {
                "author": {"login": "devin-ai-integration[bot]"},
                "state": "COMMENTED",
                "submittedAt": "2026-03-13T19:30:00Z",
                "body": "Please add a regression test.",
                "url": "https://example.com/review",
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "review_complete")
        self.assertEqual(len(summary.stale_reviews), 1)

    def test_clean_when_fresh_approval_has_no_open_threads(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].append(
            {
                "author": {"login": "devin-ai-integration[bot]"},
                "state": "APPROVED",
                "submittedAt": "2026-03-13T20:05:00Z",
                "body": "",
                "url": "https://example.com/review",
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "review_complete")

    def test_clean_when_latest_fresh_review_supersedes_changes_requested(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviews"]["nodes"].extend(
            [
                {
                    "author": {"login": "devin-ai-integration[bot]"},
                    "state": "CHANGES_REQUESTED",
                    "submittedAt": "2026-03-13T20:05:00Z",
                    "body": "",
                    "url": "https://example.com/review/1",
                },
                {
                    "author": {"login": "devin-ai-integration[bot]"},
                    "state": "APPROVED",
                    "submittedAt": "2026-03-13T20:06:00Z",
                    "body": "",
                    "url": "https://example.com/review/2",
                },
            ]
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "review_complete")
        self.assertEqual(len(summary.actionable_reviews), 0)

    def test_clean_when_only_resolved_target_threads_exist(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviewThreads"]["nodes"].append(
            {
                "isResolved": True,
                "isOutdated": False,
                "path": "src/core.py",
                "comments": {
                    "nodes": [
                        {
                            "author": {"login": "devin-ai-integration"},
                            "body": "Fixed in latest pass.",
                            "createdAt": "2026-03-13T20:01:00Z",
                            "url": "https://example.com/thread",
                            "line": 42,
                            "originalLine": 42,
                        }
                    ]
                },
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            review_loop.expand_reviewer_aliases(["devin-ai-integration[bot]"]),
        )
        self.assertEqual(summary.review_state, "review_complete")

    def test_review_complete_when_only_stale_resolved_threads_exist(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviewThreads"]["nodes"].append(
            {
                "isResolved": True,
                "isOutdated": False,
                "path": "src/core.py",
                "comments": {
                    "nodes": [
                        {
                            "author": {"login": "devin-ai-integration"},
                            "body": "Old resolved note.",
                            "createdAt": "2026-03-13T19:59:00Z",
                            "url": "https://example.com/thread",
                            "line": 42,
                            "originalLine": 42,
                        }
                    ]
                },
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            review_loop.expand_reviewer_aliases(["devin-ai-integration[bot]"]),
        )
        self.assertEqual(summary.review_state, "review_complete")

    def test_handles_missing_thread_comment_timestamps_without_crashing(self) -> None:
        pull_request = self.make_pull_request()
        pull_request["reviewThreads"]["nodes"].append(
            {
                "isResolved": False,
                "isOutdated": False,
                "path": "src/core.py",
                "comments": {
                    "nodes": [
                        {
                            "author": {"login": "devin-ai-integration[bot]"},
                            "body": "Primary finding.",
                            "createdAt": None,
                            "url": "https://example.com/thread/1",
                            "line": 42,
                            "originalLine": 42,
                        },
                        {
                            "author": {"login": "devin-ai-integration[bot]"},
                            "body": "Follow-up note.",
                            "createdAt": "2026-03-13T20:02:00Z",
                            "url": "https://example.com/thread/2",
                            "line": 43,
                            "originalLine": 43,
                        },
                    ]
                },
            }
        )
        summary = review_loop.evaluate_review_state(
            pull_request,
            {"devin-ai-integration[bot]"},
        )
        self.assertEqual(summary.review_state, "action_required")


class RunCodexReviewTests(unittest.TestCase):
    def test_writes_running_manifest_before_review_subprocess_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            manifest_path = (
                root
                / ".codex"
                / "review_artifacts"
                / "codex-pro-7-example"
                / "20260317T180000Z-codex-review.md"
            )

            def fake_run_command(
                command: list[str],
                *,
                cwd: pathlib.Path | None = None,
                stdin: str | None = None,
                timeout_seconds: int | None = None,
            ) -> subprocess.CompletedProcess[str]:
                latest_manifest = (
                    root
                    / ".codex"
                    / "review_artifacts"
                    / "codex-pro-7-example"
                    / "latest.json"
                )
                self.assertTrue(latest_manifest.exists())
                payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "running")
                self.assertEqual(payload["timeout_seconds"], 30)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout='{"type":"item.completed","item":{"type":"agent_message","text":"No findings"}}\n',
                    stderr="",
                )

            with mock.patch.object(review_loop, "repo_root", return_value=root), mock.patch.object(
                review_loop, "current_branch", return_value="codex/pro-7-example"
            ), mock.patch.object(
                review_loop, "current_commit", return_value="abc123"
            ), mock.patch.object(
                review_loop, "utc_timestamp", return_value="20260317T180000Z"
            ), mock.patch.object(
                review_loop.runtime_config,
                "build_codex_command",
                return_value=["codex", "exec", "review"],
            ), mock.patch.object(
                review_loop, "run_command", side_effect=fake_run_command
            ):
                returncode = review_loop.run_codex_review(
                    "origin/main",
                    ".codex/review_artifacts",
                    review_loop.DEFAULT_REVIEW_PROMPT,
                    "PRO-7",
                    30,
                )

            self.assertEqual(returncode, 0)
            latest_manifest = (
                root
                / ".codex"
                / "review_artifacts"
                / "codex-pro-7-example"
                / "latest.json"
            )
            payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["returncode"], 0)
            self.assertTrue(
                (
                    root
                    / ".codex"
                    / "review_artifacts"
                    / "codex-pro-7-example"
                    / "20260317T180000Z-codex-review.jsonl"
                ).exists()
            )
            self.assertEqual(
                (
                    root
                    / ".codex"
                    / "review_artifacts"
                    / "codex-pro-7-example"
                    / "20260317T180000Z-codex-review.md"
                ).read_text(encoding="utf-8").strip(),
                "No findings",
            )


class ActionableReviewCommentTests(unittest.TestCase):
    def test_bug_comment_metadata_is_actionable(self) -> None:
        self.assertTrue(
            review_loop.is_actionable_thread_comment(
                "<!-- devin-review-comment "
                '{"id": "BUG_pr-review-job_0001"} -->\n\n'
                "Bug details."
            )
        )

    def test_analysis_comment_metadata_is_not_actionable(self) -> None:
        self.assertFalse(
            review_loop.is_actionable_thread_comment(
                "<!-- devin-review-comment "
                '{"id": "ANALYSIS_pr-review-job_0001"} -->\n\n'
                "Informational note."
            )
        )


if __name__ == "__main__":
    unittest.main()
