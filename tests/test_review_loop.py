from __future__ import annotations

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
        self.assertEqual(summary.review_state, "clean")
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

        self.assertEqual(summary.review_state, "clean")
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

        self.assertEqual(summary.review_state, "clean")
        self.assertEqual(len(summary.actionable_reviews), 0)

    def test_pending_rereview_when_only_stale_review_exists(self) -> None:
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
        self.assertEqual(summary.review_state, "pending_rereview")
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
        self.assertEqual(summary.review_state, "clean")

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
        self.assertEqual(summary.review_state, "clean")
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
        self.assertEqual(summary.review_state, "clean")

    def test_pending_rereview_when_only_stale_resolved_threads_exist(self) -> None:
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
        self.assertEqual(summary.review_state, "pending_rereview")

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
