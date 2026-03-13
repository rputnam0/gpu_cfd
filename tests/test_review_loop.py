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
                            "body": "This branch misses a required guard.",
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


if __name__ == "__main__":
    unittest.main()
