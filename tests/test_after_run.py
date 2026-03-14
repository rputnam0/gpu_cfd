from __future__ import annotations

import unittest
from unittest import mock

from scripts.symphony import after_run


class AfterRunTests(unittest.TestCase):
    def test_clean_review_message_detection(self) -> None:
        self.assertTrue(
            after_run.review_message_is_clean(
                "No material findings remain on this branch."
            )
        )
        self.assertFalse(
            after_run.review_message_has_findings(
                "No material findings remain on this branch."
            )
        )

    def test_finding_review_message_detection(self) -> None:
        message = (
            "- [P2] Strip Markdown formatting from pin-manifest fields\n"
            "Downstream tooling should receive stable tokens."
        )
        self.assertFalse(after_run.review_message_is_clean(message))
        self.assertTrue(after_run.review_message_has_findings(message))

    @mock.patch("scripts.symphony.after_run.time.sleep")
    @mock.patch("scripts.symphony.after_run.time.monotonic")
    @mock.patch("scripts.symphony.after_run.github_linear_bridge.fetch_linear_issue")
    def test_wait_for_issue_state_retries_until_target_state(
        self,
        fetch_linear_issue: mock.Mock,
        monotonic: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        fetch_linear_issue.side_effect = [
            {"state": {"name": "Rework"}, "title": "Example"},
            {"state": {"name": "In Review"}, "title": "Example"},
        ]
        monotonic.side_effect = [0, 1]

        issue = after_run.wait_for_issue_state("PRO-5", "In Review")

        self.assertEqual(issue["state"]["name"], "In Review")
        self.assertEqual(fetch_linear_issue.call_count, 2)
        sleep.assert_called_once_with(after_run.STATE_SETTLE_POLL_SECONDS)

    @mock.patch("scripts.symphony.after_run.time.sleep")
    @mock.patch("scripts.symphony.after_run.time.monotonic")
    @mock.patch("scripts.symphony.after_run.github_linear_bridge.fetch_linear_issue")
    def test_wait_for_issue_state_returns_last_seen_issue_on_timeout(
        self,
        fetch_linear_issue: mock.Mock,
        monotonic: mock.Mock,
        sleep: mock.Mock,
    ) -> None:
        fetch_linear_issue.side_effect = [
            {"state": {"name": "Rework"}, "title": "Example"},
            {"state": {"name": "Rework"}, "title": "Example"},
        ]
        monotonic.side_effect = [0, 31]

        issue = after_run.wait_for_issue_state("PRO-5", "In Review")

        self.assertEqual(issue["state"]["name"], "Rework")
        self.assertEqual(fetch_linear_issue.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
