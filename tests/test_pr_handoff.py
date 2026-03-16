from __future__ import annotations

import pathlib
from unittest import mock
import unittest

from scripts.symphony import pr_handoff


class PrHandoffTests(unittest.TestCase):
    def test_clean_review_message_detection(self) -> None:
        self.assertTrue(
            pr_handoff.review_message_is_clean(
                "No material findings remain on this branch."
            )
        )
        self.assertFalse(
            pr_handoff.review_message_has_findings(
                "No material findings remain on this branch."
            )
        )

    def test_finding_review_message_detection(self) -> None:
        message = (
            "- [P2] Strip Markdown formatting from pin-manifest fields\n"
            "Downstream tooling should receive stable tokens."
        )
        self.assertFalse(pr_handoff.review_message_is_clean(message))
        self.assertTrue(pr_handoff.review_message_has_findings(message))

    def test_build_pr_title_prefixes_issue_identifier(self) -> None:
        self.assertEqual(
            pr_handoff.build_pr_title("PRO-6", "Example change"),
            "[codex] PRO-6 Example change",
        )

    def test_build_pr_body_mentions_closing_issue(self) -> None:
        body = pr_handoff.build_pr_body("PRO-6", "Example change")

        self.assertIn("Implement PRO-6: Example change", body)
        self.assertIn("Closes PRO-6", body)

    @mock.patch("scripts.symphony.pr_handoff.load_manifest_message")
    @mock.patch("scripts.symphony.pr_handoff.current_branch")
    @mock.patch("scripts.symphony.pr_handoff.subprocess.run")
    def test_nonzero_review_with_clean_message_is_not_treated_as_clean(
        self,
        mock_run: mock.Mock,
        mock_current_branch: mock.Mock,
        mock_load_manifest_message: mock.Mock,
    ) -> None:
        mock_run.return_value = mock.Mock(returncode=124)
        mock_current_branch.return_value = "codex/pro-6-example"
        mock_load_manifest_message.return_value = (
            {"message_path": ".codex/review_artifacts/latest.md"},
            "No material findings remain on this branch.",
        )

        result = pr_handoff.run_host_review(
            pathlib.Path("/tmp/workspace"),
            "PRO-6",
        )

        self.assertEqual(result.status, "unavailable")


if __name__ == "__main__":
    unittest.main()
