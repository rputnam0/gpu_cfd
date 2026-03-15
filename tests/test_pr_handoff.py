from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
