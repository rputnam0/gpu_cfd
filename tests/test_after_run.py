from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
