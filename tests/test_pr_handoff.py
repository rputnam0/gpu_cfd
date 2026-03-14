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

    def test_result_payload_includes_pr_details_when_present(self) -> None:
        payload = pr_handoff.result_payload(
            issue="PRO-6",
            branch="rputnam0/pro-6-example",
            review_result=pr_handoff.ReviewResult(
                status="in_review",
                message="No findings.",
                manifest={"jsonl_path": "artifact.jsonl", "message_path": "artifact.md"},
            ),
            pr=pr_handoff.PullRequestRef(
                number=7,
                url="https://github.com/rputnam0/gpu_cfd/pull/7",
                is_draft=False,
            ),
        )

        self.assertEqual(payload["status"], "in_review")
        self.assertEqual(payload["issue"], "PRO-6")
        self.assertEqual(payload["pr"]["number"], 7)
        self.assertEqual(payload["pr"]["url"], "https://github.com/rputnam0/gpu_cfd/pull/7")


if __name__ == "__main__":
    unittest.main()
