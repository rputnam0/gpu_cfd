from __future__ import annotations

import unittest

from scripts.symphony import devin_review_gate, review_loop


class DevinReviewGateTests(unittest.TestCase):
    def make_snapshot(self, *, state: str = "OPEN", is_draft: bool = False) -> devin_review_gate.PullRequestSnapshot:
        return devin_review_gate.PullRequestSnapshot(
            number=17,
            title="PRO-17 Example",
            body="Closes PRO-17",
            head_ref_name="codex/pro-17-example",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state=state,
            is_draft=is_draft,
        )

    def make_summary(
        self,
        *,
        review_state: str,
        actionable_threads: list[dict] | None = None,
        observed_threads: list[dict] | None = None,
        latest_commit_at: str | None = "2026-03-15T12:00:00Z",
    ) -> review_loop.ReviewSummary:
        return review_loop.ReviewSummary(
            pr_number=17,
            pr_url="https://github.com/rputnam0/gpu_cfd/pull/17",
            pr_state="OPEN",
            review_state=review_state,
            review_decision=None,
            head_oid="abc123",
            latest_commit_at=latest_commit_at,
            reviewers=["devin-ai-integration[bot]"],
            actionable_reviews=[],
            actionable_threads=actionable_threads or [],
            stale_reviews=[],
            observed_reviews=[],
            observed_threads=observed_threads or [],
        )

    def test_action_required_moves_issue_to_rework(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(),
            self.make_summary(review_state="action_required"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "failure")
        self.assertEqual(decision.target_state, "Rework")
        self.assertEqual(decision.issue_identifier, "PRO-17")

    def test_clean_marks_gate_success(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(),
            self.make_summary(review_state="clean"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "success")
        self.assertIsNone(decision.target_state)

    def test_open_pr_with_stale_feedback_stays_pending(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(),
            self.make_summary(review_state="pending_rereview"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "pending")
        self.assertIn("fresh Devin re-review", decision.description)

    def test_closed_or_draft_pr_stays_pending_without_rework(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(state="CLOSED", is_draft=True),
            self.make_summary(review_state="pending_initial_review"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "pending")
        self.assertIsNone(decision.target_state)
        self.assertIn("open ready PR", decision.description)

    def test_collect_resolvable_thread_ids_ignores_current_actionable_threads(self) -> None:
        actionable_thread = {
            "id": "thread-current",
            "is_resolved": False,
            "is_outdated": False,
            "comments": [
                {
                    "created_at": "2026-03-15T12:05:00Z",
                }
            ],
        }
        stale_thread = {
            "id": "thread-stale",
            "is_resolved": False,
            "is_outdated": False,
            "comments": [
                {
                    "created_at": "2026-03-15T11:00:00Z",
                }
            ],
        }
        resolved_thread = {
            "id": "thread-resolved",
            "is_resolved": True,
            "is_outdated": False,
            "comments": [
                {
                    "created_at": "2026-03-15T12:01:00Z",
                }
            ],
        }
        summary = self.make_summary(
            review_state="action_required",
            actionable_threads=[actionable_thread],
            observed_threads=[actionable_thread, stale_thread, resolved_thread],
        )

        self.assertEqual(
            devin_review_gate.collect_resolvable_thread_ids(summary),
            ["thread-stale"],
        )


if __name__ == "__main__":
    unittest.main()
