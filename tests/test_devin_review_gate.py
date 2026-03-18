from __future__ import annotations

import unittest

from scripts.symphony import devin_review_gate, review_loop


class DevinReviewGateTests(unittest.TestCase):
    def make_snapshot(
        self,
        *,
        state: str = "OPEN",
        is_draft: bool = False,
        merge_state_status: str | None = "CLEAN",
    ) -> devin_review_gate.PullRequestSnapshot:
        return devin_review_gate.PullRequestSnapshot(
            number=17,
            title="PRO-17 Example",
            body="Closes PRO-17",
            head_ref_name="codex/pro-17-example",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state=state,
            is_draft=is_draft,
            merge_state_status=merge_state_status,
        )

    def make_summary(
        self,
        *,
        review_state: str,
        actionable_threads: list[dict] | None = None,
        observed_threads: list[dict] | None = None,
        latest_commit_at: str | None = "2026-03-15T12:00:00Z",
        merge_state_status: str | None = "CLEAN",
    ) -> review_loop.ReviewSummary:
        return review_loop.ReviewSummary(
            pr_number=17,
            pr_url="https://github.com/rputnam0/gpu_cfd/pull/17",
            pr_state="OPEN",
            review_state=review_state,
            review_decision=None,
            merge_state_status=merge_state_status,
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

    def test_review_complete_marks_gate_success(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(),
            self.make_summary(review_state="review_complete"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "success")
        self.assertIsNone(decision.target_state)

    def test_first_round_feedback_resolved_marks_success(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(),
            self.make_summary(review_state="review_complete"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "success")
        self.assertIsNone(decision.target_state)
        self.assertIn("review is complete", decision.description)

    def test_closed_or_draft_pr_stays_pending_without_rework(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(state="CLOSED", is_draft=True),
            self.make_summary(review_state="pending_initial_review"),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "pending")
        self.assertIsNone(decision.target_state)
        self.assertIn("open ready PR", decision.description)

    def test_select_issue_identifier_prefers_explicit_marker(self) -> None:
        snapshot = devin_review_gate.PullRequestSnapshot(
            number=17,
            title="PRO-17 Example mentioning PRO-18",
            body=(
                "Closes PRO-17\n"
                "<!-- gpu-cfd-linear-issue: PRO-41 -->\n"
                "Follow-up note for PRO-18."
            ),
            head_ref_name="codex/pro-17-example",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="OPEN",
            is_draft=False,
            merge_state_status="CLEAN",
        )

        self.assertEqual(
            devin_review_gate.select_issue_identifier(snapshot),
            "PRO-41",
        )

    def test_select_issue_identifier_does_not_use_arbitrary_issue_mentions(self) -> None:
        snapshot = devin_review_gate.PullRequestSnapshot(
            number=29,
            title="[codex] Retry behind PR reconciliation",
            body=(
                "Live test note: workflow_dispatch of linear-post-merge on PR #28 moved PRO-16 "
                "from In Review to Rework.\n"
                "This maintenance PR is not itself linked to a Linear issue."
            ),
            head_ref_name="codex/behind-pr-reconcile-retry",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/29",
            state="OPEN",
            is_draft=False,
            merge_state_status="CLEAN",
        )

        self.assertIsNone(devin_review_gate.select_issue_identifier(snapshot))

    def test_select_issue_identifier_falls_back_to_branch_issue_identifier(self) -> None:
        snapshot = devin_review_gate.PullRequestSnapshot(
            number=17,
            title="[codex] Scope cleanup",
            body="No explicit marker or closing directive.",
            head_ref_name="rputnam0/pro-17-example",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="OPEN",
            is_draft=False,
            merge_state_status="CLEAN",
        )

        self.assertEqual(devin_review_gate.select_issue_identifier(snapshot), "PRO-17")

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

    def test_branch_refresh_required_moves_issue_to_rework(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(merge_state_status="BEHIND"),
            self.make_summary(
                review_state="branch_refresh_required",
                merge_state_status="BEHIND",
            ),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "failure")
        self.assertEqual(decision.target_state, "Rework")
        self.assertIn("behind", decision.description.lower())

    def test_action_required_mentions_branch_refresh_when_both_conditions_apply(self) -> None:
        decision = devin_review_gate.determine_gate_decision(
            self.make_snapshot(merge_state_status="BEHIND"),
            self.make_summary(
                review_state="action_required",
                merge_state_status="BEHIND",
            ),
            "PRO-17",
        )

        self.assertEqual(decision.status_state, "failure")
        self.assertEqual(decision.target_state, "Rework")
        self.assertIn("refresh", decision.description.lower())

    def test_collect_resolvable_thread_ids_keeps_stale_actionable_threads_open(self) -> None:
        actionable_thread = {
            "id": "thread-current",
            "is_resolved": False,
            "is_outdated": False,
            "comments": [
                {
                    "body": '<!-- devin-review-comment {"id": "BUG_pr-review-job_0001"} -->\nBug',
                    "created_at": "2026-03-15T11:00:00Z",
                }
            ],
        }
        summary = self.make_summary(
            review_state="action_required",
            actionable_threads=[actionable_thread],
            observed_threads=[actionable_thread],
        )

        self.assertEqual(devin_review_gate.collect_resolvable_thread_ids(summary), [])

    def test_collect_resolvable_thread_ids_resolves_outdated_threads(self) -> None:
        outdated_thread = {
            "id": "thread-outdated",
            "is_resolved": False,
            "is_outdated": True,
            "comments": [
                {
                    "body": '<!-- devin-review-comment {"id": "BUG_pr-review-job_0001"} -->\nBug',
                    "created_at": "2026-03-15T11:00:00Z",
                }
            ],
        }
        summary = self.make_summary(
            review_state="review_complete",
            actionable_threads=[],
            observed_threads=[outdated_thread],
        )

        self.assertEqual(
            devin_review_gate.collect_resolvable_thread_ids(summary),
            ["thread-outdated"],
        )


if __name__ == "__main__":
    unittest.main()
