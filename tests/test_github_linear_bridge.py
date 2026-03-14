from __future__ import annotations

import unittest

from scripts.symphony import github_linear_bridge, review_loop


class ExtractIssueIdentifiersTests(unittest.TestCase):
    def test_extracts_unique_issue_identifiers_in_first_seen_order(self) -> None:
        self.assertEqual(
            github_linear_bridge.extract_issue_identifiers(
                "Ref PRO-93 and PRO-93",
                "Follow-up OPS-01 touches PRO-100 too",
                "codex/pro-93-review-bridge",
            ),
            ["PRO-93", "OPS-01", "PRO-100"],
        )

    def test_prefers_body_then_title_then_branch_for_issue_selection(self) -> None:
        snapshot = github_linear_bridge.PullRequestSnapshot(
            number=2,
            title="OPS-01: Add automation bridge",
            body="Ref PRO-93",
            head_ref_name="codex/pro-77-alt-branch",
            url="https://github.com/rputnam0/gpu_cfd/pull/2",
            state="OPEN",
            is_draft=False,
            mergeable="MERGEABLE",
            merge_state_status="CLEAN",
            review_decision="",
        )

        self.assertEqual(
            github_linear_bridge.select_issue_identifier(snapshot), "PRO-93"
        )


class DetermineBridgeDecisionTests(unittest.TestCase):
    def make_snapshot(
        self, *, mergeable: str = "MERGEABLE", merge_state_status: str = "CLEAN"
    ) -> github_linear_bridge.PullRequestSnapshot:
        return github_linear_bridge.PullRequestSnapshot(
            number=2,
            title="Add automated PR review loop",
            body="Ref PRO-93",
            head_ref_name="codex/review-loop-automation",
            url="https://github.com/rputnam0/gpu_cfd/pull/2",
            state="OPEN",
            is_draft=False,
            mergeable=mergeable,
            merge_state_status=merge_state_status,
            review_decision="",
        )

    def make_summary(self, *, review_state: str) -> review_loop.ReviewSummary:
        return review_loop.ReviewSummary(
            pr_number=2,
            pr_url="https://github.com/rputnam0/gpu_cfd/pull/2",
            pr_state="OPEN",
            review_state=review_state,
            review_decision="",
            head_oid="abc123",
            latest_commit_at="2026-03-13T20:00:00Z",
            reviewers=["devin-ai-integration[bot]"],
            actionable_reviews=[],
            actionable_threads=[],
            stale_reviews=[],
            observed_reviews=[],
            observed_threads=[],
        )

    def test_moves_to_rework_when_devin_feedback_is_actionable(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(),
            self.make_summary(review_state="action_required"),
            "PRO-93",
        )

        self.assertEqual(decision.target_state, "Rework")

    def test_moves_to_ready_to_merge_when_review_is_clean_and_mergeable(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(),
            self.make_summary(review_state="clean"),
            "PRO-93",
        )

        self.assertEqual(decision.target_state, "Ready to Merge")

    def test_does_not_move_to_ready_to_merge_when_merge_state_is_blocked(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(merge_state_status="BLOCKED"),
            self.make_summary(review_state="clean"),
            "PRO-93",
        )

        self.assertIsNone(decision.target_state)

    def test_noop_when_pr_has_no_linear_issue_reference(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(),
            self.make_summary(review_state="action_required"),
            None,
        )

        self.assertIsNone(decision.target_state)
