from __future__ import annotations

import unittest
from unittest import mock

from scripts.symphony import github_linear_bridge, review_loop


class ExtractIssueIdentifiersTests(unittest.TestCase):
    def test_extracts_unique_issue_identifiers_in_first_seen_order(self) -> None:
        self.assertEqual(
            github_linear_bridge.extract_issue_identifiers(
                "Ref PRO-93 and PRO-93",
                "Follow-up OPS-01 touches PRO-100 too",
                "HTTP-404 should not be treated as a Linear issue",
            ),
            ["PRO-93", "PRO-100"],
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

    def test_ignores_non_project_identifiers_when_selecting_issue(self) -> None:
        snapshot = github_linear_bridge.PullRequestSnapshot(
            number=2,
            title="OPS-01: Add automation bridge",
            body="HTTP-404 appeared before Ref PRO-93",
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
        self,
        *,
        mergeable: str = "MERGEABLE",
        merge_state_status: str = "CLEAN",
        status_check_rollup: list[dict[str, object]] | None = None,
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
            status_check_rollup=status_check_rollup,
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

    def test_moves_to_ready_to_merge_when_required_checks_pass(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(
                merge_state_status="BLOCKED",
                status_check_rollup=[
                    {
                        "name": "review-loop-harness",
                        "status": "COMPLETED",
                        "conclusion": "SUCCESS",
                    }
                ],
            ),
            self.make_summary(review_state="clean"),
            "PRO-93",
        )

        self.assertEqual(decision.target_state, "Ready to Merge")

    def test_noop_when_pr_has_no_linear_issue_reference(self) -> None:
        decision = github_linear_bridge.determine_bridge_decision(
            self.make_snapshot(),
            self.make_summary(review_state="action_required"),
            None,
        )

        self.assertIsNone(decision.target_state)


class UpdateLinearIssueStateTests(unittest.TestCase):
    @mock.patch("scripts.symphony.github_linear_bridge.linear_graphql")
    @mock.patch("scripts.symphony.github_linear_bridge.fetch_linear_issue")
    def test_uses_internal_issue_id_for_state_update(
        self,
        fetch_linear_issue: mock.Mock,
        linear_graphql: mock.Mock,
    ) -> None:
        fetch_linear_issue.return_value = {
            "id": "35d69fd1-94de-4436-a58f-37a2faec86d9",
            "identifier": "PRO-93",
            "state": {"name": "In Review"},
            "team": {
                "key": "PRO",
                "states": {
                    "nodes": [
                        {"id": "state-rework", "name": "Rework"},
                    ]
                },
            },
        }
        linear_graphql.return_value = {
            "issueUpdate": {
                "issue": {
                    "id": "35d69fd1-94de-4436-a58f-37a2faec86d9",
                    "identifier": "PRO-93",
                    "state": {"name": "Rework"},
                }
            }
        }

        github_linear_bridge.update_linear_issue_state("PRO-93", "Rework")

        linear_graphql.assert_called_once_with(
            github_linear_bridge.LINEAR_ISSUE_UPDATE_MUTATION,
            {"id": "35d69fd1-94de-4436-a58f-37a2faec86d9", "stateId": "state-rework"},
        )


class ResolvableThreadTests(unittest.TestCase):
    def make_summary(self) -> review_loop.ReviewSummary:
        return review_loop.ReviewSummary(
            pr_number=2,
            pr_url="https://github.com/rputnam0/gpu_cfd/pull/2",
            pr_state="OPEN",
            review_state="clean",
            review_decision="",
            head_oid="abc123",
            latest_commit_at="2026-03-14T01:04:19Z",
            reviewers=["devin-ai-integration[bot]"],
            actionable_reviews=[],
            actionable_threads=[],
            stale_reviews=[],
            observed_reviews=[],
            observed_threads=[
                {
                    "id": "thread-analysis",
                    "path": "scripts/symphony/review_loop.py",
                    "is_resolved": False,
                    "is_outdated": False,
                    "comments": [
                        {
                            "author": "devin-ai-integration[bot]",
                            "body": "<!-- devin-review-comment "
                            '{"id": "ANALYSIS_pr-review-job_0001"} -->\n\n'
                            "Informational note.",
                            "created_at": "2026-03-14T01:15:41Z",
                            "line": 10,
                            "original_line": 10,
                            "url": "https://example.com/thread-analysis",
                        }
                    ],
                },
                {
                    "id": "thread-bug",
                    "path": "scripts/symphony/review_loop.py",
                    "is_resolved": False,
                    "is_outdated": False,
                    "comments": [
                        {
                            "author": "devin-ai-integration[bot]",
                            "body": "<!-- devin-review-comment "
                            '{"id": "BUG_pr-review-job_0002"} -->\n\n'
                            "Bug note.",
                            "created_at": "2026-03-14T01:15:42Z",
                            "line": 11,
                            "original_line": 11,
                            "url": "https://example.com/thread-bug",
                        }
                    ],
                },
            ],
        )

    def test_collects_only_non_actionable_unresolved_threads(self) -> None:
        summary = self.make_summary()
        summary.actionable_threads = [summary.observed_threads[1]]

        self.assertEqual(
            github_linear_bridge.collect_resolvable_thread_ids(summary),
            ["thread-analysis"],
        )


class ResolveReviewThreadTests(unittest.TestCase):
    @mock.patch("scripts.symphony.github_linear_bridge.review_loop.require_command")
    def test_surfaces_actionable_message_when_actions_token_cannot_resolve_thread(
        self, require_command: mock.Mock
    ) -> None:
        require_command.side_effect = review_loop.CommandError(
            "gh: Resource not accessible by integration"
        )

        with self.assertRaisesRegex(
            ValueError,
            "REVIEW_BRIDGE_GH_TOKEN",
        ):
            github_linear_bridge.resolve_review_thread("thread-123")
