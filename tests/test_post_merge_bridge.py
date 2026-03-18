from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from scripts.symphony import devin_review_gate, post_merge_bridge


class PostMergeBridgeTests(unittest.TestCase):
    def test_select_issue_identifier_prefers_override(self) -> None:
        snapshot = post_merge_bridge.PullRequestSnapshot(
            number=17,
            title="Ignore PRO-17 in title",
            body="Closes PRO-17",
            head_ref_name="codex/pro-17-example",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="MERGED",
            merged_at="2026-03-15T12:00:00Z",
        )

        self.assertEqual(
            post_merge_bridge.select_issue_identifier(snapshot, "PRO-99"),
            "PRO-99",
        )

    def test_select_issue_identifier_prefers_explicit_marker(self) -> None:
        snapshot = post_merge_bridge.PullRequestSnapshot(
            number=17,
            title="Mentions PRO-17 and PRO-18",
            body=(
                "Closes PRO-17\n"
                "<!-- gpu-cfd-linear-issue: PRO-42 -->\n"
                "Also references PRO-18 for context."
            ),
            head_ref_name="codex/pro-17-example",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="MERGED",
            merged_at="2026-03-15T12:00:00Z",
        )

        self.assertEqual(
            post_merge_bridge.select_issue_identifier(snapshot, None),
            "PRO-42",
        )

    def test_select_issue_identifier_uses_closing_keyword_not_arbitrary_issue_mentions(self) -> None:
        snapshot = post_merge_bridge.PullRequestSnapshot(
            number=29,
            title="[codex] Retry behind PR reconciliation",
            body=(
                "Live test note: workflow_dispatch of linear-post-merge on PR #28 moved PRO-16 "
                "from In Review to Rework.\n"
                "This maintenance PR is not itself linked to a Linear issue."
            ),
            head_ref_name="codex/behind-pr-reconcile-retry",
            url="https://github.com/rputnam0/gpu_cfd/pull/29",
            state="MERGED",
            merged_at="2026-03-18T04:43:05Z",
        )

        self.assertIsNone(post_merge_bridge.select_issue_identifier(snapshot, None))

    def test_select_issue_identifier_falls_back_to_branch_issue_identifier(self) -> None:
        snapshot = post_merge_bridge.PullRequestSnapshot(
            number=17,
            title="[codex] Scope cleanup",
            body="No explicit marker or closing directive.",
            head_ref_name="rputnam0/pro-17-example",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="MERGED",
            merged_at="2026-03-15T12:00:00Z",
        )

        self.assertEqual(
            post_merge_bridge.select_issue_identifier(snapshot, None),
            "PRO-17",
        )

    @mock.patch("scripts.symphony.post_merge_bridge.phase_cleanup.promote_ready_backlog_issues")
    @mock.patch("scripts.symphony.post_merge_bridge.phase_cleanup.resolve_section_for_issue")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.list_team_issues")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.fetch_issue")
    @mock.patch("scripts.symphony.post_merge_bridge.reconcile_open_review_pull_requests")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.post_merge_bridge.select_issue_identifier")
    @mock.patch("scripts.symphony.post_merge_bridge.fetch_pr_snapshot")
    def test_main_updates_done_and_releases_dependents(
        self,
        mock_fetch_pr_snapshot: mock.Mock,
        mock_select_issue_identifier: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_reconcile_open_review_pull_requests: mock.Mock,
        mock_fetch_issue: mock.Mock,
        mock_list_team_issues: mock.Mock,
        mock_resolve_section_for_issue: mock.Mock,
        mock_promote_ready_backlog_issues: mock.Mock,
    ) -> None:
        mock_fetch_pr_snapshot.return_value = post_merge_bridge.PullRequestSnapshot(
            number=17,
            title="PRO-17 Example",
            body="Closes PRO-17",
            head_ref_name="codex/pro-17-example",
            url="https://github.com/rputnam0/gpu_cfd/pull/17",
            state="MERGED",
            merged_at="2026-03-15T12:00:00Z",
        )
        mock_select_issue_identifier.return_value = "PRO-17"
        mock_update_issue_state.return_value = {
            "changed": True,
            "previous_state": "In Review",
            "current_state": "Done",
        }
        mock_fetch_issue.return_value = {
            "identifier": "PRO-17",
            "title": "P0-08 Example",
            "team": {"id": "team-1", "key": "PRO"},
            "state": {"name": "In Review"},
        }
        mock_list_team_issues.return_value = [
            {"identifier": "PRO-17", "title": "P0-08 Example", "state": {"name": "Done"}}
        ]
        mock_resolve_section_for_issue.return_value = None
        mock_promote_ready_backlog_issues.return_value = [
            {"identifier": "PRO-18", "changed": True}
        ]
        mock_reconcile_open_review_pull_requests.return_value = []

        with mock.patch(
            "sys.argv",
            ["post_merge_bridge.py", "--repo", "rputnam0/gpu_cfd", "--pr", "17"],
        ), contextlib.redirect_stdout(io.StringIO()):
            exit_code = post_merge_bridge.main()

        self.assertEqual(exit_code, 0)
        mock_update_issue_state.assert_called_once_with("PRO-17", "Done")
        mock_promote_ready_backlog_issues.assert_called_once()
        mock_reconcile_open_review_pull_requests.assert_called_once_with(
            "rputnam0/gpu_cfd",
            exclude_pr_number=17,
        )

    @mock.patch("scripts.symphony.post_merge_bridge.reconcile_open_review_pull_requests")
    @mock.patch("scripts.symphony.post_merge_bridge.phase_cleanup.promote_ready_backlog_issues")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.post_merge_bridge.select_issue_identifier")
    @mock.patch("scripts.symphony.post_merge_bridge.fetch_pr_snapshot")
    def test_main_skips_when_pull_request_has_no_linear_issue(
        self,
        mock_fetch_pr_snapshot: mock.Mock,
        mock_select_issue_identifier: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_promote_ready_backlog_issues: mock.Mock,
        mock_reconcile_open_review_pull_requests: mock.Mock,
    ) -> None:
        mock_fetch_pr_snapshot.return_value = post_merge_bridge.PullRequestSnapshot(
            number=18,
            title="[codex] Maintenance cleanup",
            body="No Linear issue attached.",
            head_ref_name="codex/maintenance-cleanup",
            url="https://github.com/rputnam0/gpu_cfd/pull/18",
            state="MERGED",
            merged_at="2026-03-16T16:00:00Z",
        )
        mock_select_issue_identifier.return_value = None
        mock_reconcile_open_review_pull_requests.return_value = []

        buffer = io.StringIO()
        with mock.patch(
            "sys.argv",
            ["post_merge_bridge.py", "--repo", "rputnam0/gpu_cfd", "--pr", "18"],
        ), contextlib.redirect_stdout(buffer):
            exit_code = post_merge_bridge.main()

        self.assertEqual(exit_code, 0)
        self.assertIn('"skipped": true', buffer.getvalue())
        mock_update_issue_state.assert_not_called()
        mock_promote_ready_backlog_issues.assert_not_called()
        mock_reconcile_open_review_pull_requests.assert_called_once_with(
            "rputnam0/gpu_cfd",
            exclude_pr_number=18,
        )

    @mock.patch("scripts.symphony.post_merge_bridge.devin_review_gate.process_pull_request")
    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_wakes_behind_issue_into_rework(
        self,
        mock_collect_candidates: mock.Mock,
        mock_process_pull_request: mock.Mock,
    ) -> None:
        mock_collect_candidates.return_value = [
            {
                "pr_number": 26,
                "issue_identifier": "PRO-16",
                "merge_state_status": "BEHIND",
            }
        ]
        mock_process_pull_request.return_value = {
            "decision": {"review_state": "branch_refresh_required"},
            "linear_update": {"current_state": "Refresh Required"},
        }

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_identifier"], "PRO-16")
        self.assertEqual(result[0]["merge_state_status"], "BEHIND")
        mock_process_pull_request.assert_called_once()

    @mock.patch("scripts.symphony.post_merge_bridge.devin_review_gate.process_pull_request")
    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_processes_refresh_candidates_in_one_pass(
        self,
        mock_collect_candidates: mock.Mock,
        mock_process_pull_request: mock.Mock,
    ) -> None:
        mock_collect_candidates.return_value = [
            {
                "pr_number": 26,
                "issue_identifier": "PRO-16",
                "merge_state_status": "BEHIND",
            },
            {
                "pr_number": 27,
                "issue_identifier": "PRO-17",
                "merge_state_status": "",
            },
        ]
        mock_process_pull_request.return_value = {
            "decision": {"review_state": "branch_refresh_required"},
            "linear_update": {"current_state": "Refresh Required"},
        }

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["merge_state_status"], "BEHIND")
        mock_process_pull_request.assert_called_once()

    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_returns_empty_without_candidates(
        self,
        mock_collect_candidates: mock.Mock,
    ) -> None:
        mock_collect_candidates.return_value = []

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
        )

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
