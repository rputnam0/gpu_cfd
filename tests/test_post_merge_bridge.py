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

    @mock.patch("scripts.symphony.post_merge_bridge.reconcile_open_review_pull_requests")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.release_direct_unblocked_dependents")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.post_merge_bridge.select_issue_identifier")
    @mock.patch("scripts.symphony.post_merge_bridge.fetch_pr_snapshot")
    def test_main_updates_done_and_releases_dependents(
        self,
        mock_fetch_pr_snapshot: mock.Mock,
        mock_select_issue_identifier: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_release_direct_unblocked_dependents: mock.Mock,
        mock_reconcile_open_review_pull_requests: mock.Mock,
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
        mock_release_direct_unblocked_dependents.return_value = [
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
        mock_release_direct_unblocked_dependents.assert_called_once_with("PRO-17")
        mock_reconcile_open_review_pull_requests.assert_called_once_with(
            "rputnam0/gpu_cfd",
            exclude_pr_number=17,
        )

    @mock.patch("scripts.symphony.post_merge_bridge.reconcile_open_review_pull_requests")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.release_direct_unblocked_dependents")
    @mock.patch("scripts.symphony.post_merge_bridge.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.post_merge_bridge.select_issue_identifier")
    @mock.patch("scripts.symphony.post_merge_bridge.fetch_pr_snapshot")
    def test_main_skips_when_pull_request_has_no_linear_issue(
        self,
        mock_fetch_pr_snapshot: mock.Mock,
        mock_select_issue_identifier: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_release_direct_unblocked_dependents: mock.Mock,
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
        mock_release_direct_unblocked_dependents.assert_not_called()
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
            "linear_update": {"current_state": "Rework"},
        }

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["issue_identifier"], "PRO-16")
        self.assertEqual(result[0]["merge_state_status"], "BEHIND")
        mock_process_pull_request.assert_called_once()

    @mock.patch("scripts.symphony.post_merge_bridge.time.sleep")
    @mock.patch("scripts.symphony.post_merge_bridge.devin_review_gate.process_pull_request")
    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_retries_until_candidate_becomes_behind(
        self,
        mock_collect_candidates: mock.Mock,
        mock_process_pull_request: mock.Mock,
        mock_sleep: mock.Mock,
    ) -> None:
        mock_collect_candidates.side_effect = [
            [
                {
                    "pr_number": 26,
                    "issue_identifier": "PRO-16",
                    "merge_state_status": "",
                }
            ],
            [
                {
                    "pr_number": 26,
                    "issue_identifier": "PRO-16",
                    "merge_state_status": "BEHIND",
                }
            ],
        ]
        mock_process_pull_request.return_value = {
            "decision": {"review_state": "branch_refresh_required"},
            "linear_update": {"current_state": "Rework"},
        }

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
            max_attempts=2,
            retry_delay_seconds=1.5,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["merge_state_status"], "BEHIND")
        mock_sleep.assert_called_once_with(1.5)
        mock_process_pull_request.assert_called_once()

    @mock.patch("scripts.symphony.post_merge_bridge.time.sleep")
    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_returns_empty_without_candidates(
        self,
        mock_collect_candidates: mock.Mock,
        mock_sleep: mock.Mock,
    ) -> None:
        mock_collect_candidates.return_value = []

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
        )

        self.assertEqual(result, [])
        mock_sleep.assert_not_called()

    @mock.patch("scripts.symphony.post_merge_bridge.time.sleep")
    @mock.patch("scripts.symphony.post_merge_bridge.devin_review_gate.process_pull_request")
    @mock.patch("scripts.symphony.post_merge_bridge.collect_in_review_pull_request_candidates")
    def test_reconcile_open_review_pull_requests_stops_after_retry_budget_without_refresh_state(
        self,
        mock_collect_candidates: mock.Mock,
        mock_process_pull_request: mock.Mock,
        mock_sleep: mock.Mock,
    ) -> None:
        mock_collect_candidates.side_effect = [
            [
                {
                    "pr_number": 26,
                    "issue_identifier": "PRO-16",
                    "merge_state_status": "",
                }
            ],
            [
                {
                    "pr_number": 26,
                    "issue_identifier": "PRO-16",
                    "merge_state_status": "",
                }
            ],
        ]

        result = post_merge_bridge.reconcile_open_review_pull_requests(
            "rputnam0/gpu_cfd",
            max_attempts=2,
            retry_delay_seconds=2.0,
        )

        self.assertEqual(result, [])
        mock_sleep.assert_called_once_with(2.0)
        mock_process_pull_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
