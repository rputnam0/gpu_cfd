from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from scripts.symphony import devin_review_gate, reconcile_review_state, review_loop


class ReconcileReviewStateTests(unittest.TestCase):
    @mock.patch("scripts.symphony.reconcile_review_state.review_loop.repo_root")
    @mock.patch("scripts.symphony.reconcile_review_state.review_loop.require_command")
    def test_list_open_prs_preserves_merge_state_status(
        self,
        mock_require_command: mock.Mock,
        mock_repo_root: mock.Mock,
    ) -> None:
        mock_repo_root.return_value = "/tmp/gpu_cfd"
        mock_require_command.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 34,
                        "title": "[codex] Harden Symphony harness rollout flow",
                        "body": "Closes PRO-34",
                        "headRefName": "codex/symphony-harness-hardening",
                        "headRefOid": "abc123",
                        "url": "https://github.com/rputnam0/gpu_cfd/pull/34",
                        "state": "OPEN",
                        "isDraft": False,
                        "mergeStateStatus": "BEHIND",
                    }
                ]
            ),
            stderr="",
        )

        snapshots = reconcile_review_state.list_open_prs("rputnam0/gpu_cfd")

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].merge_state_status, "BEHIND")

    @mock.patch("scripts.symphony.reconcile_review_state.devin_review_gate.sync_followup_workpad_best_effort")
    @mock.patch("scripts.symphony.reconcile_review_state.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.reconcile_review_state.linear_api.suppress_legacy_workpad_comments")
    @mock.patch("scripts.symphony.reconcile_review_state.linear_api.fetch_issue")
    @mock.patch("scripts.symphony.reconcile_review_state.review_loop.fetch_pr_summary")
    def test_reconcile_snapshot_action_required_uses_followup_workpad_sync(
        self,
        mock_fetch_pr_summary: mock.Mock,
        mock_fetch_issue: mock.Mock,
        mock_suppress_legacy_workpad_comments: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_sync_followup_workpad_best_effort: mock.Mock,
    ) -> None:
        snapshot = devin_review_gate.PullRequestSnapshot(
            number=34,
            title="[codex] Harden Symphony harness rollout flow",
            body="Closes PRO-34",
            head_ref_name="codex/symphony-harness-hardening",
            head_oid="abc123",
            url="https://github.com/rputnam0/gpu_cfd/pull/34",
            state="OPEN",
            is_draft=False,
            merge_state_status="CLEAN",
        )
        mock_fetch_pr_summary.return_value = review_loop.ReviewSummary(
            pr_number=34,
            pr_url=snapshot.url,
            pr_state="OPEN",
            review_state="action_required",
            review_decision="CHANGES_REQUESTED",
            merge_state_status="CLEAN",
            head_oid=snapshot.head_oid,
            latest_commit_at="2026-03-18T16:39:57Z",
            reviewers=["devin-ai-integration[bot]"],
            actionable_reviews=[{"author": "devin-ai-integration[bot]"}],
            actionable_threads=[],
            stale_reviews=[],
            observed_reviews=[],
            observed_threads=[],
        )
        mock_fetch_issue.return_value = {"state": {"name": "In Review"}}
        mock_suppress_legacy_workpad_comments.return_value = []
        mock_update_issue_state.return_value = {
            "changed": True,
            "previous_state": "In Review",
            "current_state": "Rework",
        }

        result = reconcile_review_state.reconcile_snapshot(
            snapshot,
            repo="rputnam0/gpu_cfd",
            reviewers=["devin-ai-integration[bot]"],
            apply_changes=True,
        )

        self.assertEqual(result["status"], "rework_required")
        mock_sync_followup_workpad_best_effort.assert_called_once_with(
            "PRO-34",
            mock_fetch_pr_summary.return_value,
            target_state=devin_review_gate.DEFAULT_REWORK_STATE,
        )


if __name__ == "__main__":
    unittest.main()
