from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from scripts.symphony import post_merge_bridge


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

        with mock.patch(
            "sys.argv",
            ["post_merge_bridge.py", "--repo", "rputnam0/gpu_cfd", "--pr", "17"],
        ), contextlib.redirect_stdout(io.StringIO()):
            exit_code = post_merge_bridge.main()

        self.assertEqual(exit_code, 0)
        mock_update_issue_state.assert_called_once_with("PRO-17", "Done")
        mock_release_direct_unblocked_dependents.assert_called_once_with("PRO-17")


if __name__ == "__main__":
    unittest.main()
