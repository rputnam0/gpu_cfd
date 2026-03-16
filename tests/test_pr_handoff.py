from __future__ import annotations

import pathlib
from unittest import mock
import unittest

from scripts.symphony import pr_handoff


class PrHandoffTests(unittest.TestCase):
    def test_local_review_complete_message_detection(self) -> None:
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

    def test_build_pr_title_prefixes_issue_identifier(self) -> None:
        self.assertEqual(
            pr_handoff.build_pr_title("PRO-6", "Example change"),
            "[codex] PRO-6 Example change",
        )

    def test_build_pr_body_mentions_closing_issue(self) -> None:
        body = pr_handoff.build_pr_body("PRO-6", "Example change")

        self.assertIn("<!-- gpu-cfd-linear-issue: PRO-6 -->", body)
        self.assertIn("Implement PRO-6: Example change", body)
        self.assertIn("Closes PRO-6", body)

    @mock.patch("scripts.symphony.pr_handoff.linear_api.upsert_workpad_comment")
    @mock.patch("scripts.symphony.pr_handoff.linear_api.merge_workpad_body")
    @mock.patch("scripts.symphony.pr_handoff.linear_api.find_workpad_comment")
    def test_sync_workpad_records_in_review_handoff_notes(
        self,
        mock_find_workpad_comment: mock.Mock,
        mock_merge_workpad_body: mock.Mock,
        mock_upsert_workpad_comment: mock.Mock,
    ) -> None:
        mock_find_workpad_comment.return_value = {"id": "comment-1", "body": "old"}
        mock_merge_workpad_body.return_value = "merged-body"
        mock_upsert_workpad_comment.return_value = {"id": "comment-1", "action": "updated"}

        pr_handoff.sync_workpad(
            issue_identifier="PRO-6",
            issue_title="Example change",
            current_status="in_review",
            validation=["Local Codex review complete."],
            review_handoff_notes=["PR opened: https://github.com/example/pull/6"],
        )

        mock_merge_workpad_body.assert_called_once()
        merge_kwargs = mock_merge_workpad_body.call_args.kwargs
        self.assertEqual(merge_kwargs["current_status"], "in_review")
        self.assertIn("PR opened: https://github.com/example/pull/6", merge_kwargs["review_handoff_notes"])
        mock_upsert_workpad_comment.assert_called_once_with("PRO-6", "merged-body")

    @mock.patch("scripts.symphony.pr_handoff.load_manifest_message")
    @mock.patch("scripts.symphony.pr_handoff.current_branch")
    @mock.patch("scripts.symphony.pr_handoff.subprocess.run")
    def test_nonzero_review_with_clean_message_is_not_treated_as_clean(
        self,
        mock_run: mock.Mock,
        mock_current_branch: mock.Mock,
        mock_load_manifest_message: mock.Mock,
    ) -> None:
        mock_run.return_value = mock.Mock(returncode=124)
        mock_current_branch.return_value = "codex/pro-6-example"
        mock_load_manifest_message.return_value = (
            {"message_path": ".codex/review_artifacts/latest.md"},
            "No material findings remain on this branch.",
        )

        result = pr_handoff.run_host_review(
            pathlib.Path("/tmp/workspace"),
            "PRO-6",
        )

        self.assertEqual(result.status, "unavailable")

    @mock.patch("builtins.print")
    @mock.patch("scripts.symphony.pr_handoff.trace.is_enabled", return_value=False)
    @mock.patch("scripts.symphony.pr_handoff.sync_workpad")
    @mock.patch("scripts.symphony.pr_handoff.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.pr_handoff.enable_auto_merge")
    @mock.patch("scripts.symphony.pr_handoff.ensure_pr")
    @mock.patch("scripts.symphony.pr_handoff.ensure_branch_pushed")
    @mock.patch("scripts.symphony.pr_handoff.current_branch")
    @mock.patch("scripts.symphony.pr_handoff.run_host_review")
    @mock.patch("scripts.symphony.pr_handoff.linear_api.fetch_issue")
    @mock.patch("scripts.symphony.pr_handoff.worktree_is_clean")
    @mock.patch("scripts.symphony.pr_handoff.parse_args")
    def test_main_returns_success_when_post_handoff_workpad_sync_fails(
        self,
        mock_parse_args: mock.Mock,
        mock_worktree_is_clean: mock.Mock,
        mock_fetch_issue: mock.Mock,
        mock_run_host_review: mock.Mock,
        mock_current_branch: mock.Mock,
        mock_ensure_branch_pushed: mock.Mock,
        mock_ensure_pr: mock.Mock,
        mock_enable_auto_merge: mock.Mock,
        mock_update_issue_state: mock.Mock,
        mock_sync_workpad: mock.Mock,
        _mock_trace_is_enabled: mock.Mock,
        _mock_print: mock.Mock,
    ) -> None:
        mock_parse_args.return_value = mock.Mock(workspace="/tmp/PRO-6")
        mock_worktree_is_clean.return_value = True
        mock_fetch_issue.return_value = {"title": "Example change"}
        mock_run_host_review.return_value = pr_handoff.ReviewResult(
            status="local_review_complete",
            message="No material findings remain on this branch.",
            manifest={"message_path": ".codex/review_artifacts/latest.md"},
        )
        mock_current_branch.return_value = "codex/pro-6-example"
        mock_ensure_pr.return_value = pr_handoff.PullRequestRef(
            number=6,
            url="https://github.com/example/pull/6",
            is_draft=False,
        )
        mock_update_issue_state.return_value = {
            "previous_state": "In Progress",
            "current_state": "In Review",
            "changed": True,
        }
        mock_sync_workpad.side_effect = RuntimeError("comment write failed")

        result = pr_handoff.main()

        self.assertEqual(result, 0)
        mock_ensure_branch_pushed.assert_called_once()
        mock_enable_auto_merge.assert_called_once_with(mock.ANY, 6)


if __name__ == "__main__":
    unittest.main()
