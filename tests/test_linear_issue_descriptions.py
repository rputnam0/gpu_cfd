from __future__ import annotations

import pathlib
import unittest
import contextlib
import io
from unittest import mock

from scripts.symphony import linear_issue_descriptions


class LinearIssueDescriptionsTests(unittest.TestCase):
    def repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def test_render_issue_description_uses_progressive_disclosure_contract(self) -> None:
        description = linear_issue_descriptions.render_issue_description(
            self.repo_root(),
            "PRO-8",
            "FND-04: Support-matrix scanner and fail-fast policy",
        )

        self.assertIn("## Worker startup contract", description)
        self.assertIn("Start with `AGENTS.md`.", description)
        self.assertIn("use `docs/tasks/pr_inventory.md` as the fallback map", description)
        self.assertNotIn("Resolve the PR card through", description)
        self.assertIn("Write or update the canonical Linear workpad before edits.", description)
        self.assertIn("docs/authority/README.md", description)
        self.assertIn("## Task card", description)
        self.assertNotIn("README_FIRST", description)

    def test_issue_metadata_comes_from_current_backlog(self) -> None:
        metadata = linear_issue_descriptions.load_issue_metadata(self.repo_root())["FND-04"]

        self.assertEqual(metadata.pr_id, "FND-04")
        self.assertEqual(metadata.title, "FND-04: Support-matrix scanner and fail-fast policy")
        self.assertEqual(metadata.section_label, "Foundation")
        self.assertEqual(metadata.depends_on, ["FND-01"])

    @mock.patch("scripts.symphony.linear_issue_descriptions.linear_api.list_team_issues")
    def test_audit_live_issue_descriptions_reports_drift(self, mock_list_team_issues: mock.Mock) -> None:
        mock_list_team_issues.return_value = [
            {
                "identifier": "PRO-8",
                "title": "FND-04: Support-matrix scanner and fail-fast policy",
                "description": "stale body",
            }
        ]

        results = linear_issue_descriptions.audit_live_issue_descriptions(
            self.repo_root(),
            team_key="PRO",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["issue_identifier"], "PRO-8")
        self.assertEqual(results[0]["status"], "drifted")
        self.assertTrue(results[0]["changed"])

    @mock.patch("scripts.symphony.linear_issue_descriptions.linear_api.update_issue_description")
    @mock.patch("scripts.symphony.linear_issue_descriptions.linear_api.list_team_issues")
    def test_sync_live_issue_descriptions_updates_only_drifted_issues(
        self,
        mock_list_team_issues: mock.Mock,
        mock_update_issue_description: mock.Mock,
    ) -> None:
        expected = linear_issue_descriptions.render_issue_description(
            self.repo_root(),
            "PRO-8",
            "FND-04: Support-matrix scanner and fail-fast policy",
        )
        mock_list_team_issues.return_value = [
            {
                "identifier": "PRO-8",
                "title": "FND-04: Support-matrix scanner and fail-fast policy",
                "description": "stale body",
            },
            {
                "identifier": "PRO-9",
                "title": "FND-05: Runtime config and state contract",
                "description": linear_issue_descriptions.render_issue_description(
                    self.repo_root(),
                    "PRO-9",
                    "FND-05: Runtime config and state contract",
                ),
            },
        ]
        mock_update_issue_description.return_value = {
            "changed": True,
            "previous_description": "stale body",
            "current_description": expected,
        }

        results = linear_issue_descriptions.sync_live_issue_descriptions(
            self.repo_root(),
            team_key="PRO",
        )

        self.assertEqual(
            [result["status"] for result in results],
            ["updated", "in_sync"],
        )
        mock_update_issue_description.assert_called_once_with("PRO-8", expected)

    @mock.patch("scripts.symphony.linear_issue_descriptions.audit_live_issue_descriptions")
    @mock.patch("scripts.symphony.linear_issue_descriptions.parse_args")
    def test_main_returns_nonzero_for_audit_drift(
        self,
        mock_parse_args: mock.Mock,
        mock_audit_live_issue_descriptions: mock.Mock,
    ) -> None:
        mock_parse_args.return_value = mock.Mock(
            mode="audit",
            team="PRO",
            output=None,
        )
        mock_audit_live_issue_descriptions.return_value = [
            {"issue_identifier": "PRO-8", "status": "drifted"}
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(linear_issue_descriptions.main(), 1)


if __name__ == "__main__":
    unittest.main()
