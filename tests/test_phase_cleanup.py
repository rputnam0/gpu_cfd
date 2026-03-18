from __future__ import annotations

import pathlib
import unittest
from unittest import mock

from scripts.symphony import linear_api, phase_cleanup


class PhaseCleanupTests(unittest.TestCase):
    def repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def make_issue(
        self,
        identifier: str,
        title: str,
        *,
        state: str,
        description: str = "",
        parent_identifier: str | None = None,
        labels: list[str] | None = None,
    ) -> dict:
        return {
            "id": f"id-{identifier}",
            "identifier": identifier,
            "title": title,
            "description": description,
            "url": f"https://linear.app/{identifier}",
            "parent": (
                {"id": f"id-{parent_identifier}", "identifier": parent_identifier}
                if parent_identifier
                else None
            ),
            "labels": {"nodes": [{"name": name} for name in (labels or [])]},
            "state": {"name": state},
        }

    def section(self, slug: str) -> phase_cleanup.SectionInfo:
        return next(
            section
            for section in phase_cleanup.load_sections(self.repo_root())
            if section.slug == slug
        )

    @mock.patch("scripts.symphony.phase_cleanup.linear_api.upsert_workpad_comment")
    @mock.patch("scripts.symphony.phase_cleanup.linear_api.update_issue")
    @mock.patch("scripts.symphony.phase_cleanup.linear_api.resolve_label_ids")
    @mock.patch("scripts.symphony.phase_cleanup.linear_api.create_issue")
    def test_ensure_cleanup_sweep_auto_closes_clean_boundary(
        self,
        mock_create_issue: mock.Mock,
        mock_resolve_label_ids: mock.Mock,
        mock_update_issue: mock.Mock,
        mock_upsert_workpad_comment: mock.Mock,
    ) -> None:
        section = self.section("foundation")
        implementation_issues = {
            pr_id: self.make_issue(f"PRO-{index}", f"{pr_id}: done", state="Done")
            for index, pr_id in enumerate(section.pr_ids, start=1)
        }
        team_issue = {
            "identifier": "PRO-1",
            "title": "FND-01: done",
            "team": {
                "id": "team-1",
                "states": {"nodes": [{"id": "todo-state", "name": "Todo"}]},
            },
            "state": {"name": "Done"},
        }
        mock_resolve_label_ids.return_value = ["label-1"]
        mock_create_issue.return_value = {
            "id": "cleanup-id",
            "identifier": "PRO-999",
            "title": "Phase cleanup: Foundation",
            "url": "https://linear.app/PRO-999",
            "state": {"name": "Todo"},
        }
        mock_update_issue.return_value = {
            "changed": True,
            "previous_state": "Todo",
            "current_state": "Done",
        }

        result = phase_cleanup.ensure_cleanup_sweep(
            section=section,
            current_issue=team_issue,
            team_issues=list(implementation_issues.values()),
            implementation_issues=implementation_issues,
        )

        self.assertTrue(result["auto_closed"])
        self.assertEqual(result["residual_followup_count"], 0)
        mock_create_issue.assert_called_once()
        mock_update_issue.assert_called_once_with("PRO-999", target_state_name="Done")
        mock_upsert_workpad_comment.assert_called_once()

    @mock.patch("scripts.symphony.phase_cleanup.linear_api.update_issue_state")
    def test_promote_ready_backlog_issues_blocks_until_cleanup_done(
        self,
        mock_update_issue_state: mock.Mock,
    ) -> None:
        phase2 = self.section("phase-2")
        team_issues = [
            self.make_issue(
                f"PRO-P2-{index}",
                f"{pr_id}: done",
                state="Done",
            )
            for index, pr_id in enumerate(phase2.pr_ids, start=1)
        ]
        team_issues.append(
            self.make_issue(
                "PRO-CLEANUP",
                "Phase cleanup: Phase 2",
                state="Todo",
                description=linear_api.render_phase_cleanup_marker(phase2.slug),
                labels=["phase-cleanup"],
            )
        )
        team_issues.append(
            self.make_issue(
                "PRO-P3-01",
                "P3-01: Synchronization and stream inventory",
                state="Backlog",
            )
        )

        promoted = phase_cleanup.promote_ready_backlog_issues(
            team_key="PRO",
            team_issues=team_issues,
            root=self.repo_root(),
        )

        self.assertEqual(promoted, [])
        mock_update_issue_state.assert_not_called()

    @mock.patch("scripts.symphony.phase_cleanup.linear_api.update_issue_state")
    def test_promote_ready_backlog_issues_releases_when_cleanup_is_done(
        self,
        mock_update_issue_state: mock.Mock,
    ) -> None:
        phase2 = self.section("phase-2")
        team_issues = [
            self.make_issue(
                f"PRO-P2-{index}",
                f"{pr_id}: done",
                state="Done",
            )
            for index, pr_id in enumerate(phase2.pr_ids, start=1)
        ]
        team_issues.append(
            self.make_issue(
                "PRO-CLEANUP",
                "Phase cleanup: Phase 2",
                state="Done",
                description=linear_api.render_phase_cleanup_marker(phase2.slug),
                labels=["phase-cleanup"],
            )
        )
        team_issues.append(
            self.make_issue(
                "PRO-P3-01",
                "P3-01: Synchronization and stream inventory",
                state="Backlog",
            )
        )
        mock_update_issue_state.return_value = {
            "changed": True,
            "previous_state": "Backlog",
            "current_state": "Todo",
        }

        promoted = phase_cleanup.promote_ready_backlog_issues(
            team_key="PRO",
            team_issues=team_issues,
            root=self.repo_root(),
        )

        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["identifier"], "PRO-P3-01")
        mock_update_issue_state.assert_called_once_with("PRO-P3-01", "Todo")

    @mock.patch("scripts.symphony.phase_cleanup.linear_api.update_issue")
    def test_finalize_cleanup_sweep_when_children_are_done(
        self,
        mock_update_issue: mock.Mock,
    ) -> None:
        phase2 = self.section("phase-2")
        sweep = self.make_issue(
            "PRO-CLEANUP",
            "Phase cleanup: Phase 2",
            state="Todo",
            description=linear_api.render_phase_cleanup_marker(phase2.slug),
        )
        child = self.make_issue(
            "PRO-CLEANUP-CHILD",
            "Cleanup child",
            state="Done",
            parent_identifier="PRO-CLEANUP",
        )
        mock_update_issue.return_value = {
            "changed": True,
            "previous_state": "Todo",
            "current_state": "Done",
        }

        result = phase_cleanup.finalize_cleanup_sweep_if_ready(phase2, [sweep, child])

        self.assertEqual(result["identifier"], "PRO-CLEANUP")
        mock_update_issue.assert_called_once_with("PRO-CLEANUP", target_state_name="Done")


if __name__ == "__main__":
    unittest.main()
