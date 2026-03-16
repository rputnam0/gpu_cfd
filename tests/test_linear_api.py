from __future__ import annotations

import unittest
from unittest import mock

from scripts.symphony import linear_api


class LinearApiTests(unittest.TestCase):
    def test_render_workpad_body_includes_marker_sections_and_guidance(self) -> None:
        body = linear_api.render_workpad_body(
            issue_identifier="PRO-17",
            issue_title="Progressive disclosure cutover",
            current_status="planning",
            execution_plan=[
                "Resolve the PR card before reading deeper docs.",
                "Record decisions in the workpad before edits.",
            ],
            scoped_sources=[
                "AGENTS.md",
                "docs/tasks/pr_inventory.md",
            ],
            decisions_and_rationale=[
                "Use the narrowest reversible implementation allowed by the task card.",
            ],
            validation=["Run the targeted Linear/workflow regression tests first."],
            review_handoff_notes=["Future agent note: do not widen the PR scope."],
        )

        self.assertIn(linear_api.WORKPAD_MARKER, body)
        self.assertIn("## Task Summary", body)
        self.assertIn("## Execution Plan", body)
        self.assertIn("## Decisions and Rationale", body)
        self.assertIn("Tried X, rejected because Y in the task card/spec", body)
        self.assertIn("Current status: planning", body)

    @mock.patch("scripts.symphony.linear_api.update_comment")
    @mock.patch("scripts.symphony.linear_api.create_comment")
    @mock.patch("scripts.symphony.linear_api.find_workpad_comment")
    def test_upsert_workpad_updates_existing_comment(
        self,
        mock_find_workpad_comment: mock.Mock,
        mock_create_comment: mock.Mock,
        mock_update_comment: mock.Mock,
    ) -> None:
        mock_find_workpad_comment.return_value = {"id": "comment-1", "body": "old"}
        mock_update_comment.return_value = {
            "id": "comment-1",
            "body": "new",
            "url": "https://linear.app/comment-1",
        }

        result = linear_api.upsert_workpad_comment(
            "PRO-17",
            body="new",
        )

        self.assertEqual(result["id"], "comment-1")
        self.assertEqual(result["action"], "updated")
        mock_create_comment.assert_not_called()
        mock_update_comment.assert_called_once_with("comment-1", "new")

    @mock.patch("scripts.symphony.linear_api.update_comment")
    @mock.patch("scripts.symphony.linear_api.create_comment")
    @mock.patch("scripts.symphony.linear_api.find_workpad_comment")
    def test_upsert_workpad_creates_when_missing(
        self,
        mock_find_workpad_comment: mock.Mock,
        mock_create_comment: mock.Mock,
        mock_update_comment: mock.Mock,
    ) -> None:
        mock_find_workpad_comment.return_value = None
        mock_create_comment.return_value = {
            "id": "comment-2",
            "body": "new",
            "url": "https://linear.app/comment-2",
        }

        result = linear_api.upsert_workpad_comment(
            "PRO-17",
            body="new",
        )

        self.assertEqual(result["id"], "comment-2")
        self.assertEqual(result["action"], "created")
        mock_update_comment.assert_not_called()
        mock_create_comment.assert_called_once_with("PRO-17", "new")

    @mock.patch("scripts.symphony.linear_api.graphql")
    def test_find_workpad_comment_searches_past_first_comments_page(
        self,
        mock_graphql: mock.Mock,
    ) -> None:
        mock_graphql.side_effect = [
            {
                "issues": {
                    "nodes": [
                        {
                            "id": "issue-1",
                            "identifier": "PRO-17",
                            "title": "Example",
                            "comments": {
                                "pageInfo": {
                                    "hasNextPage": True,
                                    "endCursor": "cursor-1",
                                },
                                "nodes": [
                                    {
                                        "id": "comment-1",
                                        "body": "ordinary comment",
                                        "url": "https://linear.app/comment-1",
                                        "createdAt": "2026-03-15T10:00:00Z",
                                        "updatedAt": "2026-03-15T10:00:00Z",
                                    }
                                ],
                            },
                        }
                    ]
                }
            },
            {
                "issues": {
                    "nodes": [
                        {
                            "id": "issue-1",
                            "identifier": "PRO-17",
                            "title": "Example",
                            "comments": {
                                "pageInfo": {
                                    "hasNextPage": False,
                                    "endCursor": None,
                                },
                                "nodes": [
                                    {
                                        "id": "comment-2",
                                        "body": (
                                            f"{linear_api.WORKPAD_MARKER}\n\n"
                                            f"{linear_api.WORKPAD_TITLE}\n"
                                        ),
                                        "url": "https://linear.app/comment-2",
                                        "createdAt": "2026-03-15T11:00:00Z",
                                        "updatedAt": "2026-03-15T11:00:00Z",
                                    }
                                ],
                            },
                        }
                    ]
                }
            },
        ]

        comment = linear_api.find_workpad_comment("PRO-17")

        self.assertIsNotNone(comment)
        assert comment is not None
        self.assertEqual(comment["id"], "comment-2")
        self.assertEqual(mock_graphql.call_count, 2)
        self.assertIsNone(mock_graphql.call_args_list[0].args[1]["after"])
        self.assertEqual(mock_graphql.call_args_list[1].args[1]["after"], "cursor-1")

    def test_parse_issue_identifier(self) -> None:
        parsed = linear_api.parse_issue_identifier("pro-17")

        self.assertEqual(parsed.team_key, "PRO")
        self.assertEqual(parsed.number, 17)

    def test_extract_issue_identifiers_deduplicates_and_normalizes(self) -> None:
        identifiers = linear_api.extract_issue_identifiers(
            "Closes pro-17 and PRO-18",
            "Follow-up for PRO-17 on another line",
        )

        self.assertEqual(identifiers, ["PRO-17", "PRO-18"])

    def test_extract_issue_identifiers_ignores_non_pro_patterns(self) -> None:
        identifiers = linear_api.extract_issue_identifiers(
            "HTTP-404 should not match",
            "OPS-01 touches FND-03 but only PRO-19 is a Linear issue here",
        )

        self.assertEqual(identifiers, ["PRO-19"])

    def test_dependent_is_unblocked_only_when_all_blockers_are_done(self) -> None:
        dependent = {
            "inverseRelations": {
                "nodes": [
                    {
                        "type": "blocks",
                        "issue": {"identifier": "PRO-5", "state": {"name": "Done"}},
                    },
                    {
                        "type": "blocks",
                        "issue": {
                            "identifier": "PRO-6",
                            "state": {"name": "In Progress"},
                        },
                    },
                ]
            }
        }

        self.assertFalse(linear_api.dependent_is_unblocked(dependent))

    @mock.patch("scripts.symphony.linear_api.update_issue_state")
    @mock.patch("scripts.symphony.linear_api.fetch_direct_blocked_dependents")
    def test_release_direct_unblocked_dependents_promotes_only_ready_backlog_items(
        self,
        mock_fetch_direct_blocked_dependents: mock.Mock,
        mock_update_issue_state: mock.Mock,
    ) -> None:
        mock_fetch_direct_blocked_dependents.return_value = [
            {
                "identifier": "PRO-7",
                "state": {"name": "Backlog"},
                "inverseRelations": {
                    "nodes": [
                        {
                            "type": "blocks",
                            "issue": {"identifier": "PRO-5", "state": {"name": "Done"}},
                        }
                    ]
                },
            },
            {
                "identifier": "PRO-8",
                "state": {"name": "Backlog"},
                "inverseRelations": {
                    "nodes": [
                        {
                            "type": "blocks",
                            "issue": {
                                "identifier": "PRO-6",
                                "state": {"name": "In Review"},
                            },
                        }
                    ]
                },
            },
            {
                "identifier": "PRO-9",
                "state": {"name": "Todo"},
                "inverseRelations": {"nodes": []},
            },
        ]
        mock_update_issue_state.return_value = {
            "previous_state": "Backlog",
            "current_state": "Todo",
            "changed": True,
        }

        promoted = linear_api.release_direct_unblocked_dependents("PRO-5")

        self.assertEqual(
            promoted,
            [
                {
                    "identifier": "PRO-7",
                    "previous_state": "Backlog",
                    "current_state": "Todo",
                    "changed": True,
                }
            ],
        )
        mock_update_issue_state.assert_called_once_with("PRO-7", "Todo")


if __name__ == "__main__":
    unittest.main()
