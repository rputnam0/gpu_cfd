from __future__ import annotations

import unittest
from unittest import mock

from scripts.symphony import linear_api


class LinearApiTests(unittest.TestCase):
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
