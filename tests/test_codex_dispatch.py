from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.symphony import codex_dispatch


class CodexDispatchTests(unittest.TestCase):
    def repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def test_render_workflow_prompt_inlines_issue_context(self) -> None:
        workflow_text = (self.repo_root() / "WORKFLOW.md").read_text(encoding="utf-8")
        rendered = codex_dispatch.render_workflow_prompt(
            workflow_text,
            {
                "identifier": "PRO-17",
                "title": "P4-08 Observability trace viewer",
                "description": "Build the trace viewer.",
                "url": "https://linear.app/example/PRO-17",
                "state": {"name": "Todo"},
                "labels": {"nodes": [{"name": "observability"}, {"name": "symphony"}]},
            },
            attempt="second run",
        )

        self.assertIn("Linear issue `PRO-17`", rendered)
        self.assertIn("Current state: `Todo`", rendered)
        self.assertIn("Labels: `observability, symphony`", rendered)
        self.assertIn("Attempt: `second run`", rendered)
        self.assertIn("Build the trace viewer.", rendered)
        self.assertTrue(rendered.startswith("You are working on Linear issue `PRO-17`"))
        self.assertNotIn("tracker:", rendered)
        self.assertNotIn("project_slug:", rendered)
        self.assertNotIn("approval_policy:", rendered)

    def test_resolve_pr_context_finds_owning_task_file_and_card(self) -> None:
        context = codex_dispatch.resolve_pr_context(
            self.repo_root(),
            {
                "identifier": "PRO-17",
                "title": "Implement P4-08 trace viewer observability",
                "description": "This issue is scoped to P4-08.",
            },
        )

        self.assertIsNotNone(context)
        assert context is not None
        self.assertEqual(context["pr_id"], "P4-08")
        self.assertTrue(context["task_file"].endswith("06_phase4_pressure_linear_algebra.md"))
        self.assertIn("## P4-08", context["card_markdown"])
        self.assertTrue(context["cited_paths"])

    @mock.patch("scripts.symphony.codex_dispatch.linear_api.fetch_issue")
    def test_fetch_issue_snapshot_raises_when_linear_issue_lookup_fails(
        self,
        mock_fetch_issue: mock.Mock,
    ) -> None:
        mock_fetch_issue.side_effect = ValueError("LINEAR_API_KEY is required")

        with self.assertRaises(codex_dispatch.DispatchError) as exc_info:
            codex_dispatch.fetch_issue_snapshot("PRO-17")

        self.assertIn("Failed to load Linear issue PRO-17", str(exc_info.exception))

    def test_fetch_issue_snapshot_uses_override_file_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_path = pathlib.Path(temp_dir) / "issue.json"
            snapshot_path.write_text(
                json.dumps(
                    {
                        "identifier": "PRO-8",
                        "title": "FND-04 override",
                        "description": "override payload",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {codex_dispatch.ISSUE_SNAPSHOT_PATH_ENV: str(snapshot_path)},
                clear=False,
            ):
                snapshot = codex_dispatch.fetch_issue_snapshot("PRO-8")

        self.assertEqual(snapshot["title"], "FND-04 override")
        self.assertEqual(snapshot["description"], "override payload")


if __name__ == "__main__":
    unittest.main()
