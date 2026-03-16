from __future__ import annotations

import pathlib
import unittest

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


if __name__ == "__main__":
    unittest.main()
