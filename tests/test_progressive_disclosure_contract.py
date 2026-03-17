from __future__ import annotations

import pathlib
import unittest


class ProgressiveDisclosureContractTests(unittest.TestCase):
    def repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def read_text(self, relative_path: str) -> str:
        return (self.repo_root() / relative_path).read_text(encoding="utf-8")

    def test_readme_first_is_removed(self) -> None:
        self.assertFalse((self.repo_root() / "docs" / "README_FIRST.md").exists())

    def test_agents_is_map_first_and_concise(self) -> None:
        agents_text = self.read_text("AGENTS.md")
        agents_lines = agents_text.splitlines()

        self.assertIn("## Repository Map", agents_text)
        self.assertIn("```text", agents_text)
        self.assertIn("├── docs/", agents_text)
        self.assertIn("└── dev/", agents_text)
        self.assertLess(
            agents_text.index("## Repository Map"),
            agents_text.index("## Source of Truth Order"),
        )
        self.assertIn("Do not read the full docs corpus by default.", agents_text)
        self.assertLessEqual(len(agents_lines), 100)
        self.assertNotIn("## Versioning and Release Notes", agents_text)
        self.assertNotIn("## Security and Secrets", agents_text)

    def test_docs_have_progressive_disclosure_indexes(self) -> None:
        authority_index = self.read_text("docs/authority/README.md")
        specs_index = self.read_text("docs/specs/README.md")

        self.assertIn("## Authority Order", authority_index)
        self.assertIn("## Consumption Rules", authority_index)
        self.assertIn("## How To Use These Specs", specs_index)

    def test_worker_contract_starts_from_agents_and_workpad(self) -> None:
        workflow_text = self.read_text("WORKFLOW.md")
        skill_text = self.read_text(".codex/skills/gpu-cfd-symphony/SKILL.md")

        self.assertIn("1. `AGENTS.md`", workflow_text)
        self.assertIn("scripts/symphony/codex_dispatch.py", workflow_text)
        self.assertNotIn("3. `docs/tasks/pr_inventory.md`", workflow_text)
        self.assertIn("use `docs/tasks/pr_inventory.md` as the fallback map", workflow_text)
        self.assertIn("Write or update the canonical Linear workpad", workflow_text)
        self.assertIn("same implementation worker", workflow_text)
        self.assertIn("moves the issue to `In Review`", workflow_text)
        self.assertIn("The finite local-review cycle is 3 total passes", workflow_text)
        self.assertIn("returns `stop_worker=true`", workflow_text)
        self.assertNotIn("README_FIRST", workflow_text)
        self.assertIn("1. Open `AGENTS.md`.", skill_text)
        self.assertNotIn("2. Open `docs/tasks/pr_inventory.md`.", skill_text)
        self.assertIn("use `docs/tasks/pr_inventory.md` as the fallback map", skill_text)
        self.assertIn("same implementation worker", skill_text)
        self.assertIn("`Backlog` issue per residual finding", skill_text)
        self.assertIn("returns `stop_worker=true`", skill_text)
        self.assertNotIn("README_FIRST", skill_text)

    def test_repo_docs_no_longer_reference_readme_first(self) -> None:
        tracked_paths = [
            "docs/README.md",
            "docs/tasks/README.md",
            "docs/tasks/pr_card_template.md",
            "docs/tasks/review_checklist.md",
            "docs/ops/symphony_runbook.md",
            "docs/authority/semantic_source_map.md",
        ]

        for relative_path in tracked_paths:
            with self.subTest(path=relative_path):
                self.assertNotIn("README_FIRST", self.read_text(relative_path))


if __name__ == "__main__":
    unittest.main()
