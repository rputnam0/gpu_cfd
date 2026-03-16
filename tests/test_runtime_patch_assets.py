from __future__ import annotations

import pathlib
import unittest


class RuntimePatchAssetTests(unittest.TestCase):
    def repo_root(self) -> pathlib.Path:
        return pathlib.Path(__file__).resolve().parents[1]

    def test_runtime_patch_includes_continuation_store_file(self) -> None:
        patch_text = (
            self.repo_root()
            / "scripts"
            / "symphony"
            / "patches"
            / "symphony-thread-resume-v2.patch"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "diff --git a/elixir/lib/symphony_elixir/continuation_store.ex "
            "b/elixir/lib/symphony_elixir/continuation_store.ex",
            patch_text,
        )
        self.assertIn("defmodule SymphonyElixir.ContinuationStore do", patch_text)

    def test_apply_runtime_patch_normalizes_to_git_toplevel(self) -> None:
        script_text = (
            self.repo_root() / "scripts" / "symphony" / "apply_runtime_patch.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('git -C "$TARGET_DIR_INPUT" rev-parse --show-toplevel', script_text)


if __name__ == "__main__":
    unittest.main()
