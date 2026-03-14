from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest

from scripts.symphony import workspace_sync


def git(path: pathlib.Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=path,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


class WorkspaceSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        self.origin = self.root / "origin.git"
        self.seed = self.root / "seed"
        self.workspace = self.root / "workspace"

        subprocess.run(
            ["git", "init", "--bare", "--initial-branch=main", self.origin.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "clone", self.origin.as_posix(), self.seed.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        )
        git(self.seed, "config", "user.name", "Codex Test")
        git(self.seed, "config", "user.email", "codex@example.com")
        (self.seed / "tracked.txt").write_text("v1\n", encoding="utf-8")
        git(self.seed, "add", "tracked.txt")
        git(self.seed, "commit", "-m", "seed")
        git(self.seed, "push", "origin", "main")
        subprocess.run(
            ["git", "clone", self.origin.as_posix(), self.workspace.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        )
        git(self.workspace, "config", "user.name", "Codex Test")
        git(self.workspace, "config", "user.email", "codex@example.com")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def push_origin_update(self, contents: str) -> str:
        (self.seed / "tracked.txt").write_text(contents, encoding="utf-8")
        git(self.seed, "add", "tracked.txt")
        git(self.seed, "commit", "-m", f"update {contents.strip()}")
        git(self.seed, "push", "origin", "main")
        return git(self.seed, "rev-parse", "HEAD")

    def test_fast_forwards_clean_main_workspace(self) -> None:
        target_head = self.push_origin_update("v2\n")

        result = workspace_sync.sync_workspace(self.workspace)

        self.assertIn("synced `main`", result)
        self.assertEqual(git(self.workspace, "rev-parse", "HEAD"), target_head)

    def test_skips_non_base_branch(self) -> None:
        self.push_origin_update("v2\n")
        git(self.workspace, "checkout", "-b", "codex/pro-5")
        original_head = git(self.workspace, "rev-parse", "HEAD")

        result = workspace_sync.sync_workspace(self.workspace)

        self.assertIn("skipped: current branch `codex/pro-5`", result)
        self.assertEqual(git(self.workspace, "rev-parse", "HEAD"), original_head)

    def test_skips_dirty_base_branch(self) -> None:
        self.push_origin_update("v2\n")
        (self.workspace / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        original_head = git(self.workspace, "rev-parse", "HEAD")

        with self.assertRaisesRegex(
            workspace_sync.WorkspaceSyncError,
            "base branch `main` has local changes",
        ):
            workspace_sync.sync_workspace(self.workspace)

        self.assertEqual(git(self.workspace, "rev-parse", "HEAD"), original_head)

    def test_switches_to_bootstrap_ref_when_requested(self) -> None:
        git(self.seed, "checkout", "-b", "codex/bootstrap-sync")
        target_head = self.push_origin_update("bootstrap\n")
        git(self.seed, "push", "-u", "origin", "codex/bootstrap-sync")
        git(self.seed, "checkout", "main")

        result = workspace_sync.sync_workspace(
            self.workspace, bootstrap_ref="codex/bootstrap-sync"
        )

        self.assertIn("synced `codex/bootstrap-sync`", result)
        self.assertEqual(
            git(self.workspace, "branch", "--show-current"), "codex/bootstrap-sync"
        )
        self.assertEqual(git(self.workspace, "rev-parse", "HEAD"), target_head)
