from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest

from scripts.symphony import workspace_sync


class WorkspaceSyncTests(unittest.TestCase):
    def git(self, root: pathlib.Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(
                f"git {' '.join(args)} failed in {root}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed.stdout

    def write_file(self, root: pathlib.Path, relative_path: str, content: str) -> None:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def make_control_and_workspace(self) -> tuple[pathlib.Path, pathlib.Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_root = pathlib.Path(temp_dir.name)
        origin = temp_root / "origin.git"
        control = temp_root / "control"
        workspace = temp_root / "workspace"

        self.git(temp_root, "init", "--bare", origin.as_posix())
        self.git(temp_root, "clone", origin.as_posix(), control.as_posix())
        self.git(control, "config", "user.email", "tests@example.com")
        self.git(control, "config", "user.name", "Workspace Sync Tests")
        self.write_file(control, "WORKFLOW.md", "canonical-v1\n")
        self.write_file(control, "scripts/symphony/runtime_config.toml", "[codex]\n")
        self.git(control, "add", ".")
        self.git(control, "commit", "-m", "initial")
        self.git(control, "branch", "-M", "main")
        self.git(control, "push", "-u", "origin", "main")
        self.git(origin, "symbolic-ref", "HEAD", "refs/heads/main")

        self.git(temp_root, "clone", origin.as_posix(), workspace.as_posix())
        self.git(workspace, "config", "user.email", "tests@example.com")
        self.git(workspace, "config", "user.name", "Workspace Sync Tests")
        self.git(workspace, "checkout", "-b", "codex/test-sync")
        return control, workspace

    def test_sync_control_plane_updates_untouched_allowlisted_files(self) -> None:
        control, workspace = self.make_control_and_workspace()

        self.write_file(control, "WORKFLOW.md", "canonical-v2\n")
        self.git(control, "add", "WORKFLOW.md")
        self.git(control, "commit", "-m", "update workflow")
        self.git(control, "push", "origin", "main")

        result = workspace_sync.sync_control_plane(control, workspace)

        self.assertEqual((workspace / "WORKFLOW.md").read_text(encoding="utf-8"), "canonical-v2\n")
        self.assertIn("WORKFLOW.md", result["synced_paths"])
        self.assertEqual(result["status"], "ok")

    def test_sync_control_plane_skips_branch_owned_files(self) -> None:
        control, workspace = self.make_control_and_workspace()

        self.write_file(workspace, "WORKFLOW.md", "branch-owned\n")
        self.git(workspace, "add", "WORKFLOW.md")
        self.git(workspace, "commit", "-m", "workspace workflow change")

        self.write_file(control, "WORKFLOW.md", "canonical-v2\n")
        self.git(control, "add", "WORKFLOW.md")
        self.git(control, "commit", "-m", "update workflow")
        self.git(control, "push", "origin", "main")

        result = workspace_sync.sync_control_plane(control, workspace)

        self.assertEqual((workspace / "WORKFLOW.md").read_text(encoding="utf-8"), "branch-owned\n")
        self.assertIn("WORKFLOW.md", result["branch_owned_paths"])
        self.assertNotIn("WORKFLOW.md", result["synced_paths"])

    def test_sync_control_plane_blocks_dirty_workspace_files(self) -> None:
        control, workspace = self.make_control_and_workspace()

        self.write_file(control, "WORKFLOW.md", "canonical-v2\n")
        self.write_file(control, "scripts/symphony/runtime_config.toml", "[codex]\nmodel = 'gpt-5.4'\n")
        self.git(control, "add", "WORKFLOW.md", "scripts/symphony/runtime_config.toml")
        self.git(control, "commit", "-m", "update control plane")
        self.git(control, "push", "origin", "main")

        self.write_file(workspace, "WORKFLOW.md", "dirty change\n")

        with self.assertRaisesRegex(
            workspace_sync.WorkspaceSyncError,
            "refusing to overwrite dirty control-plane files",
        ):
            workspace_sync.sync_control_plane(control, workspace)

        self.assertEqual((workspace / "WORKFLOW.md").read_text(encoding="utf-8"), "dirty change\n")
        self.assertEqual(
            (workspace / "scripts/symphony/runtime_config.toml").read_text(encoding="utf-8"),
            "[codex]\n",
        )


if __name__ == "__main__":
    unittest.main()
