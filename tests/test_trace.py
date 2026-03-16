from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.symphony import trace


class TraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.trace_root = pathlib.Path(self.temp_dir.name) / "traces"
        self.environ = {
            trace.TRACE_ENABLE_ENV: "1",
            trace.TRACE_ROOT_ENV: str(self.trace_root),
            trace.TRACE_MODE_ENV: "full",
        }
        self._env_patch = mock.patch.dict(os.environ, self.environ, clear=False)
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

    def test_create_run_records_events_artifacts_and_indexes(self) -> None:
        run = trace.create_run(
            issue_id="PRO-17",
            run_id="run-1",
            run_kind="implementation",
            branch="codex/pro-17-progressive-disclosure",
            pr_number=17,
            state_start="Todo",
            env_metadata={
                "workspace_path": "/tmp/workspaces/PRO-17",
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
                "LINEAR_API_KEY": "secret-value",
            },
        )

        artifact = trace.capture_text_artifact(
            issue_id="PRO-17",
            run_id="run-1",
            artifact_type="prompt",
            label="Rendered Worker Prompt",
            content="Prompt body",
            content_type="text/markdown",
            filename="rendered_prompt.md",
        )
        event = trace.capture_event(
            issue_id="PRO-17",
            run_id="run-1",
            actor="Symphony",
            stage="dispatch",
            summary="Dispatched worker with a frozen context pack",
            artifact_refs=[artifact["artifact_id"]],
        )
        finalized = trace.finalize_run(
            issue_id="PRO-17",
            run_id="run-1",
            state_end="In Progress",
        )
        index = trace.build_index(self.trace_root)

        self.assertEqual(run["run_id"], "run-1")
        self.assertEqual(event["stage"], "dispatch")
        self.assertEqual(finalized["state_end"], "In Progress")
        self.assertEqual(index["issues"][0]["issue_id"], "PRO-17")
        self.assertEqual(index["runs"]["run-1"]["issue_id"], "PRO-17")
        self.assertEqual(index["artifacts"][artifact["artifact_id"]]["run_id"], "run-1")
        self.assertNotIn("LINEAR_API_KEY", finalized["env_metadata"])

        run_manifest = json.loads(
            (self.trace_root / "issues" / "PRO-17" / "runs" / "run-1" / "run.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(run_manifest["events"]), 1)
        self.assertEqual(run_manifest["artifacts"][0]["artifact_id"], artifact["artifact_id"])

    def test_capture_workpad_revision_records_before_after_and_diff(self) -> None:
        trace.create_run(
            issue_id="PRO-42",
            run_id="run-2",
            run_kind="implementation",
            branch="codex/pro-42-workpad",
            state_start="In Progress",
        )

        revision = trace.capture_workpad_revision(
            issue_id="PRO-42",
            run_id="run-2",
            previous_body="# Old\n\n- note",
            current_body="# New\n\n- updated note",
            action="updated",
            comment_id="comment-1",
            comment_url="https://linear.app/example/comment-1",
        )

        self.assertEqual(revision["event"]["stage"], "workpad_updated")
        self.assertTrue(revision["diff"]["unified_diff"])
        self.assertIn("updated note", revision["diff"]["after"])

        run_manifest = trace.load_run_manifest(self.trace_root, "PRO-42", "run-2")
        self.assertEqual(len(run_manifest["diffs"]), 1)
        self.assertEqual(run_manifest["diffs"][0]["diff_id"], revision["diff"]["diff_id"])

    def test_redacts_secret_like_command_arguments(self) -> None:
        redacted = trace.redact_command_args(
            [
                "codex",
                "--api-key",
                "super-secret",
                "TOKEN=value",
                "--safe-flag",
            ]
        )

        self.assertEqual(redacted[0], "codex")
        self.assertEqual(redacted[2], "[REDACTED]")
        self.assertEqual(redacted[3], "TOKEN=[REDACTED]")
        self.assertEqual(redacted[4], "--safe-flag")


if __name__ == "__main__":
    unittest.main()
