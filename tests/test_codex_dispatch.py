from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock

from scripts.authority import load_authority_bundle, render_source_audit_note
from scripts.symphony import codex_dispatch
from scripts.symphony.runtime_config import CodexProfile


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

    def test_enforce_source_audit_gate_rejects_missing_workpad_metadata(self) -> None:
        pr_context = {
            "pr_id": "P5-02",
            "card_markdown": "Cites docs/authority/semantic_source_map.md for implementation.",
        }

        with self.assertRaisesRegex(
            codex_dispatch.DispatchError,
            "implementation planning requires a reviewed source-audit note",
        ):
            codex_dispatch.enforce_source_audit_gate(
                self.repo_root(),
                pr_context,
                None,
            )

    def test_enforce_source_audit_gate_accepts_reviewed_note_metadata(self) -> None:
        bundle = load_authority_bundle(self.repo_root())
        note_text = render_source_audit_note(
            bundle,
            touched_surfaces=["Alpha transport"],
            review_status="reviewed",
        )
        note_path = self.repo_root() / "phase5_source_audit_note.md"
        note_path.write_text(note_text, encoding="utf-8")
        self.addCleanup(note_path.unlink)

        codex_dispatch.enforce_source_audit_gate(
            self.repo_root(),
            {
                "pr_id": "P5-02",
                "card_markdown": "Cites docs/authority/semantic_source_map.md for implementation.",
            },
            {
                "body": "\n".join(
                    [
                        "- Source audit note: phase5_source_audit_note.md",
                        "- Touched semantic surfaces: Alpha transport",
                    ]
                )
            },
        )

    def test_main_preserves_worker_exit_code_when_trace_finalization_fetch_fails(
        self,
    ) -> None:
        initial_issue = {
            "identifier": "PRO-17",
            "title": "P4-08 Observability trace viewer",
            "url": "https://linear.app/example/PRO-17",
            "state": {"name": "Todo"},
            "labels": {"nodes": [{"name": "observability"}]},
        }
        runtime_profile = CodexProfile(model="gpt-5.4", reasoning_effort="medium")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            workspace = temp_path / "PRO-17"
            workspace.mkdir()
            stderr_buffer = io.StringIO()

            with (
                mock.patch.object(
                    codex_dispatch,
                    "parse_args",
                    return_value=argparse.Namespace(codex_args=["app-server"]),
                ),
                mock.patch.object(codex_dispatch, "repo_root", return_value=self.repo_root()),
                mock.patch.object(codex_dispatch, "workspace_root", return_value=workspace),
                mock.patch.object(
                    codex_dispatch,
                    "fetch_issue_snapshot",
                    side_effect=[
                        initial_issue,
                        codex_dispatch.DispatchError("final issue fetch failed"),
                    ],
                ),
                mock.patch.object(codex_dispatch, "current_branch", return_value="codex/test"),
                mock.patch.object(
                    codex_dispatch.runtime_config,
                    "build_codex_command",
                    return_value=["codex", "app-server"],
                ),
                mock.patch.object(
                    codex_dispatch.runtime_config,
                    "load_codex_profile",
                    return_value=runtime_profile,
                ),
                mock.patch.object(codex_dispatch.trace, "is_enabled", return_value=True),
                mock.patch.object(codex_dispatch.trace, "latest_run_summary", return_value=None),
                mock.patch.object(
                    codex_dispatch.trace,
                    "create_run",
                    return_value={"issue_id": "PRO-17", "run_id": "run-1"},
                ),
                mock.patch.object(codex_dispatch, "capture_dispatch_bundle"),
                mock.patch.object(codex_dispatch.trace, "resolve_trace_root", return_value=temp_path),
                mock.patch.object(codex_dispatch.trace, "run_dir", return_value=temp_path / "run"),
                mock.patch.object(
                    codex_dispatch,
                    "launch_codex_proxy",
                    return_value=(0, {}),
                ),
                mock.patch.object(codex_dispatch, "current_commit", side_effect=["before", "after"]),
                mock.patch.object(codex_dispatch.trace, "capture_event") as mock_capture_event,
                mock.patch.object(codex_dispatch.trace, "finalize_run") as mock_finalize_run,
                mock.patch("sys.stderr", stderr_buffer),
            ):
                exit_code = codex_dispatch.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("trace finalization: could not fetch final issue state", stderr_buffer.getvalue())
        self.assertEqual(
            mock_capture_event.call_args.kwargs["metadata"]["final_state"],
            "Todo",
        )
        self.assertEqual(mock_finalize_run.call_args.kwargs["state_end"], "Todo")

    def test_main_prefetches_issue_for_source_audit_gate_even_when_trace_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            workspace = temp_path / "PRO-17"
            workspace.mkdir()

            with (
                mock.patch.object(
                    codex_dispatch,
                    "parse_args",
                    return_value=argparse.Namespace(codex_args=["app-server"]),
                ),
                mock.patch.object(codex_dispatch, "repo_root", return_value=self.repo_root()),
                mock.patch.object(codex_dispatch, "workspace_root", return_value=workspace),
                mock.patch.object(codex_dispatch, "current_branch", return_value="main"),
                mock.patch.object(
                    codex_dispatch,
                    "fetch_issue_snapshot",
                    return_value={"identifier": "PRO-17", "title": "P4-08", "state": {"name": "Todo"}},
                ) as mock_fetch_issue,
                mock.patch.object(codex_dispatch, "resolve_pr_context", return_value=None),
                mock.patch.object(codex_dispatch, "find_workpad_snapshot", return_value=None),
                mock.patch.object(codex_dispatch, "enforce_source_audit_gate") as mock_gate,
                mock.patch.object(
                    codex_dispatch.runtime_config,
                    "build_codex_command",
                    return_value=["codex", "app-server"],
                ),
                mock.patch.object(codex_dispatch.trace, "is_enabled", return_value=False),
                mock.patch.object(
                    codex_dispatch,
                    "launch_codex_proxy",
                    return_value=(0, {}),
                ) as mock_launch,
            ):
                exit_code = codex_dispatch.main()

        self.assertEqual(exit_code, 0)
        mock_fetch_issue.assert_called_once_with("PRO-17")
        mock_gate.assert_called_once()
        self.assertEqual(
            mock_launch.call_args.kwargs["command"],
            ["codex", "app-server"],
        )
        self.assertIsNone(mock_launch.call_args.kwargs["transcript_dir"])

    def test_stream_copy_flushes_small_chunks_without_waiting_for_eof(self) -> None:
        read_fd, write_fd = os.pipe()
        source = io.BufferedReader(io.FileIO(read_fd, "rb", closefd=True))

        class Sink:
            def __init__(self) -> None:
                self.data = bytearray()

            def write(self, chunk: bytes) -> int:
                self.data.extend(chunk)
                return len(chunk)

            def flush(self) -> None:
                return None

            def close(self) -> None:
                return None

        sink = Sink()
        copy_thread = threading.Thread(
            target=codex_dispatch._stream_copy,
            args=(source, [sink]),
            daemon=True,
        )
        copy_thread.start()

        os.write(write_fd, b'{"id":1}\n')

        deadline = time.time() + 1.0
        while time.time() < deadline and sink.data != b'{"id":1}\n':
            time.sleep(0.01)

        self.assertEqual(bytes(sink.data), b'{"id":1}\n')

        os.close(write_fd)
        copy_thread.join(timeout=1.0)
        source.close()


if __name__ == "__main__":
    unittest.main()
