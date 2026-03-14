from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.symphony import resume_context


class ResumeContextTests(unittest.TestCase):
    def test_render_resume_context_includes_pr_review_and_events(self) -> None:
        snapshot = resume_context.ResumeSnapshot(
            generated_at="2026-03-14T10:00:00Z",
            issue="PRO-7",
            workspace="/tmp/PRO-7",
            branch="rputnam0/pro-7-example",
            head_commit="abc123",
            base_ref="origin/main",
            commits_ahead=["abc123 Add ladder resolver", "def456 Tighten tests"],
            changed_files=["scripts/foo.py", "tests/test_foo.py"],
            pull_request=resume_context.PullRequestSnapshot(
                number=12,
                url="https://github.com/example/repo/pull/12",
                title="FND-03: Example",
                state="OPEN",
                is_draft=False,
                review_decision=None,
            ),
            review_status="findings",
            review_message="- [P1] Example finding",
            review_artifact=".codex/review_artifacts/branch/latest.jsonl",
            recent_events=[
                {
                    "timestamp": "2026-03-14T09:55:00Z",
                    "event_type": "review_requested",
                    "message": "Waiting for external review",
                }
            ],
            warnings=["No PR comments imported yet."],
        )

        rendered = resume_context.render_resume_context(snapshot)

        self.assertIn("# Worker Resume Context", rendered)
        self.assertIn("`#12` `FND-03: Example`", rendered)
        self.assertIn("`scripts/foo.py`", rendered)
        self.assertIn("Status: `findings`", rendered)
        self.assertIn("review_requested", rendered)
        self.assertIn("No PR comments imported yet.", rendered)

    def test_latest_review_result_reads_manifest_and_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            artifact_dir = (
                workspace
                / ".codex"
                / "review_artifacts"
                / resume_context.review_loop.sanitize_path_component("feature/test")
            )
            artifact_dir.mkdir(parents=True)
            message_path = artifact_dir / "message.md"
            message_path.write_text("No material findings remain.\n", encoding="utf-8")
            latest = artifact_dir / "latest.json"
            latest.write_text(
                json.dumps(
                    {
                        "message_path": str(message_path.relative_to(workspace)),
                        "jsonl_path": ".codex/review_artifacts/feature-test/run.jsonl",
                    }
                ),
                encoding="utf-8",
            )

            status, message, artifact = resume_context.latest_review_result(
                workspace, "feature/test"
            )

            self.assertEqual(status, "clean")
            self.assertEqual(message, "No material findings remain.")
            self.assertEqual(artifact, ".codex/review_artifacts/feature-test/run.jsonl")

    def test_load_recent_events_returns_latest_issue_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            issue_log = root / "issues" / "PRO-7.jsonl"
            issue_log.parent.mkdir(parents=True)
            issue_log.write_text(
                "\n".join(
                    [
                        json.dumps({"timestamp": "t1", "event_type": "issue_started"}),
                        json.dumps({"timestamp": "t2", "event_type": "pr_opened"}),
                        json.dumps({"timestamp": "t3", "event_type": "review_requested"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            original = resume_context.telemetry.default_telemetry_root
            try:
                resume_context.telemetry.default_telemetry_root = lambda: root
                events = resume_context.load_recent_events("PRO-7", limit=2)
            finally:
                resume_context.telemetry.default_telemetry_root = original

            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["event_type"], "pr_opened")
            self.assertEqual(events[1]["event_type"], "review_requested")


if __name__ == "__main__":
    unittest.main()
