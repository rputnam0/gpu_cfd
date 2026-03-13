from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.symphony import telemetry


class TelemetryRootTests(unittest.TestCase):
    def test_prefers_explicit_telemetry_root(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GPU_CFD_TELEMETRY_ROOT": "~/custom-telemetry",
                "SYMPHONY_LOGS_ROOT": "~/logs-root",
            },
            clear=False,
        ):
            self.assertEqual(
                telemetry.default_telemetry_root(),
                pathlib.Path("~/custom-telemetry").expanduser(),
            )

    def test_falls_back_to_symphony_logs_root(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "SYMPHONY_LOGS_ROOT": "~/logs-root",
            },
            clear=True,
        ):
            self.assertEqual(
                telemetry.default_telemetry_root(),
                pathlib.Path("~/logs-root").expanduser(),
            )


class TelemetryWriteTests(unittest.TestCase):
    def test_write_event_appends_global_and_issue_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            event = telemetry.build_event(
                event_type="blocker",
                message="Missing external token",
                issue="PRO-5",
                pr=12,
                state="In Progress",
                branch="codex/pro-5",
                commit="abc123",
                details={"source": "linear", "severity": "error"},
            )

            paths = telemetry.write_event(root, event)

            self.assertEqual(paths["events"], root / "events.jsonl")
            self.assertEqual(paths["issue"], root / "issues" / "PRO-5.jsonl")
            self.assertEqual(paths["blockers"], root / "blockers.jsonl")

            events_payload = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events_payload[0]["event_type"], "blocker")
            self.assertEqual(events_payload[0]["issue"], "PRO-5")
            self.assertEqual(events_payload[0]["details"]["source"], "linear")

    def test_parse_details_rejects_invalid_key_value_pair(self) -> None:
        with self.assertRaises(ValueError):
            telemetry.parse_details(["missing_separator"])


if __name__ == "__main__":
    unittest.main()
