from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from scripts.symphony import reconcile_review_state


class ReconcileReviewStateTests(unittest.TestCase):
    @mock.patch("scripts.symphony.reconcile_review_state.review_loop.repo_root")
    @mock.patch("scripts.symphony.reconcile_review_state.review_loop.require_command")
    def test_list_open_prs_preserves_merge_state_status(
        self,
        mock_require_command: mock.Mock,
        mock_repo_root: mock.Mock,
    ) -> None:
        mock_repo_root.return_value = "/tmp/gpu_cfd"
        mock_require_command.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "number": 34,
                        "title": "[codex] Harden Symphony harness rollout flow",
                        "body": "Closes PRO-34",
                        "headRefName": "codex/symphony-harness-hardening",
                        "headRefOid": "abc123",
                        "url": "https://github.com/rputnam0/gpu_cfd/pull/34",
                        "state": "OPEN",
                        "isDraft": False,
                        "mergeStateStatus": "BEHIND",
                    }
                ]
            ),
            stderr="",
        )

        snapshots = reconcile_review_state.list_open_prs("rputnam0/gpu_cfd")

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].merge_state_status, "BEHIND")


if __name__ == "__main__":
    unittest.main()
