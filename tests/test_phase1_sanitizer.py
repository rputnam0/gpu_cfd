from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.authority import load_authority_bundle, repo_root
from scripts.authority.phase1_sanitizer import (
    Phase1MemcheckRunResult,
    main,
    parse_compute_sanitizer_memcheck_log,
    run_phase1_memcheck,
)


def repo_path() -> pathlib.Path:
    return repo_root()


class Phase1SanitizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_path()
        cls.bundle = load_authority_bundle(cls.root)

    def test_parse_compute_sanitizer_memcheck_log_tracks_clean_summary(self) -> None:
        result = parse_compute_sanitizer_memcheck_log(
            "\n".join(
                (
                    "========= COMPUTE-SANITIZER",
                    "========= Program hit cudaErrorNoKernelImageForDevice",
                    "========= ERROR SUMMARY: 0 errors",
                )
            )
        )

        self.assertTrue(result.error_summary_found)
        self.assertEqual(result.error_summary_count, 0)
        self.assertEqual(result.actionable_errors, 0)
        self.assertEqual(result.classification, "clean")

    def test_run_phase1_memcheck_emits_result_json_for_cube_linear(self) -> None:
        commands: list[tuple[tuple[str, ...], pathlib.Path]] = []

        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            commands.append((command, cwd))
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            self.assertEqual(
                command,
                ("compute-sanitizer", "--tool", "memcheck", "laplacianFoam"),
            )
            log_path.write_text(
                "\n".join(
                    (
                        "========= COMPUTE-SANITIZER",
                        "========= ERROR SUMMARY: 0 errors",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            time_dir = cwd / "0.1"
            time_dir.mkdir(parents=True, exist_ok=True)
            (time_dir / "T").write_text("0\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            scratch_root = pathlib.Path(temp_dir) / "scratch"
            result = run_phase1_memcheck(
                self.bundle,
                artifact_root=artifact_root,
                scratch_root=scratch_root,
                root=self.root,
                command_runner=command_runner,
            )

            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual([command[0][0] for command in commands], ["blockMesh", "compute-sanitizer"])
        self.assertEqual(result.status, "pass")
        self.assertEqual(result_payload["status"], "pass")
        self.assertEqual(result_payload["case_name"], "cubeLinear")
        self.assertEqual(result_payload["memcheck"]["actionable_errors"], 0)
        self.assertEqual(result_payload["command_results"][1]["log_path"].split("/")[-1], "memcheck.log")

    def test_run_phase1_memcheck_fails_when_actionable_errors_are_reported(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            log_path.write_text(
                "\n".join(
                    (
                        "========= Invalid __global__ write of size 4 bytes",
                        "========= ERROR SUMMARY: 2 errors",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            time_dir = cwd / "0.1"
            time_dir.mkdir(parents=True, exist_ok=True)
            (time_dir / "T").write_text("0\n", encoding="utf-8")
            return 1

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phase1_memcheck(
                self.bundle,
                artifact_root=pathlib.Path(temp_dir) / "artifacts",
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "fail")
        self.assertEqual(result_payload["status"], "fail")
        self.assertEqual(result_payload["memcheck"]["actionable_errors"], 2)
        self.assertIn("actionable_memcheck_errors", result_payload["failure_reasons"])

    def test_run_phase1_memcheck_rejects_non_smallest_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "cubeLinear"):
                run_phase1_memcheck(
                    self.bundle,
                    case_name="channelSteady",
                    artifact_root=pathlib.Path(temp_dir) / "artifacts",
                    scratch_root=pathlib.Path(temp_dir) / "scratch",
                    root=self.root,
                )

    @mock.patch("scripts.authority.phase1_sanitizer.load_authority_bundle")
    @mock.patch("scripts.authority.phase1_sanitizer.run_phase1_memcheck")
    def test_main_returns_non_zero_when_memcheck_run_fails(
        self,
        run_phase1_memcheck_mock: mock.Mock,
        load_authority_bundle_mock: mock.Mock,
    ) -> None:
        load_authority_bundle_mock.return_value = self.bundle
        run_phase1_memcheck_mock.return_value = Phase1MemcheckRunResult(
            case_name="cubeLinear",
            scratch_case_dir=pathlib.Path("/tmp/cubeLinear"),
            audit_report_path=pathlib.Path("/tmp/audit.json"),
            result_json_path=pathlib.Path("/tmp/memcheck_result.json"),
            memcheck_log_path=pathlib.Path("/tmp/memcheck.log"),
            status="fail",
        )

        exit_code = main(
            [
                "--root",
                str(self.root),
                "run",
                "--artifact-root",
                "/tmp/artifacts",
                "--scratch-root",
                "/tmp/scratch",
            ]
        )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
