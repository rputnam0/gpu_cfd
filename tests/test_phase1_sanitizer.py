from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from scripts.authority import load_authority_bundle, repo_root
from scripts.authority.pins import load_pin_details
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
                    "========= ERROR SUMMARY: 0 errors",
                )
            )
        )

        self.assertTrue(result.error_summary_found)
        self.assertEqual(result.error_summary_count, 0)
        self.assertEqual(result.actionable_errors, 0)
        self.assertEqual(result.classification, "clean")

    def test_parse_compute_sanitizer_memcheck_log_allows_non_actionable_noise(self) -> None:
        result = parse_compute_sanitizer_memcheck_log(
            "\n".join(
                (
                    "========= third-party library noise from an external allocator",
                    "========= lineinfo was not found for one or more kernels",
                    "========= ERROR SUMMARY: 2 errors",
                )
            )
        )

        self.assertTrue(result.error_summary_found)
        self.assertEqual(result.error_summary_count, 2)
        self.assertEqual(result.actionable_errors, 0)
        self.assertEqual(result.classification, "non_actionable_noise")
        self.assertIn("third_party_noise", result.notes)
        self.assertIn("lineinfo_absent", result.notes)

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
        self.assertEqual(result_payload["reviewed_source_tuple_id"], load_pin_details(self.bundle).reviewed_source_tuple_id)
        self.assertEqual(result_payload["runtime_base"], load_pin_details(self.bundle).runtime_base)
        self.assertEqual(result_payload["toolkit"]["selected_lane"], "primary")
        self.assertEqual(result_payload["case_name"], "cubeLinear")
        self.assertEqual(result_payload["memcheck"]["actionable_errors"], 0)
        self.assertEqual(result_payload["command_results"][1]["log_path"].split("/")[-1], "memcheck.log")

    def test_run_phase1_memcheck_keeps_outputs_out_of_smoke_artifact_namespace(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            log_path.write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")
            time_dir = cwd / "0.1"
            time_dir.mkdir(parents=True, exist_ok=True)
            (time_dir / "T").write_text("0\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            smoke_case_root = artifact_root / "cubeLinear"
            smoke_case_root.mkdir(parents=True, exist_ok=True)
            smoke_audit = smoke_case_root / "smoke_audit.json"
            block_mesh_log = smoke_case_root / "01_blockMesh.log"
            smoke_audit.write_text("{\"keep\": true}\n", encoding="utf-8")
            block_mesh_log.write_text("original smoke log\n", encoding="utf-8")

            result = run_phase1_memcheck(
                self.bundle,
                artifact_root=artifact_root,
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            smoke_audit_text = smoke_audit.read_text(encoding="utf-8")
            block_mesh_log_text = block_mesh_log.read_text(encoding="utf-8")
            result_path = result.result_json_path.as_posix()

        self.assertEqual(smoke_audit_text, "{\"keep\": true}\n")
        self.assertEqual(block_mesh_log_text, "original smoke log\n")
        self.assertIn("/compute_sanitizer/cubeLinear/", result_path)

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

    def test_run_phase1_memcheck_allows_non_actionable_noise_to_pass(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            log_path.write_text(
                "\n".join(
                    (
                        "========= third-party library noise from an external allocator",
                        "========= lineinfo was not found for one or more kernels",
                        "========= ERROR SUMMARY: 2 errors",
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
            result = run_phase1_memcheck(
                self.bundle,
                artifact_root=pathlib.Path(temp_dir) / "artifacts",
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "pass")
        self.assertEqual(result_payload["status"], "pass")
        self.assertEqual(result_payload["memcheck"]["error_summary_count"], 2)
        self.assertEqual(result_payload["memcheck"]["actionable_errors"], 0)

    def test_run_phase1_memcheck_does_not_reuse_stale_log_when_memcheck_never_runs(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            log_path.write_text("setup failed\n", encoding="utf-8")
            return 1

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            stale_log = artifact_root / "compute_sanitizer" / "cubeLinear" / "memcheck.log"
            stale_log.parent.mkdir(parents=True, exist_ok=True)
            stale_log.write_text("========= ERROR SUMMARY: 0 errors\n", encoding="utf-8")

            result = run_phase1_memcheck(
                self.bundle,
                artifact_root=artifact_root,
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "fail")
        self.assertFalse(stale_log.exists())
        self.assertFalse(result_payload["memcheck"]["error_summary_found"])
        self.assertIn("memcheck_not_run", result_payload["failure_reasons"])
        self.assertIn("missing_error_summary", result_payload["failure_reasons"])

    def test_run_phase1_memcheck_writes_failed_artifact_when_binary_is_missing(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            raise FileNotFoundError(command[0])

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
        self.assertIn("memcheck_command_failed", result_payload["failure_reasons"])
        self.assertIn("missing_error_summary", result_payload["failure_reasons"])
        self.assertFalse(result_payload["memcheck"]["error_summary_found"])

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
