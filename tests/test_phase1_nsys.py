from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import load_authority_bundle, repo_root
from scripts.authority.phase1_nsys import run_phase1_nsys_profile


def repo_path() -> pathlib.Path:
    return repo_root()


def _csv_text(*rows: tuple[str, ...]) -> str:
    return "\n".join(",".join(row) for row in rows) + "\n"


class Phase1ProfilingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_path()
        cls.bundle = load_authority_bundle(cls.root)

    def test_run_phase1_nsys_profile_emits_baseline_artifacts_and_summary_for_channel_transient(self) -> None:
        commands: list[tuple[tuple[str, ...], pathlib.Path]] = []

        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            commands.append((command, cwd))
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            if command[:2] == ("nsys", "profile"):
                log_path.write_text("profile complete\n", encoding="utf-8")
                output_stem = pathlib.Path(command[command.index("--output") + 1])
                output_stem.parent.mkdir(parents=True, exist_ok=True)
                output_stem.with_suffix(".nsys-rep").write_text("rep\n", encoding="utf-8")
                time_dir = cwd / "0.1"
                time_dir.mkdir(parents=True, exist_ok=True)
                (time_dir / "U").write_text("(0 0 0)\n", encoding="utf-8")
                (time_dir / "p").write_text("0\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "export"):
                log_path.write_text("export complete\n", encoding="utf-8")
                sqlite_path = pathlib.Path(command[command.index("--output") + 1])
                sqlite_path.write_text("sqlite\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "stats"):
                report_name = command[command.index("--report") + 1]
                output_path = pathlib.Path(command[command.index("--output") + 1])
                if report_name == "cuda_gpu_kern_sum":
                    output_path.write_text(
                        _csv_text(
                            ("Name", "Total Time (ns)"),
                            ("pimpleKernel", "42000"),
                        ),
                        encoding="utf-8",
                    )
                elif report_name == "nvtx_sum":
                    output_path.write_text(
                        _csv_text(
                            ("Range", "Instances"),
                            ("phase1:init", "1"),
                            ("phase1:caseSetup", "1"),
                            ("phase1:solveLoop", "1"),
                            ("phase1:iteration", "2"),
                            ("phase1:linearSolve", "2"),
                            ("phase1:write", "1"),
                            ("phase1:pimple:outerLoop", "2"),
                        ),
                        encoding="utf-8",
                    )
                else:
                    output_path.write_text(_csv_text(("Metric", "Value"), ("placeholder", "1")), encoding="utf-8")
                log_path.write_text(f"stats {report_name}\n", encoding="utf-8")
                return 0
            raise AssertionError(f"unexpected command: {command!r}")

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            scratch_root = pathlib.Path(temp_dir) / "scratch"
            result = run_phase1_nsys_profile(
                self.bundle,
                artifact_root=artifact_root,
                scratch_root=scratch_root,
                root=self.root,
                command_runner=command_runner,
            )

            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "pass")
        self.assertEqual(result_payload["status"], "pass")
        self.assertEqual(result_payload["profile_mode"], "basic")
        self.assertFalse(result_payload["diagnostic_only"])
        self.assertTrue(result_payload["timing_baseline_eligible"])
        self.assertTrue(result_payload["success_criteria"]["gpu_kernels_present"])
        self.assertTrue(result_payload["success_criteria"]["phase1_required_ranges_present"])
        self.assertEqual(result_payload["nvtx"]["missing_phase1_ranges"], [])
        self.assertIn("/nsight_systems/basic/channelTransient/", result.result_json_path.as_posix())
        self.assertEqual([command[0][0] for command in commands[:2]], ["blockMesh", "nsys"])

    def test_run_phase1_nsys_profile_requests_uvm_reports_and_marks_diagnostic_mode(self) -> None:
        profile_commands: list[tuple[str, ...]] = []

        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            if command[:2] == ("nsys", "profile"):
                profile_commands.append(command)
                log_path.write_text("profile complete\n", encoding="utf-8")
                output_stem = pathlib.Path(command[command.index("--output") + 1])
                output_stem.parent.mkdir(parents=True, exist_ok=True)
                output_stem.with_suffix(".nsys-rep").write_text("rep\n", encoding="utf-8")
                time_dir = cwd / "0.1"
                time_dir.mkdir(parents=True, exist_ok=True)
                (time_dir / "U").write_text("(0 0 0)\n", encoding="utf-8")
                (time_dir / "p").write_text("0\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "export"):
                sqlite_path = pathlib.Path(command[command.index("--output") + 1])
                sqlite_path.write_text("sqlite\n", encoding="utf-8")
                log_path.write_text("export complete\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "stats"):
                report_name = command[command.index("--report") + 1]
                output_path = pathlib.Path(command[command.index("--output") + 1])
                report_payloads = {
                    "cuda_gpu_kern_sum": _csv_text(("Name", "Total Time (ns)"), ("pimpleKernel", "42000")),
                    "nvtx_sum": _csv_text(
                        ("Range", "Instances"),
                        ("phase1:init", "1"),
                        ("phase1:caseSetup", "1"),
                        ("phase1:solveLoop", "1"),
                        ("phase1:iteration", "2"),
                        ("phase1:linearSolve", "2"),
                        ("phase1:write", "1"),
                        ("phase1:pimple:outerLoop", "2"),
                    ),
                    "um_sum": _csv_text(
                        ("Metric", "Value"),
                        ("GPU page faults", "1"),
                        ("GPU migrations", "4"),
                    ),
                    "um_total_sum": _csv_text(("Metric", "Value"), ("Total migrations", "4")),
                    "um_cpu_page_faults_sum": _csv_text(("Metric", "Value"), ("CPU page faults", "2")),
                }
                output_path.write_text(
                    report_payloads.get(report_name, _csv_text(("Metric", "Value"), ("placeholder", "1"))),
                    encoding="utf-8",
                )
                log_path.write_text(f"stats {report_name}\n", encoding="utf-8")
                return 0
            raise AssertionError(f"unexpected command: {command!r}")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phase1_nsys_profile(
                self.bundle,
                artifact_root=pathlib.Path(temp_dir) / "artifacts",
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                mode="um_fault",
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "pass")
        self.assertTrue(result_payload["diagnostic_only"])
        self.assertFalse(result_payload["timing_baseline_eligible"])
        self.assertEqual(result_payload["uvm"]["cpu_um_faults"], 2)
        self.assertEqual(result_payload["uvm"]["gpu_um_faults"], 1)
        self.assertEqual(result_payload["uvm"]["classification"], "documented_activity")
        self.assertTrue(any("--cuda-um-cpu-page-faults=true" in command for command in profile_commands))
        self.assertTrue(any("--cuda-um-gpu-page-faults=true" in command for command in profile_commands))

    def test_run_phase1_nsys_profile_fails_when_required_phase1_range_is_missing(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            if command[:2] == ("nsys", "profile"):
                log_path.write_text("profile complete\n", encoding="utf-8")
                output_stem = pathlib.Path(command[command.index("--output") + 1])
                output_stem.parent.mkdir(parents=True, exist_ok=True)
                output_stem.with_suffix(".nsys-rep").write_text("rep\n", encoding="utf-8")
                time_dir = cwd / "0.1"
                time_dir.mkdir(parents=True, exist_ok=True)
                (time_dir / "U").write_text("(0 0 0)\n", encoding="utf-8")
                (time_dir / "p").write_text("0\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "export"):
                sqlite_path = pathlib.Path(command[command.index("--output") + 1])
                sqlite_path.write_text("sqlite\n", encoding="utf-8")
                log_path.write_text("export complete\n", encoding="utf-8")
                return 0
            if command[:2] == ("nsys", "stats"):
                report_name = command[command.index("--report") + 1]
                output_path = pathlib.Path(command[command.index("--output") + 1])
                if report_name == "cuda_gpu_kern_sum":
                    output_path.write_text(_csv_text(("Name", "Total Time (ns)"), ("pimpleKernel", "42000")), encoding="utf-8")
                elif report_name == "nvtx_sum":
                    output_path.write_text(
                        _csv_text(
                            ("Range", "Instances"),
                            ("phase1:init", "1"),
                            ("phase1:caseSetup", "1"),
                            ("phase1:solveLoop", "1"),
                            ("phase1:iteration", "2"),
                            ("phase1:linearSolve", "2"),
                            ("phase1:pimple:outerLoop", "2"),
                        ),
                        encoding="utf-8",
                    )
                else:
                    output_path.write_text(_csv_text(("Metric", "Value"), ("placeholder", "1")), encoding="utf-8")
                log_path.write_text(f"stats {report_name}\n", encoding="utf-8")
                return 0
            raise AssertionError(f"unexpected command: {command!r}")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phase1_nsys_profile(
                self.bundle,
                artifact_root=pathlib.Path(temp_dir) / "artifacts",
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "fail")
        self.assertIn("missing_phase1_nvtx_ranges", result_payload["failure_reasons"])
        self.assertIn("phase1:write", result_payload["nvtx"]["missing_phase1_ranges"])

    def test_run_phase1_nsys_profile_rejects_non_transient_case_for_um_fault(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "channelTransient"):
                run_phase1_nsys_profile(
                    self.bundle,
                    artifact_root=pathlib.Path(temp_dir) / "artifacts",
                    scratch_root=pathlib.Path(temp_dir) / "scratch",
                    root=self.root,
                    case_name="cubeLinear",
                    mode="um_fault",
                )

    def test_run_nsys_wrapper_exists_and_invokes_phase1_nsys_module(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "run" / "run_nsys.sh"

        self.assertTrue(wrapper_path.is_file())

        wrapper = wrapper_path.read_text(encoding="utf-8")
        self.assertIn("scripts/authority/phase1_nsys.py run", wrapper)


if __name__ == "__main__":
    unittest.main()
