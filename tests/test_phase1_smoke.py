from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from scripts.authority import (
    build_phase1_smoke_manifest,
    load_authority_bundle,
    repo_root,
    run_phase1_smoke_case,
    scan_phase1_smoke_case,
)
from scripts.authority.pins import load_pin_details
from scripts.authority.phase1_smoke import Phase1SmokeAuditReport, Phase1SmokeRunResult, main


def repo_path() -> pathlib.Path:
    return repo_root()


class Phase1SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_path()
        cls.bundle = load_authority_bundle(cls.root)

    def test_manifest_lists_three_repo_local_smoke_cases(self) -> None:
        manifest = build_phase1_smoke_manifest(self.root)

        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertEqual(
            [case["name"] for case in manifest["cases"]],
            ["cubeLinear", "channelSteady", "channelTransient"],
        )
        for case in manifest["cases"]:
            self.assertTrue((self.root / case["case_dir"]).is_dir())
            self.assertTrue((self.root / case["acceptance_json"]).is_file())
            self.assertTrue(case["self_contained"])

    def test_checked_in_cases_pass_pre_run_audit(self) -> None:
        for case_name in ("cubeLinear", "channelSteady", "channelTransient"):
            with self.subTest(case_name=case_name):
                report = scan_phase1_smoke_case(self.bundle, case_name=case_name, root=self.root)
                self.assertTrue(report.startup_allowed)
                self.assertEqual(report.reject_reasons, ())
                self.assertTrue(report.support_scan["startup_allowed"])

    def test_channel_steady_checked_in_run_window_exceeds_half_timestep(self) -> None:
        control_dict = (
            self.root
            / "tools"
            / "bringup"
            / "cases"
            / "phase1_smoke"
            / "channelSteady"
            / "system"
            / "controlDict"
        ).read_text(encoding="utf-8")

        self.assertIn("endTime 0.1;", control_dict)
        self.assertIn("deltaT 0.05;", control_dict)

    def test_audit_rejects_unsupported_fvsolution_settings(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelSteady"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelSteady"
            shutil.copytree(source_case, temp_case)
            fvsolution = temp_case / "system" / "fvSolution"
            fvsolution.write_text(
                fvsolution.read_text(encoding="utf-8").replace("aDILU", "DILU"),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("unsupported_preconditioner", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_unsupported_scheme_settings(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            fv_schemes = temp_case / "system" / "fvSchemes"
            fv_schemes.write_text(
                fv_schemes.read_text(encoding="utf-8").replace("default Euler;", "default localEuler;"),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("unsupported_time_scheme", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_unsupported_gradient_and_interpolation_scheme_settings(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            fv_schemes = temp_case / "system" / "fvSchemes"
            fv_schemes.write_text(
                fv_schemes.read_text(encoding="utf-8")
                .replace("default Gauss linear;", "default leastSquares;")
                .replace("default linear;", "default cubic;"),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("unsupported_gradient_scheme", codes)
        self.assertIn("unsupported_interpolation_scheme", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_non_laminar_turbulence_settings(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelSteady"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelSteady"
            shutil.copytree(source_case, temp_case)
            turbulence_properties = temp_case / "constant" / "turbulenceProperties"
            turbulence_properties.write_text(
                turbulence_properties.read_text(encoding="utf-8").replace(
                    "simulationType laminar;", "simulationType RAS;"
                ),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("turbulence_scope_violation", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_missing_gamg_smoother(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            fv_solution = temp_case / "system" / "fvSolution"
            fv_solution.write_text(
                fv_solution.read_text(encoding="utf-8").replace("        smoother diagonal;\n", ""),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("missing_gamg_smoother", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_coded_patch_fields_from_checked_in_boundary_data(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            velocity_field = temp_case / "0" / "U"
            velocity_field.write_text(
                velocity_field.read_text(encoding="utf-8").replace("type noSlip;", "type codedFixedValue;"),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("coded_patch_field_violation", codes)
        self.assertFalse(report.startup_allowed)

    def test_audit_rejects_cyclic_patches_from_block_mesh_dict(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            block_mesh = temp_case / "system" / "blockMeshDict"
            block_mesh.write_text(
                block_mesh.read_text(encoding="utf-8").replace("type wall;", "type cyclic;", 1),
                encoding="utf-8",
            )

            report = scan_phase1_smoke_case(self.bundle, case_dir=temp_case)

        codes = tuple(reason["code"] for reason in report.reject_reasons)
        self.assertIn("cyclic_or_ami_patch_violation", codes)
        self.assertFalse(report.startup_allowed)

    def test_run_phase1_smoke_case_copies_case_and_emits_result_json(self) -> None:
        commands: list[tuple[tuple[str, ...], pathlib.Path]] = []

        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            commands.append((command, cwd))
            log_path.write_text("Execution completed successfully.\n", encoding="utf-8")
            if command[0] == "blockMesh":
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0
            time_dir = cwd / "0.1"
            time_dir.mkdir(parents=True, exist_ok=True)
            case_name = cwd.name
            if case_name == "cubeLinear":
                (time_dir / "T").write_text("0\n", encoding="utf-8")
            else:
                (time_dir / "U").write_text("(0 0 0)\n", encoding="utf-8")
                (time_dir / "p").write_text("0\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            scratch_root = pathlib.Path(temp_dir) / "scratch"
            result = run_phase1_smoke_case(
                self.bundle,
                case_name="cubeLinear",
                artifact_root=artifact_root,
                scratch_root=scratch_root,
                root=self.root,
                command_runner=command_runner,
            )

            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
            copied_case = scratch_root / "cubeLinear"
            copied_case_exists = copied_case.is_dir()

        self.assertEqual([command[0][0] for command in commands], ["blockMesh", "laplacianFoam"])
        self.assertTrue(copied_case_exists)
        self.assertTrue(result.audit_report.startup_allowed)
        self.assertEqual(result_payload["status"], "pass")
        self.assertEqual(result_payload["reviewed_source_tuple_id"], load_pin_details(self.bundle).reviewed_source_tuple_id)
        self.assertEqual(result_payload["runtime_base"], load_pin_details(self.bundle).runtime_base)
        self.assertEqual(result_payload["toolkit"]["selected_lane"], "primary")
        self.assertEqual(result_payload["case_name"], "cubeLinear")
        self.assertEqual(result_payload["command_results"][0]["command"][0], "blockMesh")
        self.assertTrue(result_payload["success_criteria"]["required_outputs_present"])

    def test_run_phase1_smoke_case_blocks_execution_when_audit_fails(self) -> None:
        source_case = self.root / "tools" / "bringup" / "cases" / "phase1_smoke" / "channelTransient"
        commands: list[tuple[str, ...]] = []

        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            commands.append(command)
            log_path.write_text("should not run\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_case = pathlib.Path(temp_dir) / "channelTransient"
            shutil.copytree(source_case, temp_case)
            control_dict = temp_case / "system" / "controlDict"
            control_dict.write_text(
                control_dict.read_text(encoding="utf-8").replace(
                    "functions\n{\n}\n",
                    "functions\n{\n    residuals1\n    {\n        type residuals;\n        executeControl timeStep;\n    }\n}\n",
                ),
                encoding="utf-8",
            )
            artifact_root = pathlib.Path(temp_dir) / "artifacts"
            scratch_root = pathlib.Path(temp_dir) / "scratch"
            result = run_phase1_smoke_case(
                self.bundle,
                case_dir=temp_case,
                artifact_root=artifact_root,
                scratch_root=scratch_root,
                command_runner=command_runner,
            )

            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(commands, [])
        self.assertEqual(result_payload["status"], "blocked")
        self.assertIn("function_object_debug_only_in_production", result_payload["reject_codes"])

    def test_run_phase1_smoke_case_marks_failed_run_when_outputs_are_missing(self) -> None:
        def command_runner(command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path) -> int:
            log_path.write_text("Execution completed successfully.\n", encoding="utf-8")
            if command[0] == "blockMesh":
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_phase1_smoke_case(
                self.bundle,
                case_name="cubeLinear",
                artifact_root=pathlib.Path(temp_dir) / "artifacts",
                scratch_root=pathlib.Path(temp_dir) / "scratch",
                root=self.root,
                command_runner=command_runner,
            )
            result_payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "fail")
        self.assertEqual(result_payload["status"], "fail")
        self.assertFalse(result_payload["success_criteria"]["required_outputs_present"])

    @mock.patch("scripts.authority.phase1_smoke.load_authority_bundle")
    @mock.patch("scripts.authority.phase1_smoke.run_phase1_smoke_case")
    def test_main_returns_non_zero_when_runtime_smoke_run_fails(
        self,
        run_phase1_smoke_case_mock: mock.Mock,
        load_authority_bundle_mock: mock.Mock,
    ) -> None:
        load_authority_bundle_mock.return_value = self.bundle
        run_phase1_smoke_case_mock.return_value = Phase1SmokeRunResult(
            case_name="cubeLinear",
            scratch_case_dir=pathlib.Path("/tmp/cubeLinear"),
            audit_report=Phase1SmokeAuditReport(
                case_name="cubeLinear",
                solver="laplacianFoam",
                case_dir=pathlib.Path("/tmp/source"),
                startup_allowed=True,
                authority_citations=(),
                issues=(),
                support_scan={"startup_allowed": True},
            ),
            audit_report_path=pathlib.Path("/tmp/audit.json"),
            result_json_path=pathlib.Path("/tmp/result.json"),
            status="fail",
        )

        exit_code = main(
            [
                "--root",
                str(self.root),
                "run",
                "--case",
                "cubeLinear",
                "--artifact-root",
                "/tmp/artifacts",
                "--scratch-root",
                "/tmp/scratch",
            ]
        )

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
