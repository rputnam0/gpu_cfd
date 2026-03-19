from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import load_authority_bundle, repo_root
from scripts.authority.phase1_acceptance import (
    PHASE1_ACCEPTANCE_MARKDOWN_NAME,
    PHASE1_ACCEPTANCE_REPORT_NAME,
    PHASE1_PTX_JIT_RESULT_NAME,
    build_phase1_acceptance_report,
    run_phase1_ptx_jit,
)


def repo_path() -> pathlib.Path:
    return repo_root()


def write_json(path: pathlib.Path, payload: dict[str, object]) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def sample_host_env(bundle) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "host_env.json",
        "consumer": "run",
        "reviewed_source_tuple_id": bundle.pins.reviewed_source_tuple_id,
        "runtime_base": bundle.pins.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": bundle.pins.primary_toolkit_lane,
            "primary_lane": bundle.pins.primary_toolkit_lane,
            "experimental_lane": bundle.pins.experimental_toolkit_lane,
            "driver_floor": bundle.pins.driver_floor,
        },
        "gpu_target": bundle.pins.gpu_target,
        "instrumentation": bundle.pins.instrumentation,
        "host_observations": {
            "hostname": "phase1-test-host",
            "gpu_csv": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
            "nvcc_version": "Cuda compilation tools, release 12.9, V12.9.86",
        },
        "authority_revisions": bundle.authority_revisions,
        "repo": {"git_commit": "abc123def456"},
    }


def sample_manifest_refs(bundle) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "manifest_refs.json",
        "consumer": "run",
        "reviewed_source_tuple_id": bundle.pins.reviewed_source_tuple_id,
        "runtime_base": bundle.pins.runtime_base,
        "required_revalidation": list(bundle.pins.required_revalidation),
        "authority_revisions": bundle.authority_revisions,
        "repo": {"git_commit": "abc123def456"},
    }


def sample_cuda_probe() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "cuda_probe.json",
        "device_name": "NVIDIA GeForce RTX 5080",
        "cc_major": 12,
        "cc_minor": 0,
        "host_env": "host_env.json",
        "manifest_refs": "manifest_refs.json",
        "native_kernel_ok": True,
        "managed_memory_probe_ok": True,
    }


def sample_build_metadata(bundle, *, build_log: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "lane": "primary",
        "mode": "relwithdebinfo",
        "have_cuda": True,
        "nvarch": 120,
        "ptx_retention_required": True,
        "selected_lane_value": bundle.pins.primary_toolkit_lane,
        "build_log": build_log,
        "reviewed_source_tuple_id": bundle.pins.reviewed_source_tuple_id,
        "required_revalidation": list(bundle.pins.required_revalidation),
        "succeeded": True,
        "returncode": 0,
        "failure_reason": None,
    }


def sample_fatbinary_report() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "required_native_sm": 120,
        "required_native_sm_found": True,
        "ptx_required": True,
        "ptx_present": True,
        "ptx_targets": [120],
        "native_sm_present": [120],
        "smoke_gate_ready": True,
    }


def sample_smoke_result(case_name: str, solver: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "smoke_result.json",
        "case_name": case_name,
        "solver": solver,
        "status": "pass",
        "failure_reasons": [],
        "success_criteria": {
            "audit_passed": True,
            "required_outputs_present": True,
            "no_nan_inf": True,
        },
    }


def sample_memcheck_result() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "memcheck_result.json",
        "case_name": "cubeLinear",
        "solver": "laplacianFoam",
        "status": "pass",
        "failure_reasons": [],
        "success_criteria": {
            "audit_passed": True,
            "required_outputs_present": True,
            "no_nan_inf": True,
            "error_summary_found": True,
            "actionable_errors": 0,
        },
        "memcheck": {
            "error_summary_found": True,
            "error_summary_count": 0,
            "actionable_errors": 0,
            "classification": "clean",
        },
    }


def sample_nsys_result(mode: str) -> dict[str, object]:
    diagnostic_only = mode == "um_fault"
    return {
        "schema_version": "1.0.0",
        "canonical_name": "nsys_profile_result.json",
        "case_name": "channelTransient",
        "solver": "pimpleFoam",
        "profile_mode": mode,
        "diagnostic_only": diagnostic_only,
        "timing_baseline_eligible": not diagnostic_only,
        "status": "pass",
        "failure_reasons": [],
        "success_criteria": {
            "audit_passed": True,
            "trace_generated": True,
            "required_outputs_present": True,
            "no_nan_inf": True,
            "gpu_kernels_present": True,
            "phase1_required_ranges_present": True,
            "uvm_evidence_present": diagnostic_only,
        },
        "nvtx": {
            "missing_phase1_ranges": [],
        },
        "uvm": {
            "diagnostic_requested": diagnostic_only,
            "evidence_present": diagnostic_only,
            "cpu_um_faults": 2 if diagnostic_only else 0,
            "gpu_um_faults": 1 if diagnostic_only else 0,
            "classification": "documented_activity" if diagnostic_only else "not_requested",
        },
    }


class Phase1AcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repo_path()
        cls.bundle = load_authority_bundle(cls.root)

    def test_run_phase1_ptx_jit_emits_result_json_and_requests_cuda_force_ptx_jit(self) -> None:
        commands: list[tuple[tuple[str, ...], pathlib.Path, dict[str, str]]] = []

        def command_runner(
            command: tuple[str, ...],
            *,
            cwd: pathlib.Path,
            log_path: pathlib.Path,
            env: dict[str, str] | None = None,
        ) -> int:
            commands.append((command, cwd, dict(env or {})))
            if command[0] == "blockMesh":
                log_path.write_text("Created mesh.\n", encoding="utf-8")
                (cwd / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
                return 0

            self.assertEqual(command, ("laplacianFoam",))
            self.assertEqual((env or {}).get("CUDA_FORCE_PTX_JIT"), "1")
            log_path.write_text("PTX JIT run complete\n", encoding="utf-8")
            time_dir = cwd / "0.1"
            time_dir.mkdir(parents=True, exist_ok=True)
            (time_dir / "T").write_text("0\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            fatbinary_report_path = write_json(
                temp_root / "build" / "fatbinary_report.json",
                sample_fatbinary_report(),
            )
            result = run_phase1_ptx_jit(
                self.bundle,
                artifact_root=temp_root / "artifacts",
                scratch_root=temp_root / "scratch",
                fatbinary_report_path=fatbinary_report_path,
                root=self.root,
                command_runner=command_runner,
            )
            payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.result_json_path.name, PHASE1_PTX_JIT_RESULT_NAME)
        self.assertEqual(payload["status"], "pass")
        self.assertEqual(payload["environment"]["CUDA_FORCE_PTX_JIT"], "1")
        self.assertTrue(payload["success_criteria"]["fatbinary_smoke_gate_ready"])
        self.assertEqual([command[0][0] for command in commands], ["blockMesh", "laplacianFoam"])
        self.assertIn("/ptx_jit/cubeLinear/", result.result_json_path.as_posix())

    def test_build_phase1_acceptance_report_emits_pass_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            host_env_path = write_json(temp_root / "discovery" / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "discovery" / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(
                temp_root / "discovery" / "cuda_probe.json",
                sample_cuda_probe(),
            )
            build_metadata_path = write_json(
                temp_root / "build" / "build_metadata_primary_relwithdebinfo.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build" / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(
                temp_root / "build" / "fatbinary_report_primary_relwithdebinfo.json",
                sample_fatbinary_report(),
            )
            smoke_result_paths = [
                write_json(
                    temp_root / "smoke" / "cubeLinear" / "smoke_result.json",
                    sample_smoke_result("cubeLinear", "laplacianFoam"),
                ),
                write_json(
                    temp_root / "smoke" / "channelSteady" / "smoke_result.json",
                    sample_smoke_result("channelSteady", "simpleFoam"),
                ),
                write_json(
                    temp_root / "smoke" / "channelTransient" / "smoke_result.json",
                    sample_smoke_result("channelTransient", "pimpleFoam"),
                ),
            ]
            memcheck_result_path = write_json(
                temp_root / "compute_sanitizer" / "cubeLinear" / "memcheck_result.json",
                sample_memcheck_result(),
            )
            nsys_result_paths = [
                write_json(
                    temp_root / "nsight_systems" / "basic" / "channelTransient" / "nsys_profile_result.json",
                    sample_nsys_result("basic"),
                ),
                write_json(
                    temp_root / "nsight_systems" / "um_fault" / "channelTransient" / "nsys_profile_result.json",
                    sample_nsys_result("um_fault"),
                ),
            ]
            ptx_jit_result_path = write_json(
                temp_root / "ptx_jit" / "cubeLinear" / PHASE1_PTX_JIT_RESULT_NAME,
                {
                    "schema_version": "1.0.0",
                    "canonical_name": PHASE1_PTX_JIT_RESULT_NAME,
                    "case_name": "cubeLinear",
                    "solver": "laplacianFoam",
                    "status": "pass",
                    "failure_reasons": [],
                    "environment": {"CUDA_FORCE_PTX_JIT": "1"},
                    "success_criteria": {
                        "audit_passed": True,
                        "fatbinary_smoke_gate_ready": True,
                        "required_outputs_present": True,
                        "no_nan_inf": True,
                    },
                },
            )

            report = build_phase1_acceptance_report(
                self.bundle,
                output_dir=temp_root / "acceptance",
                host_env_path=host_env_path,
                manifest_refs_path=manifest_refs_path,
                cuda_probe_path=cuda_probe_path,
                build_metadata_path=build_metadata_path,
                fatbinary_report_path=fatbinary_report_path,
                smoke_result_paths=smoke_result_paths,
                memcheck_result_path=memcheck_result_path,
                nsys_result_paths=nsys_result_paths,
                ptx_jit_result_path=ptx_jit_result_path,
                bringup_doc_path=docs_path,
            )
            payload = json.loads(report.json_path.read_text(encoding="utf-8"))
            markdown = report.markdown_path.read_text(encoding="utf-8")

        self.assertEqual(report.json_path.name, PHASE1_ACCEPTANCE_REPORT_NAME)
        self.assertEqual(report.markdown_path.name, PHASE1_ACCEPTANCE_MARKDOWN_NAME)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["disposition"], "pass")
        self.assertEqual(payload["lane"], "primary")
        self.assertEqual(payload["reviewed_source_tuple_id"], self.bundle.pins.reviewed_source_tuple_id)
        self.assertTrue(payload["gate_results"]["hard"]["ptx_jit_succeeds"]["passed"])
        self.assertTrue(payload["gate_results"]["hard"]["uvm_trace_captured"]["passed"])
        self.assertEqual(payload["failing_gate_ids"], [])
        self.assertIn("Status: PASS", markdown)
        self.assertIn(PHASE1_PTX_JIT_RESULT_NAME, markdown)

    def test_build_phase1_acceptance_report_fails_when_a_required_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", sample_cuda_probe())
            build_metadata_path = write_json(
                temp_root / "build_metadata.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(
                    temp_root / "cubeLinear.json",
                    sample_smoke_result("cubeLinear", "laplacianFoam"),
                ),
                write_json(
                    temp_root / "channelSteady.json",
                    sample_smoke_result("channelSteady", "simpleFoam"),
                ),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam"),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", sample_memcheck_result())
            nsys_result_paths = [
                write_json(temp_root / "basic.json", sample_nsys_result("basic")),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault")),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                {
                    "schema_version": "1.0.0",
                    "canonical_name": PHASE1_PTX_JIT_RESULT_NAME,
                    "case_name": "cubeLinear",
                    "solver": "laplacianFoam",
                    "status": "fail",
                    "failure_reasons": ["ptx_jit_command_failed"],
                    "environment": {"CUDA_FORCE_PTX_JIT": "1"},
                    "success_criteria": {
                        "audit_passed": True,
                        "fatbinary_smoke_gate_ready": True,
                        "required_outputs_present": True,
                        "no_nan_inf": False,
                    },
                },
            )

            report = build_phase1_acceptance_report(
                self.bundle,
                output_dir=temp_root / "acceptance",
                host_env_path=host_env_path,
                manifest_refs_path=manifest_refs_path,
                cuda_probe_path=cuda_probe_path,
                build_metadata_path=build_metadata_path,
                fatbinary_report_path=fatbinary_report_path,
                smoke_result_paths=smoke_result_paths,
                memcheck_result_path=memcheck_result_path,
                nsys_result_paths=nsys_result_paths,
                ptx_jit_result_path=ptx_jit_result_path,
                bringup_doc_path=docs_path,
            )
            payload = json.loads(report.json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "FAIL")
        self.assertEqual(payload["disposition"], "fail")
        self.assertIn("ptx_jit_succeeds", payload["failing_gate_ids"])
        self.assertIn("ptx_jit_succeeds", payload["reason"])


if __name__ == "__main__":
    unittest.main()
