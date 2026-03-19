from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import load_authority_bundle, repo_root
from scripts.authority.pins import load_pin_details
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
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "canonical_name": "host_env.json",
        "consumer": "run",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "gpu_target": pin_details.gpu_target,
        "instrumentation": pin_details.instrumentation,
        "host_observations": {
            "hostname": "phase1-test-host",
            "gpu_csv": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
            "nvcc_version": "Cuda compilation tools, release 12.9, V12.9.86",
        },
        "authority_revisions": bundle.authority_revisions,
        "repo": {"git_commit": "abc123def456"},
    }


def sample_manifest_refs(bundle) -> dict[str, object]:
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "canonical_name": "manifest_refs.json",
        "consumer": "run",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "required_revalidation": list(pin_details.required_revalidation),
        "authority_revisions": bundle.authority_revisions,
        "repo": {"git_commit": "abc123def456"},
    }


def sample_cuda_probe(bundle=None) -> dict[str, object]:
    if bundle is None:
        bundle = load_authority_bundle(repo_path())
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "canonical_name": "cuda_probe.json",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "device_name": "NVIDIA GeForce RTX 5080",
        "cc_major": 12,
        "cc_minor": 0,
        "host_env": "host_env.json",
        "manifest_refs": "manifest_refs.json",
        "native_kernel_ok": True,
        "managed_memory_probe_ok": True,
    }


def sample_build_metadata(bundle, *, build_log: str) -> dict[str, object]:
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "lane": "primary",
        "mode": "relwithdebinfo",
        "have_cuda": True,
        "nvarch": 120,
        "ptx_retention_required": True,
        "selected_lane_value": pin_details.primary_toolkit_lane,
        "build_log": build_log,
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "required_revalidation": list(pin_details.required_revalidation),
        "succeeded": True,
        "returncode": 0,
        "failure_reason": None,
    }


def sample_fatbinary_report(bundle=None) -> dict[str, object]:
    if bundle is None:
        bundle = load_authority_bundle(repo_path())
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "required_native_sm": 120,
        "required_native_sm_found": True,
        "ptx_required": True,
        "ptx_present": True,
        "ptx_targets": [120],
        "native_sm_present": [120],
        "smoke_gate_ready": True,
    }


def sample_smoke_result(case_name: str, solver: str, bundle=None) -> dict[str, object]:
    if bundle is None:
        bundle = load_authority_bundle(repo_path())
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "canonical_name": "smoke_result.json",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
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


def sample_memcheck_result(bundle=None) -> dict[str, object]:
    if bundle is None:
        bundle = load_authority_bundle(repo_path())
    pin_details = load_pin_details(bundle)
    return {
        "schema_version": "1.0.0",
        "canonical_name": "memcheck_result.json",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
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


def sample_nsys_result(mode: str, bundle=None) -> dict[str, object]:
    if bundle is None:
        bundle = load_authority_bundle(repo_path())
    pin_details = load_pin_details(bundle)
    diagnostic_only = mode == "um_fault"
    return {
        "schema_version": "1.0.0",
        "canonical_name": "nsys_profile_result.json",
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
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


def sample_ptx_jit_result(
    bundle,
    *,
    case_name: str = "cubeLinear",
    solver: str = "laplacianFoam",
    status: str = "pass",
    failure_reasons: list[str] | None = None,
    success_criteria_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    pin_details = load_pin_details(bundle)
    success_criteria = {
        "audit_passed": True,
        "fatbinary_smoke_gate_ready": True,
        "required_outputs_present": True,
        "no_nan_inf": True,
    }
    if success_criteria_overrides:
        success_criteria.update(success_criteria_overrides)
    return {
        "schema_version": "1.0.0",
        "canonical_name": PHASE1_PTX_JIT_RESULT_NAME,
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": "primary",
            "selected_lane_value": pin_details.primary_toolkit_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "case_name": case_name,
        "solver": solver,
        "status": status,
        "failure_reasons": list(failure_reasons or []),
        "environment": {"CUDA_FORCE_PTX_JIT": "1"},
        "success_criteria": success_criteria,
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
        self.assertEqual(payload["reviewed_source_tuple_id"], load_pin_details(self.bundle).reviewed_source_tuple_id)
        self.assertEqual(payload["runtime_base"], load_pin_details(self.bundle).runtime_base)
        self.assertEqual(payload["toolkit"]["selected_lane"], "primary")
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertEqual(
            payload["reviewed_source_tuple_id"],
            load_pin_details(self.bundle).reviewed_source_tuple_id,
        )
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
                sample_ptx_jit_result(
                    self.bundle,
                    status="fail",
                    failure_reasons=["ptx_jit_command_failed"],
                    success_criteria_overrides={"no_nan_inf": False},
                ),
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

    def test_build_phase1_acceptance_report_accepts_normalized_pin_values_from_emitters(self) -> None:
        pin_details = load_pin_details(self.bundle)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            host_env = sample_host_env(self.bundle)
            host_env["reviewed_source_tuple_id"] = pin_details.reviewed_source_tuple_id
            host_env["runtime_base"] = pin_details.runtime_base
            host_env["toolkit"]["driver_floor"] = pin_details.driver_floor
            manifest_refs = sample_manifest_refs(self.bundle)
            manifest_refs["reviewed_source_tuple_id"] = pin_details.reviewed_source_tuple_id
            manifest_refs["runtime_base"] = pin_details.runtime_base
            build_metadata = sample_build_metadata(
                self.bundle,
                build_log=(temp_root / "build" / "build.log").as_posix(),
            )
            build_metadata["reviewed_source_tuple_id"] = pin_details.reviewed_source_tuple_id

            host_env_path = write_json(temp_root / "discovery" / "host_env.json", host_env)
            manifest_refs_path = write_json(
                temp_root / "discovery" / "manifest_refs.json",
                manifest_refs,
            )
            cuda_probe_path = write_json(
                temp_root / "discovery" / "cuda_probe.json",
                sample_cuda_probe(),
            )
            build_metadata_path = write_json(
                temp_root / "build" / "build_metadata_primary_relwithdebinfo.json",
                build_metadata,
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
                sample_ptx_jit_result(self.bundle),
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

        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["reviewed_source_tuple_id"], pin_details.reviewed_source_tuple_id)
        self.assertEqual(payload["runtime_base"], pin_details.runtime_base)

    def test_build_phase1_acceptance_report_fails_when_manifest_refs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(temp_root / "manifest_refs.json", {})
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", sample_cuda_probe())
            build_metadata_path = write_json(
                temp_root / "build_metadata.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("manifest_refs_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_cuda_probe_runtime_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            failed_probe = sample_cuda_probe()
            failed_probe["managed_memory_probe_ok"] = False
            failed_probe["managed_memory_failure_reason"] = "nvidia-uvm HMM init failed"

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", failed_probe)
            build_metadata_path = write_json(
                temp_root / "build_metadata.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("cuda_probe_runtime_ready", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_cuda_probe_to_match_primary_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_probe = sample_cuda_probe(self.bundle)
            stale_probe["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_probe["runtime_base"] = "stale/runtime"
            stale_probe["toolkit"]["selected_lane"] = "experimental"
            stale_probe["toolkit"]["selected_lane_value"] = load_pin_details(self.bundle).experimental_toolkit_lane

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", stale_probe)
            build_metadata_path = write_json(
                temp_root / "build_metadata.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("cuda_probe_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_rejects_experimental_build_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            experimental_build = sample_build_metadata(
                self.bundle,
                build_log=(temp_root / "build.log").as_posix(),
            )
            experimental_build["lane"] = "experimental"
            experimental_build["selected_lane_value"] = self.bundle.pins.experimental_toolkit_lane

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", sample_cuda_probe())
            build_metadata_path = write_json(temp_root / "build_metadata.json", experimental_build)
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("build_metadata_matches_primary_lane", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_cube_linear_memcheck_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            wrong_memcheck = sample_memcheck_result()
            wrong_memcheck["case_name"] = "channelTransient"
            wrong_memcheck["solver"] = "pimpleFoam"

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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam"),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", wrong_memcheck)
            nsys_result_paths = [
                write_json(temp_root / "basic.json", sample_nsys_result("basic")),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault")),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("memcheck_matches_required_case", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_fails_when_nsys_artifacts_are_marked_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            failed_basic = sample_nsys_result("basic")
            failed_basic["status"] = "fail"
            failed_basic["failure_reasons"] = ["trace_missing"]
            failed_basic["success_criteria"]["trace_generated"] = False
            failed_um = sample_nsys_result("um_fault")
            failed_um["status"] = "fail"
            failed_um["failure_reasons"] = ["nan_or_inf_detected"]
            failed_um["success_criteria"]["no_nan_inf"] = False

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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam"),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", sample_memcheck_result())
            nsys_result_paths = [
                write_json(temp_root / "basic.json", failed_basic),
                write_json(temp_root / "um_fault.json", failed_um),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("nsys_basic_artifact_passes", payload["failing_gate_ids"])
        self.assertIn("nsys_um_fault_artifact_passes", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_rejects_stale_manifest_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_manifest_refs = sample_manifest_refs(self.bundle)
            stale_manifest_refs["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_manifest_refs["runtime_base"] = "stale/runtime"
            stale_manifest_refs["required_revalidation"] = ["wrong"]

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(temp_root / "manifest_refs.json", stale_manifest_refs)
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", sample_cuda_probe())
            build_metadata_path = write_json(
                temp_root / "build_metadata.json",
                sample_build_metadata(self.bundle, build_log=(temp_root / "build.log").as_posix()),
            )
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("manifest_refs_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_build_tuple_to_match_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_build_metadata = sample_build_metadata(
                self.bundle,
                build_log=(temp_root / "build.log").as_posix(),
            )
            stale_build_metadata["reviewed_source_tuple_id"] = "STALE_TUPLE"

            host_env_path = write_json(temp_root / "host_env.json", sample_host_env(self.bundle))
            manifest_refs_path = write_json(
                temp_root / "manifest_refs.json",
                sample_manifest_refs(self.bundle),
            )
            cuda_probe_path = write_json(temp_root / "cuda_probe.json", sample_cuda_probe())
            build_metadata_path = write_json(temp_root / "build_metadata.json", stale_build_metadata)
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", sample_fatbinary_report())
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("build_metadata_matches_reviewed_tuple", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_cube_linear_ptx_jit_evidence(self) -> None:
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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                sample_ptx_jit_result(self.bundle, case_name="channelTransient", solver="pimpleFoam"),
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
        self.assertIn("ptx_jit_matches_required_case", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_ptx_jit_to_match_primary_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_ptx_jit = sample_ptx_jit_result(self.bundle)
            stale_ptx_jit["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_ptx_jit["runtime_base"] = "stale/runtime"
            stale_ptx_jit["toolkit"]["selected_lane"] = "experimental"
            stale_ptx_jit["toolkit"]["selected_lane_value"] = load_pin_details(self.bundle).experimental_toolkit_lane

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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam")),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam")),
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
                stale_ptx_jit,
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
        self.assertIn("ptx_jit_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_fatbinary_report_to_match_primary_tuple(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_fatbinary = sample_fatbinary_report(self.bundle)
            stale_fatbinary["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_fatbinary["runtime_base"] = "stale/runtime"
            stale_fatbinary["toolkit"]["selected_lane"] = "experimental"
            stale_fatbinary["toolkit"]["selected_lane_value"] = load_pin_details(
                self.bundle
            ).experimental_toolkit_lane

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
            fatbinary_report_path = write_json(temp_root / "fatbinary_report.json", stale_fatbinary)
            smoke_result_paths = [
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam", self.bundle)),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam", self.bundle)),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam", self.bundle),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", sample_memcheck_result(self.bundle))
            nsys_result_paths = [
                write_json(temp_root / "basic.json", sample_nsys_result("basic", self.bundle)),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault", self.bundle)),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("fatbinary_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_smoke_results_to_match_primary_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_smoke = sample_smoke_result("cubeLinear", "laplacianFoam", self.bundle)
            stale_smoke["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_smoke["runtime_base"] = "stale/runtime"
            stale_smoke["toolkit"]["selected_lane"] = "experimental"
            stale_smoke["toolkit"]["selected_lane_value"] = load_pin_details(self.bundle).experimental_toolkit_lane

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
                write_json(temp_root / "cubeLinear.json", stale_smoke),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam", self.bundle)),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam", self.bundle),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", sample_memcheck_result(self.bundle))
            nsys_result_paths = [
                write_json(temp_root / "basic.json", sample_nsys_result("basic", self.bundle)),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault", self.bundle)),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("smoke_results_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_memcheck_to_match_primary_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_memcheck = sample_memcheck_result(self.bundle)
            stale_memcheck["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_memcheck["runtime_base"] = "stale/runtime"
            stale_memcheck["toolkit"]["selected_lane"] = "experimental"
            stale_memcheck["toolkit"]["selected_lane_value"] = load_pin_details(self.bundle).experimental_toolkit_lane

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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam", self.bundle)),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam", self.bundle)),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam", self.bundle),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", stale_memcheck)
            nsys_result_paths = [
                write_json(temp_root / "basic.json", sample_nsys_result("basic", self.bundle)),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault", self.bundle)),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("memcheck_traceable", payload["failing_gate_ids"])

    def test_build_phase1_acceptance_report_requires_nsys_results_to_match_primary_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            docs_path = temp_root / "docs" / "bringup" / "phase1_blackwell.md"
            docs_path.parent.mkdir(parents=True, exist_ok=True)
            docs_path.write_text("# Phase 1 Blackwell bring-up\n", encoding="utf-8")

            stale_nsys = sample_nsys_result("basic", self.bundle)
            stale_nsys["reviewed_source_tuple_id"] = "STALE_TUPLE"
            stale_nsys["runtime_base"] = "stale/runtime"
            stale_nsys["toolkit"]["selected_lane"] = "experimental"
            stale_nsys["toolkit"]["selected_lane_value"] = load_pin_details(self.bundle).experimental_toolkit_lane

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
                write_json(temp_root / "cubeLinear.json", sample_smoke_result("cubeLinear", "laplacianFoam", self.bundle)),
                write_json(temp_root / "channelSteady.json", sample_smoke_result("channelSteady", "simpleFoam", self.bundle)),
                write_json(
                    temp_root / "channelTransient.json",
                    sample_smoke_result("channelTransient", "pimpleFoam", self.bundle),
                ),
            ]
            memcheck_result_path = write_json(temp_root / "memcheck_result.json", sample_memcheck_result(self.bundle))
            nsys_result_paths = [
                write_json(temp_root / "basic.json", stale_nsys),
                write_json(temp_root / "um_fault.json", sample_nsys_result("um_fault", self.bundle)),
            ]
            ptx_jit_result_path = write_json(
                temp_root / PHASE1_PTX_JIT_RESULT_NAME,
                sample_ptx_jit_result(self.bundle),
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
        self.assertIn("nsys_results_traceable", payload["failing_gate_ids"])


if __name__ == "__main__":
    unittest.main()
