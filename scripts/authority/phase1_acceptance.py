"""Phase 1 PTX-JIT proof runner and acceptance bundle generator."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from .bundle import AuthorityBundle, load_authority_bundle, repo_root
    from .pins import load_pin_details
    from .phase1_smoke import (
        CASE_DEFINITIONS,
        MANIFEST_SCHEMA_VERSION,
        PHASE1_SMOKE_AUDIT_NAME,
        Phase1SmokeAuditReport,
        scan_phase1_smoke_case,
    )
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.authority.bundle import AuthorityBundle, load_authority_bundle, repo_root  # type: ignore
    from scripts.authority.pins import load_pin_details  # type: ignore
    from scripts.authority.phase1_smoke import (  # type: ignore
        CASE_DEFINITIONS,
        MANIFEST_SCHEMA_VERSION,
        PHASE1_SMOKE_AUDIT_NAME,
        Phase1SmokeAuditReport,
        scan_phase1_smoke_case,
    )


DEFAULT_PHASE1_PTX_JIT_CASE = "cubeLinear"
PHASE1_PTX_JIT_ARTIFACT_DIRNAME = "ptx_jit"
PHASE1_PTX_JIT_RESULT_NAME = "ptx_jit_result.json"
PHASE1_ACCEPTANCE_REPORT_NAME = "phase1_acceptance_report.json"
PHASE1_ACCEPTANCE_MARKDOWN_NAME = "phase1_acceptance_report.md"
PHASE1_ACCEPTANCE_BUNDLE_INDEX_NAME = "phase1_acceptance_bundle_index.json"
REQUIRED_PHASE1_SMOKE_CASES = ("cubeLinear", "channelSteady", "channelTransient")
REQUIRED_PHASE1_NSYS_MODES = ("basic", "um_fault")
LOG_NAN_INF_PATTERN = re.compile(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", flags=re.IGNORECASE)
VERSION_PATTERN = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclass(frozen=True)
class Phase1PtxJitRunResult:
    case_name: str
    scratch_case_dir: pathlib.Path
    audit_report_path: pathlib.Path
    result_json_path: pathlib.Path
    status: str


@dataclass(frozen=True)
class Phase1AcceptanceReportResult:
    json_path: pathlib.Path
    markdown_path: pathlib.Path
    bundle_index_path: pathlib.Path
    status: str


def run_phase1_ptx_jit(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    scratch_root: pathlib.Path | str,
    fatbinary_report_path: pathlib.Path | str,
    root: pathlib.Path | str | None = None,
    case_name: str = DEFAULT_PHASE1_PTX_JIT_CASE,
    command_runner: Callable[..., int] | None = None,
) -> Phase1PtxJitRunResult:
    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)
    pin_details = load_pin_details(bundle)
    audit_report = scan_phase1_smoke_case(bundle, case_name=case_name, root=resolved_root)
    case_definition = _definition_by_name(case_name)

    artifact_case_root = pathlib.Path(artifact_root) / PHASE1_PTX_JIT_ARTIFACT_DIRNAME / case_name
    artifact_case_root.mkdir(parents=True, exist_ok=True)
    audit_report_path = artifact_case_root / PHASE1_SMOKE_AUDIT_NAME
    _write_json(audit_report_path, audit_report.as_dict())

    scratch_case_dir = pathlib.Path(scratch_root) / case_name
    if scratch_case_dir.exists():
        shutil.rmtree(scratch_case_dir)

    result_json_path = artifact_case_root / PHASE1_PTX_JIT_RESULT_NAME
    fatbinary_report = _read_json(pathlib.Path(fatbinary_report_path))
    fatbinary_ready = bool(fatbinary_report.get("smoke_gate_ready"))
    fatbinary_ptx_present = bool(fatbinary_report.get("ptx_present"))
    fatbinary_native_sm_found = bool(fatbinary_report.get("required_native_sm_found"))

    if not audit_report.startup_allowed or not fatbinary_ready:
        failure_reasons: list[str] = []
        status = "blocked" if not audit_report.startup_allowed else "fail"
        if not audit_report.startup_allowed:
            failure_reasons.append("audit_failed")
        if not fatbinary_ready:
            failure_reasons.append("fatbinary_smoke_gate_not_ready")
        payload = _build_ptx_jit_payload(
            pin_details=pin_details,
            case_name=case_name,
            solver=case_definition.solver,
            scratch_case_dir=scratch_case_dir,
            status=status,
            command_results=(),
            audit_report=audit_report,
            required_outputs=case_definition.required_outputs,
            required_outputs_present=False,
            no_nan_inf=True,
            fatbinary_report_path=pathlib.Path(fatbinary_report_path),
            fatbinary_ready=fatbinary_ready,
            fatbinary_ptx_present=fatbinary_ptx_present,
            fatbinary_native_sm_found=fatbinary_native_sm_found,
            solver_exit_zero=False,
            failure_reasons=failure_reasons,
        )
        _write_json(result_json_path, payload)
        return Phase1PtxJitRunResult(
            case_name=case_name,
            scratch_case_dir=scratch_case_dir,
            audit_report_path=audit_report_path,
            result_json_path=result_json_path,
            status=status,
        )

    shutil.copytree(audit_report.case_dir, scratch_case_dir)
    runner = command_runner or _default_command_runner
    command_results: list[dict[str, Any]] = []

    block_mesh_log_path = artifact_case_root / "01_blockMesh.log"
    block_mesh_returncode = _run_command(
        runner,
        ("blockMesh",),
        cwd=scratch_case_dir,
        log_path=block_mesh_log_path,
    )
    command_results.append(
        {
            "command": ["blockMesh"],
            "cwd": scratch_case_dir.as_posix(),
            "log_path": block_mesh_log_path.as_posix(),
            "returncode": block_mesh_returncode,
            "env_overrides": {},
        }
    )

    solver_returncode: int | None = None
    if block_mesh_returncode == 0:
        ptx_jit_log_path = artifact_case_root / "02_ptx_jit.log"
        solver_env = {"CUDA_FORCE_PTX_JIT": "1"}
        solver_returncode = _run_command(
            runner,
            (case_definition.solver,),
            cwd=scratch_case_dir,
            log_path=ptx_jit_log_path,
            env=solver_env,
        )
        command_results.append(
            {
                "command": [case_definition.solver],
                "cwd": scratch_case_dir.as_posix(),
                "log_path": ptx_jit_log_path.as_posix(),
                "returncode": solver_returncode,
                "env_overrides": dict(solver_env),
            }
        )

    required_outputs_present = all(
        (scratch_case_dir / relative_path).exists() for relative_path in case_definition.required_outputs
    )
    no_nan_inf = _logs_are_clean(command_results)
    solver_exit_zero = solver_returncode == 0
    failure_reasons = _build_ptx_jit_failure_reasons(
        command_results=command_results,
        required_outputs_present=required_outputs_present,
        no_nan_inf=no_nan_inf,
    )
    status = "pass" if not failure_reasons else "fail"
    payload = _build_ptx_jit_payload(
        pin_details=pin_details,
        case_name=case_name,
        solver=case_definition.solver,
        scratch_case_dir=scratch_case_dir,
        status=status,
        command_results=command_results,
        audit_report=audit_report,
        required_outputs=case_definition.required_outputs,
        required_outputs_present=required_outputs_present,
        no_nan_inf=no_nan_inf,
        fatbinary_report_path=pathlib.Path(fatbinary_report_path),
        fatbinary_ready=fatbinary_ready,
        fatbinary_ptx_present=fatbinary_ptx_present,
        fatbinary_native_sm_found=fatbinary_native_sm_found,
        solver_exit_zero=solver_exit_zero,
        failure_reasons=failure_reasons,
    )
    _write_json(result_json_path, payload)
    return Phase1PtxJitRunResult(
        case_name=case_name,
        scratch_case_dir=scratch_case_dir,
        audit_report_path=audit_report_path,
        result_json_path=result_json_path,
        status=status,
    )


def build_phase1_acceptance_report(
    bundle: AuthorityBundle,
    *,
    output_dir: pathlib.Path | str,
    host_env_path: pathlib.Path | str,
    manifest_refs_path: pathlib.Path | str,
    cuda_probe_path: pathlib.Path | str,
    build_metadata_path: pathlib.Path | str,
    fatbinary_report_path: pathlib.Path | str,
    smoke_result_paths: list[pathlib.Path | str] | tuple[pathlib.Path | str, ...],
    memcheck_result_path: pathlib.Path | str,
    nsys_result_paths: list[pathlib.Path | str] | tuple[pathlib.Path | str, ...],
    ptx_jit_result_path: pathlib.Path | str,
    bringup_doc_path: pathlib.Path | str,
) -> Phase1AcceptanceReportResult:
    output_dir_path = pathlib.Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    pin_details = load_pin_details(bundle)

    resolved_paths = {
        "host_env": pathlib.Path(host_env_path),
        "manifest_refs": pathlib.Path(manifest_refs_path),
        "cuda_probe": pathlib.Path(cuda_probe_path),
        "build_metadata": pathlib.Path(build_metadata_path),
        "fatbinary_report": pathlib.Path(fatbinary_report_path),
        "memcheck_result": pathlib.Path(memcheck_result_path),
        "ptx_jit_result": pathlib.Path(ptx_jit_result_path),
        "bringup_doc": pathlib.Path(bringup_doc_path),
    }
    host_env = _read_json(resolved_paths["host_env"])
    manifest_refs = _read_json(resolved_paths["manifest_refs"])
    cuda_probe = _read_json(resolved_paths["cuda_probe"])
    build_metadata = _read_json(resolved_paths["build_metadata"])
    fatbinary_report = _read_json(resolved_paths["fatbinary_report"])
    memcheck_result = _read_json(resolved_paths["memcheck_result"])
    ptx_jit_result = _read_json(resolved_paths["ptx_jit_result"])

    smoke_results, smoke_input_inventory = _collect_named_result_inputs(
        smoke_result_paths,
        key="case_name",
        required_names=REQUIRED_PHASE1_SMOKE_CASES,
    )
    nsys_results, nsys_input_inventory = _collect_named_result_inputs(
        nsys_result_paths,
        key="profile_mode",
        required_names=REQUIRED_PHASE1_NSYS_MODES,
    )

    host_observations = _as_dict(host_env.get("host_observations"))
    toolkit = _as_dict(host_env.get("toolkit"))
    profilers = _as_dict(host_env.get("profilers"))
    required_host_observation_fields = (
        "hostname",
        "gpu_csv",
        "nvcc_version",
        "gcc_version",
        "nsys_version",
        "ncu_version",
        "compute_sanitizer_version",
        "os_release",
        "kernel",
    )
    missing_host_observation_fields = [
        field_name
        for field_name in required_host_observation_fields
        if not str(host_observations.get(field_name, "")).strip()
    ]
    missing_profiler_fields = [
        field_name
        for field_name in ("nsight_systems", "nsight_compute", "compute_sanitizer")
        if not str(profilers.get(field_name, "")).strip()
    ]
    expected_required_revalidation = list(pin_details.required_revalidation)
    basic_nsys = nsys_results.get("basic", {})
    um_fault_nsys = nsys_results.get("um_fault", {})
    basic_nsys_pass = (
        str(basic_nsys.get("status", "")).lower() == "pass"
        and _nested_bool(basic_nsys, "success_criteria", "audit_passed")
        and _nested_bool(basic_nsys, "success_criteria", "trace_generated")
        and _nested_bool(basic_nsys, "success_criteria", "required_outputs_present")
        and _nested_bool(basic_nsys, "success_criteria", "no_nan_inf")
    )
    um_fault_nsys_pass = (
        str(um_fault_nsys.get("status", "")).lower() == "pass"
        and _nested_bool(um_fault_nsys, "success_criteria", "audit_passed")
        and _nested_bool(um_fault_nsys, "success_criteria", "trace_generated")
        and _nested_bool(um_fault_nsys, "success_criteria", "required_outputs_present")
        and _nested_bool(um_fault_nsys, "success_criteria", "no_nan_inf")
    )
    gate_results = {
        "host_manifest_complete": _gate_result(
            label="Host manifest exists and is complete",
            passed=all(
                (
                    bool(host_env),
                    bool(host_env.get("reviewed_source_tuple_id")),
                    bool(host_env.get("runtime_base")),
                    bool(toolkit.get("selected_lane")),
                    bool(toolkit.get("selected_lane_value")),
                    not missing_host_observation_fields,
                    not missing_profiler_fields,
                )
            ),
            expected={
                "required_host_observation_fields": list(required_host_observation_fields),
                "required_profiler_fields": [
                    "nsight_systems",
                    "nsight_compute",
                    "compute_sanitizer",
                ],
            },
            observed={
                "path": resolved_paths["host_env"].as_posix(),
                "missing_host_observation_fields": missing_host_observation_fields,
                "missing_profiler_fields": missing_profiler_fields,
            },
            evidence=resolved_paths["host_env"].as_posix(),
        ),
        "manifest_refs_traceable": _gate_result(
            label="Manifest refs exist and match the host manifest",
            passed=all(
                (
                    bool(manifest_refs),
                    bool(manifest_refs.get("reviewed_source_tuple_id")),
                    bool(manifest_refs.get("runtime_base")),
                    manifest_refs.get("required_revalidation") == expected_required_revalidation,
                    manifest_refs.get("reviewed_source_tuple_id") == host_env.get("reviewed_source_tuple_id"),
                    manifest_refs.get("runtime_base") == host_env.get("runtime_base"),
                    manifest_refs.get("reviewed_source_tuple_id") == pin_details.reviewed_source_tuple_id,
                    manifest_refs.get("runtime_base") == pin_details.runtime_base,
                )
            ),
            expected={
                "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
                "runtime_base": pin_details.runtime_base,
                "required_revalidation": expected_required_revalidation,
            },
            observed={
                "reviewed_source_tuple_id": manifest_refs.get("reviewed_source_tuple_id"),
                "runtime_base": manifest_refs.get("runtime_base"),
                "required_revalidation": manifest_refs.get("required_revalidation"),
            },
            evidence=resolved_paths["manifest_refs"].as_posix(),
        ),
        "gpu_target_matches_workstation": _gate_result(
            label="Detected GPU is RTX 5080, CC 12.0",
            passed=(
                str(cuda_probe.get("device_name", "")).strip() == "NVIDIA GeForce RTX 5080"
                and int(cuda_probe.get("cc_major", -1)) == 12
                and int(cuda_probe.get("cc_minor", -1)) == 0
            ),
            expected="NVIDIA GeForce RTX 5080, CC 12.0",
            observed={
                "device_name": cuda_probe.get("device_name"),
                "cc_major": cuda_probe.get("cc_major"),
                "cc_minor": cuda_probe.get("cc_minor"),
            },
            evidence=resolved_paths["cuda_probe"].as_posix(),
        ),
        "cuda_probe_runtime_ready": _gate_result(
            label="CUDA probe confirms native kernel and managed-memory readiness",
            passed=bool(cuda_probe.get("native_kernel_ok")) and bool(cuda_probe.get("managed_memory_probe_ok")),
            expected={"native_kernel_ok": True, "managed_memory_probe_ok": True},
            observed={
                "native_kernel_ok": cuda_probe.get("native_kernel_ok"),
                "managed_memory_probe_ok": cuda_probe.get("managed_memory_probe_ok"),
                "managed_memory_failure_reason": cuda_probe.get("managed_memory_failure_reason"),
            },
            evidence=resolved_paths["cuda_probe"].as_posix(),
        ),
        "cuda_probe_traceable": _gate_result(
            label="CUDA probe matches the reviewed tuple, runtime base, and primary lane",
            passed=(
                cuda_probe.get("reviewed_source_tuple_id") == manifest_refs.get("reviewed_source_tuple_id")
                and cuda_probe.get("reviewed_source_tuple_id") == pin_details.reviewed_source_tuple_id
                and cuda_probe.get("runtime_base") == manifest_refs.get("runtime_base")
                and cuda_probe.get("runtime_base") == pin_details.runtime_base
                and _nested_value(cuda_probe, "toolkit", "selected_lane") == "primary"
                and _nested_value(cuda_probe, "toolkit", "selected_lane_value") == pin_details.primary_toolkit_lane
            ),
            expected={
                "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
                "runtime_base": pin_details.runtime_base,
                "selected_lane": "primary",
                "selected_lane_value": pin_details.primary_toolkit_lane,
            },
            observed={
                "reviewed_source_tuple_id": cuda_probe.get("reviewed_source_tuple_id"),
                "runtime_base": cuda_probe.get("runtime_base"),
                "selected_lane": _nested_value(cuda_probe, "toolkit", "selected_lane"),
                "selected_lane_value": _nested_value(cuda_probe, "toolkit", "selected_lane_value"),
            },
            evidence=resolved_paths["cuda_probe"].as_posix(),
        ),
        "primary_lane_toolchain": _gate_result(
            label="Primary lane toolchain is CUDA 12.9.1",
            passed=(
                toolkit.get("selected_lane") == "primary"
                and toolkit.get("selected_lane_value") == pin_details.primary_toolkit_lane
            ),
            expected={"selected_lane": "primary", "selected_lane_value": pin_details.primary_toolkit_lane},
            observed={
                "selected_lane": toolkit.get("selected_lane"),
                "selected_lane_value": toolkit.get("selected_lane_value"),
            },
            evidence=resolved_paths["host_env"].as_posix(),
        ),
        "driver_floor_satisfied": _gate_result(
            label="Driver is 595.45.04 or newer",
            passed=_driver_floor_satisfied(
                host_observations.get("gpu_csv"),
                pin_details.driver_floor,
            ),
            expected=pin_details.driver_floor,
            observed=host_observations.get("gpu_csv"),
            evidence=resolved_paths["host_env"].as_posix(),
        ),
        "build_env_recorded": _gate_result(
            label="have_cuda=true and NVARCH=120 recorded",
            passed=bool(build_metadata.get("have_cuda")) and int(build_metadata.get("nvarch", -1)) == 120,
            expected={"have_cuda": True, "nvarch": 120},
            observed={"have_cuda": build_metadata.get("have_cuda"), "nvarch": build_metadata.get("nvarch")},
            evidence=resolved_paths["build_metadata"].as_posix(),
        ),
        "build_metadata_matches_primary_lane": _gate_result(
            label="Build metadata comes from the required primary lane",
            passed=(
                build_metadata.get("lane") == "primary"
                and build_metadata.get("selected_lane_value") == pin_details.primary_toolkit_lane
            ),
            expected={"lane": "primary", "selected_lane_value": pin_details.primary_toolkit_lane},
            observed={
                "lane": build_metadata.get("lane"),
                "selected_lane_value": build_metadata.get("selected_lane_value"),
            },
            evidence=resolved_paths["build_metadata"].as_posix(),
        ),
        "build_metadata_matches_reviewed_tuple": _gate_result(
            label="Build metadata matches the reviewed source tuple",
            passed=(
                build_metadata.get("reviewed_source_tuple_id") == manifest_refs.get("reviewed_source_tuple_id")
                and build_metadata.get("reviewed_source_tuple_id") == pin_details.reviewed_source_tuple_id
            ),
            expected={"reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id},
            observed={"reviewed_source_tuple_id": build_metadata.get("reviewed_source_tuple_id")},
            evidence=resolved_paths["build_metadata"].as_posix(),
        ),
        "build_metadata_traceable": _gate_result(
            label="Build metadata matches the reviewed tuple, runtime base, and primary lane",
            passed=_artifact_traceable_to_primary_lane(
                build_metadata,
                manifest_refs=manifest_refs,
                pin_details=pin_details,
            ),
            expected=_traceability_expectation(pin_details),
            observed=_traceability_observed(build_metadata),
            evidence=resolved_paths["build_metadata"].as_posix(),
        ),
        "build_succeeded": _gate_result(
            label="SPUMA builds in required lane",
            passed=bool(build_metadata.get("succeeded")) and int(build_metadata.get("returncode", 1)) == 0,
            expected={"succeeded": True, "returncode": 0},
            observed={
                "succeeded": build_metadata.get("succeeded"),
                "returncode": build_metadata.get("returncode"),
                "failure_reason": build_metadata.get("failure_reason"),
            },
            evidence=resolved_paths["build_metadata"].as_posix(),
        ),
        "fatbinary_native_sm_present": _gate_result(
            label="Binary inspection confirms native sm_120",
            passed=bool(fatbinary_report.get("required_native_sm_found")),
            expected=True,
            observed=fatbinary_report.get("required_native_sm_found"),
            evidence=resolved_paths["fatbinary_report"].as_posix(),
        ),
        "fatbinary_ptx_present": _gate_result(
            label="Binary inspection confirms PTX present",
            passed=bool(fatbinary_report.get("ptx_present")),
            expected=True,
            observed=fatbinary_report.get("ptx_present"),
            evidence=resolved_paths["fatbinary_report"].as_posix(),
        ),
        "fatbinary_traceable": _gate_result(
            label="Fatbinary inspection matches the reviewed tuple, runtime base, and primary lane",
            passed=_artifact_traceable_to_primary_lane(
                fatbinary_report,
                manifest_refs=manifest_refs,
                pin_details=pin_details,
            ),
            expected=_traceability_expectation(pin_details),
            observed=_traceability_observed(fatbinary_report),
            evidence=resolved_paths["fatbinary_report"].as_posix(),
        ),
        "ptx_jit_succeeds": _gate_result(
            label="CUDA_FORCE_PTX_JIT=1 run succeeds",
            passed=(
                str(ptx_jit_result.get("status", "")).lower() == "pass"
                and _nested_value(ptx_jit_result, "environment", "CUDA_FORCE_PTX_JIT") == "1"
                and _nested_bool(ptx_jit_result, "success_criteria", "audit_passed")
                and _nested_bool(ptx_jit_result, "success_criteria", "fatbinary_smoke_gate_ready")
                and _nested_bool(ptx_jit_result, "success_criteria", "required_outputs_present")
                and _nested_bool(ptx_jit_result, "success_criteria", "no_nan_inf")
                and _nested_bool(ptx_jit_result, "success_criteria", "solver_exit_zero")
            ),
            expected={
                "status": "pass",
                "CUDA_FORCE_PTX_JIT": "1",
                "audit_passed": True,
                "solver_exit_zero": True,
            },
            observed={
                "status": ptx_jit_result.get("status"),
                "failure_reasons": ptx_jit_result.get("failure_reasons"),
                "environment": ptx_jit_result.get("environment"),
                "audit_passed": _nested_value(ptx_jit_result, "success_criteria", "audit_passed"),
                "solver_exit_zero": _nested_value(ptx_jit_result, "success_criteria", "solver_exit_zero"),
            },
            evidence=resolved_paths["ptx_jit_result"].as_posix(),
        ),
        "ptx_jit_matches_required_case": _gate_result(
            label="PTX-JIT evidence comes from the required cubeLinear lane",
            passed=(
                ptx_jit_result.get("case_name") == DEFAULT_PHASE1_PTX_JIT_CASE
                and ptx_jit_result.get("solver") == "laplacianFoam"
            ),
            expected={"case_name": DEFAULT_PHASE1_PTX_JIT_CASE, "solver": "laplacianFoam"},
            observed={
                "case_name": ptx_jit_result.get("case_name"),
                "solver": ptx_jit_result.get("solver"),
            },
            evidence=resolved_paths["ptx_jit_result"].as_posix(),
        ),
        "ptx_jit_traceable": _gate_result(
            label="PTX-JIT evidence matches the reviewed tuple, runtime base, and primary lane",
            passed=(
                ptx_jit_result.get("reviewed_source_tuple_id") == manifest_refs.get("reviewed_source_tuple_id")
                and ptx_jit_result.get("reviewed_source_tuple_id") == pin_details.reviewed_source_tuple_id
                and ptx_jit_result.get("runtime_base") == manifest_refs.get("runtime_base")
                and ptx_jit_result.get("runtime_base") == pin_details.runtime_base
                and _nested_value(ptx_jit_result, "toolkit", "selected_lane") == "primary"
                and _nested_value(ptx_jit_result, "toolkit", "selected_lane_value") == pin_details.primary_toolkit_lane
            ),
            expected={
                "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
                "runtime_base": pin_details.runtime_base,
                "selected_lane": "primary",
                "selected_lane_value": pin_details.primary_toolkit_lane,
            },
            observed={
                "reviewed_source_tuple_id": ptx_jit_result.get("reviewed_source_tuple_id"),
                "runtime_base": ptx_jit_result.get("runtime_base"),
                "selected_lane": _nested_value(ptx_jit_result, "toolkit", "selected_lane"),
                "selected_lane_value": _nested_value(ptx_jit_result, "toolkit", "selected_lane_value"),
            },
            evidence=resolved_paths["ptx_jit_result"].as_posix(),
        ),
        "laplacian_smoke_passes": _smoke_gate(
            smoke_results.get("cubeLinear"),
            label="laplacianFoam smoke case passes",
            expected_case_name="cubeLinear",
            expected_solver="laplacianFoam",
            evidence=_smoke_evidence_path(smoke_result_paths, "cubeLinear"),
        ),
        "simple_smoke_passes": _smoke_gate(
            smoke_results.get("channelSteady"),
            label="simpleFoam smoke case passes",
            expected_case_name="channelSteady",
            expected_solver="simpleFoam",
            evidence=_smoke_evidence_path(smoke_result_paths, "channelSteady"),
        ),
        "pimple_smoke_passes": _smoke_gate(
            smoke_results.get("channelTransient"),
            label="pimpleFoam smoke case passes",
            expected_case_name="channelTransient",
            expected_solver="pimpleFoam",
            evidence=_smoke_evidence_path(smoke_result_paths, "channelTransient"),
        ),
        "smoke_result_inventory_complete": _gate_result(
            label="Smoke result inputs cover the required cases exactly once",
            passed=bool(smoke_input_inventory["complete"]),
            expected={
                "required": list(REQUIRED_PHASE1_SMOKE_CASES),
                "missing": [],
                "unexpected": [],
                "duplicates": {},
            },
            observed=smoke_input_inventory,
            evidence=",".join(item["path"] for item in smoke_input_inventory["provided"]),
        ),
        "smoke_results_traceable": _gate_result(
            label="Smoke results match the reviewed tuple, runtime base, and primary lane",
            passed=all(
                _artifact_traceable_to_primary_lane(
                    smoke_results.get(case_name),
                    manifest_refs=manifest_refs,
                    pin_details=pin_details,
                )
                for case_name in ("cubeLinear", "channelSteady", "channelTransient")
            ),
            expected=_traceability_expectation(pin_details),
            observed={
                case_name: _traceability_observed(smoke_results.get(case_name))
                for case_name in ("cubeLinear", "channelSteady", "channelTransient")
            },
            evidence=",".join(
                _smoke_evidence_path(smoke_result_paths, case_name)
                for case_name in ("cubeLinear", "channelSteady", "channelTransient")
            ),
        ),
        "nvtx_visible_in_nsys": _gate_result(
            label="NVTX3 ranges visible in Nsight Systems",
            passed=basic_nsys_pass and _nested_bool(basic_nsys, "success_criteria", "phase1_required_ranges_present"),
            expected=True,
            observed=_nested_value(basic_nsys, "nvtx", "missing_phase1_ranges"),
            evidence=_nsys_evidence_path(nsys_result_paths, "basic"),
        ),
        "gpu_kernels_visible_in_nsys": _gate_result(
            label="GPU kernels visible in Nsight Systems",
            passed=basic_nsys_pass and _nested_bool(basic_nsys, "success_criteria", "gpu_kernels_present"),
            expected=True,
            observed=_nested_value(basic_nsys, "success_criteria", "gpu_kernels_present"),
            evidence=_nsys_evidence_path(nsys_result_paths, "basic"),
        ),
        "nsys_basic_artifact_passes": _gate_result(
            label="Baseline Nsight artifact completed successfully",
            passed=basic_nsys_pass,
            expected={
                "status": "pass",
                "audit_passed": True,
                "trace_generated": True,
                "required_outputs_present": True,
                "no_nan_inf": True,
            },
            observed={
                "status": basic_nsys.get("status"),
                "failure_reasons": basic_nsys.get("failure_reasons"),
                "success_criteria": _nested_value(basic_nsys, "success_criteria"),
            },
            evidence=_nsys_evidence_path(nsys_result_paths, "basic"),
        ),
        "memcheck_matches_required_case": _gate_result(
            label="Memcheck evidence comes from the required cubeLinear lane",
            passed=(
                memcheck_result.get("case_name") == DEFAULT_PHASE1_PTX_JIT_CASE
                and memcheck_result.get("solver") == "laplacianFoam"
            ),
            expected={"case_name": DEFAULT_PHASE1_PTX_JIT_CASE, "solver": "laplacianFoam"},
            observed={
                "case_name": memcheck_result.get("case_name"),
                "solver": memcheck_result.get("solver"),
            },
            evidence=resolved_paths["memcheck_result"].as_posix(),
        ),
        "memcheck_passes": _gate_result(
            label="Compute Sanitizer memcheck passes on smallest case",
            passed=(
                str(memcheck_result.get("status", "")).lower() == "pass"
                and _nested_bool(memcheck_result, "success_criteria", "audit_passed")
                and _nested_bool(memcheck_result, "success_criteria", "required_outputs_present")
                and _nested_bool(memcheck_result, "success_criteria", "no_nan_inf")
                and _nested_bool(memcheck_result, "success_criteria", "error_summary_found")
                and int(_nested_value(memcheck_result, "memcheck", "actionable_errors") or 0) == 0
            ),
            expected={
                "status": "pass",
                "audit_passed": True,
                "required_outputs_present": True,
                "no_nan_inf": True,
                "error_summary_found": True,
                "actionable_errors": 0,
            },
            observed={
                "status": memcheck_result.get("status"),
                "audit_passed": _nested_value(memcheck_result, "success_criteria", "audit_passed"),
                "required_outputs_present": _nested_value(
                    memcheck_result, "success_criteria", "required_outputs_present"
                ),
                "no_nan_inf": _nested_value(memcheck_result, "success_criteria", "no_nan_inf"),
                "error_summary_found": _nested_value(
                    memcheck_result, "success_criteria", "error_summary_found"
                ),
                "actionable_errors": _nested_value(memcheck_result, "memcheck", "actionable_errors"),
                "failure_reasons": memcheck_result.get("failure_reasons"),
            },
            evidence=resolved_paths["memcheck_result"].as_posix(),
        ),
        "memcheck_traceable": _gate_result(
            label="Memcheck evidence matches the reviewed tuple, runtime base, and primary lane",
            passed=_artifact_traceable_to_primary_lane(
                memcheck_result,
                manifest_refs=manifest_refs,
                pin_details=pin_details,
            ),
            expected=_traceability_expectation(pin_details),
            observed=_traceability_observed(memcheck_result),
            evidence=resolved_paths["memcheck_result"].as_posix(),
        ),
        "uvm_trace_captured": _gate_result(
            label="UVM-fault trace captured on one case",
            passed=um_fault_nsys_pass and _nested_bool(um_fault_nsys, "success_criteria", "uvm_evidence_present"),
            expected=True,
            observed=_nested_value(um_fault_nsys, "uvm", "classification"),
            evidence=_nsys_evidence_path(nsys_result_paths, "um_fault"),
        ),
        "no_unexplained_recurring_page_migrations": _gate_result(
            label="No unexplained recurring page migrations remain",
            passed=um_fault_nsys_pass and str(_nested_value(um_fault_nsys, "uvm", "classification")) in {
                "clean",
                "documented_activity",
            },
            expected=["clean", "documented_activity"],
            observed=_nested_value(um_fault_nsys, "uvm", "classification"),
            evidence=_nsys_evidence_path(nsys_result_paths, "um_fault"),
        ),
        "nsys_um_fault_artifact_passes": _gate_result(
            label="UVM diagnostic Nsight artifact completed successfully",
            passed=um_fault_nsys_pass,
            expected={
                "status": "pass",
                "audit_passed": True,
                "trace_generated": True,
                "required_outputs_present": True,
                "no_nan_inf": True,
            },
            observed={
                "status": um_fault_nsys.get("status"),
                "failure_reasons": um_fault_nsys.get("failure_reasons"),
                "success_criteria": _nested_value(um_fault_nsys, "success_criteria"),
            },
            evidence=_nsys_evidence_path(nsys_result_paths, "um_fault"),
        ),
        "nsys_result_inventory_complete": _gate_result(
            label="Nsight result inputs cover the required modes exactly once",
            passed=bool(nsys_input_inventory["complete"]),
            expected={
                "required": list(REQUIRED_PHASE1_NSYS_MODES),
                "missing": [],
                "unexpected": [],
                "duplicates": {},
            },
            observed=nsys_input_inventory,
            evidence=",".join(item["path"] for item in nsys_input_inventory["provided"]),
        ),
        "nsys_results_traceable": _gate_result(
            label="Nsight artifacts match the reviewed tuple, runtime base, and primary lane",
            passed=all(
                _artifact_traceable_to_primary_lane(
                    nsys_results.get(mode),
                    manifest_refs=manifest_refs,
                    pin_details=pin_details,
                )
                for mode in ("basic", "um_fault")
            ),
            expected=_traceability_expectation(pin_details),
            observed={
                mode: _traceability_observed(nsys_results.get(mode))
                for mode in ("basic", "um_fault")
            },
            evidence=",".join(_nsys_evidence_path(nsys_result_paths, mode) for mode in ("basic", "um_fault")),
        ),
        "bringup_doc_present": _gate_result(
            label="Bring-up documentation is archived with the acceptance bundle",
            passed=resolved_paths["bringup_doc"].exists(),
            expected=True,
            observed=resolved_paths["bringup_doc"].exists(),
            evidence=resolved_paths["bringup_doc"].as_posix(),
        ),
    }

    gate_order = tuple(gate_results)
    failing_gate_ids = [gate_id for gate_id in gate_order if not bool(gate_results[gate_id]["passed"])]
    disposition = "pass" if not failing_gate_ids else "fail"
    status = "PASS" if disposition == "pass" else "FAIL"
    reason = (
        "All Phase 1 hard acceptance checks passed."
        if not failing_gate_ids
        else "Hard gate failures: " + ", ".join(failing_gate_ids) + "."
    )
    accepted_phase1_proposal = {
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "gpu_target": pin_details.gpu_target,
        "workstation_target": pin_details.workstation_target,
        "instrumentation": pin_details.instrumentation,
        "profilers": {
            "nsight_systems": pin_details.nsight_systems,
            "nsight_compute": pin_details.nsight_compute,
            "compute_sanitizer": pin_details.compute_sanitizer,
        },
    }

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": PHASE1_ACCEPTANCE_REPORT_NAME,
        "phase_gate": "Phase 1",
        "card_id": "P1-07",
        "status": status,
        "disposition": disposition,
        "reason": reason,
        "reviewed_source_tuple_id": manifest_refs.get("reviewed_source_tuple_id")
        or host_env.get("reviewed_source_tuple_id"),
        "runtime_base": manifest_refs.get("runtime_base") or host_env.get("runtime_base"),
        "lane": toolkit.get("selected_lane"),
        "lane_value": toolkit.get("selected_lane_value"),
        "workstation": {
            "hostname": host_observations.get("hostname"),
            "gpu_csv": host_observations.get("gpu_csv"),
            "device_name": cuda_probe.get("device_name"),
            "compute_capability": f"{cuda_probe.get('cc_major')}.{cuda_probe.get('cc_minor')}",
        },
        "accepted_phase1_proposal": accepted_phase1_proposal,
        "authority_revisions": host_env.get("authority_revisions")
        or manifest_refs.get("authority_revisions")
        or bundle.authority_revisions,
        "required_revalidation": manifest_refs.get("required_revalidation", []),
        "input_inventory": {
            "smoke_results": smoke_input_inventory,
            "nsys_results": nsys_input_inventory,
        },
        "artifact_paths": {
            **{name: path.as_posix() for name, path in resolved_paths.items()},
            "build_log": build_metadata.get("build_log"),
            "fatbinary_artifacts": _fatbinary_artifacts(fatbinary_report),
            "smoke_result_inputs": list(smoke_input_inventory["provided"]),
            "nsys_result_inputs": list(nsys_input_inventory["provided"]),
            "audit_reports": {
                "ptx_jit": _sibling_artifact_path(resolved_paths["ptx_jit_result"], "smoke_audit.json"),
                "memcheck": _sibling_artifact_path(resolved_paths["memcheck_result"], "smoke_audit.json"),
                "smoke_cases": {
                    str((_read_json(pathlib.Path(item)).get("case_name") or pathlib.Path(item).stem)): _sibling_artifact_path(
                        pathlib.Path(item),
                        "smoke_audit.json",
                    )
                    for item in smoke_result_paths
                },
                "nsys": {
                    str((_read_json(pathlib.Path(item)).get("profile_mode") or pathlib.Path(item).stem)): _sibling_artifact_path(
                        pathlib.Path(item),
                        "smoke_audit.json",
                    )
                    for item in nsys_result_paths
                },
            },
            "memcheck_logs": _command_log_paths(memcheck_result),
            "ptx_jit_logs": _ptx_jit_log_paths(ptx_jit_result),
            "smoke_logs": {
                str((_read_json(pathlib.Path(item)).get("case_name") or pathlib.Path(item).stem)): _command_log_paths(
                    _read_json(pathlib.Path(item))
                )
                for item in smoke_result_paths
            },
            "nsys_logs": {
                str((_read_json(pathlib.Path(item)).get("profile_mode") or pathlib.Path(item).stem)): _command_log_paths(
                    _read_json(pathlib.Path(item))
                )
                for item in nsys_result_paths
            },
            "nsys_supporting_artifacts": {
                str((_read_json(pathlib.Path(item)).get("profile_mode") or pathlib.Path(item).stem)): _nsys_supporting_artifacts(
                    pathlib.Path(item)
                )
                for item in nsys_result_paths
            },
            "nsys_trace_artifacts": {
                str((_read_json(pathlib.Path(item)).get("profile_mode") or pathlib.Path(item).stem)): _trace_artifacts(
                    _read_json(pathlib.Path(item))
                )
                for item in nsys_result_paths
            },
            "smoke_results": {
                str((_read_json(pathlib.Path(item)).get("case_name") or pathlib.Path(item).stem)): pathlib.Path(item).as_posix()
                for item in smoke_result_paths
            },
            "nsys_results": {
                str((_read_json(pathlib.Path(item)).get("profile_mode") or pathlib.Path(item).stem)): pathlib.Path(item).as_posix()
                for item in nsys_result_paths
            },
        },
        "gate_results": {
            "hard": gate_results,
            "soft": {},
        },
        "failing_gate_ids": failing_gate_ids,
        "checklist_order": list(gate_order),
        "acceptance_manifest_revision": bundle.authority_revisions["acceptance_manifest"]["sha256"],
        "accepted_tuple_id": None,
        "tuple_admission": "Phase 1 uses the FND-05 disposition vocabulary but does not map to an admitted acceptance tuple row yet.",
    }

    json_path = output_dir_path / PHASE1_ACCEPTANCE_REPORT_NAME
    markdown_path = output_dir_path / PHASE1_ACCEPTANCE_MARKDOWN_NAME
    bundle_index_path = output_dir_path / PHASE1_ACCEPTANCE_BUNDLE_INDEX_NAME
    bundle_index = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "phase_gate": "Phase 1",
        "card_id": "P1-07",
        "status": status,
        "disposition": disposition,
        "reason": reason,
        "failing_gate_ids": list(failing_gate_ids),
        "reviewed_source_tuple_id": payload["reviewed_source_tuple_id"],
        "runtime_base": payload["runtime_base"],
        "lane": payload["lane"],
        "lane_value": payload["lane_value"],
        "authority_revisions": payload["authority_revisions"],
        "required_revalidation": payload["required_revalidation"],
        "acceptance_manifest_revision": payload["acceptance_manifest_revision"],
        "accepted_phase1_proposal": payload["accepted_phase1_proposal"],
        "accepted_tuple_id": payload["accepted_tuple_id"],
        "tuple_admission": payload["tuple_admission"],
        "input_inventory": payload["input_inventory"],
        "workstation": dict(payload["workstation"]),
        "workstation_manifests": {
            "host_env": resolved_paths["host_env"].as_posix(),
            "manifest_refs": resolved_paths["manifest_refs"].as_posix(),
            "cuda_probe": resolved_paths["cuda_probe"].as_posix(),
        },
        "supporting_artifacts": {
            "build_log": payload["artifact_paths"]["build_log"],
            "fatbinary_artifacts": payload["artifact_paths"]["fatbinary_artifacts"],
            "audit_reports": payload["artifact_paths"]["audit_reports"],
            "ptx_jit_logs": payload["artifact_paths"]["ptx_jit_logs"],
            "memcheck_logs": payload["artifact_paths"]["memcheck_logs"],
            "smoke_logs": payload["artifact_paths"]["smoke_logs"],
            "nsys_logs": payload["artifact_paths"]["nsys_logs"],
            "nsys_supporting_artifacts": payload["artifact_paths"]["nsys_supporting_artifacts"],
            "nsys_trace_artifacts": payload["artifact_paths"]["nsys_trace_artifacts"],
        },
        "outputs": {
            "phase1_acceptance_report_json": json_path.as_posix(),
            "phase1_acceptance_report_markdown": markdown_path.as_posix(),
            "phase1_acceptance_bundle_index": bundle_index_path.as_posix(),
        },
        "inputs": {
            name: path.as_posix()
            for name, path in resolved_paths.items()
        },
        "smoke_results": payload["artifact_paths"]["smoke_results"],
        "nsys_results": payload["artifact_paths"]["nsys_results"],
    }
    _write_json(json_path, payload)
    markdown_path.write_text(_render_acceptance_markdown(payload), encoding="utf-8")
    _write_json(bundle_index_path, bundle_index)
    return Phase1AcceptanceReportResult(
        json_path=json_path,
        markdown_path=markdown_path,
        bundle_index_path=bundle_index_path,
        status=status,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ptx_jit_parser = subparsers.add_parser("ptx-jit", help="Run the Phase 1 PTX-JIT compatibility proof.")
    ptx_jit_parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    ptx_jit_parser.add_argument("--scratch-root", type=pathlib.Path, required=True)
    ptx_jit_parser.add_argument("--fatbinary-report", type=pathlib.Path, required=True)
    ptx_jit_parser.add_argument(
        "--case",
        default=DEFAULT_PHASE1_PTX_JIT_CASE,
        choices=[definition.name for definition in CASE_DEFINITIONS],
    )

    report_parser = subparsers.add_parser("report", help="Build the Phase 1 acceptance report bundle.")
    report_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    report_parser.add_argument("--host-env-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--manifest-refs-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--cuda-probe-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--build-metadata-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--fatbinary-report-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--smoke-result", type=pathlib.Path, action="append", default=[])
    report_parser.add_argument("--memcheck-result-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--nsys-result", type=pathlib.Path, action="append", default=[])
    report_parser.add_argument("--ptx-jit-result-json", type=pathlib.Path, required=True)
    report_parser.add_argument("--bringup-doc", type=pathlib.Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_root = repo_root(args.root)
    bundle = load_authority_bundle(resolved_root)

    if args.command == "ptx-jit":
        result = run_phase1_ptx_jit(
            bundle,
            artifact_root=args.artifact_root,
            scratch_root=args.scratch_root,
            fatbinary_report_path=args.fatbinary_report,
            root=resolved_root,
            case_name=args.case,
        )
        print(result.result_json_path.as_posix())
        return 0 if result.status == "pass" else 1

    report = build_phase1_acceptance_report(
        bundle,
        output_dir=args.output_dir,
        host_env_path=args.host_env_json,
        manifest_refs_path=args.manifest_refs_json,
        cuda_probe_path=args.cuda_probe_json,
        build_metadata_path=args.build_metadata_json,
        fatbinary_report_path=args.fatbinary_report_json,
        smoke_result_paths=args.smoke_result,
        memcheck_result_path=args.memcheck_result_json,
        nsys_result_paths=args.nsys_result,
        ptx_jit_result_path=args.ptx_jit_result_json,
        bringup_doc_path=args.bringup_doc,
    )
    print(report.json_path.as_posix())
    return 0 if report.status == "PASS" else 1


def _definition_by_name(case_name: str):
    for definition in CASE_DEFINITIONS:
        if definition.name == case_name:
            return definition
    raise ValueError(f"unknown Phase 1 smoke case {case_name!r}")


def _build_ptx_jit_failure_reasons(
    *,
    command_results: list[dict[str, Any]],
    required_outputs_present: bool,
    no_nan_inf: bool,
) -> list[str]:
    failure_reasons: list[str] = []
    if not command_results:
        failure_reasons.append("missing_commands")
        return failure_reasons
    if command_results[0]["returncode"] != 0:
        failure_reasons.append("setup_command_failed")
    if len(command_results) < 2:
        failure_reasons.append("ptx_jit_not_run")
    elif command_results[1]["returncode"] != 0:
        failure_reasons.append("ptx_jit_command_failed")
    if not required_outputs_present:
        failure_reasons.append("required_outputs_missing")
    if not no_nan_inf:
        failure_reasons.append("nan_or_inf_detected")
    return failure_reasons


def _build_ptx_jit_payload(
    *,
    pin_details: Any,
    case_name: str,
    solver: str,
    scratch_case_dir: pathlib.Path,
    status: str,
    command_results: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    audit_report: Phase1SmokeAuditReport,
    required_outputs: tuple[str, ...],
    required_outputs_present: bool,
    no_nan_inf: bool,
    fatbinary_report_path: pathlib.Path,
    fatbinary_ready: bool,
    fatbinary_ptx_present: bool,
    fatbinary_native_sm_found: bool,
    solver_exit_zero: bool,
    failure_reasons: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
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
        "scratch_case_dir": scratch_case_dir.as_posix(),
        "command_results": list(command_results),
        "required_outputs": list(required_outputs),
        "reject_codes": [reason["code"] for reason in audit_report.reject_reasons],
        "failure_reasons": list(failure_reasons),
        "environment": {
            "CUDA_FORCE_PTX_JIT": "1",
        },
        "fatbinary": {
            "report_path": fatbinary_report_path.as_posix(),
            "smoke_gate_ready": fatbinary_ready,
            "ptx_present": fatbinary_ptx_present,
            "required_native_sm_found": fatbinary_native_sm_found,
        },
        "success_criteria": {
            "audit_passed": audit_report.startup_allowed,
            "fatbinary_smoke_gate_ready": fatbinary_ready,
            "required_outputs_present": required_outputs_present,
            "no_nan_inf": no_nan_inf,
            "solver_exit_zero": solver_exit_zero,
        },
    }


def _smoke_gate(
    payload: Mapping[str, Any] | None,
    *,
    label: str,
    expected_case_name: str,
    expected_solver: str,
    evidence: str,
) -> dict[str, Any]:
    payload = dict(payload or {})
    return _gate_result(
        label=label,
        passed=(
            str(payload.get("case_name", "")) == expected_case_name
            and str(payload.get("solver", "")) == expected_solver
            and str(payload.get("status", "")).lower() == "pass"
            and _nested_bool(payload, "success_criteria", "audit_passed")
            and _nested_bool(payload, "success_criteria", "required_outputs_present")
            and _nested_bool(payload, "success_criteria", "no_nan_inf")
        ),
        expected={
            "case_name": expected_case_name,
            "solver": expected_solver,
            "status": "pass",
        },
        observed={
            "case_name": payload.get("case_name"),
            "solver": payload.get("solver"),
            "status": payload.get("status"),
            "failure_reasons": payload.get("failure_reasons"),
        },
        evidence=evidence,
    )


def _gate_result(
    *,
    label: str,
    passed: bool,
    expected: Any,
    observed: Any,
    evidence: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "passed": passed,
        "expected": expected,
        "observed": observed,
        "evidence": evidence,
    }


def _traceability_expectation(pin_details: Any) -> dict[str, Any]:
    return {
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "selected_lane": "primary",
        "selected_lane_value": pin_details.primary_toolkit_lane,
    }


def _traceability_observed(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "reviewed_source_tuple_id": _nested_value(payload, "reviewed_source_tuple_id"),
        "runtime_base": _nested_value(payload, "runtime_base"),
        "selected_lane": _nested_value(payload, "toolkit", "selected_lane"),
        "selected_lane_value": _nested_value(payload, "toolkit", "selected_lane_value"),
    }


def _artifact_traceable_to_primary_lane(
    payload: Mapping[str, Any] | None,
    *,
    manifest_refs: Mapping[str, Any],
    pin_details: Any,
) -> bool:
    return (
        _nested_value(payload, "reviewed_source_tuple_id") == manifest_refs.get("reviewed_source_tuple_id")
        and _nested_value(payload, "reviewed_source_tuple_id") == pin_details.reviewed_source_tuple_id
        and _nested_value(payload, "runtime_base") == manifest_refs.get("runtime_base")
        and _nested_value(payload, "runtime_base") == pin_details.runtime_base
        and _nested_value(payload, "toolkit", "selected_lane") == "primary"
        and _nested_value(payload, "toolkit", "selected_lane_value") == pin_details.primary_toolkit_lane
    )


def _smoke_evidence_path(
    smoke_result_paths: list[pathlib.Path | str] | tuple[pathlib.Path | str, ...],
    case_name: str,
) -> str:
    for item in smoke_result_paths:
        path = pathlib.Path(item)
        payload = _read_json(path)
        if str(payload.get("case_name") or "") == case_name:
            return path.as_posix()
    return case_name


def _nsys_evidence_path(
    nsys_result_paths: list[pathlib.Path | str] | tuple[pathlib.Path | str, ...],
    mode: str,
) -> str:
    for item in nsys_result_paths:
        path = pathlib.Path(item)
        payload = _read_json(path)
        if str(payload.get("profile_mode") or "") == mode:
            return path.as_posix()
    return mode


def _collect_named_result_inputs(
    paths: list[pathlib.Path | str] | tuple[pathlib.Path | str, ...],
    *,
    key: str,
    required_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    names_to_paths: dict[str, list[str]] = {}
    provided: list[dict[str, str]] = []
    for item in paths:
        path = pathlib.Path(item)
        payload = _read_json(path)
        name = str(payload.get(key) or path.stem)
        payloads.setdefault(name, payload)
        names_to_paths.setdefault(name, []).append(path.as_posix())
        provided.append({"name": name, "path": path.as_posix()})
    duplicates = {
        name: artifact_paths
        for name, artifact_paths in names_to_paths.items()
        if len(artifact_paths) > 1
    }
    missing = [name for name in required_names if name not in names_to_paths]
    unexpected = sorted(name for name in names_to_paths if name not in required_names)
    inventory = {
        "required": list(required_names),
        "provided": provided,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "complete": not missing and not unexpected and not duplicates,
    }
    return payloads, inventory


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_value(payload: Mapping[str, Any] | None, *keys: str) -> Any:
    current: Any = payload or {}
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_bool(payload: Mapping[str, Any] | None, *keys: str) -> bool:
    return bool(_nested_value(payload, *keys))


def _driver_floor_satisfied(observed_gpu_csv: Any, driver_floor: str) -> bool:
    observed_match = VERSION_PATTERN.search(str(observed_gpu_csv or ""))
    required_match = VERSION_PATTERN.search(str(driver_floor or ""))
    if observed_match is None or required_match is None:
        return False
    observed = tuple(int(group) for group in observed_match.groups())
    required = tuple(int(group) for group in required_match.groups())
    return observed >= required


def _append_input_inventory_markdown(
    lines: list[str],
    title: str,
    inventory: Mapping[str, Any],
) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
        ]
    )
    for entry in inventory.get("provided", []):
        if isinstance(entry, Mapping):
            lines.append(f"- `{entry.get('name')}`: `{entry.get('path')}`")
    missing = inventory.get("missing", [])
    unexpected = inventory.get("unexpected", [])
    duplicates = inventory.get("duplicates", {})
    if missing:
        lines.append(f"- Missing: `{', '.join(str(name) for name in missing)}`")
    if unexpected:
        lines.append(f"- Unexpected: `{', '.join(str(name) for name in unexpected)}`")
    if isinstance(duplicates, Mapping):
        for name, artifact_paths in duplicates.items():
            joined_paths = ", ".join(f"`{artifact_path}`" for artifact_path in artifact_paths)
            lines.append(f"- Duplicate `{name}` inputs: {joined_paths}")
    lines.append("")


def _render_acceptance_markdown(payload: Mapping[str, Any]) -> str:
    workstation = payload["workstation"]
    proposal = payload["accepted_phase1_proposal"]
    proposal_toolkit = proposal["toolkit"]
    proposal_profilers = proposal["profilers"]
    lines = [
        "# Phase 1 Acceptance Report",
        "",
        f"Status: {payload['status']}",
        f"Disposition: {payload['disposition']}",
        f"Reason: {payload['reason']}",
        "",
        "## Context",
        "",
        f"- Phase gate: `{payload['phase_gate']}`",
        f"- Card: `{payload['card_id']}`",
        f"- Reviewed source tuple: `{payload['reviewed_source_tuple_id']}`",
        f"- Runtime base: `{payload['runtime_base']}`",
        f"- Lane: `{payload['lane']}` (`{payload['lane_value']}`)",
        f"- Host: `{workstation['hostname']}`",
        f"- GPU: `{workstation['device_name']}`",
        f"- GPU observation: `{workstation['gpu_csv']}`",
        "",
        "## Contract Traceability",
        "",
        f"- Acceptance manifest revision: `{payload['acceptance_manifest_revision']}`",
        f"- Required revalidation: `{', '.join(payload['required_revalidation'])}`",
        "",
        "### Authority Revisions",
        "",
    ]
    for authority_name, revision in payload["authority_revisions"].items():
        lines.append(f"- `{authority_name}`: `{revision.get('sha256')}`")
    lines.extend(
        [
            "",
        "## Accepted Proposal",
        "",
        f"- Reviewed source tuple: `{proposal['reviewed_source_tuple_id']}`",
        f"- Runtime base: `{proposal['runtime_base']}`",
        f"- Primary toolkit lane: `{proposal_toolkit['primary_lane']}`",
        f"- Experimental toolkit lane: `{proposal_toolkit['experimental_lane']}`",
        f"- Driver floor: `{proposal_toolkit['driver_floor']}`",
        f"- GPU target: `{proposal['gpu_target']}`",
        f"- Workstation target: `{proposal['workstation_target']}`",
        f"- Instrumentation: `{proposal['instrumentation']}`",
        f"- Nsight Systems: `{proposal_profilers['nsight_systems']}`",
        f"- Nsight Compute: `{proposal_profilers['nsight_compute']}`",
        f"- Compute Sanitizer: `{proposal_profilers['compute_sanitizer']}`",
        "",
        "## Checklist",
        "",
        ]
    )
    for gate_id in payload["checklist_order"]:
        gate = payload["gate_results"]["hard"][gate_id]
        marker = "x" if gate["passed"] else " "
        lines.append(f"- [{marker}] `{gate_id}`: {gate['label']}")
        lines.append(f"  - Evidence: `{gate['evidence']}`")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- `host_env.json`: `{payload['artifact_paths']['host_env']}`",
            f"- `manifest_refs.json`: `{payload['artifact_paths']['manifest_refs']}`",
            f"- `cuda_probe.json`: `{payload['artifact_paths']['cuda_probe']}`",
            f"- Build metadata: `{payload['artifact_paths']['build_metadata']}`",
            f"- Build log: `{payload['artifact_paths']['build_log']}`",
            f"- `fatbinary_report.json`: `{payload['artifact_paths']['fatbinary_report']}`",
            f"- Memcheck result: `{payload['artifact_paths']['memcheck_result']}`",
            f"- PTX-JIT result: `{payload['artifact_paths']['ptx_jit_result']}`",
            f"- Bring-up doc: `{payload['artifact_paths']['bringup_doc']}`",
            "",
        ]
    )
    fatbinary_artifacts = payload["artifact_paths"]["fatbinary_artifacts"]
    if fatbinary_artifacts:
        lines.extend(
            [
                "### Fatbinary Artifacts",
                "",
                *[f"- `{name}`: `{artifact_path}`" for name, artifact_path in fatbinary_artifacts.items()],
                "",
            ]
        )
    audit_reports = payload["artifact_paths"]["audit_reports"]
    if audit_reports:
        lines.extend(
            [
                "### Audit Reports",
                "",
                f"- `ptx_jit`: `{audit_reports.get('ptx_jit')}`",
                f"- `memcheck`: `{audit_reports.get('memcheck')}`",
            ]
        )
        for case_name, artifact_path in audit_reports.get("smoke_cases", {}).items():
            lines.append(f"- `smoke` `{case_name}`: `{artifact_path}`")
        for profile_mode, artifact_path in audit_reports.get("nsys", {}).items():
            lines.append(f"- `nsys` `{profile_mode}`: `{artifact_path}`")
        lines.append("")
    smoke_results = payload["artifact_paths"]["smoke_results"]
    if smoke_results:
        lines.extend(
            [
                "### Smoke Results",
                "",
                *[f"- `{case_name}`: `{artifact_path}`" for case_name, artifact_path in smoke_results.items()],
                "",
            ]
        )
    smoke_input_inventory = _as_dict(_nested_value(payload, "input_inventory", "smoke_results"))
    if smoke_input_inventory:
        _append_input_inventory_markdown(lines, "Smoke Input Inventory", smoke_input_inventory)
    smoke_logs = payload["artifact_paths"]["smoke_logs"]
    if smoke_logs:
        lines.extend(
            [
                "### Smoke Logs",
                "",
            ]
        )
        for case_name, logs in smoke_logs.items():
            for command_name, artifact_path in logs.items():
                lines.append(f"- `{case_name}` `{command_name}`: `{artifact_path}`")
        lines.append("")
    nsys_results = payload["artifact_paths"]["nsys_results"]
    if nsys_results:
        lines.extend(
            [
                "### Nsight Results",
                "",
                *[f"- `{profile_mode}`: `{artifact_path}`" for profile_mode, artifact_path in nsys_results.items()],
                "",
            ]
        )
    nsys_input_inventory = _as_dict(_nested_value(payload, "input_inventory", "nsys_results"))
    if nsys_input_inventory:
        _append_input_inventory_markdown(lines, "Nsight Input Inventory", nsys_input_inventory)
    nsys_logs = payload["artifact_paths"]["nsys_logs"]
    if nsys_logs:
        lines.extend(
            [
                "### Nsight Logs",
                "",
            ]
        )
        for profile_mode, logs in nsys_logs.items():
            for command_name, artifact_path in logs.items():
                lines.append(f"- `{profile_mode}` `{command_name}`: `{artifact_path}`")
        lines.append("")
    nsys_supporting_artifacts = payload["artifact_paths"]["nsys_supporting_artifacts"]
    if nsys_supporting_artifacts:
        lines.extend(
            [
                "### Nsight Supporting Artifacts",
                "",
            ]
        )
        for profile_mode, artifacts in nsys_supporting_artifacts.items():
            lines.append(f"- `{profile_mode}` summary: `{artifacts.get('summary')}`")
            lines.append(f"- `{profile_mode}` NVTX report: `{artifacts.get('nvtx_report')}`")
        lines.append("")
    ptx_jit_logs = payload["artifact_paths"]["ptx_jit_logs"]
    if ptx_jit_logs:
        lines.extend(
            [
                "### PTX-JIT Logs",
                "",
                *[f"- `{command_name}`: `{artifact_path}`" for command_name, artifact_path in ptx_jit_logs.items()],
                "",
            ]
        )
    memcheck_logs = payload["artifact_paths"]["memcheck_logs"]
    if memcheck_logs:
        lines.extend(
            [
                "### Memcheck Logs",
                "",
                *[f"- `{command_name}`: `{artifact_path}`" for command_name, artifact_path in memcheck_logs.items()],
                "",
            ]
        )
    nsys_trace_artifacts = payload["artifact_paths"]["nsys_trace_artifacts"]
    if nsys_trace_artifacts:
        lines.extend(
            [
                "### Nsight Trace Artifacts",
                "",
            ]
        )
        for profile_mode, artifacts in nsys_trace_artifacts.items():
            lines.append(f"- `{profile_mode}` trace: `{artifacts.get('trace')}`")
            lines.append(f"- `{profile_mode}` sqlite: `{artifacts.get('sqlite')}`")
            lines.append(f"- `{profile_mode}` stats: `{artifacts.get('stats_dir')}`")
        lines.append("")
    if payload["failing_gate_ids"]:
        lines.extend(
            [
                "## Failing Gates",
                "",
                *[f"- `{gate_id}`" for gate_id in payload["failing_gate_ids"]],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _default_command_runner(
    command: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    env: dict[str, str] | None = None,
) -> int:
    execution_env = None
    if env:
        execution_env = dict(os.environ)
        execution_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=execution_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def _run_command(
    runner: Callable[..., int],
    command: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    env: dict[str, str] | None = None,
) -> int:
    try:
        return runner(command, cwd=cwd, log_path=log_path, env=env)
    except TypeError:
        return runner(command, cwd=cwd, log_path=log_path)
    except OSError as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"command launch failed: {exc}\n", encoding="utf-8")
        return 127 if isinstance(exc, FileNotFoundError) else 126


def _logs_are_clean(command_results: list[dict[str, Any]]) -> bool:
    for result in command_results:
        path = pathlib.Path(result["log_path"])
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False
        if LOG_NAN_INF_PATTERN.search(text):
            return False
    return True


def _ptx_jit_log_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    return _command_log_paths(payload)


def _command_log_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    log_paths: dict[str, str] = {}
    for index, entry in enumerate(payload.get("command_results", []), start=1):
        if not isinstance(entry, Mapping):
            continue
        command = entry.get("command")
        command_name: str | None = None
        if isinstance(command, (list, tuple)) and command:
            command_name = str(command[0])
        log_path = entry.get("log_path")
        if not command_name or not log_path:
            continue
        key = command_name if command_name not in log_paths else f"{index}_{command_name}"
        log_paths[key] = str(log_path)
    return log_paths


def _trace_artifacts(payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = payload.get("trace_artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        key: str(value)
        for key, value in artifacts.items()
        if value
    }


def _fatbinary_artifacts(payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    return {
        key: str(value)
        for key, value in artifacts.items()
        if key != "report" and value
    }


def _nsys_supporting_artifacts(result_path: pathlib.Path) -> dict[str, str]:
    parent = result_path.parent
    return {
        "summary": (parent / "nsys_profile_summary.txt").as_posix(),
        "nvtx_report": (parent / "nvtx_range_report.json").as_posix(),
    }


def _sibling_artifact_path(result_path: pathlib.Path, filename: str) -> str:
    return (result_path.parent / filename).as_posix()


def _write_json(path: pathlib.Path | str, payload: Mapping[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
