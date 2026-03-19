"""Phase 1 Nsight Systems baseline and UVM diagnostic profiling lanes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

try:
    from .bundle import AuthorityBundle, load_authority_bundle, repo_root
    from .graph_registry import load_graph_stage_registry
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
    from scripts.authority.graph_registry import load_graph_stage_registry  # type: ignore
    from scripts.authority.phase1_smoke import (  # type: ignore
        CASE_DEFINITIONS,
        MANIFEST_SCHEMA_VERSION,
        PHASE1_SMOKE_AUDIT_NAME,
        Phase1SmokeAuditReport,
        scan_phase1_smoke_case,
    )


DEFAULT_PHASE1_PROFILE_CASE = "channelTransient"
PHASE1_NSYS_ARTIFACT_DIRNAME = "nsight_systems"
PHASE1_NSYS_RESULT_NAME = "nsys_profile_result.json"
PHASE1_NSYS_SUMMARY_NAME = "nsys_profile_summary.txt"
PHASE1_NSYS_NVTX_REPORT_NAME = "nvtx_range_report.json"
PHASE1_NSYS_STATS_DIRNAME = "stats"
PHASE1_NSYS_TRACE_NAME = "trace.nsys-rep"
PHASE1_NSYS_SQLITE_NAME = "trace.sqlite"
LOG_NAN_INF_PATTERN = re.compile(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", flags=re.IGNORECASE)
PHASE1_BASELINE_REQUIRED_RANGES = (
    "phase1:init",
    "phase1:caseSetup",
    "phase1:solveLoop",
    "phase1:iteration",
    "phase1:linearSolve",
    "phase1:write",
)
PHASE1_PIMPLE_REQUIRED_RANGE = "phase1:pimple:outerLoop"
PROFILE_MODE_CONFIG = {
    "basic": {
        "diagnostic_only": False,
        "reports": ("cuda_gpu_kern_sum", "nvtx_sum", "nvtx_gpu_proj_sum"),
        "extra_profile_args": (),
    },
    "um_fault": {
        "diagnostic_only": True,
        "reports": (
            "cuda_gpu_kern_sum",
            "nvtx_sum",
            "nvtx_gpu_proj_sum",
            "um_sum",
            "um_total_sum",
            "um_cpu_page_faults_sum",
        ),
        "extra_profile_args": (
            "--cuda-um-cpu-page-faults=true",
            "--cuda-um-gpu-page-faults=true",
        ),
    },
}


@dataclass(frozen=True)
class Phase1NsysRunResult:
    case_name: str
    mode: str
    scratch_case_dir: pathlib.Path
    audit_report_path: pathlib.Path
    result_json_path: pathlib.Path
    summary_path: pathlib.Path
    trace_path: pathlib.Path
    status: str


@dataclass(frozen=True)
class NvtxRangeReport:
    required_phase1_ranges: tuple[str, ...]
    present_phase1_ranges: tuple[str, ...]
    missing_phase1_ranges: tuple[str, ...]
    authority_required_orchestration_ranges: tuple[str, ...]
    present_authority_orchestration_ranges: tuple[str, ...]
    missing_authority_orchestration_ranges: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_phase1_ranges": list(self.required_phase1_ranges),
            "present_phase1_ranges": list(self.present_phase1_ranges),
            "missing_phase1_ranges": list(self.missing_phase1_ranges),
            "authority_required_orchestration_ranges": list(self.authority_required_orchestration_ranges),
            "present_authority_orchestration_ranges": list(self.present_authority_orchestration_ranges),
            "missing_authority_orchestration_ranges": list(self.missing_authority_orchestration_ranges),
        }


@dataclass(frozen=True)
class UvmTriageReport:
    diagnostic_requested: bool
    evidence_present: bool
    cpu_um_faults: int
    gpu_um_faults: int
    classification: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_requested": self.diagnostic_requested,
            "evidence_present": self.evidence_present,
            "cpu_um_faults": self.cpu_um_faults,
            "gpu_um_faults": self.gpu_um_faults,
            "classification": self.classification,
        }


def run_phase1_nsys_profile(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    scratch_root: pathlib.Path | str,
    root: pathlib.Path | str | None = None,
    case_name: str = DEFAULT_PHASE1_PROFILE_CASE,
    mode: str = "basic",
    command_runner: Callable[..., int] | None = None,
    nsys_command: tuple[str, ...] = ("nsys",),
) -> Phase1NsysRunResult:
    if mode not in PROFILE_MODE_CONFIG:
        raise ValueError(f"unsupported Phase 1 Nsight Systems mode {mode!r}")
    if mode == "um_fault" and case_name != DEFAULT_PHASE1_PROFILE_CASE:
        raise ValueError(
            f"Phase 1 UVM diagnostic mode is limited to the smallest transient smoke case {DEFAULT_PHASE1_PROFILE_CASE!r}"
        )

    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)
    audit_report = scan_phase1_smoke_case(bundle, case_name=case_name, root=resolved_root)
    case_definition = _definition_by_name(case_name)
    profile_root = pathlib.Path(artifact_root) / PHASE1_NSYS_ARTIFACT_DIRNAME / mode / case_name
    profile_root.mkdir(parents=True, exist_ok=True)
    audit_report_path = profile_root / PHASE1_SMOKE_AUDIT_NAME
    _write_json(audit_report_path, audit_report.as_dict())

    scratch_case_dir = pathlib.Path(scratch_root) / case_name
    if scratch_case_dir.exists():
        shutil.rmtree(scratch_case_dir)

    result_json_path = profile_root / PHASE1_NSYS_RESULT_NAME
    summary_path = profile_root / PHASE1_NSYS_SUMMARY_NAME
    trace_path = profile_root / PHASE1_NSYS_TRACE_NAME
    sqlite_path = profile_root / PHASE1_NSYS_SQLITE_NAME
    stats_dir = profile_root / PHASE1_NSYS_STATS_DIRNAME
    nvtx_report_path = profile_root / PHASE1_NSYS_NVTX_REPORT_NAME
    if trace_path.exists():
        trace_path.unlink()
    if sqlite_path.exists():
        sqlite_path.unlink()
    if stats_dir.exists():
        shutil.rmtree(stats_dir)

    if not audit_report.startup_allowed:
        nvtx_report = NvtxRangeReport(
            required_phase1_ranges=_required_phase1_ranges(case_name),
            present_phase1_ranges=(),
            missing_phase1_ranges=_required_phase1_ranges(case_name),
            authority_required_orchestration_ranges=_authority_orchestration_ranges(resolved_root),
            present_authority_orchestration_ranges=(),
            missing_authority_orchestration_ranges=_authority_orchestration_ranges(resolved_root),
        )
        uvm = UvmTriageReport(
            diagnostic_requested=bool(PROFILE_MODE_CONFIG[mode]["diagnostic_only"]),
            evidence_present=False,
            cpu_um_faults=0,
            gpu_um_faults=0,
            classification="not_requested" if mode == "basic" else "missing_diagnostic_evidence",
        )
        payload = _build_result_payload(
            case_name=case_name,
            solver=case_definition.solver,
            mode=mode,
            scratch_case_dir=scratch_case_dir,
            status="blocked",
            command_results=(),
            audit_report=audit_report,
            required_outputs=case_definition.required_outputs,
            required_outputs_present=False,
            no_nan_inf=True,
            gpu_kernels_present=False,
            trace_generated=False,
            nvtx_report=nvtx_report,
            uvm=uvm,
            failure_reasons=("audit_failed",),
            trace_path=trace_path,
            sqlite_path=sqlite_path,
            stats_dir=stats_dir,
        )
        _write_json(nvtx_report_path, nvtx_report.as_dict())
        _write_json(result_json_path, payload)
        summary_path.write_text(_render_summary(payload), encoding="utf-8")
        return Phase1NsysRunResult(
            case_name=case_name,
            mode=mode,
            scratch_case_dir=scratch_case_dir,
            audit_report_path=audit_report_path,
            result_json_path=result_json_path,
            summary_path=summary_path,
            trace_path=trace_path,
            status="blocked",
        )

    shutil.copytree(audit_report.case_dir, scratch_case_dir)
    runner = command_runner or _default_command_runner
    stats_dir.mkdir(parents=True, exist_ok=True)
    command_results: list[dict[str, Any]] = []

    block_mesh_log_path = profile_root / "01_blockMesh.log"
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
        }
    )

    if block_mesh_returncode == 0:
        profile_log_path = profile_root / "02_nsys_profile.log"
        trace_output_stem = trace_path.with_suffix("")
        profile_command = (
            *nsys_command,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--output",
            trace_output_stem.as_posix(),
            *PROFILE_MODE_CONFIG[mode]["extra_profile_args"],
            case_definition.solver,
        )
        profile_returncode = _run_command(
            runner,
            profile_command,
            cwd=scratch_case_dir,
            log_path=profile_log_path,
        )
        command_results.append(
            {
                "command": list(profile_command),
                "cwd": scratch_case_dir.as_posix(),
                "log_path": profile_log_path.as_posix(),
                "returncode": profile_returncode,
            }
        )

        if profile_returncode == 0 and trace_path.exists():
            export_log_path = profile_root / "03_nsys_export.log"
            export_command = (
                *nsys_command,
                "export",
                "--type",
                "sqlite",
                "--output",
                sqlite_path.as_posix(),
                trace_path.as_posix(),
            )
            export_returncode = _run_command(
                runner,
                export_command,
                cwd=scratch_case_dir,
                log_path=export_log_path,
            )
            command_results.append(
                {
                    "command": list(export_command),
                    "cwd": scratch_case_dir.as_posix(),
                    "log_path": export_log_path.as_posix(),
                    "returncode": export_returncode,
                }
            )
            if export_returncode == 0 and sqlite_path.exists():
                for index, report_name in enumerate(PROFILE_MODE_CONFIG[mode]["reports"], start=4):
                    stats_log_path = profile_root / f"{index:02d}_{report_name}.log"
                    output_path = stats_dir / f"{report_name}.csv"
                    stats_command = (
                        *nsys_command,
                        "stats",
                        "--report",
                        report_name,
                        "--format",
                        "csv",
                        "--output",
                        output_path.as_posix(),
                        sqlite_path.as_posix(),
                    )
                    stats_returncode = _run_command(
                        runner,
                        stats_command,
                        cwd=scratch_case_dir,
                        log_path=stats_log_path,
                    )
                    command_results.append(
                        {
                            "command": list(stats_command),
                            "cwd": scratch_case_dir.as_posix(),
                            "log_path": stats_log_path.as_posix(),
                            "returncode": stats_returncode,
                        }
                    )
                    if stats_returncode != 0:
                        break

    gpu_kernels_present = _gpu_kernels_present(stats_dir / "cuda_gpu_kern_sum.csv")
    nvtx_report = _build_nvtx_report(
        stats_dir / "nvtx_sum.csv",
        required_phase1_ranges=_required_phase1_ranges(case_name),
        authority_required_orchestration_ranges=_authority_orchestration_ranges(resolved_root),
    )
    uvm = _build_uvm_report(mode, stats_dir)
    required_outputs_present = all(
        (scratch_case_dir / relative_path).exists() for relative_path in case_definition.required_outputs
    )
    no_nan_inf = _logs_are_clean(command_results)
    trace_generated = trace_path.exists()
    failure_reasons = tuple(
        _build_failure_reasons(
            command_results=command_results,
            required_outputs_present=required_outputs_present,
            no_nan_inf=no_nan_inf,
            trace_generated=trace_generated,
            gpu_kernels_present=gpu_kernels_present,
            nvtx_report=nvtx_report,
            uvm=uvm,
            mode=mode,
        )
    )
    status = "pass" if not failure_reasons else "fail"
    payload = _build_result_payload(
        case_name=case_name,
        solver=case_definition.solver,
        mode=mode,
        scratch_case_dir=scratch_case_dir,
        status=status,
        command_results=command_results,
        audit_report=audit_report,
        required_outputs=case_definition.required_outputs,
        required_outputs_present=required_outputs_present,
        no_nan_inf=no_nan_inf,
        gpu_kernels_present=gpu_kernels_present,
        trace_generated=trace_generated,
        nvtx_report=nvtx_report,
        uvm=uvm,
        failure_reasons=failure_reasons,
        trace_path=trace_path,
        sqlite_path=sqlite_path,
        stats_dir=stats_dir,
    )
    _write_json(nvtx_report_path, nvtx_report.as_dict())
    _write_json(result_json_path, payload)
    summary_path.write_text(_render_summary(payload), encoding="utf-8")
    return Phase1NsysRunResult(
        case_name=case_name,
        mode=mode,
        scratch_case_dir=scratch_case_dir,
        audit_report_path=audit_report_path,
        result_json_path=result_json_path,
        summary_path=summary_path,
        trace_path=trace_path,
        status=status,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a Phase 1 Nsight Systems capture.")
    run_parser.add_argument(
        "--case",
        default=DEFAULT_PHASE1_PROFILE_CASE,
        choices=[definition.name for definition in CASE_DEFINITIONS],
    )
    run_parser.add_argument(
        "--mode",
        default="basic",
        choices=tuple(PROFILE_MODE_CONFIG),
    )
    run_parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    run_parser.add_argument("--scratch-root", type=pathlib.Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_root = repo_root(args.root)
    resolved_bundle = load_authority_bundle(resolved_root)
    result = run_phase1_nsys_profile(
        resolved_bundle,
        case_name=args.case,
        mode=args.mode,
        artifact_root=args.artifact_root,
        scratch_root=args.scratch_root,
        root=resolved_root,
    )
    print(result.result_json_path.as_posix())
    return 0 if result.status == "pass" else 1


def _required_phase1_ranges(case_name: str) -> tuple[str, ...]:
    required_ranges = list(PHASE1_BASELINE_REQUIRED_RANGES)
    if case_name == DEFAULT_PHASE1_PROFILE_CASE:
        required_ranges.append(PHASE1_PIMPLE_REQUIRED_RANGE)
    return tuple(required_ranges)


def _authority_orchestration_ranges(root: pathlib.Path) -> tuple[str, ...]:
    registry = load_graph_stage_registry(root)
    return tuple(registry.required_orchestration_ranges)


def _definition_by_name(case_name: str) -> Any:
    for definition in CASE_DEFINITIONS:
        if definition.name == case_name:
            return definition
    raise ValueError(f"unknown Phase 1 smoke case {case_name!r}")


def _gpu_kernels_present(csv_path: pathlib.Path) -> bool:
    rows = _read_csv_rows(csv_path)
    return bool(rows)


def _build_nvtx_report(
    csv_path: pathlib.Path,
    *,
    required_phase1_ranges: tuple[str, ...],
    authority_required_orchestration_ranges: tuple[str, ...],
) -> NvtxRangeReport:
    observed_values = {value for row in _read_csv_rows(csv_path) for value in row.values()}
    present_phase1 = tuple(
        sorted(range_name for range_name in required_phase1_ranges if range_name in observed_values)
    )
    missing_phase1 = tuple(
        sorted(range_name for range_name in required_phase1_ranges if range_name not in observed_values)
    )
    present_authority = tuple(
        sorted(range_name for range_name in authority_required_orchestration_ranges if range_name in observed_values)
    )
    missing_authority = tuple(
        sorted(range_name for range_name in authority_required_orchestration_ranges if range_name not in observed_values)
    )
    return NvtxRangeReport(
        required_phase1_ranges=required_phase1_ranges,
        present_phase1_ranges=present_phase1,
        missing_phase1_ranges=missing_phase1,
        authority_required_orchestration_ranges=authority_required_orchestration_ranges,
        present_authority_orchestration_ranges=present_authority,
        missing_authority_orchestration_ranges=missing_authority,
    )


def _build_uvm_report(mode: str, stats_dir: pathlib.Path) -> UvmTriageReport:
    if mode != "um_fault":
        return UvmTriageReport(
            diagnostic_requested=False,
            evidence_present=False,
            cpu_um_faults=0,
            gpu_um_faults=0,
            classification="not_requested",
        )

    um_sum_rows = _read_csv_rows(stats_dir / "um_sum.csv")
    um_total_rows = _read_csv_rows(stats_dir / "um_total_sum.csv")
    cpu_fault_rows = _read_csv_rows(stats_dir / "um_cpu_page_faults_sum.csv")
    evidence_present = bool(um_sum_rows or um_total_rows or cpu_fault_rows)
    cpu_um_faults = _sum_numeric_cells(cpu_fault_rows)
    gpu_um_faults = _sum_numeric_cells(
        [
            row
            for row in um_sum_rows
            if "gpu" in " ".join(row.values()).lower() and "fault" in " ".join(row.values()).lower()
        ]
    )
    if not evidence_present:
        classification = "missing_diagnostic_evidence"
    elif cpu_um_faults or gpu_um_faults or _rows_contain_activity(um_sum_rows, um_total_rows):
        classification = "documented_activity"
    else:
        classification = "clean"
    return UvmTriageReport(
        diagnostic_requested=True,
        evidence_present=evidence_present,
        cpu_um_faults=cpu_um_faults,
        gpu_um_faults=gpu_um_faults,
        classification=classification,
    )


def _rows_contain_activity(*row_groups: list[dict[str, str]]) -> bool:
    activity_tokens = ("fault", "migration", "htod", "dtoh", "um")
    for rows in row_groups:
        for row in rows:
            lowered = " ".join(row.values()).lower()
            if any(token in lowered for token in activity_tokens):
                return True
    return False


def _sum_numeric_cells(rows: list[dict[str, str]]) -> int:
    total = 0
    for row in rows:
        for value in row.values():
            if re.fullmatch(r"-?\d+", value.strip()):
                total += int(value.strip())
    return total


def _build_failure_reasons(
    *,
    command_results: list[dict[str, Any]],
    required_outputs_present: bool,
    no_nan_inf: bool,
    trace_generated: bool,
    gpu_kernels_present: bool,
    nvtx_report: NvtxRangeReport,
    uvm: UvmTriageReport,
    mode: str,
) -> list[str]:
    failure_reasons: list[str] = []
    if not command_results:
        failure_reasons.append("missing_commands")
        return failure_reasons
    if command_results[0]["returncode"] != 0:
        failure_reasons.append("setup_command_failed")
    if len(command_results) < 2:
        failure_reasons.append("profile_not_run")
        return failure_reasons
    if command_results[1]["returncode"] != 0:
        failure_reasons.append("profile_command_failed")
    if not trace_generated:
        failure_reasons.append("missing_trace_artifact")
    if any(
        result["command"][:2] == ["nsys", "export"] and result["returncode"] != 0
        for result in command_results
    ):
        failure_reasons.append("export_command_failed")
    if any(
        result["command"][:2] == ["nsys", "stats"] and result["returncode"] != 0
        for result in command_results
    ):
        failure_reasons.append("stats_command_failed")
    if not gpu_kernels_present:
        failure_reasons.append("missing_gpu_kernels")
    if nvtx_report.missing_phase1_ranges:
        failure_reasons.append("missing_phase1_nvtx_ranges")
    if mode == "um_fault" and not uvm.evidence_present:
        failure_reasons.append("missing_uvm_diagnostic_evidence")
    if not required_outputs_present:
        failure_reasons.append("required_outputs_missing")
    if not no_nan_inf:
        failure_reasons.append("nan_or_inf_detected")
    return failure_reasons


def _read_csv_rows(csv_path: pathlib.Path) -> list[dict[str, str]]:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                if row is None:
                    continue
                normalized = {
                    str(key).strip(): str(value or "").strip()
                    for key, value in row.items()
                    if key is not None
                }
                if any(normalized.values()):
                    rows.append(normalized)
            return rows
    except OSError:
        return []


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


def _build_result_payload(
    *,
    case_name: str,
    solver: str,
    mode: str,
    scratch_case_dir: pathlib.Path,
    status: str,
    command_results: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    audit_report: Phase1SmokeAuditReport,
    required_outputs: tuple[str, ...],
    required_outputs_present: bool,
    no_nan_inf: bool,
    gpu_kernels_present: bool,
    trace_generated: bool,
    nvtx_report: NvtxRangeReport,
    uvm: UvmTriageReport,
    failure_reasons: tuple[str, ...] | list[str],
    trace_path: pathlib.Path,
    sqlite_path: pathlib.Path,
    stats_dir: pathlib.Path,
) -> dict[str, Any]:
    diagnostic_only = bool(PROFILE_MODE_CONFIG[mode]["diagnostic_only"])
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": PHASE1_NSYS_RESULT_NAME,
        "case_name": case_name,
        "solver": solver,
        "profile_mode": mode,
        "diagnostic_only": diagnostic_only,
        "timing_baseline_eligible": not diagnostic_only,
        "status": status,
        "scratch_case_dir": scratch_case_dir.as_posix(),
        "command_results": list(command_results),
        "required_outputs": list(required_outputs),
        "reject_codes": [reason["code"] for reason in audit_report.reject_reasons],
        "failure_reasons": list(failure_reasons),
        "trace_artifacts": {
            "trace": trace_path.as_posix(),
            "sqlite": sqlite_path.as_posix(),
            "stats_dir": stats_dir.as_posix(),
        },
        "success_criteria": {
            "audit_passed": audit_report.startup_allowed,
            "trace_generated": trace_generated,
            "required_outputs_present": required_outputs_present,
            "no_nan_inf": no_nan_inf,
            "gpu_kernels_present": gpu_kernels_present,
            "phase1_required_ranges_present": not nvtx_report.missing_phase1_ranges,
            "uvm_evidence_present": uvm.evidence_present,
        },
        "nvtx": nvtx_report.as_dict(),
        "uvm": uvm.as_dict(),
    }


def _render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "Phase 1 Nsight Systems summary",
        f"mode={payload['profile_mode']}",
        f"case={payload['case_name']}",
        f"status={payload['status']}",
        f"timing_baseline_eligible={payload['timing_baseline_eligible']}",
        f"gpu_kernels_present={payload['success_criteria']['gpu_kernels_present']}",
        f"phase1_required_ranges_present={payload['success_criteria']['phase1_required_ranges_present']}",
        f"cpu_um_faults={payload['uvm']['cpu_um_faults']}",
        f"gpu_um_faults={payload['uvm']['gpu_um_faults']}",
        f"uvm_classification={payload['uvm']['classification']}",
    ]
    missing_ranges = payload["nvtx"]["missing_phase1_ranges"]
    if missing_ranges:
        lines.append("missing_phase1_ranges=" + ",".join(missing_ranges))
    if payload["failure_reasons"]:
        lines.append("failure_reasons=" + ",".join(payload["failure_reasons"]))
    return "\n".join(lines) + "\n"


def _default_command_runner(
    command: tuple[str, ...],
    *,
    cwd: pathlib.Path,
    log_path: pathlib.Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
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
) -> int:
    try:
        return runner(command, cwd=cwd, log_path=log_path)
    except OSError as exc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"command launch failed: {exc}\n", encoding="utf-8")
        return 127 if isinstance(exc, FileNotFoundError) else 126


def _write_json(path: pathlib.Path | str, payload: Mapping[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
