"""Phase 1 Compute Sanitizer memcheck lane for the smallest smoke case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

try:
    from .bundle import AuthorityBundle, load_authority_bundle, repo_root
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
    from scripts.authority.phase1_smoke import (  # type: ignore
        CASE_DEFINITIONS,
        MANIFEST_SCHEMA_VERSION,
        PHASE1_SMOKE_AUDIT_NAME,
        Phase1SmokeAuditReport,
        scan_phase1_smoke_case,
    )


SMALLEST_PHASE1_SMOKE_CASE = "cubeLinear"
PHASE1_SANITIZER_ARTIFACT_DIRNAME = "compute_sanitizer"
PHASE1_MEMCHECK_RESULT_NAME = "memcheck_result.json"
PHASE1_MEMCHECK_LOG_NAME = "memcheck.log"
ERROR_SUMMARY_PATTERN = re.compile(r"ERROR SUMMARY:\s*(?P<count>\d+)\s+errors?", flags=re.IGNORECASE)
MEMCHECK_EVENT_PATTERN = re.compile(r"^=+\s+(?P<message>.+)$")
LOG_NAN_INF_PATTERN = re.compile(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", flags=re.IGNORECASE)
NON_ACTIONABLE_NOTE_PATTERNS = (
    ("lineinfo_absent", re.compile(r"lineinfo", flags=re.IGNORECASE)),
    ("debug_symbols_missing", re.compile(r"debug symbols?", flags=re.IGNORECASE)),
    ("third_party_noise", re.compile(r"third[- ]party|external allocator|system library", flags=re.IGNORECASE)),
)
ACTIONABLE_EVENT_PATTERN = re.compile(
    r"invalid|misaligned|uninitialized|out of bounds|leak|error|cudaError",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Phase1MemcheckParseResult:
    error_summary_found: bool
    error_summary_count: int | None
    actionable_errors: int
    classification: str
    notes: tuple[str, ...]
    summary_line: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_summary_found": self.error_summary_found,
            "error_summary_count": self.error_summary_count,
            "actionable_errors": self.actionable_errors,
            "classification": self.classification,
            "notes": list(self.notes),
            "summary_line": self.summary_line,
        }


@dataclass(frozen=True)
class Phase1MemcheckRunResult:
    case_name: str
    scratch_case_dir: pathlib.Path
    audit_report_path: pathlib.Path
    result_json_path: pathlib.Path
    memcheck_log_path: pathlib.Path
    status: str


def parse_compute_sanitizer_memcheck_log(log_text: str) -> Phase1MemcheckParseResult:
    summary_match = ERROR_SUMMARY_PATTERN.search(log_text)
    notes = tuple(
        code
        for code, pattern in NON_ACTIONABLE_NOTE_PATTERNS
        if pattern.search(log_text)
    )
    if not summary_match:
        return Phase1MemcheckParseResult(
            error_summary_found=False,
            error_summary_count=None,
            actionable_errors=0,
            classification="missing_error_summary",
            notes=notes,
            summary_line=None,
        )

    error_summary_count = int(summary_match.group("count"))
    summary_line = next(
        (
            line.strip()
            for line in log_text.splitlines()
            if ERROR_SUMMARY_PATTERN.search(line)
        ),
        None,
    )
    actionable_events = _collect_actionable_events(log_text)
    actionable_errors = error_summary_count
    if error_summary_count == 0:
        classification = "clean"
    elif not actionable_events and notes:
        actionable_errors = 0
        classification = "non_actionable_noise"
    else:
        classification = "actionable_errors"
    return Phase1MemcheckParseResult(
        error_summary_found=True,
        error_summary_count=error_summary_count,
        actionable_errors=actionable_errors,
        classification=classification,
        notes=notes,
        summary_line=summary_line,
    )


def run_phase1_memcheck(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    scratch_root: pathlib.Path | str,
    root: pathlib.Path | str | None = None,
    case_name: str = SMALLEST_PHASE1_SMOKE_CASE,
    command_runner: Callable[..., int] | None = None,
    compute_sanitizer_command: tuple[str, ...] = ("compute-sanitizer",),
) -> Phase1MemcheckRunResult:
    if case_name != SMALLEST_PHASE1_SMOKE_CASE:
        raise ValueError(
            f"Phase 1 memcheck is limited to the smallest supported smoke case {SMALLEST_PHASE1_SMOKE_CASE!r}"
        )

    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)
    audit_report = scan_phase1_smoke_case(bundle, case_name=case_name, root=resolved_root)
    case_definition = _definition_by_name(case_name)
    artifact_case_root = (
        pathlib.Path(artifact_root)
        / PHASE1_SANITIZER_ARTIFACT_DIRNAME
        / case_name
    )
    artifact_case_root.mkdir(parents=True, exist_ok=True)
    audit_report_path = artifact_case_root / PHASE1_SMOKE_AUDIT_NAME
    _write_json(audit_report_path, audit_report.as_dict())

    scratch_case_dir = pathlib.Path(scratch_root) / case_name
    if scratch_case_dir.exists():
        shutil.rmtree(scratch_case_dir)

    memcheck_log_path = artifact_case_root / PHASE1_MEMCHECK_LOG_NAME
    if memcheck_log_path.exists():
        memcheck_log_path.unlink()
    if not audit_report.startup_allowed:
        result_payload = _build_result_payload(
            case_name=case_name,
            solver=case_definition.solver,
            scratch_case_dir=scratch_case_dir,
            status="blocked",
            command_results=(),
            audit_report=audit_report,
            required_outputs=case_definition.required_outputs,
            required_outputs_present=False,
            no_nan_inf=True,
            memcheck=parse_compute_sanitizer_memcheck_log(""),
            failure_reasons=("audit_failed",),
        )
        result_json_path = artifact_case_root / PHASE1_MEMCHECK_RESULT_NAME
        _write_json(result_json_path, result_payload)
        return Phase1MemcheckRunResult(
            case_name=case_name,
            scratch_case_dir=scratch_case_dir,
            audit_report_path=audit_report_path,
            result_json_path=result_json_path,
            memcheck_log_path=memcheck_log_path,
            status="blocked",
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
        }
    )
    if block_mesh_returncode == 0:
        memcheck_command = (*compute_sanitizer_command, "--tool", "memcheck", case_definition.solver)
        memcheck_returncode = _run_command(
            runner,
            memcheck_command,
            cwd=scratch_case_dir,
            log_path=memcheck_log_path,
        )
        command_results.append(
            {
                "command": list(memcheck_command),
                "cwd": scratch_case_dir.as_posix(),
                "log_path": memcheck_log_path.as_posix(),
                "returncode": memcheck_returncode,
            }
        )

    memcheck = _parse_memcheck_log(memcheck_log_path)
    required_outputs_present = all(
        (scratch_case_dir / relative_path).exists() for relative_path in case_definition.required_outputs
    )
    no_nan_inf = _logs_are_clean(command_results)
    failure_reasons = tuple(
        _build_failure_reasons(
            command_results=command_results,
            required_outputs_present=required_outputs_present,
            no_nan_inf=no_nan_inf,
            memcheck=memcheck,
        )
    )
    status = "pass" if not failure_reasons else "fail"
    result_payload = _build_result_payload(
        case_name=case_name,
        solver=case_definition.solver,
        scratch_case_dir=scratch_case_dir,
        status=status,
        command_results=command_results,
        audit_report=audit_report,
        required_outputs=case_definition.required_outputs,
        required_outputs_present=required_outputs_present,
        no_nan_inf=no_nan_inf,
        memcheck=memcheck,
        failure_reasons=failure_reasons,
    )
    result_json_path = artifact_case_root / PHASE1_MEMCHECK_RESULT_NAME
    _write_json(result_json_path, result_payload)
    return Phase1MemcheckRunResult(
        case_name=case_name,
        scratch_case_dir=scratch_case_dir,
        audit_report_path=audit_report_path,
        result_json_path=result_json_path,
        memcheck_log_path=memcheck_log_path,
        status=status,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run Phase 1 memcheck for the smallest smoke case.")
    run_parser.add_argument(
        "--case",
        default=SMALLEST_PHASE1_SMOKE_CASE,
        choices=[SMALLEST_PHASE1_SMOKE_CASE],
    )
    run_parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    run_parser.add_argument("--scratch-root", type=pathlib.Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_root = repo_root(args.root)
    resolved_bundle = load_authority_bundle(resolved_root)
    result = run_phase1_memcheck(
        resolved_bundle,
        case_name=args.case,
        artifact_root=args.artifact_root,
        scratch_root=args.scratch_root,
        root=resolved_root,
    )
    print(result.result_json_path.as_posix())
    return 0 if result.status == "pass" else 1


def _definition_by_name(case_name: str) -> Any:
    for definition in CASE_DEFINITIONS:
        if definition.name == case_name:
            return definition
    raise ValueError(f"unknown Phase 1 smoke case {case_name!r}")


def _build_failure_reasons(
    *,
    command_results: list[dict[str, Any]],
    required_outputs_present: bool,
    no_nan_inf: bool,
    memcheck: Phase1MemcheckParseResult,
) -> list[str]:
    failure_reasons: list[str] = []
    if not command_results:
        failure_reasons.append("missing_commands")
        return failure_reasons
    if command_results[0]["returncode"] != 0:
        failure_reasons.append("setup_command_failed")
    if len(command_results) < 2:
        failure_reasons.append("memcheck_not_run")
    elif command_results[1]["returncode"] != 0:
        failure_reasons.append("memcheck_command_failed")
    if not memcheck.error_summary_found:
        failure_reasons.append("missing_error_summary")
    if memcheck.actionable_errors:
        failure_reasons.append("actionable_memcheck_errors")
    if not required_outputs_present:
        failure_reasons.append("required_outputs_missing")
    if not no_nan_inf:
        failure_reasons.append("nan_or_inf_detected")
    return failure_reasons


def _collect_actionable_events(log_text: str) -> list[str]:
    actionable_events: list[str] = []
    for line in log_text.splitlines():
        match = MEMCHECK_EVENT_PATTERN.match(line.strip())
        if not match:
            continue
        message = match.group("message").strip()
        if ERROR_SUMMARY_PATTERN.search(message):
            continue
        if any(pattern.search(message) for _, pattern in NON_ACTIONABLE_NOTE_PATTERNS):
            continue
        if ACTIONABLE_EVENT_PATTERN.search(message):
            actionable_events.append(message)
    return actionable_events


def _parse_memcheck_log(log_path: pathlib.Path) -> Phase1MemcheckParseResult:
    try:
        return parse_compute_sanitizer_memcheck_log(log_path.read_text(encoding="utf-8"))
    except OSError:
        return parse_compute_sanitizer_memcheck_log("")


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
    scratch_case_dir: pathlib.Path,
    status: str,
    command_results: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    audit_report: Phase1SmokeAuditReport,
    required_outputs: tuple[str, ...],
    required_outputs_present: bool,
    no_nan_inf: bool,
    memcheck: Phase1MemcheckParseResult,
    failure_reasons: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": PHASE1_MEMCHECK_RESULT_NAME,
        "case_name": case_name,
        "solver": solver,
        "status": status,
        "scratch_case_dir": scratch_case_dir.as_posix(),
        "command_results": list(command_results),
        "required_outputs": list(required_outputs),
        "reject_codes": [reason["code"] for reason in audit_report.reject_reasons],
        "failure_reasons": list(failure_reasons),
        "success_criteria": {
            "audit_passed": audit_report.startup_allowed,
            "required_outputs_present": required_outputs_present,
            "no_nan_inf": no_nan_inf,
            "error_summary_found": memcheck.error_summary_found,
            "actionable_errors": memcheck.actionable_errors,
        },
        "memcheck": memcheck.as_dict(),
    }


def _write_json(path: pathlib.Path | str, payload: Mapping[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
