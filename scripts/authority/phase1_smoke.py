"""Phase 1 repo-local smoke-case planning, audit, and execution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any, Callable, Mapping

try:
    from .bundle import AuthorityBundle, repo_root
    from .support_scanner import SupportFunctionObject, SupportScanRequest, scan_support_matrix
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.authority.bundle import AuthorityBundle, repo_root  # type: ignore
    from scripts.authority.support_scanner import (  # type: ignore
        SupportFunctionObject,
        SupportScanRequest,
        scan_support_matrix,
    )


MANIFEST_SCHEMA_VERSION = "1.0.0"
PHASE1_SMOKE_CASES_DIR = pathlib.Path("tools/bringup/cases/phase1_smoke")
PHASE1_SMOKE_MANIFEST_NAME = "phase1_smoke_manifest.json"
PHASE1_SMOKE_AUDIT_NAME = "smoke_audit.json"
PHASE1_SMOKE_RESULT_NAME = "smoke_result.json"
CONTROL_DICT_CITATION = "docs/specs/phase1_blackwell_bringup_spec.md#step-11"
FV_SOLUTION_CITATION = "docs/specs/phase1_blackwell_bringup_spec.md#step-10"
PACK_CITATION = "docs/specs/phase1_blackwell_bringup_spec.md#step-9"
TASK_CARD_CITATION = "docs/tasks/03_phase1_blackwell_bringup.md#p1-04"
DISALLOWED_PRECONDITIONERS = frozenset({"DIC", "DILU"})
ALLOWED_GAMG_SMOOTHERS = frozenset({"Richardson", "twoStageGaussSeidel", "diagonal"})
INCLUDE_PATTERN = re.compile(r"^\s*#include(?:Etc|Func|IfPresent)?\b", flags=re.MULTILINE)
LOG_NAN_INF_PATTERN = re.compile(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", flags=re.IGNORECASE)
TOKEN_PATTERN = re.compile(r'"[^"]*"|[{};]|\S+')
CASE_FILE_SUFFIXES = frozenset({"", ".dict"})


@dataclass(frozen=True)
class Phase1SmokeCaseDefinition:
    name: str
    solver: str
    relative_case_dir: pathlib.Path
    required_outputs: tuple[str, ...]
    required_files: tuple[str, ...]

    def case_dir(self, root: pathlib.Path) -> pathlib.Path:
        return root / self.relative_case_dir

    def acceptance_json_path(self, root: pathlib.Path) -> pathlib.Path:
        return self.case_dir(root) / "acceptance.json"


@dataclass(frozen=True)
class Phase1SmokeIssue:
    code: str
    message: str
    citations: tuple[str, ...]
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "citations": list(self.citations),
        }
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class Phase1SmokeAuditReport:
    case_name: str
    solver: str
    case_dir: pathlib.Path
    startup_allowed: bool
    authority_citations: tuple[str, ...]
    issues: tuple[Phase1SmokeIssue, ...]
    support_scan: dict[str, Any]

    @property
    def reject_reasons(self) -> tuple[dict[str, Any], ...]:
        return tuple(issue.as_dict() for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "case_name": self.case_name,
            "solver": self.solver,
            "case_dir": self.case_dir.as_posix(),
            "startup_allowed": self.startup_allowed,
            "authority_citations": list(self.authority_citations),
            "reject_reasons": [issue.as_dict() for issue in self.issues],
            "support_scan": self.support_scan,
        }


@dataclass(frozen=True)
class Phase1SmokeRunResult:
    case_name: str
    scratch_case_dir: pathlib.Path
    audit_report: Phase1SmokeAuditReport
    audit_report_path: pathlib.Path
    result_json_path: pathlib.Path


CASE_DEFINITIONS = (
    Phase1SmokeCaseDefinition(
        name="cubeLinear",
        solver="laplacianFoam",
        relative_case_dir=PHASE1_SMOKE_CASES_DIR / "cubeLinear",
        required_outputs=("0.1/T",),
        required_files=(
            "0/T",
            "constant/transportProperties",
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "acceptance.json",
        ),
    ),
    Phase1SmokeCaseDefinition(
        name="channelSteady",
        solver="simpleFoam",
        relative_case_dir=PHASE1_SMOKE_CASES_DIR / "channelSteady",
        required_outputs=("0.1/U", "0.1/p"),
        required_files=(
            "0/U",
            "0/p",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "acceptance.json",
        ),
    ),
    Phase1SmokeCaseDefinition(
        name="channelTransient",
        solver="pimpleFoam",
        relative_case_dir=PHASE1_SMOKE_CASES_DIR / "channelTransient",
        required_outputs=("0.1/U", "0.1/p"),
        required_files=(
            "0/U",
            "0/p",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "acceptance.json",
        ),
    ),
)


def build_phase1_smoke_manifest(root: pathlib.Path | str | None = None) -> dict[str, Any]:
    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)
    cases = []
    for definition in CASE_DEFINITIONS:
        case_dir = definition.case_dir(resolved_root)
        acceptance_path = definition.acceptance_json_path(resolved_root)
        cases.append(
            {
                "name": definition.name,
                "solver": definition.solver,
                "case_dir": definition.relative_case_dir.as_posix(),
                "acceptance_json": acceptance_path.relative_to(resolved_root).as_posix(),
                "required_outputs": list(definition.required_outputs),
                "self_contained": not _find_external_dependencies(case_dir),
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": PHASE1_SMOKE_MANIFEST_NAME,
        "cases": cases,
    }


def scan_phase1_smoke_case(
    bundle: AuthorityBundle,
    *,
    case_name: str | None = None,
    case_dir: pathlib.Path | str | None = None,
    root: pathlib.Path | str | None = None,
    execution_mode: str = "production",
) -> Phase1SmokeAuditReport:
    resolved_root = repo_root(pathlib.Path(root) if root is not None else None)
    definition, resolved_case_dir = _resolve_case_definition(
        case_name=case_name,
        case_dir=case_dir,
        root=resolved_root,
    )
    issues: list[Phase1SmokeIssue] = []

    for relative_path in definition.required_files:
        target = resolved_case_dir / relative_path
        if not target.exists():
            issues.append(
                Phase1SmokeIssue(
                    code="missing_required_file",
                    message="Phase 1 smoke cases must include the required checked-in case assets.",
                    citations=(PACK_CITATION, TASK_CARD_CITATION),
                    detail={"path": relative_path},
                )
            )

    acceptance_payload: dict[str, Any] = {}
    acceptance_path = resolved_case_dir / "acceptance.json"
    if acceptance_path.exists():
        acceptance_payload = _read_json(acceptance_path, issues=issues, relative_path="acceptance.json")
        if acceptance_payload:
            expected_outputs = acceptance_payload.get("required_outputs")
            if expected_outputs != list(definition.required_outputs):
                issues.append(
                    Phase1SmokeIssue(
                        code="acceptance_required_outputs_mismatch",
                        message="acceptance.json must declare the checked-in smoke-case outputs exactly.",
                        citations=(PACK_CITATION, CONTROL_DICT_CITATION),
                        detail={
                            "expected": list(definition.required_outputs),
                            "observed": expected_outputs,
                        },
                    )
                )
            if acceptance_payload.get("case_name") != definition.name:
                issues.append(
                    Phase1SmokeIssue(
                        code="acceptance_case_name_mismatch",
                        message="acceptance.json must match the registered smoke-case name.",
                        citations=(PACK_CITATION,),
                        detail={
                            "expected": definition.name,
                            "observed": acceptance_payload.get("case_name"),
                        },
                    )
                )
            if acceptance_payload.get("solver") != definition.solver:
                issues.append(
                    Phase1SmokeIssue(
                        code="acceptance_solver_mismatch",
                        message="acceptance.json must match the registered solver.",
                        citations=(PACK_CITATION, CONTROL_DICT_CITATION),
                        detail={
                            "expected": definition.solver,
                            "observed": acceptance_payload.get("solver"),
                        },
                    )
                )

    dependency_hits = _find_external_dependencies(resolved_case_dir)
    if dependency_hits:
        issues.append(
            Phase1SmokeIssue(
                code="external_case_dependency",
                message="Repo-local smoke cases must not depend on external include assets or tutorials.",
                citations=(PACK_CITATION, TASK_CARD_CITATION),
                detail={"hits": dependency_hits},
            )
        )

    control_dict = _read_foam_dictionary(resolved_case_dir / "system" / "controlDict", issues=issues)
    if control_dict:
        observed_solver = _clean_token(control_dict.get("application"))
        if observed_solver != definition.solver:
            issues.append(
                Phase1SmokeIssue(
                    code="control_dict_solver_mismatch",
                    message="controlDict application must match the registered Phase 1 smoke solver.",
                    citations=(CONTROL_DICT_CITATION,),
                    detail={"expected": definition.solver, "observed": observed_solver},
                )
            )

    function_objects = _extract_function_objects(control_dict)
    support_report = scan_support_matrix(
        bundle,
        SupportScanRequest(
            execution_mode=execution_mode,
            fallback_policy="failFast",
            backend="native",
            scheme_audit=False,
            function_objects=function_objects,
        ),
    )

    fv_solution = _read_foam_dictionary(resolved_case_dir / "system" / "fvSolution", issues=issues)
    issues.extend(_audit_fvsolution(fv_solution))
    issues.extend(_wrap_support_scan_issues(support_report))

    all_issues = tuple(sorted(issues, key=lambda issue: issue.code))
    citations = tuple(
        dict.fromkeys(
            [
                PACK_CITATION,
                FV_SOLUTION_CITATION,
                CONTROL_DICT_CITATION,
                *support_report.authority_citations,
            ]
        )
    )
    return Phase1SmokeAuditReport(
        case_name=definition.name,
        solver=definition.solver,
        case_dir=resolved_case_dir,
        startup_allowed=not all_issues and support_report.startup_allowed,
        authority_citations=citations,
        issues=all_issues,
        support_scan=support_report.as_dict(),
    )


def run_phase1_smoke_case(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    scratch_root: pathlib.Path | str,
    case_name: str | None = None,
    case_dir: pathlib.Path | str | None = None,
    root: pathlib.Path | str | None = None,
    execution_mode: str = "production",
    command_runner: Callable[..., int] | None = None,
) -> Phase1SmokeRunResult:
    audit_report = scan_phase1_smoke_case(
        bundle,
        case_name=case_name,
        case_dir=case_dir,
        root=root,
        execution_mode=execution_mode,
    )
    artifact_case_root = pathlib.Path(artifact_root) / audit_report.case_name
    artifact_case_root.mkdir(parents=True, exist_ok=True)
    audit_report_path = artifact_case_root / PHASE1_SMOKE_AUDIT_NAME
    _write_json(audit_report_path, audit_report.as_dict())

    scratch_case_dir = pathlib.Path(scratch_root) / audit_report.case_name
    if scratch_case_dir.exists():
        shutil.rmtree(scratch_case_dir)

    if not audit_report.startup_allowed:
        result_payload = _build_result_payload(
            case_name=audit_report.case_name,
            solver=audit_report.solver,
            scratch_case_dir=scratch_case_dir,
            status="blocked",
            command_results=(),
            required_outputs=(),
            audit_report=audit_report,
            required_outputs_present=False,
            no_nan_inf=True,
        )
        result_json_path = artifact_case_root / PHASE1_SMOKE_RESULT_NAME
        _write_json(result_json_path, result_payload)
        return Phase1SmokeRunResult(
            case_name=audit_report.case_name,
            scratch_case_dir=scratch_case_dir,
            audit_report=audit_report,
            audit_report_path=audit_report_path,
            result_json_path=result_json_path,
        )

    shutil.copytree(audit_report.case_dir, scratch_case_dir)
    runner = command_runner or _default_command_runner
    case_definition = _definition_by_name(audit_report.case_name)
    command_results = []
    for command in (("blockMesh",), (audit_report.solver,)):
        log_path = artifact_case_root / f"{len(command_results) + 1:02d}_{command[0]}.log"
        returncode = runner(command, cwd=scratch_case_dir, log_path=log_path)
        command_results.append(
            {
                "command": list(command),
                "cwd": scratch_case_dir.as_posix(),
                "log_path": log_path.as_posix(),
                "returncode": returncode,
            }
        )
        if returncode != 0:
            result_payload = _build_result_payload(
                case_name=audit_report.case_name,
                solver=audit_report.solver,
                scratch_case_dir=scratch_case_dir,
                status="fail",
                command_results=tuple(command_results),
                required_outputs=case_definition.required_outputs,
                audit_report=audit_report,
                required_outputs_present=False,
                no_nan_inf=_logs_are_clean(command_results),
            )
            result_json_path = artifact_case_root / PHASE1_SMOKE_RESULT_NAME
            _write_json(result_json_path, result_payload)
            return Phase1SmokeRunResult(
                case_name=audit_report.case_name,
                scratch_case_dir=scratch_case_dir,
                audit_report=audit_report,
                audit_report_path=audit_report_path,
                result_json_path=result_json_path,
            )

    required_outputs_present = all(
        (scratch_case_dir / relative_path).exists() for relative_path in case_definition.required_outputs
    )
    no_nan_inf = _logs_are_clean(command_results)
    status = "pass" if required_outputs_present and no_nan_inf else "fail"
    result_payload = _build_result_payload(
        case_name=audit_report.case_name,
        solver=audit_report.solver,
        scratch_case_dir=scratch_case_dir,
        status=status,
        command_results=tuple(command_results),
        required_outputs=case_definition.required_outputs,
        audit_report=audit_report,
        required_outputs_present=required_outputs_present,
        no_nan_inf=no_nan_inf,
    )
    result_json_path = artifact_case_root / PHASE1_SMOKE_RESULT_NAME
    _write_json(result_json_path, result_payload)
    return Phase1SmokeRunResult(
        case_name=audit_report.case_name,
        scratch_case_dir=scratch_case_dir,
        audit_report=audit_report,
        audit_report_path=audit_report_path,
        result_json_path=result_json_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Print the smoke-case manifest JSON.")
    manifest_parser.add_argument("--json-out", type=pathlib.Path, default=None)

    audit_parser = subparsers.add_parser("audit", help="Audit one checked-in Phase 1 smoke case.")
    audit_parser.add_argument("--case", required=True, choices=[definition.name for definition in CASE_DEFINITIONS])

    run_parser = subparsers.add_parser("run", help="Audit and execute one Phase 1 smoke case.")
    run_parser.add_argument("--case", required=True, choices=[definition.name for definition in CASE_DEFINITIONS])
    run_parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    run_parser.add_argument("--scratch-root", type=pathlib.Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    resolved_root = repo_root(args.root)

    if args.command == "manifest":
        payload = build_phase1_smoke_manifest(resolved_root)
        if args.json_out is not None:
            _write_json(args.json_out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        from .bundle import load_authority_bundle  # type: ignore
    except ImportError:  # pragma: no cover - script execution fallback
        from scripts.authority.bundle import load_authority_bundle  # type: ignore

    resolved_bundle = load_authority_bundle(resolved_root)
    if args.command == "audit":
        report = scan_phase1_smoke_case(resolved_bundle, case_name=args.case, root=resolved_root)
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0 if report.startup_allowed else 1

    result = run_phase1_smoke_case(
        resolved_bundle,
        case_name=args.case,
        artifact_root=args.artifact_root,
        scratch_root=args.scratch_root,
        root=resolved_root,
    )
    print(result.result_json_path.as_posix())
    return 0 if result.audit_report.startup_allowed else 1


def _definition_by_name(case_name: str) -> Phase1SmokeCaseDefinition:
    for definition in CASE_DEFINITIONS:
        if definition.name == case_name:
            return definition
    raise ValueError(f"unknown Phase 1 smoke case {case_name!r}")


def _resolve_case_definition(
    *,
    case_name: str | None,
    case_dir: pathlib.Path | str | None,
    root: pathlib.Path,
) -> tuple[Phase1SmokeCaseDefinition, pathlib.Path]:
    if case_dir is not None:
        resolved_case_dir = pathlib.Path(case_dir).resolve()
        inferred_name = case_name or resolved_case_dir.name
        return _definition_by_name(inferred_name), resolved_case_dir
    if case_name is None:
        raise ValueError("case_name or case_dir is required")
    definition = _definition_by_name(case_name)
    return definition, definition.case_dir(root)


def _find_external_dependencies(case_dir: pathlib.Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for path in sorted(case_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if INCLUDE_PATTERN.search(line):
                hits.append(
                    {
                        "path": path.relative_to(case_dir).as_posix(),
                        "line": line_no,
                        "line_text": line.strip(),
                    }
                )
    return hits


def _read_json(
    path: pathlib.Path,
    *,
    issues: list[Phase1SmokeIssue],
    relative_path: str,
) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(
            Phase1SmokeIssue(
                code="invalid_json_payload",
                message="Smoke-case JSON artifacts must be valid JSON.",
                citations=(PACK_CITATION,),
                detail={"path": relative_path, "error": str(exc)},
            )
        )
        return {}


def _read_foam_dictionary(
    path: pathlib.Path,
    *,
    issues: list[Phase1SmokeIssue],
) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(
            Phase1SmokeIssue(
                code="unreadable_case_dictionary",
                message="Smoke-case dictionaries must be readable from the repo checkout.",
                citations=(PACK_CITATION,),
                detail={"path": path.as_posix(), "error": str(exc)},
            )
        )
        return {}
    try:
        return _parse_foam_dictionary(text)
    except ValueError as exc:
        issues.append(
            Phase1SmokeIssue(
                code="invalid_case_dictionary",
                message="Smoke-case dictionaries must parse cleanly for pre-run audit.",
                citations=(PACK_CITATION, FV_SOLUTION_CITATION, CONTROL_DICT_CITATION),
                detail={"path": path.as_posix(), "error": str(exc)},
            )
        )
        return {}


def _parse_foam_dictionary(text: str) -> dict[str, Any]:
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    stripped = re.sub(r"//.*", "", stripped)
    stripped = stripped.replace("{", " { ").replace("}", " } ").replace(";", " ; ")
    tokens = TOKEN_PATTERN.findall(stripped)
    parsed, index = _parse_mapping(tokens, 0)
    if index != len(tokens):
        raise ValueError("unexpected trailing tokens in OpenFOAM dictionary")
    return parsed


def _parse_mapping(tokens: list[str], index: int) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {}
    while index < len(tokens):
        token = tokens[index]
        if token == "}":
            return payload, index + 1
        if token in {"{", ";"}:
            index += 1
            continue
        key = token
        index += 1
        if index >= len(tokens):
            raise ValueError(f"missing value for token {key!r}")
        if tokens[index] == "{":
            nested, index = _parse_mapping(tokens, index + 1)
            payload[_clean_token(key)] = nested
            continue
        values: list[str] = []
        while index < len(tokens) and tokens[index] != ";":
            values.append(tokens[index])
            index += 1
        if index >= len(tokens):
            raise ValueError(f"missing semicolon for token {key!r}")
        index += 1
        payload[_clean_token(key)] = _clean_token(" ".join(values))
    return payload, index


def _clean_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def _extract_function_objects(control_dict: Mapping[str, Any]) -> tuple[SupportFunctionObject, ...]:
    functions = control_dict.get("functions")
    if not isinstance(functions, Mapping):
        return ()
    items = []
    for name, payload in functions.items():
        if not isinstance(payload, Mapping):
            continue
        class_name = _clean_token(payload.get("type") or payload.get("class"))
        execute_control = _clean_token(payload.get("executeControl") or "timeStep")
        if not class_name:
            continue
        items.append(
            SupportFunctionObject(
                name=_clean_token(name),
                class_name=class_name,
                execute_control=execute_control,
            )
        )
    return tuple(items)


def _audit_fvsolution(fv_solution: Mapping[str, Any]) -> tuple[Phase1SmokeIssue, ...]:
    solvers = fv_solution.get("solvers")
    if not isinstance(solvers, Mapping):
        return (
            Phase1SmokeIssue(
                code="missing_solver_block",
                message="fvSolution must define a solvers block for Phase 1 smoke-case audit.",
                citations=(FV_SOLUTION_CITATION,),
            ),
        )
    issues: list[Phase1SmokeIssue] = []
    for field_name, config in solvers.items():
        if not isinstance(config, Mapping):
            continue
        solver = _clean_token(config.get("solver"))
        preconditioner = _clean_token(config.get("preconditioner"))
        smoother = _clean_token(config.get("smoother"))
        if solver == "PBiCGStab":
            issues.append(
                Phase1SmokeIssue(
                    code="unsupported_solver_setting",
                    message="PBiCGStab is not admitted for the audited Phase 1 smoke cases.",
                    citations=(FV_SOLUTION_CITATION,),
                    detail={"field": _clean_token(field_name), "solver": solver},
                )
            )
        if preconditioner in DISALLOWED_PRECONDITIONERS:
            issues.append(
                Phase1SmokeIssue(
                    code="unsupported_preconditioner",
                    message="DIC and DILU preconditioners must be replaced with GPU-safe alternatives.",
                    citations=(FV_SOLUTION_CITATION,),
                    detail={"field": _clean_token(field_name), "preconditioner": preconditioner},
                )
            )
        if solver == "GAMG" and smoother and smoother not in ALLOWED_GAMG_SMOOTHERS:
            issues.append(
                Phase1SmokeIssue(
                    code="unsupported_gamg_smoother",
                    message="GAMG smoother must stay within the audited Phase 1 allow-list.",
                    citations=(FV_SOLUTION_CITATION,),
                    detail={
                        "field": _clean_token(field_name),
                        "smoother": smoother,
                        "allowed": sorted(ALLOWED_GAMG_SMOOTHERS),
                    },
                )
            )
    return tuple(issues)


def _wrap_support_scan_issues(report: Any) -> tuple[Phase1SmokeIssue, ...]:
    wrapped = []
    for issue in report.issues:
        wrapped.append(
            Phase1SmokeIssue(
                code=issue.code,
                message=issue.message,
                citations=issue.citations,
                detail=dict(issue.detail),
            )
        )
    return tuple(wrapped)


def _default_command_runner(
    command: tuple[str, ...], *, cwd: pathlib.Path, log_path: pathlib.Path
) -> int:
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
    required_outputs: tuple[str, ...],
    audit_report: Phase1SmokeAuditReport,
    required_outputs_present: bool,
    no_nan_inf: bool,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": PHASE1_SMOKE_RESULT_NAME,
        "case_name": case_name,
        "solver": solver,
        "status": status,
        "scratch_case_dir": scratch_case_dir.as_posix(),
        "command_results": list(command_results),
        "required_outputs": list(required_outputs),
        "reject_codes": [reason["code"] for reason in audit_report.reject_reasons],
        "success_criteria": {
            "audit_passed": audit_report.startup_allowed,
            "required_outputs_present": required_outputs_present,
            "no_nan_inf": no_nan_inf,
        },
    }


def _write_json(path: pathlib.Path | str, payload: Mapping[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
