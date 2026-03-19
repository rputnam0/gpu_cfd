"""Phase 1 Blackwell build-wrapper helpers built on top of frozen pin manifests."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import shlex
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Any

try:
    from .bundle import AuthorityBundle, load_authority_bundle
    from .pins import emit_environment_manifests, load_pin_details, write_json
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.authority.bundle import AuthorityBundle, load_authority_bundle  # type: ignore
    from scripts.authority.pins import (  # type: ignore
        emit_environment_manifests,
        load_pin_details,
        write_json,
    )


BUILD_METADATA_SCHEMA_VERSION = "1.0.0"
FATBINARY_REPORT_SCHEMA_VERSION = "1.0.0"
SUPPORTED_BUILD_MODES = {"debug", "relwithdebinfo"}
SKIP_AUDIT_DIR_NAMES = {".git", "__pycache__", ".venv", "build", "artifacts"}
FATBINARY_CANDIDATE_DIR_NAMES = {"platforms", "bin", "lib"}
FATBINARY_REPORT_NAME = "fatbinary_report.json"
PTX_DUMP_NAME = "ptx.txt"
SASS_DUMP_NAME = "sass.txt"
WSL_DRIVER_LIB_DIR = "/usr/lib/wsl/lib"
AUDIT_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}
NVTX2_INCLUDE_PATTERN = re.compile(
    r'^\s*#\s*include\s*[<"]\s*nvtoolsext\.h\s*[>"]',
    flags=re.IGNORECASE | re.MULTILINE,
)
SASS_SM_PATTERN = re.compile(r"(?:arch\s*=\s*sm_|code\s+for\s+sm_)(\d+)")
PTX_SM_PATTERN = re.compile(
    r"(?:\.target\s+sm_|arch\s*=\s*sm_|code\s+for\s+sm_)(\d+)"
)


@dataclass(frozen=True)
class Phase1BuildPlan:
    source_root: pathlib.Path
    output_dir: pathlib.Path
    lane: str
    mode: str
    build_command: tuple[str, ...]
    bashrc_path: pathlib.Path | None
    host_env_path: pathlib.Path
    manifest_refs_path: pathlib.Path
    metadata_path: pathlib.Path
    log_path: pathlib.Path
    fatbinary_report_path: pathlib.Path
    ptx_dump_path: pathlib.Path
    sass_dump_path: pathlib.Path
    env_exports: dict[str, str]
    nvtx_audit_hits: tuple[str, ...]
    discovery_provenance: dict[str, str]
    recorded_tool_paths: dict[str, str]
    prebuild_candidate_snapshot: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class Phase1BuildResult:
    succeeded: bool
    returncode: int
    log_path: pathlib.Path
    metadata_path: pathlib.Path


@dataclass(frozen=True)
class Phase1FatbinaryInspectionResult:
    smoke_gate_ready: bool
    report_path: pathlib.Path
    ptx_dump_path: pathlib.Path
    sass_dump_path: pathlib.Path


class Phase1BuildError(ValueError):
    """Raised when Phase 1 build-wrapper inputs or execution are invalid."""


REQUIRED_RECORDED_TOOL_PATH_FIELDS = (
    "nvc_path",
    "nvcc_path",
    "nsys_path",
    "compute_sanitizer_path",
    "nvidia_smi_path",
)


def plan_phase1_build(
    bundle: AuthorityBundle,
    *,
    source_root: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    discovery_host_env_path: pathlib.Path | str,
    discovery_manifest_refs_path: pathlib.Path | str,
    cuda_probe_path: pathlib.Path | str,
    lane: str = "primary",
    mode: str = "relwithdebinfo",
    build_entrypoint: str = "Allwmake",
    cuda_visible_devices: str = "0",
) -> Phase1BuildPlan:
    if lane not in {"primary", "experimental"}:
        raise Phase1BuildError(f"unsupported build lane {lane!r}")
    if mode not in SUPPORTED_BUILD_MODES:
        raise Phase1BuildError(f"unsupported build mode {mode!r}")

    source_root_path = pathlib.Path(source_root).resolve()
    if not source_root_path.exists():
        raise Phase1BuildError(f"source_root does not exist: {source_root_path}")

    output_dir_path = pathlib.Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    discovery_host_env = _read_json_payload(pathlib.Path(discovery_host_env_path))
    discovery_manifest_refs = _read_json_payload(pathlib.Path(discovery_manifest_refs_path))
    cuda_probe = _read_json_payload(pathlib.Path(cuda_probe_path))

    pin_details = load_pin_details(bundle)
    nvarch = _extract_nvarch(pin_details.gpu_target)
    selected_lane_value = (
        pin_details.primary_toolkit_lane
        if lane == "primary"
        else pin_details.experimental_toolkit_lane
    )

    _validate_discovery_host_env(
        discovery_host_env,
        pin_details=pin_details,
    )
    discovery_provenance = _extract_discovery_provenance(
        discovery_host_env,
        discovery_manifest_refs,
        cuda_probe,
    )
    local_mirror_refs = _extract_local_mirror_refs(discovery_manifest_refs)
    recorded_tool_paths = _extract_recorded_tool_paths(
        discovery_host_env,
        discovery_manifest_refs,
    )
    _validate_cuda_probe(
        cuda_probe,
        expected_nvarch=nvarch,
        expected_gpu_target=pin_details.workstation_target,
    )

    emitted = emit_environment_manifests(
        bundle,
        consumer="build",
        output_dir=output_dir_path,
        lane=lane,
        host_observations=dict(discovery_host_env["host_observations"]),
        local_mirror_refs=local_mirror_refs,
        repo_commit=_extract_repo_commit(discovery_host_env, discovery_manifest_refs),
        provenance=discovery_provenance,
    )

    build_command = _resolve_build_command(
        source_root_path,
        build_entrypoint=build_entrypoint,
    )
    bashrc_path = _resolve_repo_bashrc(source_root_path)
    nvtx_audit_hits = audit_nvtx_includes(source_root_path)
    if nvtx_audit_hits:
        formatted_hits = ", ".join(nvtx_audit_hits)
        raise Phase1BuildError(
            "NVTX2-era includes are forbidden; remove direct nvToolsExt usage from: "
            + formatted_hits
        )

    env_exports = _build_env_exports(
        selected_lane_value=selected_lane_value,
        recorded_tool_paths=recorded_tool_paths,
        nvarch=nvarch,
        mode=mode,
        cuda_visible_devices=cuda_visible_devices,
        host_env_path=emitted.host_env_path,
        manifest_refs_path=emitted.manifest_refs_path,
        cuda_probe_path=pathlib.Path(cuda_probe_path).resolve(),
    )

    log_path = output_dir_path / f"build_{lane}_{mode}.log"
    metadata_path = output_dir_path / f"build_metadata_{lane}_{mode}.json"
    fatbinary_report_path = output_dir_path / f"fatbinary_report_{lane}_{mode}.json"
    ptx_dump_path = output_dir_path / f"ptx_{lane}_{mode}.txt"
    sass_dump_path = output_dir_path / f"sass_{lane}_{mode}.txt"
    prebuild_candidate_snapshot = _capture_fatbinary_candidate_snapshot(source_root_path)
    metadata = _build_metadata_payload(
        bundle=bundle,
        pin_details=pin_details,
        lane=lane,
        mode=mode,
        nvarch=nvarch,
        selected_lane_value=selected_lane_value,
        build_command=build_command,
        bashrc_path=bashrc_path,
        source_root=source_root_path,
        host_env_path=emitted.host_env_path,
        manifest_refs_path=emitted.manifest_refs_path,
        cuda_probe_path=pathlib.Path(cuda_probe_path).resolve(),
        log_path=log_path,
        env_exports=env_exports,
        nvtx_audit_hits=nvtx_audit_hits,
        discovery_provenance=discovery_provenance,
        recorded_tool_paths=recorded_tool_paths,
        fatbinary_report_path=fatbinary_report_path,
        ptx_dump_path=ptx_dump_path,
        sass_dump_path=sass_dump_path,
        prebuild_candidate_snapshot=prebuild_candidate_snapshot,
    )
    write_json(metadata_path, metadata)

    return Phase1BuildPlan(
        source_root=source_root_path,
        output_dir=output_dir_path,
        lane=lane,
        mode=mode,
        build_command=build_command,
        bashrc_path=bashrc_path,
        host_env_path=emitted.host_env_path,
        manifest_refs_path=emitted.manifest_refs_path,
        metadata_path=metadata_path,
        log_path=log_path,
        fatbinary_report_path=fatbinary_report_path,
        ptx_dump_path=ptx_dump_path,
        sass_dump_path=sass_dump_path,
        env_exports=env_exports,
        nvtx_audit_hits=nvtx_audit_hits,
        discovery_provenance=discovery_provenance,
        recorded_tool_paths=recorded_tool_paths,
        prebuild_candidate_snapshot=prebuild_candidate_snapshot,
    )


def render_phase1_env_exports(plan: Phase1BuildPlan) -> str:
    lines: list[str] = []
    path_export = _render_dynamic_path_export(plan)
    if path_export:
        lines.append(path_export)
    for key, value in sorted(plan.env_exports.items()):
        if key == "LD_LIBRARY_PATH":
            lines.append(_render_ld_library_path_export())
            continue
        lines.append(f"export {key}={shlex.quote(value)}")
    if "SPUMA_EXTRA_CXX_FLAGS" in plan.env_exports:
        lines.append(
            'export FOAM_EXTRA_CXXFLAGS="${FOAM_EXTRA_CXXFLAGS:+${FOAM_EXTRA_CXXFLAGS} }${SPUMA_EXTRA_CXX_FLAGS}"'
        )
    return "\n".join(lines)


def run_phase1_build(plan: Phase1BuildPlan) -> Phase1BuildResult:
    try:
        _validate_repo_native_toolchain(plan)
    except Phase1BuildError as exc:
        _write_build_log_message(plan.log_path, str(exc))
        _record_build_outcome(
            plan,
            returncode=None,
            succeeded=False,
            failure_stage="preflight",
            failure_reason=str(exc),
        )
        raise

    env = os.environ.copy()
    with _patched_cuda_rules(plan):
        with plan.log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                _build_shell_command(plan),
                cwd=plan.source_root,
                env=env,
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

    result = Phase1BuildResult(
        succeeded=completed.returncode == 0,
        returncode=completed.returncode,
        log_path=plan.log_path,
        metadata_path=plan.metadata_path,
    )
    if completed.returncode != 0:
        failure_reason = (
            f"build command failed with exit code {completed.returncode}; see {plan.log_path}"
        )
        _record_build_outcome(
            plan,
            returncode=completed.returncode,
            succeeded=False,
            failure_stage="build",
            failure_reason=failure_reason,
        )
        raise Phase1BuildError(
            failure_reason
        )
    try:
        inspect_phase1_build_fatbinaries(plan)
    except Phase1BuildError as exc:
        _record_build_outcome(
            plan,
            returncode=completed.returncode,
            succeeded=False,
            failure_stage="inspection",
            failure_reason=str(exc),
        )
        raise
    _record_build_outcome(
        plan,
        returncode=completed.returncode,
        succeeded=True,
        failure_stage=None,
        failure_reason=None,
    )
    return result


def inspect_phase1_build_fatbinaries(
    plan: Phase1BuildPlan,
    *,
    use_prebuild_snapshot: bool = True,
) -> Phase1FatbinaryInspectionResult:
    cuobjdump_path = _resolve_cuobjdump_path(plan)
    targets = _discover_fatbinary_targets(
        plan.source_root,
        previous_snapshot=plan.prebuild_candidate_snapshot if use_prebuild_snapshot else None,
    )
    if not targets:
        report = _write_fatbinary_report(
            plan,
            cuobjdump_path=cuobjdump_path,
            inspected_targets=[],
            native_sm_present=[],
            ptx_targets=[],
            ptx_present=False,
            smoke_gate_ready=False,
            failure_reason="no candidate built binaries were found for fatbinary inspection",
            smoke_gate_targets=[],
        )
        raise Phase1BuildError(str(report["failure_reason"]))

    sass_sections: list[str] = []
    ptx_sections: list[str] = []
    inspected_targets: list[dict[str, Any]] = []
    native_sm_present: set[int] = set()
    ptx_targets: set[int] = set()
    required_native_sm = int(plan.env_exports["NVARCH"])
    smoke_gate_targets: list[str] = []

    for target_path in targets:
        relative_target = target_path.relative_to(plan.source_root).as_posix()
        sass_completed = subprocess.run(
            [cuobjdump_path, "-sass", target_path.as_posix()],
            text=True,
            capture_output=True,
            check=False,
        )
        ptx_completed = subprocess.run(
            [cuobjdump_path, "-ptx", target_path.as_posix()],
            text=True,
            capture_output=True,
            check=False,
        )
        sass_output = sass_completed.stdout or ""
        ptx_output = ptx_completed.stdout or ""
        target_native_sms = sorted(set(_extract_sm_targets(sass_output, pattern=SASS_SM_PATTERN)))
        target_ptx_targets = sorted(set(_extract_sm_targets(ptx_output, pattern=PTX_SM_PATTERN)))
        target_ptx_present = bool(ptx_output.strip())
        target_smoke_gate_ready = (
            required_native_sm in target_native_sms and target_ptx_present
        )
        native_sm_present.update(target_native_sms)
        ptx_targets.update(target_ptx_targets)
        sass_sections.append(_render_dump_section(relative_target, sass_output))
        ptx_sections.append(_render_dump_section(relative_target, ptx_output))
        if target_smoke_gate_ready:
            smoke_gate_targets.append(relative_target)
        inspected_targets.append(
            {
                "path": relative_target,
                "sass_returncode": sass_completed.returncode,
                "ptx_returncode": ptx_completed.returncode,
                "native_sm_present": target_native_sms,
                "ptx_targets": target_ptx_targets,
                "ptx_present": target_ptx_present,
                "smoke_gate_ready": target_smoke_gate_ready,
            }
        )

    plan.sass_dump_path.write_text("".join(sass_sections), encoding="utf-8")
    plan.ptx_dump_path.write_text("".join(ptx_sections), encoding="utf-8")

    any_ptx = any(target["ptx_present"] for target in inspected_targets)
    smoke_gate_ready = bool(smoke_gate_targets)
    failure_reason = None
    if not smoke_gate_ready and required_native_sm not in native_sm_present:
        failure_reason = (
            f"fatbinary inspection requires native sm_{required_native_sm} coverage"
        )
    elif not smoke_gate_ready and not any_ptx:
        failure_reason = "fatbinary inspection requires retained PTX before smoke execution"
    elif not smoke_gate_ready:
        failure_reason = (
            "fatbinary inspection requires at least one current build artifact to retain "
            f"both native sm_{required_native_sm} cubin and PTX"
        )

    report = _write_fatbinary_report(
        plan,
        cuobjdump_path=cuobjdump_path,
        inspected_targets=inspected_targets,
        native_sm_present=sorted(native_sm_present),
        ptx_targets=sorted(ptx_targets),
        ptx_present=any_ptx,
        smoke_gate_ready=smoke_gate_ready,
        failure_reason=failure_reason,
        smoke_gate_targets=smoke_gate_targets,
    )
    if failure_reason:
        raise Phase1BuildError(failure_reason)
    return Phase1FatbinaryInspectionResult(
        smoke_gate_ready=bool(report["smoke_gate_ready"]),
        report_path=plan.fatbinary_report_path,
        ptx_dump_path=plan.ptx_dump_path,
        sass_dump_path=plan.sass_dump_path,
    )


def audit_nvtx_includes(source_root: pathlib.Path | str) -> tuple[str, ...]:
    root = pathlib.Path(source_root).resolve()
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_AUDIT_DIR_NAMES for part in relative_parts):
            continue
        if path.suffix.lower() not in AUDIT_SOURCE_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise Phase1BuildError(f"unable to audit {path}: {exc}") from exc
        if NVTX2_INCLUDE_PATTERN.search(content):
            hits.append(path.relative_to(root).as_posix())
    return tuple(sorted(hits))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("env", help="Print sourceable Phase 1 env exports.")
    build_parser = subparsers.add_parser("build", help="Run the Phase 1 build wrapper.")
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect built binaries for native cubin/PTX coverage.",
    )
    for subparser in (env_parser, build_parser, inspect_parser):
        subparser.add_argument(
            "--root",
            type=pathlib.Path,
            default=None,
            help="Repository root containing docs/authority.",
        )
        subparser.add_argument(
            "--source-root",
            type=pathlib.Path,
            required=True,
            help="SPUMA source root containing the repo-native build entrypoint.",
        )
        subparser.add_argument(
            "--output-dir",
            type=pathlib.Path,
            required=True,
            help="Directory for build manifests, logs, and metadata.",
        )
        subparser.add_argument(
            "--host-env-json",
            type=pathlib.Path,
            required=True,
            help="Path to the canonical P1-01 host_env.json artifact.",
        )
        subparser.add_argument(
            "--manifest-refs-json",
            type=pathlib.Path,
            required=True,
            help="Path to the canonical P1-01 manifest_refs.json artifact.",
        )
        subparser.add_argument(
            "--cuda-probe-json",
            type=pathlib.Path,
            required=True,
            help="Path to the canonical P1-01 cuda_probe.json artifact.",
        )
        subparser.add_argument(
            "--lane",
            choices=("primary", "experimental"),
            default="primary",
            help="Frozen toolkit lane to build. Defaults to primary.",
        )
        subparser.add_argument(
            "--mode",
            choices=tuple(sorted(SUPPORTED_BUILD_MODES)),
            default="relwithdebinfo",
            help="Build mode to apply. Defaults to relwithdebinfo.",
        )
        subparser.add_argument(
            "--build-entrypoint",
            default="Allwmake",
            help="Repo-native build entrypoint relative to source-root. Defaults to Allwmake.",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    plan = plan_phase1_build(
        bundle,
        source_root=args.source_root,
        output_dir=args.output_dir,
        discovery_host_env_path=args.host_env_json,
        discovery_manifest_refs_path=args.manifest_refs_json,
        cuda_probe_path=args.cuda_probe_json,
        lane=args.lane,
        mode=args.mode,
        build_entrypoint=args.build_entrypoint,
    )
    if args.command == "env":
        print(render_phase1_env_exports(plan))
        return 0
    if args.command == "inspect":
        result = inspect_phase1_build_fatbinaries(plan, use_prebuild_snapshot=False)
        print(
            json.dumps(
                {
                    "smoke_gate_ready": result.smoke_gate_ready,
                    "fatbinary_report": result.report_path.as_posix(),
                    "ptx_dump": result.ptx_dump_path.as_posix(),
                    "sass_dump": result.sass_dump_path.as_posix(),
                    "metadata_path": plan.metadata_path.as_posix(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = run_phase1_build(plan)
    print(
        json.dumps(
            {
                "succeeded": result.succeeded,
                "returncode": result.returncode,
                "log_path": result.log_path.as_posix(),
                "metadata_path": result.metadata_path.as_posix(),
                "host_env": plan.host_env_path.as_posix(),
                "manifest_refs": plan.manifest_refs_path.as_posix(),
                "fatbinary_report": plan.fatbinary_report_path.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_env_exports(
    *,
    selected_lane_value: str,
    recorded_tool_paths: dict[str, str],
    nvarch: int,
    mode: str,
    cuda_visible_devices: str,
    host_env_path: pathlib.Path,
    manifest_refs_path: pathlib.Path,
    cuda_probe_path: pathlib.Path,
) -> dict[str, str]:
    cuda_home = _infer_cuda_home(
        selected_lane_value,
        nvcc_path=recorded_tool_paths.get("nvcc_path"),
    )
    nvcc_flags, cxx_flags = _build_mode_flags(mode, nvarch=nvarch)
    env = {
        "have_cuda": "true",
        "NVARCH": str(nvarch),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CUDA_HOME": cuda_home,
        "LD_LIBRARY_PATH": _prepend_env_path(
            "LD_LIBRARY_PATH",
            f"{WSL_DRIVER_LIB_DIR}:{cuda_home}/lib64",
        ),
        "SPUMA_ENABLE_NVTX": "1",
        "SPUMA_EXTRA_NVCC_FLAGS": nvcc_flags,
        "SPUMA_EXTRA_CXX_FLAGS": cxx_flags,
        "GPU_CFD_HOST_ENV": host_env_path.as_posix(),
        "GPU_CFD_MANIFEST_REFS": manifest_refs_path.as_posix(),
        "GPU_CFD_CUDA_PROBE": cuda_probe_path.as_posix(),
    }
    return env


def _build_metadata_payload(
    *,
    bundle: AuthorityBundle,
    pin_details: Any,
    lane: str,
    mode: str,
    nvarch: int,
    selected_lane_value: str,
    build_command: tuple[str, ...],
    bashrc_path: pathlib.Path | None,
    source_root: pathlib.Path,
    host_env_path: pathlib.Path,
    manifest_refs_path: pathlib.Path,
    cuda_probe_path: pathlib.Path,
    log_path: pathlib.Path,
    env_exports: dict[str, str],
    nvtx_audit_hits: tuple[str, ...],
    discovery_provenance: dict[str, str],
    recorded_tool_paths: dict[str, str],
    fatbinary_report_path: pathlib.Path,
    ptx_dump_path: pathlib.Path,
    sass_dump_path: pathlib.Path,
    prebuild_candidate_snapshot: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_METADATA_SCHEMA_VERSION,
        "runtime_base": pin_details.runtime_base,
        "lane": lane,
        "mode": mode,
        "have_cuda": True,
        "nvarch": nvarch,
        "ptx_retention_required": True,
        "instrumentation": pin_details.instrumentation,
        "gpu_target": pin_details.gpu_target,
        "selected_lane_value": selected_lane_value,
        "toolkit": {
            "selected_lane": lane,
            "selected_lane_value": selected_lane_value,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "build_entrypoint": build_command[0],
        "build_command": list(build_command),
        "bashrc_path": bashrc_path.as_posix() if bashrc_path else None,
        "source_root": source_root.as_posix(),
        "host_env": host_env_path.as_posix(),
        "manifest_refs": manifest_refs_path.as_posix(),
        "cuda_probe": cuda_probe_path.as_posix(),
        "build_log": log_path.as_posix(),
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "required_revalidation": list(pin_details.required_revalidation),
        "required_cuda_flags": env_exports["SPUMA_EXTRA_NVCC_FLAGS"].split(),
        "required_host_flags": env_exports["SPUMA_EXTRA_CXX_FLAGS"].split(),
        "discovery_provenance": dict(discovery_provenance),
        "recorded_tool_paths": dict(recorded_tool_paths),
        "nvtx_audit": {
            "passed": not nvtx_audit_hits,
            "banned_include_hits": list(nvtx_audit_hits),
        },
        "fatbinary_artifacts": {
            "report": fatbinary_report_path.as_posix(),
            "ptx_dump": ptx_dump_path.as_posix(),
            "sass_dump": sass_dump_path.as_posix(),
        },
        "fatbinary_inspection": {
            "required_native_sm": nvarch,
            "ptx_required": True,
            "smoke_gate_ready": None,
            "report_path": fatbinary_report_path.as_posix(),
        },
        "fatbinary_candidate_snapshot_before_build": {
            path: {"mtime_ns": values[0], "size": values[1]}
            for path, values in sorted(prebuild_candidate_snapshot.items())
        },
        "returncode": None,
        "succeeded": None,
        "failure_stage": None,
        "failure_reason": None,
        "authority_revisions": bundle.authority_revisions,
    }


def _record_build_outcome(
    plan: Phase1BuildPlan,
    *,
    returncode: int | None,
    succeeded: bool,
    failure_stage: str | None,
    failure_reason: str | None,
) -> None:
    metadata = _read_json_payload(plan.metadata_path)
    metadata["returncode"] = returncode
    metadata["succeeded"] = succeeded
    metadata["failure_stage"] = failure_stage
    metadata["failure_reason"] = failure_reason
    write_json(plan.metadata_path, metadata)


def _write_fatbinary_report(
    plan: Phase1BuildPlan,
    *,
    cuobjdump_path: str,
    inspected_targets: list[dict[str, Any]],
    native_sm_present: list[int],
    ptx_targets: list[int],
    ptx_present: bool,
    smoke_gate_ready: bool,
    failure_reason: str | None,
    smoke_gate_targets: list[str],
) -> dict[str, Any]:
    host_env = _read_json_payload(plan.host_env_path)
    report = {
        "schema_version": FATBINARY_REPORT_SCHEMA_VERSION,
        "reviewed_source_tuple_id": host_env.get("reviewed_source_tuple_id"),
        "runtime_base": host_env.get("runtime_base"),
        "toolkit": dict(host_env.get("toolkit") or {}),
        "source_root": plan.source_root.as_posix(),
        "cuobjdump_path": cuobjdump_path,
        "required_native_sm": int(plan.env_exports["NVARCH"]),
        "required_native_sm_found": int(plan.env_exports["NVARCH"]) in native_sm_present,
        "ptx_required": True,
        "ptx_present": ptx_present,
        "ptx_targets": ptx_targets,
        "native_sm_present": native_sm_present,
        "inspected_binary_count": len(inspected_targets),
        "inspection_targets": inspected_targets,
        "smoke_gate_targets": smoke_gate_targets,
        "artifacts": {
            "report": plan.fatbinary_report_path.as_posix(),
            "ptx_dump": plan.ptx_dump_path.as_posix(),
            "sass_dump": plan.sass_dump_path.as_posix(),
        },
        "smoke_gate_ready": smoke_gate_ready,
        "failure_reason": failure_reason,
        "discovery_provenance": dict(plan.discovery_provenance),
    }
    write_json(plan.fatbinary_report_path, report)
    metadata = _read_json_payload(plan.metadata_path)
    metadata["fatbinary_inspection"] = {
        "required_native_sm": int(plan.env_exports["NVARCH"]),
        "ptx_required": True,
        "smoke_gate_ready": smoke_gate_ready,
        "report_path": plan.fatbinary_report_path.as_posix(),
        "failure_reason": failure_reason,
    }
    write_json(plan.metadata_path, metadata)
    return report


def _write_build_log_message(log_path: pathlib.Path, message: str) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def _resolve_cuobjdump_path(plan: Phase1BuildPlan) -> str:
    nvcc_path = pathlib.Path(plan.recorded_tool_paths["nvcc_path"])
    sibling = nvcc_path.with_name("cuobjdump")
    if sibling.is_file():
        return sibling.as_posix()
    discovered = shutil.which("cuobjdump")
    if discovered:
        return discovered
    raise Phase1BuildError(
        "cuobjdump is unavailable; fatbinary inspection requires the CUDA binary utilities"
    )


def _capture_fatbinary_candidate_snapshot(
    source_root: pathlib.Path,
) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in _candidate_fatbinary_paths(source_root):
        stat_result = path.stat()
        snapshot[path.relative_to(source_root).as_posix()] = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
        )
    return snapshot


def _discover_fatbinary_targets(
    source_root: pathlib.Path,
    *,
    previous_snapshot: dict[str, tuple[int, int]] | None,
) -> tuple[pathlib.Path, ...]:
    current_snapshot = _capture_fatbinary_candidate_snapshot(source_root)
    current_targets = tuple(sorted(source_root / path for path in current_snapshot))
    if previous_snapshot is None:
        return current_targets
    targets: list[pathlib.Path] = []
    for relative_path, fingerprint in current_snapshot.items():
        if previous_snapshot.get(relative_path) == fingerprint:
            continue
        targets.append(source_root / relative_path)
    if targets:
        return tuple(sorted(targets))
    return current_targets


def _candidate_fatbinary_paths(source_root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    targets: list[pathlib.Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(source_root).parts
        if any(part in SKIP_AUDIT_DIR_NAMES for part in relative_parts):
            continue
        if not any(part in FATBINARY_CANDIDATE_DIR_NAMES for part in relative_parts):
            continue
        if not os.access(path, os.X_OK):
            continue
        targets.append(path)
    return tuple(sorted(targets))


def _render_dump_section(relative_target: str, content: str) -> str:
    header = f"## {relative_target}\n"
    body = content if content.endswith("\n") or not content else content + "\n"
    return header + body


def _extract_sm_targets(content: str, *, pattern: re.Pattern[str]) -> list[int]:
    return [int(match) for match in pattern.findall(content)]


def _build_mode_flags(mode: str, *, nvarch: int) -> tuple[str, str]:
    arch_flags = (
        f"-gencode=arch=compute_{nvarch},code=sm_{nvarch} "
        f"-gencode=arch=compute_{nvarch},code=compute_{nvarch}"
    )
    if mode == "debug":
        return (
            f"-O0 -g -lineinfo {arch_flags}",
            "-O0 -g -fno-omit-frame-pointer",
        )
    if mode == "relwithdebinfo":
        return (
            f"-O3 -g -lineinfo {arch_flags}",
            "-O3 -g -fno-omit-frame-pointer",
        )
    raise Phase1BuildError(f"unsupported build mode {mode!r}")


def _validate_discovery_host_env(discovery_host_env: dict[str, Any], *, pin_details: Any) -> None:
    host_observations = discovery_host_env.get("host_observations")
    if not isinstance(host_observations, dict):
        raise Phase1BuildError("discovery host_env.json is missing host_observations")
    reviewed_source_tuple_id = str(discovery_host_env.get("reviewed_source_tuple_id", "")).strip()
    if reviewed_source_tuple_id != pin_details.reviewed_source_tuple_id:
        raise Phase1BuildError(
            "discovery host_env.json reviewed_source_tuple_id does not match frozen authority"
        )


def _validate_cuda_probe(
    cuda_probe: dict[str, Any],
    *,
    expected_nvarch: int,
    expected_gpu_target: str,
) -> None:
    cc_major = _coerce_int(cuda_probe.get("cc_major"), field_name="cc_major")
    cc_minor = _coerce_int(cuda_probe.get("cc_minor"), field_name="cc_minor")
    expected_cc_major = expected_nvarch // 10
    expected_cc_minor = expected_nvarch % 10
    if (cc_major, cc_minor) != (expected_cc_major, expected_cc_minor):
        raise Phase1BuildError(
            "cuda_probe.json compute capability does not match the frozen GPU target"
        )
    if not _coerce_bool(cuda_probe.get("native_kernel_ok"), field_name="native_kernel_ok"):
        raise Phase1BuildError("cuda_probe.json requires native_kernel_ok=true")
    if not _coerce_bool(
        cuda_probe.get("managed_memory_probe_ok"),
        field_name="managed_memory_probe_ok",
    ):
        raise Phase1BuildError("cuda_probe.json requires managed_memory_probe_ok=true")
    device_name = str(cuda_probe.get("device_name", "")).strip()
    if "RTX 5080" not in device_name:
        raise Phase1BuildError(
            f"cuda_probe.json must realize the frozen workstation GPU; found {device_name!r}"
        )
    if "RTX 5080" not in expected_gpu_target:
        raise Phase1BuildError("frozen workstation target no longer matches the Phase 1 expectation")


def _extract_local_mirror_refs(discovery_manifest_refs: dict[str, Any]) -> dict[str, str]:
    raw_components = discovery_manifest_refs.get("source_components")
    if not isinstance(raw_components, dict):
        raise Phase1BuildError("manifest_refs.json is missing source_components")

    local_mirror_refs: dict[str, str] = {}
    for component, payload in raw_components.items():
        if not isinstance(payload, dict):
            raise Phase1BuildError(
                f"manifest_refs.json has invalid source component payload for {component!r}"
            )
        local_commit = str(payload.get("local_mirror_commit", "")).strip()
        if not local_commit:
            raise Phase1BuildError(
                f"manifest_refs.json is missing local_mirror_commit for {component!r}"
            )
        local_mirror_refs[str(component)] = local_commit
    return local_mirror_refs


def _extract_repo_commit(
    discovery_host_env: dict[str, Any],
    discovery_manifest_refs: dict[str, Any],
) -> str | None:
    for payload in (discovery_host_env, discovery_manifest_refs):
        repo = payload.get("repo")
        if isinstance(repo, dict):
            commit = str(repo.get("git_commit", "")).strip()
            if commit:
                return commit
    return None


def _resolve_build_command(
    source_root: pathlib.Path,
    *,
    build_entrypoint: str,
) -> tuple[str, ...]:
    entrypoint_path = source_root / build_entrypoint
    if not entrypoint_path.exists():
        raise Phase1BuildError(
            f"missing repo-native build entrypoint at {entrypoint_path}"
        )
    if not os.access(entrypoint_path, os.X_OK):
        raise Phase1BuildError(
            f"repo-native build entrypoint is not executable: {entrypoint_path}"
        )
    return (f"./{build_entrypoint}",)


def _resolve_repo_bashrc(source_root: pathlib.Path) -> pathlib.Path | None:
    bashrc_path = source_root / "etc" / "bashrc"
    if bashrc_path.is_file():
        return bashrc_path
    return None


def _build_shell_command(plan: Phase1BuildPlan) -> list[str]:
    segments = _build_shell_preamble(plan)
    command_text = " ".join(shlex.quote(part) for part in plan.build_command)
    segments.append(f"exec {command_text}")
    return ["bash", "--noprofile", "--norc", "-c", "\n".join(segments)]


def _build_shell_preamble(plan: Phase1BuildPlan) -> list[str]:
    segments: list[str] = []
    if plan.bashrc_path is not None:
        segments.append(f". {shlex.quote(plan.bashrc_path.as_posix())}")
    path_export = _render_dynamic_path_export(plan)
    if path_export:
        segments.append(path_export)
    segments.extend(
        (
            'if command -v gcc-12 >/dev/null 2>&1; then export CC="$(command -v gcc-12)"; fi',
            'if command -v g++-12 >/dev/null 2>&1; then export CXX="$(command -v g++-12)"; export CUDAHOSTCXX="$(command -v g++-12)"; fi',
        )
    )
    for key, value in sorted(plan.env_exports.items()):
        if key == "LD_LIBRARY_PATH":
            segments.append(_render_ld_library_path_export())
            continue
        segments.append(f"export {key}={shlex.quote(value)}")
    if "SPUMA_EXTRA_CXX_FLAGS" in plan.env_exports:
        segments.append(
            'export FOAM_EXTRA_CXXFLAGS="${FOAM_EXTRA_CXXFLAGS:+${FOAM_EXTRA_CXXFLAGS} }${SPUMA_EXTRA_CXX_FLAGS}"'
        )
    return segments


def _render_ld_library_path_export() -> str:
    return (
        f'if [ -d "{WSL_DRIVER_LIB_DIR}" ]; then '
        f'export LD_LIBRARY_PATH="{WSL_DRIVER_LIB_DIR}:$CUDA_HOME/lib64${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        'else '
        'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
        "fi"
    )


@contextlib.contextmanager
def _patched_cuda_rules(plan: Phase1BuildPlan):
    cuda_rules_path = plan.source_root / "wmake" / "rules" / "General" / "cuda"
    if not cuda_rules_path.is_file():
        yield
        return

    original_text = cuda_rules_path.read_text(encoding="utf-8")
    patched_text = _inject_spuma_extra_nvcc_flags(original_text)
    if patched_text == original_text:
        yield
        return

    cuda_rules_path.write_text(patched_text, encoding="utf-8")
    try:
        yield
    finally:
        cuda_rules_path.write_text(original_text, encoding="utf-8")


def _inject_spuma_extra_nvcc_flags(cuda_rules_text: str) -> str:
    if "$(SPUMA_EXTRA_NVCC_FLAGS)" in cuda_rules_text:
        return cuda_rules_text
    marker = "$(FOAM_EXTRA_CXXFLAGS) $(CU_LIB_HEADER_DIRS)"
    # Keep host-only flags on the normal C++ path while ensuring nvcc sees only
    # the explicit CUDA/Blackwell flag set.
    replacement = "$(SPUMA_EXTRA_NVCC_FLAGS) $(CU_LIB_HEADER_DIRS)"
    if marker not in cuda_rules_text:
        raise Phase1BuildError(
            "unable to patch SPUMA CUDA rules for deterministic extra nvcc flags"
        )
    return cuda_rules_text.replace(marker, replacement, 1)


def _validate_repo_native_toolchain(plan: Phase1BuildPlan) -> None:
    host_env = _read_json_payload(plan.host_env_path)
    host_observations = host_env.get("host_observations")
    toolkit = host_env.get("toolkit")
    if not isinstance(host_observations, dict) or not isinstance(toolkit, dict):
        raise Phase1BuildError(
            "build host_env.json is missing host_observations or toolkit metadata"
        )
    _validate_live_host_discovery_provenance(plan.discovery_provenance)
    missing_or_inaccessible = _recorded_tool_path_issues(plan.recorded_tool_paths)
    if missing_or_inaccessible:
        raise Phase1BuildError("; ".join(missing_or_inaccessible))
    probe_segments = _build_shell_preamble(plan)
    probe_segments.append(
        'printf "WM_COMPILER=%s\\nNVC=%s\\nNVCC=%s\\nNVCC_VERSION=%s\\nNSYS=%s\\nNSYS_VERSION=%s\\nCOMPUTE_SANITIZER=%s\\nCOMPUTE_SANITIZER_VERSION=%s\\nNVIDIA_SMI=%s\\nNVIDIA_SMI_GPU=%s\\n" '
        '"${WM_COMPILER:-}" '
        '"$(command -v nvc || true)" '
        '"$(command -v nvcc || true)" '
        '"$(nvcc --version 2>/dev/null | tr \'\\n\' \' \' || true)" '
        '"$(command -v nsys || true)" '
        '"$(nsys --version 2>/dev/null | tr \'\\n\' \' \' || true)" '
        '"$(command -v compute-sanitizer || true)" '
        '"$(compute-sanitizer --version 2>/dev/null | tr \'\\n\' \' \' || true)" '
        '"$(command -v nvidia-smi || true)" '
        '"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n 1 || true)"'
    )
    completed = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", "\n".join(probe_segments)],
        cwd=plan.source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Phase1BuildError(
            "unable to validate the repo-native build environment before running Allwmake"
        )
    compiler = ""
    nvc_path = ""
    nvcc_path = ""
    nvcc_version = ""
    nsys_path = ""
    nsys_version = ""
    compute_sanitizer_path = ""
    compute_sanitizer_version = ""
    nvidia_smi_path = ""
    nvidia_smi_gpu = ""
    for line in completed.stdout.splitlines():
        if line.startswith("WM_COMPILER="):
            compiler = line.partition("=")[2].strip()
        if line.startswith("NVC="):
            nvc_path = line.partition("=")[2].strip()
        if line.startswith("NVCC="):
            nvcc_path = line.partition("=")[2].strip()
        if line.startswith("NVCC_VERSION="):
            nvcc_version = line.partition("=")[2].strip()
        if line.startswith("NSYS="):
            nsys_path = line.partition("=")[2].strip()
        if line.startswith("NSYS_VERSION="):
            nsys_version = line.partition("=")[2].strip()
        if line.startswith("COMPUTE_SANITIZER="):
            compute_sanitizer_path = line.partition("=")[2].strip()
        if line.startswith("COMPUTE_SANITIZER_VERSION="):
            compute_sanitizer_version = line.partition("=")[2].strip()
        if line.startswith("NVIDIA_SMI="):
            nvidia_smi_path = line.partition("=")[2].strip()
        if line.startswith("NVIDIA_SMI_GPU="):
            nvidia_smi_gpu = line.partition("=")[2].strip()

    issues: list[str] = []
    if compiler == "Nvidia" and not nvc_path:
        issues.append(
            "repo-native environment selects WM_COMPILER=Nvidia but nvc is unavailable; "
            "the SPUMA GPU build requires Nvidia HPC plus CUDA"
        )

    expected_toolkit = str(toolkit.get("selected_lane_value", "")).strip().removeprefix("CUDA ").strip()
    expected_nsys = str(host_observations.get("nsys_version", "")).strip()
    expected_compute_sanitizer = str(
        host_observations.get("compute_sanitizer_version", "")
    ).strip()
    expected_gpu_csv = str(host_observations.get("gpu_csv", "")).strip()
    expected_paths = plan.recorded_tool_paths

    if not nvcc_path:
        issues.append("nvcc is unavailable in the active build environment")
    elif not _paths_match(nvcc_path, expected_paths["nvcc_path"]):
        issues.append(
            "repo-native environment resolved nvcc from "
            f"{nvcc_path!r} instead of recorded nvcc_path {expected_paths['nvcc_path']!r}"
        )
    elif expected_toolkit and not _matches_expected_version(
        nvcc_version,
        expected_toolkit,
        allow_release_line_match=True,
    ):
        issues.append(
            f"nvcc version must realize frozen toolkit lane {expected_toolkit!r}; found {nvcc_version!r}"
        )
    if not nsys_path:
        issues.append("nsys is unavailable in the active build environment")
    elif not _paths_match(nsys_path, expected_paths["nsys_path"]):
        issues.append(
            "repo-native environment resolved nsys from "
            f"{nsys_path!r} instead of recorded nsys_path {expected_paths['nsys_path']!r}"
        )
    elif expected_nsys and not _matches_expected_version(nsys_version, expected_nsys):
        issues.append(
            f"nsys version must realize frozen value {expected_nsys!r}; found {nsys_version!r}"
        )
    if not compute_sanitizer_path:
        issues.append("compute-sanitizer is unavailable in the active build environment")
    elif not _paths_match(
        compute_sanitizer_path,
        expected_paths["compute_sanitizer_path"],
    ):
        issues.append(
            "repo-native environment resolved compute-sanitizer from "
            f"{compute_sanitizer_path!r} instead of recorded compute_sanitizer_path "
            f"{expected_paths['compute_sanitizer_path']!r}"
        )
    elif expected_compute_sanitizer and not _matches_expected_version(
        compute_sanitizer_version,
        expected_compute_sanitizer,
    ):
        issues.append(
            "compute-sanitizer version must realize frozen value "
            f"{expected_compute_sanitizer!r}; found {compute_sanitizer_version!r}"
        )
    if not nvidia_smi_path:
        issues.append("nvidia-smi is unavailable; gpu_csv cannot be verified against the frozen workstation target")
    elif not _paths_match(nvidia_smi_path, expected_paths["nvidia_smi_path"]):
        issues.append(
            "repo-native environment resolved nvidia-smi from "
            f"{nvidia_smi_path!r} instead of recorded nvidia_smi_path {expected_paths['nvidia_smi_path']!r}"
        )
    elif expected_gpu_csv and nvidia_smi_gpu != expected_gpu_csv:
        issues.append(
            f"gpu_csv must realize frozen workstation observations {expected_gpu_csv!r}; found {nvidia_smi_gpu!r}"
        )

    if issues:
        raise Phase1BuildError("; ".join(issues))


def _matches_expected_version(
    observed_value: str,
    expected_value: str,
    *,
    allow_release_line_match: bool = False,
) -> bool:
    observed = observed_value.strip()
    expected = expected_value.strip()
    observed_versions = _extract_version_tuples(observed)
    expected_versions = _extract_version_tuples(expected)
    if observed_versions and expected_versions:
        return any(
            _version_tuple_matches(
                observed_tuple,
                expected_tuple,
                allow_release_line_match=allow_release_line_match,
            )
            for observed_tuple in observed_versions
            for expected_tuple in expected_versions
        )
    if observed_versions:
        return expected in {
            ".".join(str(component) for component in version_tuple)
            for version_tuple in observed_versions
        }
    return observed == expected


def _extract_version_tuples(value: str) -> list[tuple[int, ...]]:
    return [
        tuple(int(component) for component in token.split("."))
        for token in re.findall(r"\d+(?:\.\d+)+", value)
    ]


def _version_tuple_matches(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
    *,
    allow_release_line_match: bool = False,
) -> bool:
    if len(observed) >= len(expected) and observed[: len(expected)] == expected:
        return True
    return (
        allow_release_line_match
        and len(expected) >= 3
        and len(observed) == 2
        and observed == expected[:2]
    )


def _extract_discovery_provenance(
    discovery_host_env: dict[str, Any],
    discovery_manifest_refs: dict[str, Any],
    cuda_probe: dict[str, Any],
) -> dict[str, str]:
    resolved: dict[str, str] | None = None
    for payload_name, payload in (
        ("host_env.json", discovery_host_env),
        ("manifest_refs.json", discovery_manifest_refs),
        ("cuda_probe.json", cuda_probe),
    ):
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise Phase1BuildError(f"{payload_name} is missing provenance")
        normalized = {
            key: str(provenance.get(key, "")).strip()
            for key in (
                "collection_mode",
                "emitter_hostname",
                "emitter_time",
                "repo_commit",
            )
        }
        missing = [key for key, value in normalized.items() if not value]
        if missing:
            raise Phase1BuildError(
                f"{payload_name} provenance is missing required fields: {', '.join(missing)}"
            )
        if resolved is None:
            resolved = normalized
            continue
        if normalized != resolved:
            raise Phase1BuildError(
                f"{payload_name} provenance does not match the paired discovery artifacts"
            )
    assert resolved is not None
    return resolved


def _extract_recorded_tool_paths(
    discovery_host_env: dict[str, Any],
    discovery_manifest_refs: dict[str, Any],
) -> dict[str, str]:
    host_observations = discovery_host_env.get("host_observations")
    if not isinstance(host_observations, dict):
        raise Phase1BuildError("discovery host_env.json is missing host_observations")
    invoked_tool_paths = discovery_manifest_refs.get("invoked_tool_paths")
    if not isinstance(invoked_tool_paths, dict):
        raise Phase1BuildError("manifest_refs.json is missing invoked_tool_paths")
    recorded_tool_paths: dict[str, str] = {}
    for field_name in REQUIRED_RECORDED_TOOL_PATH_FIELDS:
        host_value = str(host_observations.get(field_name, "")).strip()
        manifest_value = str(invoked_tool_paths.get(field_name, "")).strip()
        if host_value and manifest_value and host_value != manifest_value:
            raise Phase1BuildError(
                f"discovery artifact mismatch for recorded tool path {field_name!r}: "
                f"host_env.json has {host_value!r} while manifest_refs.json has {manifest_value!r}"
            )
        recorded_tool_paths[field_name] = host_value or manifest_value
    return recorded_tool_paths


def _validate_live_host_discovery_provenance(provenance: dict[str, str]) -> None:
    collection_mode = provenance["collection_mode"]
    if collection_mode != "live_host":
        raise Phase1BuildError(
            "Phase 1 build execution requires live-collected discovery artifacts; "
            f"found provenance.collection_mode={collection_mode!r}"
        )
    current_hostname = _current_hostname()
    if provenance["emitter_hostname"] != current_hostname:
        raise Phase1BuildError(
            "Phase 1 build execution requires discovery artifacts emitted on the current host; "
            f"found provenance.emitter_hostname={provenance['emitter_hostname']!r}, "
            f"current host is {current_hostname!r}"
        )


def _recorded_tool_path_issues(recorded_tool_paths: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for field_name in REQUIRED_RECORDED_TOOL_PATH_FIELDS:
        recorded_path = str(recorded_tool_paths.get(field_name, "")).strip()
        if not recorded_path:
            issues.append(
                f"discovery artifacts are missing required recorded tool path {field_name}"
            )
            continue
        if not pathlib.Path(recorded_path).exists():
            issues.append(
                f"recorded {field_name} missing on current host: {recorded_path}"
            )
    return issues


def _current_hostname() -> str:
    return socket.gethostname().strip()


def _paths_match(observed_path: str, expected_path: str) -> bool:
    observed = observed_path.strip()
    expected = expected_path.strip()
    if not observed or not expected:
        return False
    try:
        return pathlib.Path(observed).resolve() == pathlib.Path(expected).resolve()
    except OSError:
        return observed == expected


def _render_dynamic_path_export(plan: Phase1BuildPlan) -> str | None:
    path_prefixes = _tool_path_prefixes(plan)
    if not path_prefixes:
        return None
    joined_prefixes = ":".join(path_prefixes)
    return f"export PATH={shlex.quote(joined_prefixes)}${{PATH:+:$PATH}}"


def _tool_path_prefixes(plan: Phase1BuildPlan) -> tuple[str, ...]:
    prefixes: list[str] = []
    preferred_field_order = (
        "nvidia_smi_path",
        "nvc_path",
        "nsys_path",
        "compute_sanitizer_path",
        "nvcc_path",
    )
    for field_name in preferred_field_order:
        recorded_path = str(plan.recorded_tool_paths.get(field_name, "")).strip()
        if not recorded_path:
            continue
        parent = pathlib.Path(recorded_path).parent.as_posix()
        if not any(_paths_match(parent, existing) for existing in prefixes):
            prefixes.append(parent)
    cuda_bin = f"{plan.env_exports['CUDA_HOME']}/bin"
    if not any(_paths_match(cuda_bin, existing) for existing in prefixes):
        prefixes.append(cuda_bin)
    return tuple(prefixes)


def _infer_cuda_home(selected_lane_value: str, *, nvcc_path: str | None = None) -> str:
    resolved_nvcc = str(nvcc_path or "").strip()
    if resolved_nvcc:
        candidate = pathlib.Path(resolved_nvcc).resolve()
        if candidate.name == "nvcc" and candidate.parent.name == "bin":
            return candidate.parent.parent.as_posix()
    match = re.search(r"(\d+\.\d+)", selected_lane_value)
    if not match:
        raise Phase1BuildError(
            f"unable to infer CUDA_HOME from selected lane value {selected_lane_value!r}"
        )
    return f"/usr/local/cuda-{match.group(1)}"


def _extract_nvarch(gpu_target: str) -> int:
    match = re.search(r"NVARCH\s*=\s*(\d+)", gpu_target)
    if not match:
        raise Phase1BuildError(f"unable to extract NVARCH from gpu_target {gpu_target!r}")
    nvarch = int(match.group(1))
    if f"sm_{nvarch}" not in gpu_target or "PTX" not in gpu_target:
        raise Phase1BuildError("gpu_target must declare native sm target plus PTX retention")
    return nvarch


def _prepend_env_path(variable_name: str, prefix: str) -> str:
    current_value = str(os.environ.get(variable_name, "")).strip()
    if not current_value:
        return prefix
    return f"{prefix}:{current_value}"


def _read_json_payload(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Phase1BuildError(f"unable to read JSON payload {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Phase1BuildError(f"invalid JSON payload {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase1BuildError(f"JSON payload {path} must contain an object")
    return payload


def _coerce_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise Phase1BuildError(f"{field_name} must be an integer") from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise Phase1BuildError(f"{field_name} must be a boolean")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
