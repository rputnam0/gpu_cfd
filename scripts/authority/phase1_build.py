"""Phase 1 Blackwell build-wrapper helpers built on top of frozen pin manifests."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import shlex
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
SUPPORTED_BUILD_MODES = {"debug", "relwithdebinfo"}
SKIP_AUDIT_DIR_NAMES = {".git", "__pycache__", ".venv", "build", "artifacts"}
NVTX2_INCLUDE_PATTERN = re.compile(r"nvtoolsext\.h", flags=re.IGNORECASE)


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
    env_exports: dict[str, str]
    nvtx_audit_hits: tuple[str, ...]


@dataclass(frozen=True)
class Phase1BuildResult:
    succeeded: bool
    returncode: int
    log_path: pathlib.Path
    metadata_path: pathlib.Path


class Phase1BuildError(ValueError):
    """Raised when Phase 1 build-wrapper inputs or execution are invalid."""


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
    local_mirror_refs = _extract_local_mirror_refs(discovery_manifest_refs)
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
        nvarch=nvarch,
        mode=mode,
        cuda_visible_devices=cuda_visible_devices,
        host_env_path=emitted.host_env_path,
        manifest_refs_path=emitted.manifest_refs_path,
        cuda_probe_path=pathlib.Path(cuda_probe_path).resolve(),
    )

    log_path = output_dir_path / f"build_{lane}_{mode}.log"
    metadata_path = output_dir_path / f"build_metadata_{lane}_{mode}.json"
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
        env_exports=env_exports,
        nvtx_audit_hits=nvtx_audit_hits,
    )


def render_phase1_env_exports(plan: Phase1BuildPlan) -> str:
    lines: list[str] = []
    for key, value in sorted(plan.env_exports.items()):
        if key == "PATH":
            lines.append('export PATH="$CUDA_HOME/bin:${PATH:-}"')
            continue
        if key == "LD_LIBRARY_PATH":
            lines.append('export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"')
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
    _record_build_outcome(
        plan,
        returncode=completed.returncode,
        succeeded=True,
        failure_stage=None,
        failure_reason=None,
    )
    return result


def audit_nvtx_includes(source_root: pathlib.Path | str) -> tuple[str, ...]:
    root = pathlib.Path(source_root).resolve()
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_AUDIT_DIR_NAMES for part in path.parts):
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
    for subparser in (env_parser, build_parser):
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_env_exports(
    *,
    selected_lane_value: str,
    nvarch: int,
    mode: str,
    cuda_visible_devices: str,
    host_env_path: pathlib.Path,
    manifest_refs_path: pathlib.Path,
    cuda_probe_path: pathlib.Path,
) -> dict[str, str]:
    cuda_home = _infer_cuda_home(selected_lane_value)
    nvcc_flags, cxx_flags = _build_mode_flags(mode, nvarch=nvarch)
    env = {
        "have_cuda": "true",
        "NVARCH": str(nvarch),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "CUDA_HOME": cuda_home,
        "PATH": _prepend_env_path("PATH", f"{cuda_home}/bin"),
        "LD_LIBRARY_PATH": _prepend_env_path("LD_LIBRARY_PATH", f"{cuda_home}/lib64"),
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
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_METADATA_SCHEMA_VERSION,
        "lane": lane,
        "mode": mode,
        "have_cuda": True,
        "nvarch": nvarch,
        "ptx_retention_required": True,
        "instrumentation": pin_details.instrumentation,
        "gpu_target": pin_details.gpu_target,
        "selected_lane_value": selected_lane_value,
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
        "nvtx_audit": {
            "passed": not nvtx_audit_hits,
            "banned_include_hits": list(nvtx_audit_hits),
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


def _write_build_log_message(log_path: pathlib.Path, message: str) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


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
    for key, value in sorted(plan.env_exports.items()):
        if key == "PATH":
            segments.append('export PATH="$CUDA_HOME/bin:${PATH:-}"')
            continue
        if key == "LD_LIBRARY_PATH":
            segments.append(
                'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
            )
            continue
        segments.append(f"export {key}={shlex.quote(value)}")
    if "SPUMA_EXTRA_CXX_FLAGS" in plan.env_exports:
        segments.append(
            'export FOAM_EXTRA_CXXFLAGS="${FOAM_EXTRA_CXXFLAGS:+${FOAM_EXTRA_CXXFLAGS} }${SPUMA_EXTRA_CXX_FLAGS}"'
        )
    return segments


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
    replacement = "$(FOAM_EXTRA_CXXFLAGS) $(SPUMA_EXTRA_NVCC_FLAGS) $(CU_LIB_HEADER_DIRS)"
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

    if not nvcc_path:
        issues.append("nvcc is unavailable in the active build environment")
    elif expected_toolkit and not _matches_expected_version(nvcc_version, expected_toolkit):
        issues.append(
            f"nvcc version must realize frozen toolkit lane {expected_toolkit!r}; found {nvcc_version!r}"
        )
    if not nsys_path:
        issues.append("nsys is unavailable in the active build environment")
    elif expected_nsys and not _matches_expected_version(nsys_version, expected_nsys):
        issues.append(
            f"nsys version must realize frozen value {expected_nsys!r}; found {nsys_version!r}"
        )
    if not compute_sanitizer_path:
        issues.append("compute-sanitizer is unavailable in the active build environment")
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
    elif expected_gpu_csv and nvidia_smi_gpu != expected_gpu_csv:
        issues.append(
            f"gpu_csv must realize frozen workstation observations {expected_gpu_csv!r}; found {nvidia_smi_gpu!r}"
        )

    if issues:
        raise Phase1BuildError("; ".join(issues))


def _matches_expected_version(observed_value: str, expected_value: str) -> bool:
    observed = observed_value.strip()
    expected = expected_value.strip()
    if expected and expected in observed:
        return True
    observed_versions = set(re.findall(r"\d+(?:\.\d+)+", observed))
    expected_versions = set(re.findall(r"\d+(?:\.\d+)+", expected))
    if expected_versions:
        return bool(observed_versions & expected_versions)
    if observed_versions:
        return expected in observed_versions
    return observed == expected


def _infer_cuda_home(selected_lane_value: str) -> str:
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
