"""Phase 1 host and CUDA discovery helpers built on top of Foundation manifests."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from .bundle import AuthorityBundle, AuthorityConflictError, load_authority_bundle
    from .pins import (
        emit_environment_manifests,
        load_pin_details,
        normalize_host_observations,
        write_json,
    )
except ImportError:  # pragma: no cover - script execution fallback
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from scripts.authority.bundle import (
        AuthorityBundle,
        AuthorityConflictError,
        load_authority_bundle,
    )
    from scripts.authority.pins import (
        emit_environment_manifests,
        load_pin_details,
        normalize_host_observations,
        write_json,
    )


CUDA_PROBE_SCHEMA_VERSION = "1.0.0"
CUDA_PROBE_ARTIFACT_NAME = "cuda_probe.json"
REQUIRED_CUDA_PROBE_FIELDS = (
    "device_index",
    "device_name",
    "cc_major",
    "cc_minor",
    "total_global_mem_bytes",
    "managed_memory",
    "concurrent_managed_access",
    "unified_addressing",
    "native_kernel_ok",
    "managed_memory_probe_ok",
)


@dataclass(frozen=True)
class EmittedPhase1DiscoveryArtifacts:
    host_env_path: pathlib.Path
    manifest_refs_path: pathlib.Path
    alias_paths: dict[str, pathlib.Path]
    cuda_probe_path: pathlib.Path


def emit_phase1_discovery_artifacts(
    bundle: AuthorityBundle,
    *,
    output_dir: pathlib.Path | str,
    lane: str = "primary",
    host_observations: Mapping[str, Any] | None = None,
    cuda_probe: Mapping[str, Any] | None = None,
    local_mirror_refs: Mapping[str, str] | None = None,
    repo_commit: str | None = None,
) -> EmittedPhase1DiscoveryArtifacts:
    emitted_manifests = emit_environment_manifests(
        bundle,
        consumer="run",
        output_dir=output_dir,
        lane=lane,
        host_observations=dict(host_observations or {}),
        local_mirror_refs=dict(local_mirror_refs or {}),
        repo_commit=repo_commit,
    )
    probe_payload = build_cuda_probe_payload(
        bundle,
        lane=lane,
        cuda_probe=dict(cuda_probe or {}),
    )
    target_dir = pathlib.Path(output_dir)
    cuda_probe_path = target_dir / CUDA_PROBE_ARTIFACT_NAME
    probe_payload["host_env"] = emitted_manifests.host_env_path.name
    probe_payload["manifest_refs"] = emitted_manifests.manifest_refs_path.name
    write_json(cuda_probe_path, probe_payload)
    return EmittedPhase1DiscoveryArtifacts(
        host_env_path=emitted_manifests.host_env_path,
        manifest_refs_path=emitted_manifests.manifest_refs_path,
        alias_paths=emitted_manifests.alias_paths,
        cuda_probe_path=cuda_probe_path,
    )


def build_cuda_probe_payload(
    bundle: AuthorityBundle,
    *,
    lane: str,
    cuda_probe: Mapping[str, Any],
) -> dict[str, Any]:
    pin_details = load_pin_details(bundle)
    _validate_cuda_probe_payload(pin_details, cuda_probe)
    selected_lane = (
        pin_details.primary_toolkit_lane
        if lane == "primary"
        else pin_details.experimental_toolkit_lane
    )
    payload = {
        "schema_version": CUDA_PROBE_SCHEMA_VERSION,
        "canonical_name": CUDA_PROBE_ARTIFACT_NAME,
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "toolkit": {
            "selected_lane": lane,
            "selected_lane_value": selected_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "gpu_target": pin_details.gpu_target,
        "workstation_target": pin_details.workstation_target,
        "authority_revisions": bundle.authority_revisions,
    }
    payload.update({field_name: cuda_probe[field_name] for field_name in REQUIRED_CUDA_PROBE_FIELDS})
    for optional_field in (
        "managed_memory_failure_reason",
        "managed_memory_workaround",
    ):
        if str(cuda_probe.get(optional_field, "")).strip():
            payload[optional_field] = str(cuda_probe[optional_field]).strip()
    return payload


def collect_host_observations(
    *,
    command_runner: Callable[[list[str]], str] | None = None,
    os_release_path: pathlib.Path | str = "/etc/os-release",
) -> dict[str, str]:
    runner = command_runner or _run_command
    observations = {
        "hostname": runner(["hostname"]).strip(),
        "gpu_csv": _single_gpu_csv(
            runner(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        ),
        "nvcc_version": runner(["nvcc", "--version"]).strip(),
        "gcc_version": _first_line(runner(["gcc", "--version"])),
        "nsys_version": runner(["nsys", "--version"]).strip(),
        "ncu_version": runner(["ncu", "--version"]).strip(),
        "compute_sanitizer_version": runner(["compute-sanitizer", "--version"]).strip(),
        "os_release": _read_os_release(pathlib.Path(os_release_path)),
        "kernel": runner(["uname", "-r"]).strip(),
    }
    for tool_name, key_name in (
        ("nvcc", "nvcc_path"),
        ("nsys", "nsys_path"),
        ("ncu", "ncu_path"),
        ("compute-sanitizer", "compute_sanitizer_path"),
    ):
        tool_path = shutil.which(tool_name)
        if not tool_path:
            raise ValueError(f"required tool {tool_name!r} was not found in PATH")
        observations[key_name] = tool_path
    return normalize_host_observations(observations)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Repository root containing docs/authority.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where host_env.json, manifest_refs.json, and cuda_probe.json are written.",
    )
    parser.add_argument(
        "--lane",
        choices=("primary", "experimental"),
        default="primary",
        help="Frozen toolkit lane to validate and emit. Defaults to 'primary'.",
    )
    parser.add_argument(
        "--cuda-probe-json",
        type=pathlib.Path,
        required=True,
        help="Path to raw CUDA probe JSON that will be validated and rewritten canonically.",
    )
    parser.add_argument(
        "--host-observations-json",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file with pre-collected host observations. If omitted, collect from the current host.",
    )
    parser.add_argument(
        "--repo-commit",
        default=None,
        help="Optional repo commit override when git metadata is unavailable from the repo root.",
    )
    parser.add_argument(
        "--local-mirror-ref",
        action="append",
        default=[],
        metavar="COMPONENT=SHA",
        help="Repeat for each frozen source component commit required by manifest_refs.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    host_observations = (
        _read_json_file(args.host_observations_json)
        if args.host_observations_json is not None
        else collect_host_observations()
    )
    cuda_probe = _read_json_file(args.cuda_probe_json)
    emitted = emit_phase1_discovery_artifacts(
        bundle,
        output_dir=args.output_dir,
        lane=args.lane,
        host_observations=host_observations,
        cuda_probe=cuda_probe,
        local_mirror_refs=_parse_local_mirror_refs(args.local_mirror_ref),
        repo_commit=args.repo_commit,
    )
    print(
        json.dumps(
            {
                "host_env": emitted.host_env_path.as_posix(),
                "manifest_refs": emitted.manifest_refs_path.as_posix(),
                "cuda_probe": emitted.cuda_probe_path.as_posix(),
                "compatibility_aliases": {
                    alias_name: alias_path.as_posix()
                    for alias_name, alias_path in emitted.alias_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_cuda_probe_payload(pin_details: Any, cuda_probe: Mapping[str, Any]) -> None:
    missing_fields = [
        field_name
        for field_name in REQUIRED_CUDA_PROBE_FIELDS
        if field_name not in cuda_probe or cuda_probe[field_name] in (None, "")
    ]
    if missing_fields:
        raise AuthorityConflictError(
            "missing required CUDA probe field(s): " + ", ".join(missing_fields)
        )

    expected_gpu_name = _expected_gpu_name(pin_details.workstation_target)
    observed_gpu_name = str(cuda_probe["device_name"]).strip()
    if expected_gpu_name and not _gpu_name_matches(observed_gpu_name, expected_gpu_name):
        raise AuthorityConflictError(
            f"CUDA probe device_name must realize workstation target {expected_gpu_name!r}; "
            f"found {observed_gpu_name!r}"
        )

    expected_cc = _expected_compute_capability(pin_details.workstation_target)
    observed_cc = (
        _coerce_int(cuda_probe["cc_major"], field_name="cc_major"),
        _coerce_int(cuda_probe["cc_minor"], field_name="cc_minor"),
    )
    if expected_cc is not None and observed_cc != expected_cc:
        raise AuthorityConflictError(
            "CUDA probe compute capability must realize workstation target "
            f"{expected_cc[0]}.{expected_cc[1]}; found {observed_cc[0]}.{observed_cc[1]}"
        )

    for field_name in (
        "managed_memory",
        "concurrent_managed_access",
        "unified_addressing",
        "native_kernel_ok",
        "managed_memory_probe_ok",
    ):
        _coerce_bool(cuda_probe[field_name], field_name=field_name)

    if not _coerce_bool(cuda_probe["native_kernel_ok"], field_name="native_kernel_ok"):
        raise AuthorityConflictError(
            "CUDA probe native-kernel check failed; raw CUDA execution is not viable"
        )

    if not _coerce_bool(
        cuda_probe["managed_memory_probe_ok"],
        field_name="managed_memory_probe_ok",
    ):
        failure_reason = str(cuda_probe.get("managed_memory_failure_reason", "")).strip()
        if failure_reason:
            raise AuthorityConflictError(
                "CUDA probe managed-memory validation failed; "
                f"environment blocker reported: {failure_reason}"
            )
        raise AuthorityConflictError(
            "CUDA probe managed-memory validation failed; environment blocker reported"
        )

    total_global_mem_bytes = _coerce_int(
        cuda_probe["total_global_mem_bytes"],
        field_name="total_global_mem_bytes",
    )
    if total_global_mem_bytes <= 0:
        raise AuthorityConflictError(
            "CUDA probe total_global_mem_bytes must be a positive integer"
        )


def _expected_gpu_name(workstation_target: str) -> str | None:
    first_clause = workstation_target.split(",", 1)[0].strip()
    normalized = re.sub(r"^(single|dual|multi)\s+", "", first_clause, flags=re.IGNORECASE)
    return normalized or None


def _expected_compute_capability(workstation_target: str) -> tuple[int, int] | None:
    match = re.search(
        r"compute capability\s+(\d+)\.(\d+)",
        workstation_target,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _gpu_name_matches(observed: str, expected: str) -> bool:
    normalized_observed = " ".join(observed.lower().split())
    normalized_expected = " ".join(expected.lower().split())
    if normalized_expected not in normalized_observed:
        return False
    disallowed_tokens = {"laptop", "mobile", "notebook", "max-q"}
    return not any(token in normalized_observed for token in disallowed_tokens)


def _coerce_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise AuthorityConflictError(f"CUDA probe {field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AuthorityConflictError(
            f"CUDA probe {field_name} must be an integer"
        ) from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise AuthorityConflictError(f"CUDA probe {field_name} must be a boolean")


def _run_command(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        message = stderr or f"command exited with status {completed.returncode}"
        raise ValueError(f"failed to collect {' '.join(args)!r}: {message}")
    stdout = completed.stdout.strip()
    if not stdout:
        raise ValueError(f"failed to collect {' '.join(args)!r}: command returned no output")
    return stdout


def _read_os_release(path: pathlib.Path) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"required OS release file is missing: {path}") from exc
    for line in content.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.partition("=")[2].strip().strip('"')
    raise ValueError(f"PRETTY_NAME is missing from {path}")


def _first_line(value: str) -> str:
    for line in value.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _single_gpu_csv(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) > 1:
        raise ValueError(
            "nvidia-smi returned multiple GPU rows; Phase 1 host discovery requires a single-GPU workstation baseline"
        )
    return lines[0]


def _parse_local_mirror_refs(entries: list[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for entry in entries:
        component, separator, sha = entry.partition("=")
        if not separator or not component.strip() or not sha.strip():
            raise ValueError(
                f"invalid --local-mirror-ref value {entry!r}; expected COMPONENT=SHA"
            )
        refs[component.strip()] = sha.strip()
    return refs


def _read_json_file(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
