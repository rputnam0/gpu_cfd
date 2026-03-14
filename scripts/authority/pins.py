"""Pin-manifest resolution and manifest emission helpers for foundation tooling."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
from dataclasses import dataclass
from typing import Any

from .bundle import (
    AuthorityBundle,
    AuthorityConflictError,
)


MANIFEST_SCHEMA_VERSION = "1.0.0"
CANONICAL_HOST_ENV_NAME = "host_env.json"
CANONICAL_MANIFEST_REFS_NAME = "manifest_refs.json"
COMPATIBILITY_ALIASES = {"env.json": CANONICAL_HOST_ENV_NAME}
SUPPORTED_CONSUMERS = {"build", "run", "profiling"}
CANONICAL_HOST_OBSERVATION_FIELDS = (
    "gpu_csv",
    "nvcc_version",
    "gcc_version",
    "nsys_version",
    "ncu_version",
    "compute_sanitizer_version",
    "os_release",
    "kernel",
    "nvcc_path",
    "nsys_path",
    "ncu_path",
    "compute_sanitizer_path",
)
HOST_OBSERVATION_ALIASES = {
    "gpu_query": "gpu_csv",
    "compiler_version": "gcc_version",
}


@dataclass(frozen=True)
class SourceComponent:
    component: str
    upstream_object: str
    frozen_ref_kind: str
    frozen_ref: str
    resolved_commit: str


@dataclass(frozen=True)
class PinDetails:
    runtime_base: str
    openfoam_release: str
    primary_toolkit_lane: str
    experimental_toolkit_lane: str
    driver_floor: str
    gpu_target: str
    workstation_target: str
    instrumentation: str
    nsight_systems: str
    nsight_compute: str
    compute_sanitizer: str
    reviewed_source_tuple_id: str
    source_components: dict[str, SourceComponent]
    required_revalidation: tuple[str, ...]


@dataclass(frozen=True)
class ConsumerPinResolution:
    consumer: str
    shared_resolution_key: str
    host_env: dict[str, Any]
    manifest_refs: dict[str, Any]


@dataclass(frozen=True)
class EmittedEnvironmentManifests:
    host_env_path: pathlib.Path
    manifest_refs_path: pathlib.Path
    alias_paths: dict[str, pathlib.Path]


def resolve_consumer_pin_manifest(
    bundle: AuthorityBundle,
    *,
    consumer: str,
    lane: str = "primary",
    overrides: dict[str, str] | None = None,
    host_observations: dict[str, Any] | None = None,
    local_mirror_refs: dict[str, str] | None = None,
    repo_commit: str | None = None,
) -> ConsumerPinResolution:
    if consumer not in SUPPORTED_CONSUMERS:
        raise ValueError(f"unsupported consumer {consumer!r}")
    if lane not in {"primary", "experimental"}:
        raise ValueError(f"unsupported lane {lane!r}")

    pin_details = load_pin_details(bundle)
    overrides = overrides or {}
    if not host_observations:
        raise ValueError("host_observations are required to emit host_env.json")
    host_observations = normalize_host_observations(host_observations)
    local_mirror_refs = local_mirror_refs or {}

    _validate_overrides(pin_details, overrides)
    _validate_host_observations(pin_details, host_observations, lane=lane)
    _validate_local_mirror_refs(pin_details, local_mirror_refs)

    selected_lane = (
        pin_details.primary_toolkit_lane
        if lane == "primary"
        else pin_details.experimental_toolkit_lane
    )
    repo_git_commit = resolve_repo_git_commit(bundle.root, repo_commit=repo_commit)

    host_env = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_HOST_ENV_NAME,
        "consumer": consumer,
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "openfoam_release": pin_details.openfoam_release,
        "toolkit": {
            "selected_lane": lane,
            "selected_lane_value": selected_lane,
            "primary_lane": pin_details.primary_toolkit_lane,
            "experimental_lane": pin_details.experimental_toolkit_lane,
            "driver_floor": pin_details.driver_floor,
        },
        "gpu_target": pin_details.gpu_target,
        "instrumentation": pin_details.instrumentation,
        "profilers": {
            "nsight_systems": pin_details.nsight_systems,
            "nsight_compute": pin_details.nsight_compute,
            "compute_sanitizer": pin_details.compute_sanitizer,
        },
        "host_observations": host_observations,
        "manifest_refs": CANONICAL_MANIFEST_REFS_NAME,
        "compatibility_aliases": dict(COMPATIBILITY_ALIASES),
        "authority_revisions": bundle.authority_revisions,
        "repo": {
            "git_commit": repo_git_commit,
        },
    }

    manifest_refs = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_MANIFEST_REFS_NAME,
        "consumer": consumer,
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "source_components": {
            component: {
                "upstream_object": source.upstream_object,
                "frozen_ref_kind": source.frozen_ref_kind,
                "frozen_ref": source.frozen_ref,
                "resolved_commit": source.resolved_commit,
                "local_mirror_commit": local_mirror_refs.get(component),
            }
            for component, source in pin_details.source_components.items()
        },
        "authority_revisions": bundle.authority_revisions,
        "invoked_tool_paths": {
            field_name: host_observations[field_name]
            for field_name in (
                "nvcc_path",
                "nsys_path",
                "ncu_path",
                "compute_sanitizer_path",
            )
            if str(host_observations.get(field_name, "")).strip()
        },
        "required_revalidation": list(pin_details.required_revalidation),
        "repo": {
            "git_commit": repo_git_commit,
        },
    }
    shared_resolution_key = build_shared_resolution_key(pin_details, lane=lane)
    return ConsumerPinResolution(
        consumer=consumer,
        shared_resolution_key=shared_resolution_key,
        host_env=host_env,
        manifest_refs=manifest_refs,
    )


def emit_environment_manifests(
    bundle: AuthorityBundle,
    *,
    consumer: str,
    output_dir: pathlib.Path | str,
    lane: str = "primary",
    overrides: dict[str, str] | None = None,
    host_observations: dict[str, Any] | None = None,
    local_mirror_refs: dict[str, str] | None = None,
    repo_commit: str | None = None,
) -> EmittedEnvironmentManifests:
    pin_details = load_pin_details(bundle)
    resolved_host_observations = normalize_host_observations(host_observations or {})
    resolved_local_mirror_refs = local_mirror_refs or {}
    _validate_host_observations(pin_details, resolved_host_observations, lane=lane)
    _validate_local_mirror_refs(
        pin_details,
        resolved_local_mirror_refs,
        require_complete=True,
    )

    resolution = resolve_consumer_pin_manifest(
        bundle,
        consumer=consumer,
        lane=lane,
        overrides=overrides,
        host_observations=resolved_host_observations,
        local_mirror_refs=resolved_local_mirror_refs,
        repo_commit=repo_commit,
    )
    target_dir = pathlib.Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    host_env_path = target_dir / CANONICAL_HOST_ENV_NAME
    manifest_refs_path = target_dir / CANONICAL_MANIFEST_REFS_NAME
    write_json(host_env_path, resolution.host_env)
    write_json(manifest_refs_path, resolution.manifest_refs)

    alias_paths: dict[str, pathlib.Path] = {}
    for alias_name, canonical_name in COMPATIBILITY_ALIASES.items():
        alias_payload = dict(resolution.host_env)
        alias_payload["alias_name"] = alias_name
        alias_payload["canonical_name"] = canonical_name
        alias_path = target_dir / alias_name
        write_json(alias_path, alias_payload)
        alias_paths[alias_name] = alias_path

    return EmittedEnvironmentManifests(
        host_env_path=host_env_path,
        manifest_refs_path=manifest_refs_path,
        alias_paths=alias_paths,
    )


def load_pin_details(bundle: AuthorityBundle) -> PinDetails:
    runtime_base = bundle.pins.runtime_base
    runtime_family, _, mapped_release = runtime_base.partition(" mapped to ")
    return PinDetails(
        runtime_base=_normalize_value(runtime_family),
        openfoam_release=_normalize_value(mapped_release),
        primary_toolkit_lane=_normalize_value(bundle.pins.primary_toolkit_lane),
        experimental_toolkit_lane=_normalize_value(bundle.pins.experimental_toolkit_lane),
        driver_floor=_normalize_value(bundle.pins.driver_floor),
        gpu_target=_normalize_value(bundle.pins.gpu_target),
        workstation_target=_normalize_value(bundle.pins.workstation_target),
        instrumentation=_normalize_value(bundle.pins.instrumentation),
        nsight_systems=_normalize_value(bundle.pins.nsight_systems),
        nsight_compute=_normalize_value(bundle.pins.nsight_compute),
        compute_sanitizer=_normalize_value(bundle.pins.compute_sanitizer),
        reviewed_source_tuple_id=_normalize_value(bundle.pins.reviewed_source_tuple_id),
        source_components={
            component: SourceComponent(
                component=component,
                upstream_object=_normalize_value(source.upstream_object),
                frozen_ref_kind=_normalize_value(source.frozen_ref_kind),
                frozen_ref=_normalize_value(source.frozen_ref),
                resolved_commit=_normalize_value(source.resolved_commit),
            )
            for component, source in bundle.pins.source_components.items()
        },
        required_revalidation=tuple(
            _normalize_value(step) for step in bundle.pins.required_revalidation
        ),
    )


def build_shared_resolution_key(pin_details: PinDetails, *, lane: str) -> str:
    material = "|".join(
        [
            pin_details.reviewed_source_tuple_id,
            lane,
            pin_details.primary_toolkit_lane,
            pin_details.experimental_toolkit_lane,
            pin_details.driver_floor,
            pin_details.gpu_target,
            pin_details.instrumentation,
            pin_details.nsight_systems,
            pin_details.nsight_compute,
            pin_details.compute_sanitizer,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def detect_git_value(args: list[str], *, cwd: pathlib.Path | None = None) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def normalize_host_observations(host_observations: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    allowed_keys = set(CANONICAL_HOST_OBSERVATION_FIELDS)
    for raw_key, value in host_observations.items():
        canonical_key = HOST_OBSERVATION_ALIASES.get(raw_key, raw_key)
        if canonical_key not in allowed_keys:
            raise ValueError(
                f"unsupported host observation field {raw_key!r}; expected canonical Phase 1 host_env keys"
            )
        existing = normalized.get(canonical_key)
        if existing is not None and existing != value:
            raise ValueError(
                f"conflicting host observation values for {canonical_key!r}"
            )
        normalized[canonical_key] = value
    return normalized


def resolve_repo_git_commit(root: pathlib.Path, *, repo_commit: str | None) -> str:
    resolved_commit = repo_commit or detect_git_value(["rev-parse", "HEAD"], cwd=root)
    if not resolved_commit:
        raise ValueError(
            "repo_commit is required when git metadata cannot be resolved from the bundle root"
        )
    return resolved_commit


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compute_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_overrides(pin_details: PinDetails, overrides: dict[str, str]) -> None:
    allowed_fields = {
        "reviewed_source_tuple_id": pin_details.reviewed_source_tuple_id,
        "runtime_base": pin_details.runtime_base,
        "primary_toolkit_lane": pin_details.primary_toolkit_lane,
        "experimental_toolkit_lane": pin_details.experimental_toolkit_lane,
        "driver_floor": pin_details.driver_floor,
        "gpu_target": pin_details.gpu_target,
        "instrumentation": pin_details.instrumentation,
        "nsight_systems": pin_details.nsight_systems,
        "nsight_compute": pin_details.nsight_compute,
        "compute_sanitizer": pin_details.compute_sanitizer,
    }
    for field_name, attempted_value in overrides.items():
        if field_name not in allowed_fields:
            raise AuthorityConflictError(
                f"unknown pin override field {field_name!r}; frozen pin overrides are not allowed"
            )
        frozen_value = allowed_fields[field_name]
        if attempted_value != frozen_value:
            raise AuthorityConflictError(
                f"{field_name} must remain {frozen_value!r}; found conflicting override {attempted_value!r}"
            )


def _validate_local_mirror_refs(
    pin_details: PinDetails,
    local_mirror_refs: dict[str, str],
    *,
    require_complete: bool = False,
) -> None:
    if require_complete:
        missing_components = [
            component
            for component in pin_details.source_components
            if component not in local_mirror_refs
        ]
        if missing_components:
            raise AuthorityConflictError(
                "missing local mirror refs for frozen source components: "
                + ", ".join(missing_components)
            )
    for component, local_commit in local_mirror_refs.items():
        if component not in pin_details.source_components:
            raise AuthorityConflictError(
                f"unknown source component {component!r} in local mirror refs"
            )
        resolved_commit = pin_details.source_components[component].resolved_commit
        if local_commit != resolved_commit:
            raise AuthorityConflictError(
                f"{component} must realize frozen commit {resolved_commit!r}; found {local_commit!r}"
            )


def _validate_host_observations(
    pin_details: PinDetails,
    host_observations: dict[str, Any],
    *,
    lane: str,
) -> None:
    required_fields = (
        "gpu_csv",
        "nvcc_version",
        "gcc_version",
        "nsys_version",
        "ncu_version",
        "compute_sanitizer_version",
        "os_release",
        "kernel",
    )
    missing_fields = [
        field_name
        for field_name in required_fields
        if not str(host_observations.get(field_name, "")).strip()
    ]
    if missing_fields:
        raise AuthorityConflictError(
            "missing required host observation(s): " + ", ".join(missing_fields)
        )

    selected_toolkit_lane = (
        pin_details.primary_toolkit_lane
        if lane == "primary"
        else pin_details.experimental_toolkit_lane
    )
    version_expectations = {
        "nvcc_version": selected_toolkit_lane.removeprefix("CUDA ").strip(),
        "nsys_version": pin_details.nsight_systems,
        "ncu_version": pin_details.nsight_compute,
        "compute_sanitizer_version": pin_details.compute_sanitizer,
    }
    for field_name, expected_value in version_expectations.items():
        observed_value = str(host_observations[field_name]).strip()
        if not _matches_frozen_version(observed_value, expected_value):
            raise AuthorityConflictError(
                f"{field_name} must realize frozen value {expected_value!r}; "
                f"found {observed_value!r}"
            )

    gpu_name, driver_version, gpu_memory_mib = _parse_gpu_csv(host_observations["gpu_csv"])
    expected_gpu, expected_memory_mib = _parse_workstation_target(pin_details.workstation_target)
    if expected_gpu and not _gpu_name_matches_workstation(gpu_name, expected_gpu):
        raise AuthorityConflictError(
            f"gpu_csv must realize workstation target {expected_gpu!r}; found {gpu_name!r}"
        )
    if expected_memory_mib is not None:
        if gpu_memory_mib is None:
            raise AuthorityConflictError(
                "gpu_csv must include GPU memory for the frozen workstation target"
            )
        memory_tolerance_mib = max(256, int(expected_memory_mib * 0.02))
        if abs(gpu_memory_mib - expected_memory_mib) > memory_tolerance_mib:
            raise AuthorityConflictError(
                f"gpu_csv memory {gpu_memory_mib} MiB does not satisfy frozen workstation target memory {expected_memory_mib} MiB"
            )
    if lane == "primary" and not _driver_meets_floor(driver_version, pin_details.driver_floor):
        raise AuthorityConflictError(
            f"gpu_csv driver {driver_version!r} does not satisfy frozen driver floor {pin_details.driver_floor!r}"
        )


def _parse_gpu_csv(gpu_csv: str) -> tuple[str, str, int | None]:
    parts = [part.strip() for part in gpu_csv.split(",")]
    if len(parts) < 2:
        raise AuthorityConflictError(
            f"gpu_csv must include GPU name and driver version; found {gpu_csv!r}"
        )
    memory_mib = _parse_memory_mib(parts[2]) if len(parts) >= 3 else None
    return parts[0], parts[1], memory_mib


def _parse_workstation_target(workstation_target: str) -> tuple[str | None, int | None]:
    first_clause = workstation_target.split(",", 1)[0].strip()
    normalized = re.sub(r"^(single|dual|multi)\s+", "", first_clause, flags=re.IGNORECASE)
    memory_match = re.search(r"(\d+(?:\.\d+)?)\s*GB", workstation_target, flags=re.IGNORECASE)
    expected_memory_mib = None
    if memory_match:
        expected_memory_mib = int(float(memory_match.group(1)) * 1024)
    return normalized or None, expected_memory_mib


def _gpu_name_matches_workstation(gpu_name: str, expected_gpu: str) -> bool:
    normalized_name = gpu_name.lower()
    normalized_expected = _normalize_gpu_model(expected_gpu)
    actual_model = _extract_gpu_model(gpu_name)
    if actual_model:
        if _normalize_gpu_model(actual_model) != normalized_expected:
            return False
    elif normalized_expected.lower() not in normalized_name:
        return False
    disallowed_tokens = {"laptop", "mobile", "notebook", "max-q"}
    return not any(token in normalized_name for token in disallowed_tokens)


def _matches_frozen_version(observed_value: str, expected_value: str) -> bool:
    expected = expected_value.strip()
    observed_versions = set(re.findall(r"\d+(?:\.\d+)+", observed_value))
    if observed_versions:
        return expected in observed_versions
    return observed_value.strip() == expected


def _parse_memory_mib(value: str) -> int | None:
    mib_match = re.search(r"(\d+)\s*MiB", value, flags=re.IGNORECASE)
    if mib_match:
        return int(mib_match.group(1))
    gib_match = re.search(r"(\d+(?:\.\d+)?)\s*GiB", value, flags=re.IGNORECASE)
    if gib_match:
        return int(float(gib_match.group(1)) * 1024)
    gb_match = re.search(r"(\d+(?:\.\d+)?)\s*GB", value, flags=re.IGNORECASE)
    if gb_match:
        return int(float(gb_match.group(1)) * 1024)
    return None


def _extract_gpu_model(value: str) -> str | None:
    match = re.search(
        r"\b(?:RTX|GTX)\s+\d{3,4}(?:\s+(?:Ti|SUPER))?\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return " ".join(match.group(0).split())


def _normalize_gpu_model(value: str) -> str:
    return " ".join(value.lower().split())


def _driver_meets_floor(driver_version: str, driver_floor: str) -> bool:
    required_version = driver_floor.removeprefix(">=").strip()
    observed_parts = tuple(int(part) for part in driver_version.split("."))
    required_parts = tuple(int(part) for part in required_version.split("."))
    length = max(len(observed_parts), len(required_parts))
    padded_observed = observed_parts + (0,) * (length - len(observed_parts))
    padded_required = required_parts + (0,) * (length - len(required_parts))
    return padded_observed >= padded_required


def _normalize_value(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("`") and normalized.endswith("`") and normalized.count("`") == 2:
        return normalized[1:-1]
    return normalized
