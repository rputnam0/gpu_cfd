"""Phase 0 OpenFOAM environment probe helpers built on top of Foundation manifests."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .bundle import AuthorityBundle, AuthorityConflictError
from .pins import (
    CANONICAL_HOST_ENV_NAME,
    CANONICAL_MANIFEST_REFS_NAME,
    COMPATIBILITY_ALIASES,
    MANIFEST_SCHEMA_VERSION,
    load_pin_details,
    normalize_host_observations,
    resolve_repo_git_commit,
    write_json,
)


PROBE_SCHEMA_VERSION = "1.0.0"
PROBE_ARTIFACT_NAME = "probe.json"
REQUIRED_OPENFOAM_COMMANDS = (
    "foamRun",
    "incompressibleVoF",
    "interIsoFoam",
    "interFoam",
    "setFields",
    "checkMesh",
    "potentialFoam",
    "foamListTimes",
    "decomposePar",
)
REQUIRED_OPENFOAM_ENV_FIELDS = (
    "WM_PROJECT",
    "WM_PROJECT_VERSION",
    "WM_OPTIONS",
)
REQUIRED_LIBRARY_HINTS = (
    "libpetscFoam.so",
    "amgx",
)


@dataclass(frozen=True)
class BaselineProbeRequest:
    lane: str | None
    runtime_base: str | None = None
    bashrc_path: str | None = None
    consumer: str = "run"
    host_observations: Mapping[str, Any] | None = None
    local_mirror_refs: Mapping[str, str] | None = None
    repo_commit: str | None = None
    command_paths: Mapping[str, str | None] | None = None
    library_hints: Mapping[str, str | None] | None = None
    openfoam_env: Mapping[str, str | None] | None = None


@dataclass(frozen=True)
class OpenFOAMBaselineProbeReport:
    schema_version: str
    compatibility_aliases: dict[str, str]
    records: dict[str, dict[str, Any]]


def probe_openfoam_baselines(
    bundle: AuthorityBundle,
    *,
    output_dir: pathlib.Path | str,
    baselines: Mapping[str, BaselineProbeRequest],
) -> OpenFOAMBaselineProbeReport:
    target_dir = pathlib.Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}

    for baseline_name, request in baselines.items():
        records[baseline_name] = _probe_baseline(bundle, target_dir, baseline_name, request)

    return OpenFOAMBaselineProbeReport(
        schema_version=PROBE_SCHEMA_VERSION,
        compatibility_aliases=dict(COMPATIBILITY_ALIASES),
        records=records,
    )


def _probe_baseline(
    bundle: AuthorityBundle,
    output_root: pathlib.Path,
    baseline_name: str,
    request: BaselineProbeRequest,
) -> dict[str, Any]:
    pin_details = load_pin_details(bundle)
    diagnostics: list[str] = []
    commands = _normalize_commands(request.command_paths or {}, diagnostics)
    library_hints = _normalize_library_hints(request.library_hints or {})
    openfoam_env = _normalize_openfoam_env(request.openfoam_env or {}, diagnostics)
    runtime_base = _resolve_runtime_base(request.runtime_base, openfoam_env)
    bashrc_path = _normalize_optional_string(request.bashrc_path)

    record: dict[str, Any] = {
        "baseline": baseline_name,
        "consumer": request.consumer,
        "requested_lane": request.lane,
        "runtime_base": runtime_base,
        "activation": {"bashrc_path": bashrc_path},
        "commands": commands,
        "library_hints": library_hints,
        "openfoam_env": openfoam_env,
        "artifacts": {},
        "diagnostics": diagnostics,
    }

    if not bashrc_path:
        diagnostics.append(
            f"missing baseline environment activation for {baseline_name}; bashrc_path is required"
        )

    if request.lane not in {"primary", "experimental"}:
        diagnostics.append(
            f"unresolved toolkit lane for {baseline_name}: {request.lane!r}; expected 'primary' or 'experimental'"
        )
        record["status"] = "diagnostic"
        return record

    if not bashrc_path:
        record["status"] = "diagnostic"
        return record

    baseline_dir = output_root / _baseline_dir_name(baseline_name)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    local_mirror_refs = _normalize_local_mirror_refs(request.local_mirror_refs or {})
    try:
        host_observations = normalize_host_observations(dict(request.host_observations or {}))
        repo_git_commit = resolve_repo_git_commit(bundle.root, repo_commit=request.repo_commit)
        host_env_manifest = _build_host_env_manifest(
            bundle,
            pin_details=pin_details,
            baseline_name=baseline_name,
            request=request,
            runtime_base=runtime_base,
            openfoam_env=openfoam_env,
            host_observations=host_observations,
            repo_git_commit=repo_git_commit,
            local_mirror_refs=local_mirror_refs,
        )
        manifest_refs = _build_manifest_refs(
            bundle,
            pin_details=pin_details,
            request=request,
            runtime_base=runtime_base,
            openfoam_env=openfoam_env,
            host_observations=host_observations,
            repo_git_commit=repo_git_commit,
            local_mirror_refs=local_mirror_refs,
        )
    except (AuthorityConflictError, ValueError) as exc:
        diagnostics.append(str(exc))
        record["status"] = "diagnostic"
        return record

    host_env_path = baseline_dir / CANONICAL_HOST_ENV_NAME
    manifest_refs_path = baseline_dir / CANONICAL_MANIFEST_REFS_NAME
    write_json(host_env_path, host_env_manifest)
    write_json(manifest_refs_path, manifest_refs)

    alias_paths: dict[str, pathlib.Path] = {}
    for alias_name, canonical_name in COMPATIBILITY_ALIASES.items():
        alias_payload = dict(host_env_manifest)
        alias_payload["alias_name"] = alias_name
        alias_payload["canonical_name"] = canonical_name
        alias_path = baseline_dir / alias_name
        write_json(alias_path, alias_payload)
        alias_paths[alias_name] = alias_path

    probe_payload = _build_probe_payload(
        record,
        runtime_base=runtime_base,
        openfoam_release=openfoam_env.get("WM_PROJECT_VERSION"),
    )
    probe_path = baseline_dir / PROBE_ARTIFACT_NAME
    write_json(probe_path, probe_payload)

    record["probe"] = probe_payload
    record["host_env"] = host_env_manifest
    record["manifest_refs"] = manifest_refs
    record["artifacts"] = {
        "directory": _baseline_dir_name(baseline_name),
        "probe": _relative_artifact_path(output_root, probe_path),
        "host_env": _relative_artifact_path(output_root, host_env_path),
        "manifest_refs": _relative_artifact_path(output_root, manifest_refs_path),
        "compatibility_aliases": {
            alias_name: _relative_artifact_path(output_root, alias_path)
            for alias_name, alias_path in alias_paths.items()
        },
    }
    record["status"] = "ok" if not diagnostics else "diagnostic"
    record["probe"]["status"] = record["status"]
    return record


def _build_probe_payload(
    record: Mapping[str, Any],
    *,
    runtime_base: str | None,
    openfoam_release: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "baseline": record["baseline"],
        "consumer": record["consumer"],
        "requested_lane": record["requested_lane"],
        "runtime_base": runtime_base,
        "openfoam_release": openfoam_release,
        "activation": record["activation"],
        "commands": record["commands"],
        "library_hints": record["library_hints"],
        "openfoam_env": record["openfoam_env"],
        "diagnostics": list(record["diagnostics"]),
    }


def _build_host_env_manifest(
    bundle: AuthorityBundle,
    *,
    pin_details: Any,
    baseline_name: str,
    request: BaselineProbeRequest,
    runtime_base: str | None,
    openfoam_env: Mapping[str, str | None],
    host_observations: Mapping[str, Any],
    repo_git_commit: str,
    local_mirror_refs: Mapping[str, str],
) -> dict[str, Any]:
    tuple_details = _resolve_tuple_details(
        pin_details=pin_details,
        runtime_base=runtime_base,
        openfoam_release=openfoam_env.get("WM_PROJECT_VERSION"),
        local_mirror_refs=local_mirror_refs,
    )
    selected_lane = (
        pin_details.primary_toolkit_lane
        if request.lane == "primary"
        else pin_details.experimental_toolkit_lane
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_HOST_ENV_NAME,
        "consumer": request.consumer,
        "baseline": baseline_name,
        "reviewed_source_tuple_id": tuple_details["reviewed_source_tuple_id"],
        "runtime_base": runtime_base,
        "openfoam_release": openfoam_env.get("WM_PROJECT_VERSION"),
        "toolkit": {
            "selected_lane": request.lane,
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
        "host_observations": dict(host_observations),
        "manifest_refs": CANONICAL_MANIFEST_REFS_NAME,
        "probe_payload": PROBE_ARTIFACT_NAME,
        "compatibility_aliases": dict(COMPATIBILITY_ALIASES),
        "authority_revisions": bundle.authority_revisions,
        "repo": {
            "git_commit": repo_git_commit,
        },
    }


def _build_manifest_refs(
    bundle: AuthorityBundle,
    *,
    pin_details: Any,
    request: BaselineProbeRequest,
    runtime_base: str | None,
    openfoam_env: Mapping[str, str | None],
    host_observations: Mapping[str, Any],
    repo_git_commit: str,
    local_mirror_refs: Mapping[str, str],
) -> dict[str, Any]:
    tuple_details = _resolve_tuple_details(
        pin_details=pin_details,
        runtime_base=runtime_base,
        openfoam_release=openfoam_env.get("WM_PROJECT_VERSION"),
        local_mirror_refs=local_mirror_refs,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "canonical_name": CANONICAL_MANIFEST_REFS_NAME,
        "consumer": request.consumer,
        "reviewed_source_tuple_id": tuple_details["reviewed_source_tuple_id"],
        "runtime_base": runtime_base,
        "source_components": _build_source_component_refs(pin_details, local_mirror_refs),
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
        "required_revalidation": (
            list(pin_details.required_revalidation)
            if tuple_details["reviewed_source_tuple_id"]
            else []
        ),
        "repo": {
            "git_commit": repo_git_commit,
        },
    }


def _resolve_tuple_details(
    *,
    pin_details: Any,
    runtime_base: str | None,
    openfoam_release: str | None,
    local_mirror_refs: Mapping[str, str],
) -> dict[str, str | None]:
    exact_runtime = runtime_base == pin_details.runtime_base
    exact_release = _release_matches_pin_details(openfoam_release, pin_details.openfoam_release)
    exact_sources = all(
        local_mirror_refs.get(component) == source.resolved_commit
        for component, source in pin_details.source_components.items()
    )
    return {
        "reviewed_source_tuple_id": (
            pin_details.reviewed_source_tuple_id
            if exact_runtime and exact_release and exact_sources
            else None
        )
    }


def _release_matches_pin_details(
    observed_release: str | None,
    frozen_release: str,
) -> bool:
    normalized_observed = _normalize_optional_string(observed_release)
    normalized_frozen = _normalize_optional_string(frozen_release)
    if not normalized_observed or not normalized_frozen:
        return False
    sanitized_observed = _sanitize_release_token(normalized_observed)
    sanitized_frozen = _sanitize_release_token(normalized_frozen)
    return (
        sanitized_observed == sanitized_frozen
        or sanitized_frozen.endswith(sanitized_observed)
        or sanitized_observed.endswith(sanitized_frozen)
    )


def _sanitize_release_token(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", value.lower())


def _build_source_component_refs(
    pin_details: Any,
    local_mirror_refs: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    component_refs: dict[str, dict[str, Any]] = {}
    for component, source in pin_details.source_components.items():
        local_commit = _normalize_optional_string(local_mirror_refs.get(component))
        component_refs[component] = {
            "upstream_object": source.upstream_object,
            "frozen_ref_kind": source.frozen_ref_kind,
            "frozen_ref": source.frozen_ref,
            "resolved_commit": local_commit or source.resolved_commit,
            "frozen_resolved_commit": source.resolved_commit,
            "local_mirror_commit": local_commit,
            "matches_frozen_resolved_commit": local_commit == source.resolved_commit,
        }
    return component_refs


def _normalize_local_mirror_refs(
    local_mirror_refs: Mapping[str, str],
) -> dict[str, str]:
    return {
        component: normalized
        for component, value in local_mirror_refs.items()
        if (normalized := _normalize_optional_string(value)) is not None
    }


def _relative_artifact_path(root: pathlib.Path, artifact_path: pathlib.Path) -> str:
    return artifact_path.relative_to(root).as_posix()


def _baseline_dir_name(baseline_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", baseline_name.lower()).strip("_")
    return normalized or "baseline"


def _normalize_commands(
    command_paths: Mapping[str, str | None],
    diagnostics: list[str],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    missing_commands: list[str] = []
    for command_name in REQUIRED_OPENFOAM_COMMANDS:
        path = _normalize_optional_string(command_paths.get(command_name))
        found = bool(path)
        normalized[command_name] = {"path": path, "found": found}
        if not found:
            missing_commands.append(command_name)
    if missing_commands:
        diagnostics.append(
            "missing OpenFOAM command(s): " + ", ".join(missing_commands)
        )
    return normalized


def _normalize_library_hints(
    library_hints: Mapping[str, str | None],
) -> dict[str, str | None]:
    return {
        hint_name: _normalize_optional_string(library_hints.get(hint_name))
        for hint_name in REQUIRED_LIBRARY_HINTS
    }


def _normalize_openfoam_env(
    openfoam_env: Mapping[str, str | None],
    diagnostics: list[str],
) -> dict[str, str | None]:
    normalized = {
        field_name: _normalize_optional_string(openfoam_env.get(field_name))
        for field_name in REQUIRED_OPENFOAM_ENV_FIELDS
    }
    missing_fields = [field_name for field_name, value in normalized.items() if not value]
    if missing_fields:
        diagnostics.append(
            "missing OpenFOAM environment value(s): " + ", ".join(missing_fields)
        )
    return normalized


def _resolve_runtime_base(
    runtime_base: str | None,
    openfoam_env: Mapping[str, str | None],
) -> str | None:
    normalized_runtime = _normalize_optional_string(runtime_base)
    if normalized_runtime:
        return normalized_runtime
    project = _normalize_optional_string(openfoam_env.get("WM_PROJECT"))
    version = _normalize_optional_string(openfoam_env.get("WM_PROJECT_VERSION"))
    if project and version:
        return f"{project} {version}"
    return project or version


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
