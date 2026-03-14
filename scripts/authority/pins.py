"""Pin-manifest resolution and manifest emission helpers for foundation tooling."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any

from .bundle import (
    AuthorityBundle,
    AuthorityConflictError,
    markdown_section,
    markdown_table_by_first_column,
)


MANIFEST_SCHEMA_VERSION = "1.0.0"
CANONICAL_HOST_ENV_NAME = "host_env.json"
CANONICAL_MANIFEST_REFS_NAME = "manifest_refs.json"
COMPATIBILITY_ALIASES = {"env.json": CANONICAL_HOST_ENV_NAME}
SUPPORTED_CONSUMERS = {"build", "run", "profiling"}


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
    local_mirror_refs = local_mirror_refs or {}

    _validate_overrides(pin_details, overrides)
    _validate_local_mirror_refs(pin_details, local_mirror_refs)

    selected_lane = (
        pin_details.primary_toolkit_lane
        if lane == "primary"
        else pin_details.experimental_toolkit_lane
    )
    authority_revisions = build_authority_revisions(bundle.root)

    repo_git_commit = repo_commit or detect_git_value(["rev-parse", "HEAD"], cwd=bundle.root)

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
        "authority_revisions": authority_revisions,
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
        "authority_revisions": authority_revisions,
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
    resolution = resolve_consumer_pin_manifest(
        bundle,
        consumer=consumer,
        lane=lane,
        overrides=overrides,
        host_observations=host_observations,
        local_mirror_refs=local_mirror_refs,
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
    manifest_path = bundle.root / "docs" / "authority" / "master_pin_manifest.md"
    text = manifest_path.read_text(encoding="utf-8")
    frozen_defaults = markdown_table_by_first_column(text, "Frozen Defaults")
    source_rows = markdown_table_by_first_column(text, "Resolved Frozen Source Tuple")
    required_revalidation = tuple(_parse_required_revalidation(text))
    runtime_base = frozen_defaults["Runtime base"]["Frozen value"]
    runtime_family, _, mapped_release = runtime_base.partition(" mapped to ")
    return PinDetails(
        runtime_base=_normalize_value(runtime_family),
        openfoam_release=_normalize_value(mapped_release),
        primary_toolkit_lane=_normalize_value(
            frozen_defaults["Primary toolkit lane"]["Frozen value"]
        ),
        experimental_toolkit_lane=_normalize_value(
            frozen_defaults["Experimental toolkit lane"]["Frozen value"]
        ),
        driver_floor=_normalize_value(frozen_defaults["Driver floor"]["Frozen value"]),
        gpu_target=_normalize_value(frozen_defaults["GPU target"]["Frozen value"]),
        instrumentation=_normalize_value(frozen_defaults["Instrumentation"]["Frozen value"]),
        nsight_systems=_normalize_value(frozen_defaults["Nsight Systems"]["Frozen value"]),
        nsight_compute=_normalize_value(frozen_defaults["Nsight Compute"]["Frozen value"]),
        compute_sanitizer=_normalize_value(
            frozen_defaults["Compute Sanitizer"]["Frozen value"]
        ),
        reviewed_source_tuple_id=_normalize_value(
            frozen_defaults["Reviewed source tuple ID"]["Frozen value"]
        ),
        source_components={
            component: SourceComponent(
                component=component,
                upstream_object=_normalize_value(row["Upstream object"]),
                frozen_ref_kind=_normalize_value(row["Frozen ref kind"]),
                frozen_ref=_normalize_value(row["Frozen ref / version"]),
                resolved_commit=_normalize_value(row["Exact resolved commit / snapshot"]),
            )
            for component, row in source_rows.items()
        },
        required_revalidation=required_revalidation,
    )


def build_authority_revisions(root: pathlib.Path) -> dict[str, dict[str, str]]:
    tracked = {
        "master_pin_manifest": root / "docs" / "authority" / "master_pin_manifest.md",
        "acceptance_manifest": root / "docs" / "authority" / "acceptance_manifest.json",
        "support_matrix": root / "docs" / "authority" / "support_matrix.json",
        "graph_capture_support_matrix": root
        / "docs"
        / "authority"
        / "graph_capture_support_matrix.json",
    }
    return {
        key: {
            "path": path.relative_to(root).as_posix(),
            "sha256": compute_sha256(path),
        }
        for key, path in tracked.items()
    }


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
) -> None:
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


def _parse_required_revalidation(text: str) -> list[str]:
    section = markdown_section(text, "Required Revalidation If This Manifest Changes")
    results: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        number, separator, content = line.partition(". ")
        if separator and number.isdigit():
            results.append(content)
    return results


def _normalize_value(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("`") and normalized.endswith("`") and normalized.count("`") == 2:
        return normalized[1:-1]
    return normalized
