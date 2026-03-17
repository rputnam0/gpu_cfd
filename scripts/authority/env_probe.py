"""Phase 0 OpenFOAM environment probe helpers built on top of Foundation manifests."""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .bundle import AuthorityBundle, AuthorityConflictError
from .pins import COMPATIBILITY_ALIASES, emit_environment_manifests


PROBE_SCHEMA_VERSION = "1.0.0"
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
    diagnostics: list[str] = []
    commands = _normalize_commands(request.command_paths or {}, diagnostics)
    library_hints = _normalize_library_hints(request.library_hints or {})
    openfoam_env = _normalize_openfoam_env(request.openfoam_env or {}, diagnostics)
    bashrc_path = _normalize_optional_string(request.bashrc_path)

    record: dict[str, Any] = {
        "baseline": baseline_name,
        "consumer": request.consumer,
        "requested_lane": request.lane,
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
    try:
        emitted = emit_environment_manifests(
            bundle,
            consumer=request.consumer,
            output_dir=baseline_dir,
            lane=request.lane,
            host_observations=dict(request.host_observations or {}),
            local_mirror_refs=dict(request.local_mirror_refs or {}),
            repo_commit=request.repo_commit,
        )
    except (AuthorityConflictError, ValueError) as exc:
        diagnostics.append(str(exc))
        record["status"] = "diagnostic"
        return record

    record["host_env"] = json.loads(emitted.host_env_path.read_text(encoding="utf-8"))
    record["manifest_refs"] = json.loads(emitted.manifest_refs_path.read_text(encoding="utf-8"))
    record["artifacts"] = {
        "directory": _baseline_dir_name(baseline_name),
        "host_env": _relative_artifact_path(output_root, emitted.host_env_path),
        "manifest_refs": _relative_artifact_path(output_root, emitted.manifest_refs_path),
        "compatibility_aliases": {
            alias_name: _relative_artifact_path(output_root, alias_path)
            for alias_name, alias_path in emitted.alias_paths.items()
        },
    }
    record["status"] = "ok" if not diagnostics else "diagnostic"
    return record


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


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
