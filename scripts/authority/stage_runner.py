"""Environment-neutral stage runner helpers built on top of probe artifacts."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from .env_probe import OpenFOAMBaselineProbeReport


@dataclass(frozen=True)
class StageRunnerContext:
    baseline_name: str
    bashrc_path: str
    runtime_base: str | None
    reviewed_source_tuple_id: str | None
    probe_payload_path: pathlib.Path
    host_env_path: pathlib.Path
    manifest_refs_path: pathlib.Path


class StageRunnerResolutionError(ValueError):
    """Raised when a stage runner cannot resolve an explicit baseline environment."""


def resolve_stage_runner_context(
    output_root: pathlib.Path | str,
    *,
    probe_report: OpenFOAMBaselineProbeReport | Mapping[str, Any],
    baseline_name: str,
    bashrc_path: str | None = None,
    probe_payload_path: pathlib.Path | str | None = None,
    host_env_path: pathlib.Path | str | None = None,
    manifest_refs_path: pathlib.Path | str | None = None,
) -> StageRunnerContext:
    output_root_path = pathlib.Path(output_root)
    record = _resolve_probe_record(probe_report, baseline_name=baseline_name)
    resolved_bashrc = _normalize_optional_string(
        bashrc_path or record.get("activation", {}).get("bashrc_path")
    )
    if not resolved_bashrc:
        raise StageRunnerResolutionError(
            f"missing baseline environment activation for {baseline_name}; bashrc_path is required"
        )

    resolved_probe_payload_path = _resolve_artifact_path(
        output_root_path,
        explicit_path=probe_payload_path,
        artifact_ref=record.get("artifacts", {}).get("probe"),
        artifact_label="probe payload",
        baseline_name=baseline_name,
    )
    resolved_host_env_path = _resolve_artifact_path(
        output_root_path,
        explicit_path=host_env_path,
        artifact_ref=record.get("artifacts", {}).get("host_env"),
        artifact_label="host_env manifest",
        baseline_name=baseline_name,
    )
    resolved_manifest_refs_path = _resolve_artifact_path(
        output_root_path,
        explicit_path=manifest_refs_path,
        artifact_ref=record.get("artifacts", {}).get("manifest_refs"),
        artifact_label="manifest_refs artifact",
        baseline_name=baseline_name,
    )

    probe_payload = _read_json_payload(
        resolved_probe_payload_path,
        artifact_label="probe payload",
        baseline_name=baseline_name,
    )
    host_env_payload = _read_json_payload(
        resolved_host_env_path,
        artifact_label="host_env manifest",
        baseline_name=baseline_name,
    )
    _read_json_payload(
        resolved_manifest_refs_path,
        artifact_label="manifest_refs artifact",
        baseline_name=baseline_name,
    )

    _validate_payload_baseline(host_env_payload, baseline_name=baseline_name)

    runtime_base = _normalize_optional_string(
        host_env_payload.get("runtime_base")
        or probe_payload.get("runtime_base")
        or record.get("runtime_base")
    )
    reviewed_source_tuple_id = _normalize_optional_string(
        host_env_payload.get("reviewed_source_tuple_id")
    )

    return StageRunnerContext(
        baseline_name=baseline_name,
        bashrc_path=resolved_bashrc,
        runtime_base=runtime_base,
        reviewed_source_tuple_id=reviewed_source_tuple_id,
        probe_payload_path=resolved_probe_payload_path,
        host_env_path=resolved_host_env_path,
        manifest_refs_path=resolved_manifest_refs_path,
    )


def wrap_stage_command(
    context: StageRunnerContext,
    *,
    case_dir: pathlib.Path | str,
    stage: Mapping[str, Any],
) -> str:
    stage_name = _normalize_optional_string(stage.get("name"))
    command = _normalize_optional_string(stage.get("cmd"))
    if not stage_name or not command:
        raise StageRunnerResolutionError("stage must define non-empty name and cmd values")

    cwd_value = stage.get("cwd", ".")
    cwd = pathlib.Path(case_dir) / str(cwd_value)
    env_prefix = _normalize_optional_string(stage.get("env_prefix"))
    command_segment = f"{env_prefix} {command}" if env_prefix else command

    segments = [
        f'. "{_escape_double_quoted(context.bashrc_path)}"',
        f'export GPU_CFD_BASELINE="{_escape_double_quoted(context.baseline_name)}"',
        f'export GPU_CFD_HOST_ENV="{_escape_double_quoted(context.host_env_path.as_posix())}"',
        f'export GPU_CFD_MANIFEST_REFS="{_escape_double_quoted(context.manifest_refs_path.as_posix())}"',
        f'export GPU_CFD_PROBE_PAYLOAD="{_escape_double_quoted(context.probe_payload_path.as_posix())}"',
        f'cd "{_escape_double_quoted(cwd.as_posix())}"',
    ]
    if context.reviewed_source_tuple_id:
        segments.append(
            "export GPU_CFD_REVIEWED_SOURCE_TUPLE_ID="
            + f'"{_escape_double_quoted(context.reviewed_source_tuple_id)}"'
        )
    segments.append(command_segment)
    return "bash -lc '" + " && ".join(segments) + "'"


def render_stage_runner_log_context(
    context: StageRunnerContext,
    *,
    stage_name: str,
) -> str:
    lines = [
        f"stage={stage_name}",
        f"baseline={context.baseline_name}",
        f"bashrc_path={context.bashrc_path}",
        f"runtime_base={context.runtime_base or ''}",
        f"host_env_manifest={context.host_env_path.as_posix()}",
        f"manifest_refs={context.manifest_refs_path.as_posix()}",
        f"probe_payload={context.probe_payload_path.as_posix()}",
    ]
    if context.reviewed_source_tuple_id:
        lines.append(f"reviewed_source_tuple_id={context.reviewed_source_tuple_id}")
    return "\n".join(lines)


def _resolve_probe_record(
    probe_report: OpenFOAMBaselineProbeReport | Mapping[str, Any],
    *,
    baseline_name: str,
) -> Mapping[str, Any]:
    if isinstance(probe_report, OpenFOAMBaselineProbeReport):
        records = probe_report.records
    elif "records" in probe_report:
        records = probe_report["records"]
    else:
        records = probe_report
    record = records.get(baseline_name)
    if not isinstance(record, Mapping):
        raise StageRunnerResolutionError(
            f"baseline {baseline_name!r} is missing from the supplied probe report"
        )
    return record


def _resolve_artifact_path(
    output_root: pathlib.Path,
    *,
    explicit_path: pathlib.Path | str | None,
    artifact_ref: Any,
    artifact_label: str,
    baseline_name: str,
) -> pathlib.Path:
    if explicit_path is not None:
        path = pathlib.Path(explicit_path)
    else:
        artifact = _normalize_optional_string(artifact_ref)
        if not artifact:
            raise StageRunnerResolutionError(
                f"missing {artifact_label} for {baseline_name}"
            )
        path = output_root / artifact
    if not path.exists():
        raise StageRunnerResolutionError(
            f"{artifact_label} for {baseline_name} does not exist: {path}"
        )
    return path


def _read_json_payload(
    path: pathlib.Path,
    *,
    artifact_label: str,
    baseline_name: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StageRunnerResolutionError(
            f"unable to read {artifact_label} for {baseline_name}: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise StageRunnerResolutionError(
            f"invalid JSON in {artifact_label} for {baseline_name}: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise StageRunnerResolutionError(
            f"{artifact_label} for {baseline_name} must contain a JSON object"
        )
    return payload


def _validate_payload_baseline(payload: Mapping[str, Any], *, baseline_name: str) -> None:
    payload_baseline = _normalize_optional_string(payload.get("baseline"))
    if payload_baseline and payload_baseline != baseline_name:
        raise StageRunnerResolutionError(
            f"artifact baseline mismatch: expected {baseline_name!r}, found {payload_baseline!r}"
        )


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _escape_double_quoted(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
