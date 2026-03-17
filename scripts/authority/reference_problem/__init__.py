from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

from ..bundle import AuthorityBundle
from ..cases import validate_case_meta, validate_stage_plan
from .extractor import build_feature_extractor_metrics_json
from .field_signatures import compute_field_signatures
from .fingerprints import compute_mesh_patch_fingerprint
from .models import (
    BUILD_FINGERPRINT_ARTIFACT_NAME,
    FIELD_SIGNATURES_ARTIFACT_NAME,
    METRICS_ARTIFACT_NAME,
    ReferenceProblemArtifacts,
)


def emit_reference_problem_artifacts(
    bundle: AuthorityBundle,
    *,
    case_dir: pathlib.Path | str,
    artifact_root: pathlib.Path | str,
    case_meta_path: pathlib.Path | str,
    stage_plan_path: pathlib.Path | str,
    normalized_steady_root: pathlib.Path | str,
    normalized_transient_root: pathlib.Path | str,
    run_meta_path: pathlib.Path | str | None = None,
    metrics: Mapping[str, Any] | None = None,
    metric_sources: Mapping[str, str] | None = None,
    time_windows: Mapping[str, Any] | None = None,
    angle_source: str | None = None,
) -> ReferenceProblemArtifacts:
    case_meta = validate_case_meta(bundle, _read_json_dict(pathlib.Path(case_meta_path)))
    stage_plan = validate_stage_plan(bundle, _read_json_dict(pathlib.Path(stage_plan_path)))
    _validate_identity_alignment(case_meta, stage_plan)
    case_identity = _build_case_identity(case_meta)
    shared_provenance = {
        "case_meta": pathlib.Path(case_meta_path).as_posix(),
        "stage_plan": pathlib.Path(stage_plan_path).as_posix(),
        "normalized_steady_root": pathlib.Path(normalized_steady_root).as_posix(),
        "normalized_transient_root": pathlib.Path(normalized_transient_root).as_posix(),
    }
    if run_meta_path is not None:
        shared_provenance["run_meta"] = pathlib.Path(run_meta_path).as_posix()

    artifact_root_path = pathlib.Path(artifact_root)
    artifact_root_path.mkdir(parents=True, exist_ok=True)

    build_fingerprint_payload = compute_mesh_patch_fingerprint(
        case_dir,
        case_identity=case_identity,
        provenance=shared_provenance,
    )
    field_signatures_payload = compute_field_signatures(
        normalized_steady_root=normalized_steady_root,
        normalized_transient_root=normalized_transient_root,
        case_identity=case_identity,
        provenance=shared_provenance,
    )

    build_fingerprint_path = artifact_root_path / BUILD_FINGERPRINT_ARTIFACT_NAME
    field_signatures_path = artifact_root_path / FIELD_SIGNATURES_ARTIFACT_NAME
    _write_json(build_fingerprint_path, build_fingerprint_payload)
    _write_json(field_signatures_path, field_signatures_payload)

    metrics_path = None
    if metrics is not None:
        metrics_payload = build_feature_extractor_metrics_json(
            case_identity=case_identity,
            metrics=metrics,
            metric_sources=metric_sources or {},
            time_windows=time_windows,
            angle_source=angle_source,
            provenance=shared_provenance,
        )
        metrics_path = artifact_root_path / METRICS_ARTIFACT_NAME
        _write_json(metrics_path, metrics_payload)

    return ReferenceProblemArtifacts(
        build_fingerprint_path=build_fingerprint_path,
        field_signatures_path=field_signatures_path,
        metrics_path=metrics_path,
    )


def _build_case_identity(case_meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_meta["case_id"],
        "case_role": case_meta["case_role"],
        "baseline": case_meta["baseline"],
        "runtime_base": case_meta["runtime_base"],
        "reviewed_source_tuple_id": case_meta["reviewed_source_tuple_id"],
    }


def _validate_identity_alignment(
    case_meta: Mapping[str, Any],
    stage_plan: Mapping[str, Any],
) -> None:
    for field_name in (
        "case_id",
        "case_role",
        "baseline",
        "runtime_base",
        "reviewed_source_tuple_id",
    ):
        if case_meta[field_name] != stage_plan[field_name]:
            raise ValueError(
                f"reference problem artifacts require matching {field_name} values across case metadata"
            )


def _read_json_dict(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"unable to read JSON payload: {path.as_posix()}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON payload: {path.as_posix()}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path.as_posix()}")
    return payload


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "ReferenceProblemArtifacts",
    "build_feature_extractor_metrics_json",
    "compute_field_signatures",
    "compute_mesh_patch_fingerprint",
    "emit_reference_problem_artifacts",
]
