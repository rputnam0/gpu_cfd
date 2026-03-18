"""Phase 0 Baseline A/B reference-freeze bundle publishing helpers."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .bundle import AuthorityBundle
from .bundle import load_authority_bundle
from .cases import (
    resolve_reference_case,
    validate_case_meta,
    validate_stage_plan,
)
from .reference_problem.models import (
    BUILD_FINGERPRINT_ARTIFACT_NAME,
    FIELD_SIGNATURES_ARTIFACT_NAME,
    METRICS_ARTIFACT_NAME,
)


REFERENCE_FREEZE_SCHEMA_VERSION = "1.0.0"
PHASE0_ARTIFACT_ROOT = "artifacts/reference_phase0"
PHASE0_CONTROL_INDEX_NAME = "control_index.json"
PHASE0_BUNDLE_MANIFEST_NAME = "frozen_bundle_manifest.json"
PHASE0_VERDICT_NAME = "baseline_verdict.json"

REQUIRED_CASE_ARTIFACTS = (
    "probe.json",
    "case_meta.json",
    "stage_plan.json",
    "reference_freeze_overlay.json",
    BUILD_FINGERPRINT_ARTIFACT_NAME,
    FIELD_SIGNATURES_ARTIFACT_NAME,
    METRICS_ARTIFACT_NAME,
)
OPTIONAL_CASE_ARTIFACTS = ("pressure_backend_patch.json",)


@dataclass(frozen=True)
class _CheckSpec:
    check_id: str
    gate_class: str
    comparator: str
    expected: Any
    source_paths: tuple[str, ...]
    validation_level: str
    missing_disposition: str = "fail"


_CASE_VALIDATION_LEVELS = {
    "R2": ("V0", "V1"),
    "R1-core": ("V0", "V2"),
    "R1": ("V0", "V3"),
    "R0": ("V0", "V4"),
}

_COMMON_SPECS = (
    _CheckSpec(
        check_id="probe_status_ok",
        gate_class="hard",
        comparator="exact",
        expected="ok",
        source_paths=("probe.status",),
        validation_level="V0",
    ),
)

_CASE_SPECS = {
    "R2": (
        _CheckSpec(
            check_id="solver_ok",
            gate_class="hard",
            comparator="exact",
            expected=1,
            source_paths=("metrics.solver_ok",),
            validation_level="V1",
        ),
        _CheckSpec(
            check_id="alpha_min",
            gate_class="hard",
            comparator="gte",
            expected=-1e-6,
            source_paths=("metrics.alpha_min",),
            validation_level="V1",
        ),
        _CheckSpec(
            check_id="alpha_max",
            gate_class="hard",
            comparator="lte",
            expected=1 + 1e-6,
            source_paths=("metrics.alpha_max",),
            validation_level="V1",
        ),
        _CheckSpec(
            check_id="alpha_integral_change_pct",
            gate_class="hard",
            comparator="lte",
            expected=0.5,
            source_paths=("metrics.alpha_integral_change_pct",),
            validation_level="V1",
        ),
        _CheckSpec(
            check_id="same_baseline_field_hash_exact",
            gate_class="review",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.same_baseline_field_hash_exact",),
            validation_level="V1",
            missing_disposition="pending",
        ),
    ),
    "R1-core": (
        _CheckSpec(
            check_id="solver_ok",
            gate_class="hard",
            comparator="exact",
            expected=1,
            source_paths=("metrics.solver_ok",),
            validation_level="V2",
        ),
        _CheckSpec(
            check_id="generic_phase5_subset_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=(
                "metrics.generic_phase5_subset_exact",
                "build_fingerprint.generic_phase5_subset_exact",
            ),
            validation_level="V2",
            missing_disposition="pending",
        ),
        _CheckSpec(
            check_id="patch_schema_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=(
                "metrics.patch_schema_exact",
                "build_fingerprint.patch_schema_exact",
                "build_fingerprint.semantic_patches",
            ),
            validation_level="V0",
        ),
        _CheckSpec(
            check_id="fingerprint_repeatable",
            gate_class="review",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.fingerprint_repeatable", "build_fingerprint.fingerprint_repeatable"),
            validation_level="V2",
            missing_disposition="pending",
        ),
    ),
    "R1": (
        _CheckSpec(
            check_id="solver_ok",
            gate_class="hard",
            comparator="exact",
            expected=1,
            source_paths=("metrics.solver_ok",),
            validation_level="V3",
        ),
        _CheckSpec(
            check_id="mass_imbalance_pct",
            gate_class="hard",
            comparator="lte",
            expected=1.0,
            source_paths=("metrics.mass_imbalance_pct",),
            validation_level="V3",
        ),
        _CheckSpec(
            check_id="patch_schema_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=(
                "metrics.patch_schema_exact",
                "build_fingerprint.patch_schema_exact",
                "build_fingerprint.semantic_patches",
            ),
            validation_level="V0",
        ),
        _CheckSpec(
            check_id="startup_provenance_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.startup_provenance_exact",),
            validation_level="V0",
            missing_disposition="pending",
        ),
        _CheckSpec(
            check_id="precondition_field_copy_provenance_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.precondition_field_copy_provenance_exact",),
            validation_level="V3",
            missing_disposition="pending",
        ),
    ),
    "R0": (
        _CheckSpec(
            check_id="solver_ok",
            gate_class="hard",
            comparator="exact",
            expected=1,
            source_paths=("metrics.solver_ok",),
            validation_level="V4",
        ),
        _CheckSpec(
            check_id="mass_imbalance_pct",
            gate_class="hard",
            comparator="lte",
            expected=1.0,
            source_paths=("metrics.mass_imbalance_pct",),
            validation_level="V4",
        ),
        _CheckSpec(
            check_id="patch_schema_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=(
                "metrics.patch_schema_exact",
                "build_fingerprint.patch_schema_exact",
                "build_fingerprint.semantic_patches",
            ),
            validation_level="V0",
        ),
        _CheckSpec(
            check_id="resolved_solver_family_recorded",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=("case_meta.resolved_vof_solver_exec",),
            validation_level="V0",
        ),
        _CheckSpec(
            check_id="startup_provenance_exact",
            gate_class="hard",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.startup_provenance_exact",),
            validation_level="V0",
            missing_disposition="pending",
        ),
        _CheckSpec(
            check_id="resolved_direct_slot_numerics_exact",
            gate_class="hard",
            comparator="exact",
            expected=True,
            source_paths=("metrics.resolved_direct_slot_numerics_exact",),
            validation_level="V4",
        ),
        _CheckSpec(
            check_id="manufacturing_sanity_reasonable",
            gate_class="review",
            comparator="truthy",
            expected=True,
            source_paths=("metrics.manufacturing_sanity_reasonable",),
            validation_level="V4",
            missing_disposition="pending",
        ),
    ),
}


def freeze_case_artifact(
    bundle: AuthorityBundle,
    *,
    artifact_dir: pathlib.Path | str,
    baseline_name: str,
) -> dict[str, Any]:
    artifact_dir_path = pathlib.Path(artifact_dir)
    case_payloads = _load_case_payloads(bundle, artifact_dir_path, baseline_name=baseline_name)
    verdict = _build_baseline_verdict(
        case_payloads["case_meta"]["case_role"],
        payloads=case_payloads,
    )
    verdict_path = artifact_dir_path / PHASE0_VERDICT_NAME
    _write_json(verdict_path, verdict)

    artifact_records = _build_artifact_records(artifact_dir_path)
    manifest = {
        "schema_version": REFERENCE_FREEZE_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "baseline": baseline_name,
        "case_role": case_payloads["case_meta"]["case_role"],
        "case_id": case_payloads["case_meta"]["case_id"],
        "validation_levels": list(
            _CASE_VALIDATION_LEVELS[case_payloads["case_meta"]["case_role"]]
        ),
        "artifact_dir": artifact_dir_path.as_posix(),
        "status": verdict["status"],
        "authority_revisions": _authority_revisions(bundle),
        "required_artifacts": list(REQUIRED_CASE_ARTIFACTS) + [PHASE0_VERDICT_NAME],
        "optional_artifacts": list(OPTIONAL_CASE_ARTIFACTS),
        "artifacts": artifact_records,
        "baseline_verdict": PHASE0_VERDICT_NAME,
    }
    _write_json(artifact_dir_path / PHASE0_BUNDLE_MANIFEST_NAME, manifest)
    return manifest


def publish_baseline_control_index(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    baseline_name: str,
    case_roles: Sequence[str] | None = None,
) -> dict[str, Any]:
    baseline_slug = baseline_artifact_slug(baseline_name)
    artifact_root_path = pathlib.Path(artifact_root)
    baseline_root = artifact_root_path / baseline_slug
    ordered_case_roles = tuple(case_roles or bundle.ladder.ordered_case_ids)
    manifests = []
    for case_role in ordered_case_roles:
        resolved = resolve_reference_case(bundle, case_role=case_role)
        manifests.append(
            freeze_case_artifact(
                bundle,
                artifact_dir=baseline_root / resolved.frozen_id,
                baseline_name=baseline_name,
            )
        )

    status = _summarize_status(manifest["status"] for manifest in manifests)
    control_index = {
        "schema_version": REFERENCE_FREEZE_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "phase_gate": "Phase 0",
        "baseline": baseline_name,
        "baseline_slug": baseline_slug,
        "status": status,
        "ordered_ladder": list(ordered_case_roles),
        "authority_revisions": _authority_revisions(bundle),
        "cases": [
            {
                "case_role": manifest["case_role"],
                "case_id": manifest["case_id"],
                "status": manifest["status"],
                "artifact_dir": manifest["artifact_dir"],
                "bundle_manifest": PHASE0_BUNDLE_MANIFEST_NAME,
                "baseline_verdict": PHASE0_VERDICT_NAME,
                "validation_levels": manifest["validation_levels"],
            }
            for manifest in manifests
        ],
    }
    baseline_root.mkdir(parents=True, exist_ok=True)
    _write_json(baseline_root / PHASE0_CONTROL_INDEX_NAME, control_index)
    return control_index


def baseline_artifact_slug(baseline_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", baseline_name.lower()).strip("_")
    if not slug:
        raise ValueError("baseline name must contain alphanumeric content")
    return slug


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Repository root containing docs/authority.",
    )
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=pathlib.Path(PHASE0_ARTIFACT_ROOT),
        help="Artifact root containing per-baseline Phase 0 case bundles.",
    )
    parser.add_argument(
        "--baseline-name",
        required=True,
        help="Baseline to publish, for example 'Baseline A'.",
    )
    parser.add_argument(
        "--case-role",
        action="append",
        dest="case_roles",
        help="Optional case-role subset to publish. Defaults to the full Phase 0 ladder.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the published control index as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    control_index = publish_baseline_control_index(
        bundle,
        artifact_root=args.artifact_root,
        baseline_name=args.baseline_name,
        case_roles=args.case_roles,
    )
    if args.json:
        print(json.dumps(control_index, indent=2, sort_keys=True))
    else:
        print(_format_text_report(control_index))
    return 0


def _format_text_report(control_index: Mapping[str, Any]) -> str:
    lines = [
        "Phase 0 reference-freeze control index",
        f"Baseline: {control_index['baseline']}",
        f"Status: {control_index['status']}",
    ]
    for case_record in control_index["cases"]:
        lines.append(
            f"- {case_record['case_role']} -> {case_record['case_id']} ({case_record['status']})"
        )
    return "\n".join(lines)


def _load_case_payloads(
    bundle: AuthorityBundle,
    artifact_dir: pathlib.Path,
    *,
    baseline_name: str,
) -> dict[str, Any]:
    phase0_baselines = tuple(
        str(item) for item in bundle.cases.phase_gate_mapping["Phase 0"]["baselines"]
    )
    if baseline_name not in phase0_baselines:
        raise ValueError(
            f"baseline {baseline_name!r} is not admitted for Phase 0; expected one of {phase0_baselines}"
        )

    payloads: dict[str, Any] = {}
    for filename in REQUIRED_CASE_ARTIFACTS:
        path = artifact_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing required Phase 0 artifact: {path.name}")
        payloads[path.stem] = _read_json(path)

    case_meta = validate_case_meta(bundle, dict(payloads["case_meta"]))
    stage_plan = validate_stage_plan(bundle, dict(payloads["stage_plan"]))
    if case_meta["baseline"] != baseline_name or stage_plan["baseline"] != baseline_name:
        raise ValueError(
            f"case bundle baseline mismatch for {artifact_dir}: expected {baseline_name!r}"
        )
    if case_meta["case_id"] != artifact_dir.name:
        raise ValueError(
            f"artifact directory name {artifact_dir.name!r} must match case_id {case_meta['case_id']!r}"
        )

    resolved_case = resolve_reference_case(bundle, case_role=case_meta["case_role"])
    if resolved_case.frozen_id != case_meta["case_id"]:
        raise ValueError(
            f"case bundle {case_meta['case_role']!r} must resolve to {resolved_case.frozen_id!r}"
        )
    if stage_plan["case_id"] != case_meta["case_id"] or stage_plan["case_role"] != case_meta["case_role"]:
        raise ValueError("case_meta.json and stage_plan.json must describe the same frozen case")
    if stage_plan["phase_gate"] != "Phase 0":
        raise ValueError(
            f"stage_plan.json phase_gate must remain 'Phase 0' for freeze publication, got {stage_plan['phase_gate']!r}"
        )

    probe_payload = dict(payloads["probe"])
    if probe_payload.get("baseline") not in (None, baseline_name):
        raise ValueError(
            f"probe.json baseline mismatch for {artifact_dir}: expected {baseline_name!r}"
        )

    build_fingerprint = dict(payloads["build_fingerprint"])
    patch_schema = build_fingerprint.get("patch_schema")
    if patch_schema is None:
        patch_schema = build_fingerprint.get("semantic_patches")
    if not isinstance(patch_schema, list) or not patch_schema:
        raise ValueError("build_fingerprint.json must define a non-empty patch_schema list")
    build_fingerprint.setdefault("patch_schema", patch_schema)
    build_fingerprint.setdefault("patch_schema_exact", bool(patch_schema))

    field_signatures = payloads["field_signatures"]
    if _field_signature_count(field_signatures) == 0:
        raise ValueError("field_signatures.json must define a non-empty signature list")

    metrics_payload = payloads["metrics"]
    if not isinstance(metrics_payload, Mapping) or not metrics_payload:
        raise ValueError("metrics.json must define a non-empty metrics object")

    payloads["case_meta"] = case_meta
    payloads["stage_plan"] = stage_plan
    payloads["probe"] = probe_payload
    payloads["build_fingerprint"] = build_fingerprint
    payloads["field_signatures"] = field_signatures
    payloads["metrics"] = _normalize_metrics(metrics_payload)
    return payloads


def _build_baseline_verdict(case_role: str, *, payloads: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for spec in (*_COMMON_SPECS, *_CASE_SPECS[case_role]):
        checks.append(_evaluate_check(spec, payloads))

    hard_failures = [check["check_id"] for check in checks if check["gate_class"] == "hard" and check["status"] == "fail"]
    hard_pending = [check["check_id"] for check in checks if check["gate_class"] == "hard" and check["status"] == "pending"]
    review_items = [check["check_id"] for check in checks if check["gate_class"] == "review" and check["status"] in {"fail", "pending"}]
    if hard_failures:
        status = "fail"
    elif hard_pending or review_items:
        status = "review"
    else:
        status = "pass"

    return {
        "schema_version": REFERENCE_FREEZE_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "baseline": payloads["case_meta"]["baseline"],
        "case_role": case_role,
        "case_id": payloads["case_meta"]["case_id"],
        "status": status,
        "validation_levels": list(_CASE_VALIDATION_LEVELS[case_role]),
        "checks": checks,
        "summary": {
            "hard_failures": hard_failures,
            "hard_pending": hard_pending,
            "review_items": review_items,
            "field_signature_count": _field_signature_count(payloads["field_signatures"]),
            "patch_count": len(payloads["build_fingerprint"]["patch_schema"]),
        },
    }


def _evaluate_check(spec: _CheckSpec, payloads: Mapping[str, Any]) -> dict[str, Any]:
    observation, source_path = _resolve_observation(spec.source_paths, payloads)
    if source_path is None:
        status = "pending" if spec.missing_disposition == "pending" else "fail"
        return {
            "check_id": spec.check_id,
            "gate_class": spec.gate_class,
            "validation_level": spec.validation_level,
            "status": status,
            "rule": _rule_text(spec),
            "source_path": None,
            "observation": None,
            "details": f"missing required observation from {', '.join(spec.source_paths)}",
        }

    passed = _compare(spec.comparator, observation, spec.expected)
    return {
        "check_id": spec.check_id,
        "gate_class": spec.gate_class,
        "validation_level": spec.validation_level,
        "status": "pass" if passed else "fail",
        "rule": _rule_text(spec),
        "source_path": source_path,
        "observation": observation,
        "details": None if passed else f"expected {spec.comparator} {spec.expected!r}",
    }


def _rule_text(spec: _CheckSpec) -> str:
    if spec.comparator == "exact":
        return f"exact {spec.expected!r}"
    if spec.comparator == "lte":
        return f"<= {spec.expected!r}"
    if spec.comparator == "gte":
        return f">= {spec.expected!r}"
    return "truthy"


def _resolve_observation(
    source_paths: Sequence[str],
    payloads: Mapping[str, Any],
) -> tuple[Any | None, str | None]:
    for source_path in source_paths:
        root_name, _, nested = source_path.partition(".")
        root_payload = payloads.get(root_name)
        if root_payload is None:
            continue
        value = _get_nested_value(root_payload, nested)
        if value is not None:
            return value, source_path
    return None, None


def _get_nested_value(payload: Any, path: str) -> Any | None:
    if not path:
        return payload
    current = payload
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _compare(comparator: str, observed: Any, expected: Any) -> bool:
    if comparator == "exact":
        return observed == expected
    if comparator == "truthy":
        return bool(observed)
    if comparator == "lte":
        return float(observed) <= float(expected)
    if comparator == "gte":
        return float(observed) >= float(expected)
    raise ValueError(f"unsupported comparator {comparator!r}")


def _normalize_metrics(metrics_payload: Mapping[str, Any]) -> dict[str, Any]:
    metric_records = metrics_payload.get("metrics", metrics_payload)
    if not isinstance(metric_records, Mapping):
        raise ValueError("metrics.json metrics payload must be an object")
    normalized = {}
    for metric_name, metric_value in metric_records.items():
        if isinstance(metric_value, Mapping) and "value" in metric_value:
            normalized[metric_name] = metric_value["value"]
        else:
            normalized[metric_name] = metric_value
    return normalized


def _field_signature_count(field_signatures: Any) -> int:
    if isinstance(field_signatures, list):
        return len(field_signatures)
    if not isinstance(field_signatures, Mapping):
        return 0
    count = 0
    for window_name in ("steady_precondition", "transient_latest"):
        window_payload = field_signatures.get(window_name)
        if not isinstance(window_payload, Mapping):
            continue
        fields = window_payload.get("fields", {})
        if isinstance(fields, Mapping):
            count += len(fields)
    return count


def _build_artifact_records(artifact_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    artifact_paths = [artifact_dir / name for name in REQUIRED_CASE_ARTIFACTS]
    artifact_paths.append(artifact_dir / PHASE0_VERDICT_NAME)
    for name in OPTIONAL_CASE_ARTIFACTS:
        path = artifact_dir / name
        if path.exists():
            artifact_paths.append(path)
    logs_dir = artifact_dir / "logs"
    if logs_dir.exists():
        artifact_paths.extend(sorted(path for path in logs_dir.rglob("*") if path.is_file()))
    return {
        path.relative_to(artifact_dir).as_posix(): {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(artifact_paths)
    }


def _authority_revisions(bundle: AuthorityBundle) -> dict[str, str]:
    revisions = {
        key: value["sha256"]
        for key, value in bundle.authority_revisions.items()
    }
    authority_root = pathlib.Path(bundle.report.root) / "docs" / "authority"
    for filename in (
        "reference_case_contract.md",
        "reference_case_contract.json",
        "validation_ladder.md",
    ):
        revisions[filename] = _sha256_file(authority_root / filename)
    return revisions


def _summarize_status(statuses: Sequence[str] | Any) -> str:
    normalized = tuple(statuses)
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status != "pass" for status in normalized):
        return "review"
    return "pass"


def _read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _timestamp_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
