"""Aggregate Baseline B build and R2 smoke evidence into a nozzle go/no-go packet."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any, Mapping, Sequence

from .bundle import AuthorityBundle, load_authority_bundle
from .cases import resolve_reference_case, validate_case_meta, validate_stage_plan
from .reference_freeze import baseline_artifact_slug


BRINGUP_PACKET_SCHEMA_VERSION = "1.0.0"
BRINGUP_PACKET_NAME = "baseline_bringup_packet.json"
BRINGUP_SUMMARY_NAME = "baseline_bringup_summary.md"
LEGACY_OF12_PATH = "/".join(("", "opt", "openfoam12"))
SMOKE_METRICS_THRESHOLDS = {
    "alpha_min": -1e-6,
    "alpha_max": 1 + 1e-6,
    "alpha_integral_change_pct": 0.5,
}


def publish_baseline_bringup_packet(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    baseline_name: str = "Baseline B",
    build_case_roles: Sequence[str] | None = None,
    smoke_case_role: str = "R2",
    json_out: pathlib.Path | str | None = None,
    markdown_out: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    artifact_root_path = pathlib.Path(artifact_root)
    baseline_root = artifact_root_path / baseline_artifact_slug(baseline_name)
    if not baseline_root.exists():
        raise FileNotFoundError(
            f"baseline artifact root does not exist for {baseline_name}: {baseline_root}"
        )

    selected_build_roles = tuple(build_case_roles or bundle.ladder.ordered_case_ids)
    smoke_case = resolve_reference_case(bundle, case_role=smoke_case_role)
    json_path = (
        pathlib.Path(json_out)
        if json_out is not None
        else baseline_root / BRINGUP_PACKET_NAME
    )
    markdown_path = (
        pathlib.Path(markdown_out)
        if markdown_out is not None
        else baseline_root / BRINGUP_SUMMARY_NAME
    )
    case_records = []

    for case_role in selected_build_roles:
        record = _collect_case_record(
            bundle,
            baseline_root=baseline_root,
            baseline_name=baseline_name,
            case_role=case_role,
        )
        case_records.append(record)

    case_records_by_role = {record["case_role"]: record for record in case_records}
    referenced_paths = _collect_referenced_paths(case_records)
    tuple_ids = _collect_reviewed_tuple_ids(case_records)
    runtime_summary = _build_runtime_summary(case_records)
    legacy_path_hits = _scan_legacy_of12_paths(
        targets=[baseline_root, *sorted(referenced_paths)],
        skip_paths=_default_bringup_output_paths(
            baseline_root=baseline_root,
            json_path=json_path,
            markdown_path=markdown_path,
        ),
    )
    tuple_traceability = _build_tuple_traceability(case_records, tuple_ids=tuple_ids)
    smoke_record = _build_smoke_record(
        case_records_by_role.get(smoke_case_role),
        expected_case_id=smoke_case.frozen_id,
    )
    build_summary = _build_build_summary(case_records)
    backend_policy = _build_backend_policy(bundle, smoke_record=smoke_record)
    contingency_runtime = _uses_contingency_runtime(runtime_summary["runtime_bases"])
    decision = _build_decision(
        build_summary=build_summary,
        legacy_path_hits=legacy_path_hits,
        tuple_traceability=tuple_traceability,
        runtime_summary=runtime_summary,
        smoke_record=smoke_record,
        contingency_runtime=contingency_runtime,
    )

    packet = {
        "schema_version": BRINGUP_PACKET_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "phase_gate": "Phase 0",
        "baseline": baseline_name,
        "baseline_slug": baseline_artifact_slug(baseline_name),
        "artifact_root": baseline_root.as_posix(),
        "build_only_case_roles": list(selected_build_roles),
        "smoke_case_role": smoke_case_role,
        "build_summary": build_summary,
        "tuple_traceability": tuple_traceability,
        "runtime_summary": {
            **runtime_summary,
            "contingency_runtime_used": contingency_runtime,
        },
        "legacy_path_audit": {
            "legacy_literal": LEGACY_OF12_PATH,
            "hit_count": len(legacy_path_hits),
            "hits": legacy_path_hits,
        },
        "backend_policy": backend_policy,
        "cases": [_strip_referenced_paths(record) for record in case_records],
        "r2_smoke": smoke_record,
        "decision": decision,
    }

    _write_json(json_path, packet)
    markdown_path.write_text(render_baseline_bringup_summary(packet), encoding="utf-8")
    return packet


def render_baseline_bringup_summary(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Baseline B Bring-up Summary",
        "",
        f"- Baseline: `{packet['baseline']}`",
        f"- Phase gate: `{packet['phase_gate']}`",
        f"- Decision: `{packet['decision']['disposition']}`",
        f"- Proceed to nozzle freeze: `{packet['decision']['proceed_to_nozzle_freeze']}`",
        f"- Smoke status: `{packet['r2_smoke']['status']}`",
        "",
        "## Build-only readiness",
        "",
    ]
    for record in packet["cases"]:
        lines.append(
            f"- `{record['case_role']}` / `{record['case_id']}`: `{record['status']}`"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            *[f"- {reason}" for reason in packet["decision"]["reasons"]],
        ]
    )
    if not packet["decision"]["reasons"]:
        lines.append("- No blocking or review-only notes.")

    if packet["legacy_path_audit"]["hits"]:
        lines.extend(
            [
                "",
                "## Legacy path hits",
                "",
                *[
                    f"- `{hit['path']}`:{hit['line']}: {hit['snippet']}"
                    for hit in packet["legacy_path_audit"]["hits"]
                ],
            ]
        )

    return "\n".join(lines) + "\n"


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
        required=True,
        help="Artifact root containing per-baseline Phase 0 case bundles.",
    )
    parser.add_argument(
        "--baseline-name",
        default="Baseline B",
        help="Baseline to summarize. Defaults to 'Baseline B'.",
    )
    parser.add_argument(
        "--build-case-role",
        action="append",
        dest="build_case_roles",
        help="Optional case-role subset for build-only validation.",
    )
    parser.add_argument(
        "--smoke-case-role",
        default="R2",
        help="Case role to treat as the smoke execution case. Defaults to R2.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional JSON output path. Defaults under the baseline artifact root.",
    )
    parser.add_argument(
        "--markdown-out",
        type=pathlib.Path,
        default=None,
        help="Optional Markdown output path. Defaults under the baseline artifact root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    packet = publish_baseline_bringup_packet(
        bundle,
        artifact_root=args.artifact_root,
        baseline_name=args.baseline_name,
        build_case_roles=args.build_case_roles,
        smoke_case_role=args.smoke_case_role,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
    )
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


def _collect_case_record(
    bundle: AuthorityBundle,
    *,
    baseline_root: pathlib.Path,
    baseline_name: str,
    case_role: str,
) -> dict[str, Any]:
    resolved_case = resolve_reference_case(bundle, case_role=case_role)
    case_dir = baseline_root / resolved_case.frozen_id
    record = {
        "case_role": case_role,
        "case_id": resolved_case.frozen_id,
        "artifact_dir": case_dir.as_posix(),
        "status": "blocked",
        "notes": [],
        "referenced_paths": set(),
    }
    if not case_dir.exists():
        record["notes"].append("case artifact directory is missing")
        return record

    case_meta_path = case_dir / "case_meta.json"
    stage_plan_path = case_dir / "stage_plan.json"
    if not case_meta_path.exists() or not stage_plan_path.exists():
        if not case_meta_path.exists():
            record["notes"].append("missing case_meta.json")
        if not stage_plan_path.exists():
            record["notes"].append("missing stage_plan.json")
        return record

    try:
        case_meta = validate_case_meta(bundle, _read_json(case_meta_path))
        stage_plan = validate_stage_plan(bundle, _read_json(stage_plan_path))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        record["notes"].append(str(exc))
        return record

    record["runtime_base"] = case_meta.get("runtime_base")
    record["reviewed_source_tuple_id"] = case_meta.get("reviewed_source_tuple_id")
    record["resolved_pressure_backend"] = case_meta.get("resolved_pressure_backend")
    record["openfoam_bashrc_used"] = case_meta.get("openfoam_bashrc_used")
    record["provenance"] = dict(case_meta.get("provenance", {}))
    record["available_commands"] = dict(case_meta.get("available_commands", {}))

    alignment_notes = _bundle_alignment_notes(case_meta, stage_plan)
    if alignment_notes:
        record["notes"].extend(alignment_notes)
        return record
    if case_meta["baseline"] != baseline_name or stage_plan["baseline"] != baseline_name:
        record["notes"].append(
            f"baseline mismatch: expected {baseline_name!r}"
        )
        return record
    if stage_plan["case_id"] != resolved_case.frozen_id:
        record["notes"].append("stage plan case_id does not match the frozen case")
        return record
    if not str(record.get("runtime_base") or "").strip():
        record["notes"].append("missing resolved runtime_base traceability")

    for field_name in ("probe_payload", "host_env", "manifest_refs"):
        raw_ref = case_meta["provenance"].get(field_name)
        resolved_path = _resolve_reference_path(
            raw_ref,
            case_dir=case_dir,
            baseline_root=baseline_root,
        )
        if resolved_path is None or not resolved_path.exists():
            record["notes"].append(f"missing provenance artifact: {field_name}")
            continue
        record["referenced_paths"].add(resolved_path)

    metrics_path = case_dir / "metrics.json"
    if metrics_path.exists():
        try:
            record["metrics"] = _read_json(metrics_path)
        except json.JSONDecodeError as exc:
            record["notes"].append(f"invalid metrics.json: {exc}")

    verdict_path = case_dir / "baseline_verdict.json"
    if verdict_path.exists():
        try:
            record["baseline_verdict"] = _read_json(verdict_path)
        except json.JSONDecodeError as exc:
            record["notes"].append(f"invalid baseline_verdict.json: {exc}")

    patch_path = case_dir / "pressure_backend_patch.json"
    if patch_path.exists():
        try:
            record["pressure_backend_patch"] = _read_json(patch_path)
        except json.JSONDecodeError as exc:
            record["notes"].append(f"invalid pressure_backend_patch.json: {exc}")

    record["status"] = "ok" if not record["notes"] else "blocked"
    return record


def _build_smoke_record(
    case_record: Mapping[str, Any] | None,
    *,
    expected_case_id: str,
) -> dict[str, Any]:
    if case_record is None:
        return {
            "case_role": "R2",
            "case_id": expected_case_id,
            "status": "blocked",
            "notes": ["smoke case bundle is missing from the build summary"],
        }

    verdict = {
        "case_role": case_record["case_role"],
        "case_id": case_record["case_id"],
        "resolved_pressure_backend": case_record.get("resolved_pressure_backend"),
        "fallback_applied": "pressure_backend_patch" in case_record,
        "status": "blocked",
        "notes": list(case_record.get("notes", [])),
    }
    if case_record["status"] != "ok":
        verdict["notes"].append("smoke case is not build-ready")
        return verdict

    baseline_verdict = case_record.get("baseline_verdict")
    if isinstance(baseline_verdict, Mapping):
        status = str(baseline_verdict.get("status") or "").strip()
        if status:
            if status == "pass":
                verdict["status"] = "pass"
            elif status == "review":
                verdict["status"] = "review"
            else:
                verdict["status"] = "blocked"
                verdict["notes"].append(
                    f"baseline_verdict.json reported status {status!r}"
                )
            verdict["notes"].extend(_baseline_verdict_notes(baseline_verdict))
            if verdict["fallback_applied"] and verdict["status"] == "pass":
                verdict["status"] = "review"
                verdict["notes"].append("pressure backend fallback was applied")
            verdict["notes"] = _unique_in_order(verdict["notes"])
            return verdict

    metrics_payload = case_record.get("metrics")
    if not isinstance(metrics_payload, Mapping):
        verdict["notes"].append("metrics.json is required to evaluate the smoke case")
        return verdict

    metric_values: dict[str, float] = {}
    missing_metrics = []
    for metric_name in ("solver_ok", *SMOKE_METRICS_THRESHOLDS):
        metric_value = _metric_value(metrics_payload, metric_name)
        if metric_value is None:
            missing_metrics.append(metric_name)
            continue
        metric_values[metric_name] = metric_value
    if missing_metrics:
        verdict["notes"].append(
            "missing smoke metrics: " + ", ".join(sorted(missing_metrics))
        )
        return verdict

    if metric_values["solver_ok"] != 1:
        verdict["notes"].append("solver_ok must equal 1 for the smoke case")
        return verdict
    if metric_values["alpha_min"] < SMOKE_METRICS_THRESHOLDS["alpha_min"]:
        verdict["notes"].append("alpha_min fell below the allowed boundedness floor")
        return verdict
    if metric_values["alpha_max"] > SMOKE_METRICS_THRESHOLDS["alpha_max"]:
        verdict["notes"].append("alpha_max exceeded the allowed boundedness ceiling")
        return verdict
    if (
        metric_values["alpha_integral_change_pct"]
        > SMOKE_METRICS_THRESHOLDS["alpha_integral_change_pct"]
    ):
        verdict["notes"].append(
            "alpha_integral_change_pct exceeded the allowed conservation tolerance"
        )
        return verdict

    verdict["status"] = "review" if verdict["fallback_applied"] else "pass"
    if verdict["fallback_applied"]:
        verdict["notes"].append("pressure backend fallback was applied")
    verdict["metrics"] = metric_values
    return verdict


def _build_backend_policy(
    bundle: AuthorityBundle,
    *,
    smoke_record: Mapping[str, Any],
) -> dict[str, Any]:
    policy = dict(bundle.support.raw.get("backend_operational_policy", {}))
    return {
        "native_pressure_required_baseline_on_every_accepted_case": policy.get(
            "native_pressure_required_baseline_on_every_accepted_case"
        ),
        "native_pressure_required_baseline": policy.get(
            "native_pressure_required_baseline_on_every_accepted_case"
        ),
        "amgx_supported_secondary_backend_via_phase4_bridge_only": policy.get(
            "amgx_supported_secondary_backend_via_phase4_bridge_only"
        ),
        "amgx_production_claim_requires_device_direct": policy.get(
            "amgx_production_claim_requires_device_direct"
        ),
        "smoke_resolved_pressure_backend": smoke_record.get("resolved_pressure_backend"),
        "smoke_backend_fallback_applied": smoke_record.get("fallback_applied", False),
    }


def _build_build_summary(case_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocked_roles = [
        record["case_role"]
        for record in case_records
        if record.get("status") != "ok"
    ]
    return {
        "case_count": len(case_records),
        "build_ready": not blocked_roles,
        "blocked_case_roles": blocked_roles,
    }


def _build_tuple_traceability(
    case_records: Sequence[Mapping[str, Any]],
    *,
    tuple_ids: set[str],
) -> dict[str, Any]:
    missing_roles = [
        record["case_role"]
        for record in case_records
        if not str(record.get("reviewed_source_tuple_id") or "").strip()
    ]
    return {
        "consistent": len(tuple_ids) == 1 and not missing_roles,
        "reviewed_source_tuple_ids": sorted(tuple_ids),
        "missing_case_roles": missing_roles,
    }


def _build_runtime_summary(case_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runtime_bases: set[str] = set()
    missing_roles = []
    for record in case_records:
        runtime_base = str(record.get("runtime_base") or "").strip()
        if runtime_base:
            runtime_bases.add(runtime_base)
            continue
        missing_roles.append(record["case_role"])
    return {
        "runtime_bases": sorted(runtime_bases),
        "missing_runtime_case_roles": sorted(missing_roles),
    }


def _build_decision(
    *,
    build_summary: Mapping[str, Any],
    legacy_path_hits: Sequence[Mapping[str, Any]],
    tuple_traceability: Mapping[str, Any],
    runtime_summary: Mapping[str, Any],
    smoke_record: Mapping[str, Any],
    contingency_runtime: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    disposition = "go"

    if legacy_path_hits:
        disposition = "blocked"
        reasons.append(
            f"legacy {LEGACY_OF12_PATH} references remain in Baseline B artifacts"
        )
    if not tuple_traceability["consistent"]:
        disposition = "blocked"
        reasons.append("reviewed source tuple traceability is missing or inconsistent")
    if runtime_summary["missing_runtime_case_roles"]:
        disposition = "blocked"
        reasons.append(
            "runtime base traceability is missing for: "
            + ", ".join(runtime_summary["missing_runtime_case_roles"])
        )
    if not build_summary["build_ready"]:
        disposition = "blocked"
        reasons.append(
            "build-only validation is incomplete for: "
            + ", ".join(build_summary["blocked_case_roles"])
        )

    smoke_status = smoke_record.get("status")
    if smoke_status == "blocked":
        disposition = "blocked"
        reasons.extend(str(note) for note in smoke_record.get("notes", []))
    elif smoke_status == "review" and disposition != "blocked":
        disposition = "review"
        reasons.extend(str(note) for note in smoke_record.get("notes", []))

    if contingency_runtime and disposition == "go":
        disposition = "review"
        reasons.append(
            "Baseline B used a contingency runtime instead of the canonical SPUMA line"
        )

    return {
        "disposition": disposition,
        "proceed_to_nozzle_freeze": disposition == "go",
        "reasons": _unique_in_order(reasons),
    }


def _metric_value(metrics_payload: Mapping[str, Any], metric_name: str) -> float | None:
    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, Mapping):
        record = metrics.get(metric_name)
        if isinstance(record, Mapping):
            value = record.get("value")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        try:
            return float(record)
        except (TypeError, ValueError):
            return None

    record = metrics_payload.get(metric_name)
    if isinstance(record, Mapping):
        value = record.get("value")
    else:
        value = record
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_reference_path(
    raw_ref: Any,
    *,
    case_dir: pathlib.Path,
    baseline_root: pathlib.Path,
) -> pathlib.Path | None:
    if raw_ref is None:
        return None
    candidate = pathlib.Path(str(raw_ref))
    if candidate.is_absolute():
        return candidate
    for root in (case_dir, baseline_root, baseline_root.parent):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (baseline_root / candidate).resolve()


def _scan_legacy_of12_paths(
    *,
    targets: Sequence[pathlib.Path],
    skip_paths: set[pathlib.Path] | None = None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen_paths: set[pathlib.Path] = set()
    ignored_paths = {path.resolve() for path in (skip_paths or set())}
    for target in targets:
        if not target.exists():
            continue
        resolved_target = target.resolve()
        if resolved_target in seen_paths or resolved_target in ignored_paths:
            continue
        seen_paths.add(resolved_target)
        if resolved_target.is_dir():
            for path in sorted(path for path in resolved_target.rglob("*") if path.is_file()):
                if path.resolve() in ignored_paths:
                    continue
                hits.extend(_scan_file_for_legacy_path(path))
        else:
            hits.extend(_scan_file_for_legacy_path(resolved_target))
    return hits


def _collect_referenced_paths(
    case_records: Sequence[Mapping[str, Any]],
) -> set[pathlib.Path]:
    referenced_paths: set[pathlib.Path] = set()
    for record in case_records:
        referenced_paths.update(record.get("referenced_paths", set()))
    return referenced_paths


def _collect_reviewed_tuple_ids(
    case_records: Sequence[Mapping[str, Any]],
) -> set[str]:
    tuple_ids: set[str] = set()
    for record in case_records:
        tuple_id = str(record.get("reviewed_source_tuple_id") or "").strip()
        if tuple_id:
            tuple_ids.add(tuple_id)
    return tuple_ids


def _default_bringup_output_paths(
    *,
    baseline_root: pathlib.Path,
    json_path: pathlib.Path,
    markdown_path: pathlib.Path,
) -> set[pathlib.Path]:
    return {
        json_path.resolve(),
        markdown_path.resolve(),
        (baseline_root / BRINGUP_PACKET_NAME).resolve(),
        (baseline_root / BRINGUP_SUMMARY_NAME).resolve(),
    }


def _scan_file_for_legacy_path(path: pathlib.Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if LEGACY_OF12_PATH not in text:
        return []

    hits = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if LEGACY_OF12_PATH not in line:
            continue
        hits.append(
            {
                "path": path.as_posix(),
                "line": line_number,
                "snippet": line.strip(),
            }
        )
    return hits


def _uses_contingency_runtime(runtime_bases: Sequence[str]) -> bool:
    normalized = [runtime.lower() for runtime in runtime_bases]
    return bool(normalized) and any("spuma" not in runtime for runtime in normalized)


def _bundle_alignment_notes(
    case_meta: Mapping[str, Any],
    stage_plan: Mapping[str, Any],
) -> list[str]:
    notes = []
    for field_name in (
        "case_id",
        "case_role",
        "baseline",
        "runtime_base",
        "reviewed_source_tuple_id",
    ):
        if case_meta.get(field_name) != stage_plan.get(field_name):
            notes.append(
                f"case bundle {field_name} mismatch between case_meta.json and stage_plan.json"
            )
    for field_name in ("probe_payload", "host_env", "manifest_refs"):
        case_meta_provenance = dict(case_meta.get("provenance", {}))
        stage_plan_provenance = dict(stage_plan.get("provenance", {}))
        if case_meta_provenance.get(field_name) != stage_plan_provenance.get(field_name):
            notes.append(
                f"case bundle provenance {field_name} mismatch between case_meta.json and stage_plan.json"
            )
    if case_meta.get("io_normalization") != stage_plan.get("io_normalization"):
        notes.append(
            "case bundle io_normalization mismatch between case_meta.json and stage_plan.json"
        )
    return _unique_in_order(notes)


def _baseline_verdict_notes(baseline_verdict: Mapping[str, Any]) -> list[str]:
    notes = []
    for field_name in ("notes", "reasons"):
        raw_value = baseline_verdict.get(field_name)
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            notes.extend(str(item) for item in raw_value if str(item).strip())
    checks = baseline_verdict.get("checks")
    if isinstance(checks, Sequence) and not isinstance(checks, (str, bytes)):
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            status = str(check.get("status") or "").strip()
            if status not in {"fail", "pending"}:
                continue
            check_id = str(check.get("check_id") or "baseline_check").strip()
            details = str(check.get("details") or "").strip()
            notes.append(f"{check_id}: {details}" if details else check_id)
    return _unique_in_order(notes)


def _strip_referenced_paths(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("referenced_paths", None)
    return payload


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timestamp_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _unique_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
