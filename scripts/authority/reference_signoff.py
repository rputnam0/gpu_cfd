"""Phase 0 Baseline A/B comparison and sign-off packet publishing helpers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from typing import Any, Mapping, Sequence

from .bundle import AuthorityBundle, load_authority_bundle
from .cases import resolve_reference_case, validate_case_meta, validate_stage_plan
from .reference_freeze import (
    PHASE0_BUNDLE_MANIFEST_NAME,
    PHASE0_CONTROL_INDEX_NAME,
    PHASE0_VERDICT_NAME,
    baseline_artifact_slug,
    publish_baseline_control_index,
)


PHASE0_SIGNOFF_SCHEMA_VERSION = "1.0.0"
PHASE0_COMPARE_JSON_NAME = "compare.json"
PHASE0_COMPARE_MARKDOWN_NAME = "compare.md"
PHASE0_ARCHIVE_INDEX_NAME = "archive_index.json"
PHASE0_PROVENANCE_MANIFEST_NAME = "provenance_manifest.json"
DEFAULT_SIGNOFF_CASE_ROLES = ("R1", "R0")


def publish_phase0_signoff_packet(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path | str,
    baseline_a_name: str = "Baseline A",
    baseline_b_name: str = "Baseline B",
    case_roles: Sequence[str] | None = None,
    reviewer: str | None = None,
    review_status: str = "pending",
    review_notes: Sequence[str] | None = None,
    json_out: pathlib.Path | str | None = None,
    markdown_out: pathlib.Path | str | None = None,
    archive_index_out: pathlib.Path | str | None = None,
    provenance_manifest_out: pathlib.Path | str | None = None,
) -> dict[str, Any]:
    artifact_root_path = pathlib.Path(artifact_root)
    selected_case_roles = tuple(case_roles or DEFAULT_SIGNOFF_CASE_ROLES)
    comparison_root = artifact_root_path / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)

    baseline_a_index = publish_baseline_control_index(
        bundle,
        artifact_root=artifact_root_path,
        baseline_name=baseline_a_name,
        case_roles=selected_case_roles,
    )
    baseline_b_index = publish_baseline_control_index(
        bundle,
        artifact_root=artifact_root_path,
        baseline_name=baseline_b_name,
        case_roles=selected_case_roles,
    )

    baseline_a_cases = {
        str(case_record["case_role"]): case_record for case_record in baseline_a_index["cases"]
    }
    baseline_b_cases = {
        str(case_record["case_role"]): case_record for case_record in baseline_b_index["cases"]
    }
    case_reports = []
    provenance_cases = []
    for case_role in selected_case_roles:
        resolved = resolve_reference_case(bundle, case_role=case_role)
        baseline_a_payload = _load_case_inputs(
            bundle,
            artifact_root=artifact_root_path,
            baseline_name=baseline_a_name,
            case_record=baseline_a_cases[case_role],
        )
        baseline_b_payload = _load_case_inputs(
            bundle,
            artifact_root=artifact_root_path,
            baseline_name=baseline_b_name,
            case_record=baseline_b_cases[case_role],
        )
        case_reports.append(
            _build_case_report(
                case_role=case_role,
                expected_case_id=resolved.frozen_id,
                baseline_a=baseline_a_payload,
                baseline_b=baseline_b_payload,
            )
        )
        provenance_cases.append(
            _build_provenance_record(
                case_role=case_role,
                expected_case_id=resolved.frozen_id,
                baseline_a=baseline_a_payload,
                baseline_b=baseline_b_payload,
            )
        )

    status = _summarize_status(case_report["status"] for case_report in case_reports)
    human_signoff = {
        "status": review_status,
        "reviewer": reviewer,
        "recorded_at": _timestamp_now() if reviewer else None,
        "notes": [str(note) for note in (review_notes or ()) if str(note).strip()],
        "required": True,
    }

    packet = {
        "schema_version": PHASE0_SIGNOFF_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "phase_gate": "Phase 0",
        "status": status,
        "compared_case_roles": list(selected_case_roles),
        "baselines": [
            {
                "baseline": baseline_a_name,
                "baseline_slug": baseline_artifact_slug(baseline_a_name),
                "control_index": PHASE0_CONTROL_INDEX_NAME,
                "status": baseline_a_index["status"],
            },
            {
                "baseline": baseline_b_name,
                "baseline_slug": baseline_artifact_slug(baseline_b_name),
                "control_index": PHASE0_CONTROL_INDEX_NAME,
                "status": baseline_b_index["status"],
            },
        ],
        "authority_revisions": dict(baseline_a_index["authority_revisions"]),
        "archive_index": PHASE0_ARCHIVE_INDEX_NAME,
        "provenance_manifest": PHASE0_PROVENANCE_MANIFEST_NAME,
        "human_signoff": human_signoff,
        "cases": case_reports,
    }

    archive_index = {
        "schema_version": PHASE0_SIGNOFF_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "phase_gate": "Phase 0",
        "comparison_root": comparison_root.as_posix(),
        "outputs": {
            "compare_json": PHASE0_COMPARE_JSON_NAME,
            "compare_markdown": PHASE0_COMPARE_MARKDOWN_NAME,
            "archive_index": PHASE0_ARCHIVE_INDEX_NAME,
            "provenance_manifest": PHASE0_PROVENANCE_MANIFEST_NAME,
        },
        "baselines": [
            {
                "baseline": baseline_a_name,
                "baseline_slug": baseline_artifact_slug(baseline_a_name),
                "control_index_path": (
                    artifact_root_path
                    / baseline_artifact_slug(baseline_a_name)
                    / PHASE0_CONTROL_INDEX_NAME
                ).as_posix(),
                "cases": [
                    _case_archive_record(case_record)
                    for case_record in baseline_a_index["cases"]
                ],
            },
            {
                "baseline": baseline_b_name,
                "baseline_slug": baseline_artifact_slug(baseline_b_name),
                "control_index_path": (
                    artifact_root_path
                    / baseline_artifact_slug(baseline_b_name)
                    / PHASE0_CONTROL_INDEX_NAME
                ).as_posix(),
                "cases": [
                    _case_archive_record(case_record)
                    for case_record in baseline_b_index["cases"]
                ],
            },
        ],
    }
    provenance_manifest = {
        "schema_version": PHASE0_SIGNOFF_SCHEMA_VERSION,
        "generated_at": _timestamp_now(),
        "case_count": len(provenance_cases),
        "cases": provenance_cases,
    }

    json_path = (
        pathlib.Path(json_out)
        if json_out is not None
        else comparison_root / PHASE0_COMPARE_JSON_NAME
    )
    markdown_path = (
        pathlib.Path(markdown_out)
        if markdown_out is not None
        else comparison_root / PHASE0_COMPARE_MARKDOWN_NAME
    )
    archive_index_path = (
        pathlib.Path(archive_index_out)
        if archive_index_out is not None
        else comparison_root / PHASE0_ARCHIVE_INDEX_NAME
    )
    provenance_manifest_path = (
        pathlib.Path(provenance_manifest_out)
        if provenance_manifest_out is not None
        else comparison_root / PHASE0_PROVENANCE_MANIFEST_NAME
    )

    _write_json(json_path, packet)
    markdown_path.write_text(render_phase0_signoff_summary(packet), encoding="utf-8")
    _write_json(archive_index_path, archive_index)
    _write_json(provenance_manifest_path, provenance_manifest)
    return packet


def render_phase0_signoff_summary(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 0 Compare Report",
        "",
        f"- Phase gate: `{packet['phase_gate']}`",
        f"- Status: `{packet['status']}`",
        f"- Compared case roles: `{', '.join(packet['compared_case_roles'])}`",
        f"- Human sign-off: `{packet['human_signoff']['status']}`",
        "",
    ]
    for case_report in packet["cases"]:
        lines.extend(
            [
                f"## `{case_report['case_role']}` / `{case_report['case_id']}`",
                "",
                f"- Status: `{case_report['status']}`",
                f"- Baseline A bundle: `{case_report['baseline_a']['bundle_status']}`",
                f"- Baseline B bundle: `{case_report['baseline_b']['bundle_status']}`",
                "",
                "### Checks",
                "",
            ]
        )
        for check in case_report["checks"]:
            details = f" ({check['details']})" if check.get("details") else ""
            lines.append(
                f"- `{check['check_id']}` [{check['gate_class']}] -> `{check['status']}`{details}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
        "--baseline-a-name",
        default="Baseline A",
        help="Reference baseline name. Defaults to 'Baseline A'.",
    )
    parser.add_argument(
        "--baseline-b-name",
        default="Baseline B",
        help="Comparison baseline name. Defaults to 'Baseline B'.",
    )
    parser.add_argument(
        "--case-role",
        action="append",
        dest="case_roles",
        help="Optional nozzle case-role subset. Defaults to R1 and R0.",
    )
    parser.add_argument(
        "--reviewer",
        default=None,
        help="Optional reviewer recorded in the sign-off packet.",
    )
    parser.add_argument(
        "--review-status",
        default="pending",
        help="Human sign-off status. Defaults to 'pending'.",
    )
    parser.add_argument(
        "--review-note",
        action="append",
        dest="review_notes",
        help="Optional human sign-off note to record in the packet.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional compare JSON output path.",
    )
    parser.add_argument(
        "--markdown-out",
        type=pathlib.Path,
        default=None,
        help="Optional compare Markdown output path.",
    )
    parser.add_argument(
        "--archive-index-out",
        type=pathlib.Path,
        default=None,
        help="Optional archive index output path.",
    )
    parser.add_argument(
        "--provenance-manifest-out",
        type=pathlib.Path,
        default=None,
        help="Optional provenance manifest output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the comparison packet JSON to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    bundle = load_authority_bundle(args.root)
    packet = publish_phase0_signoff_packet(
        bundle,
        artifact_root=args.artifact_root,
        baseline_a_name=args.baseline_a_name,
        baseline_b_name=args.baseline_b_name,
        case_roles=args.case_roles,
        reviewer=args.reviewer,
        review_status=args.review_status,
        review_notes=args.review_notes,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
        archive_index_out=args.archive_index_out,
        provenance_manifest_out=args.provenance_manifest_out,
    )
    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    else:
        print(render_phase0_signoff_summary(packet))
    return 0


def _build_case_report(
    *,
    case_role: str,
    expected_case_id: str,
    baseline_a: Mapping[str, Any],
    baseline_b: Mapping[str, Any],
) -> dict[str, Any]:
    checks = [
        _bundle_status_check("baseline_a_bundle_status", baseline_a, "Baseline A"),
        _bundle_status_check("baseline_b_bundle_status", baseline_b, "Baseline B"),
        _exact_check(
            "case_id_exact",
            expected=expected_case_id,
            baseline_a=baseline_a["case_meta"]["case_id"],
            baseline_b=baseline_b["case_meta"]["case_id"],
            gate_class="hard",
        ),
        _truthy_check(
            "patch_schema_exact",
            gate_class="hard",
            observations=(
                bool(baseline_a["metrics"].get("patch_schema_exact")),
                bool(baseline_b["metrics"].get("patch_schema_exact")),
            ),
            details=None,
        ),
        _exact_check(
            "patch_schema_cross_baseline_exact",
            expected=baseline_a["build_fingerprint"].get("patch_schema"),
            baseline_a=baseline_a["build_fingerprint"].get("patch_schema"),
            baseline_b=baseline_b["build_fingerprint"].get("patch_schema"),
            gate_class="hard",
        ),
        _exact_check(
            "solver_ok_exact",
            expected=1,
            baseline_a=baseline_a["metrics"].get("solver_ok"),
            baseline_b=baseline_b["metrics"].get("solver_ok"),
            gate_class="hard",
        ),
        _threshold_check(
            "mass_imbalance_pct_within_threshold",
            threshold=1.0,
            baseline_a=baseline_a["metrics"].get("mass_imbalance_pct"),
            baseline_b=baseline_b["metrics"].get("mass_imbalance_pct"),
            gate_class="hard",
        ),
    ]

    if case_role == "R1":
        checks.extend(
            [
                _truthy_check(
                    "startup_provenance_exact",
                    gate_class="hard",
                    observations=(
                        baseline_a["metrics"].get("startup_provenance_exact"),
                        baseline_b["metrics"].get("startup_provenance_exact"),
                    ),
                    details=None,
                ),
                _truthy_check(
                    "precondition_field_copy_provenance_exact",
                    gate_class="hard",
                    observations=(
                        baseline_a["metrics"].get("precondition_field_copy_provenance_exact"),
                        baseline_b["metrics"].get("precondition_field_copy_provenance_exact"),
                    ),
                    details=None,
                ),
                _relative_threshold_check(
                    "water_flow_cfd_gph_within_threshold",
                    threshold_pct=3.0,
                    baseline_a=baseline_a["metrics"].get("water_flow_cfd_gph"),
                    baseline_b=baseline_b["metrics"].get("water_flow_cfd_gph"),
                    gate_class="hard",
                ),
                _relative_or_absolute_check(
                    "cd_mass_cfd_within_threshold",
                    threshold_pct=3.0,
                    threshold_abs=0.03,
                    baseline_a=baseline_a["metrics"].get("cd_mass_cfd"),
                    baseline_b=baseline_b["metrics"].get("cd_mass_cfd"),
                    gate_class="hard",
                ),
                _review_threshold_check(
                    "air_core_area_ratio_within_threshold",
                    threshold_abs=0.05,
                    threshold_pct=10.0,
                    baseline_a=baseline_a["metrics"].get("air_core_area_ratio"),
                    baseline_b=baseline_b["metrics"].get("air_core_area_ratio"),
                ),
                _review_threshold_check(
                    "swirl_number_proxy_within_threshold",
                    threshold_abs=None,
                    threshold_pct=10.0,
                    baseline_a=baseline_a["metrics"].get("swirl_number_proxy"),
                    baseline_b=baseline_b["metrics"].get("swirl_number_proxy"),
                ),
                _review_threshold_check(
                    "recirculation_fraction_within_threshold",
                    threshold_abs=0.05,
                    threshold_pct=None,
                    baseline_a=baseline_a["metrics"].get("recirculation_fraction"),
                    baseline_b=baseline_b["metrics"].get("recirculation_fraction"),
                ),
            ]
        )
    elif case_role == "R0":
        checks.extend(
            [
                _truthy_check(
                    "resolved_solver_family_recorded",
                    gate_class="hard",
                    observations=(
                        baseline_a["case_meta"].get("resolved_vof_solver_exec"),
                        baseline_b["case_meta"].get("resolved_vof_solver_exec"),
                    ),
                    details=None,
                ),
                _truthy_check(
                    "startup_provenance_exact",
                    gate_class="hard",
                    observations=(
                        baseline_a["metrics"].get("startup_provenance_exact"),
                        baseline_b["metrics"].get("startup_provenance_exact"),
                    ),
                    details=None,
                ),
                _truthy_check(
                    "resolved_direct_slot_numerics_exact",
                    gate_class="hard",
                    observations=(
                        baseline_a["metrics"].get("resolved_direct_slot_numerics_exact"),
                        baseline_b["metrics"].get("resolved_direct_slot_numerics_exact"),
                    ),
                    details=None,
                ),
                _relative_threshold_check(
                    "water_flow_cfd_gph_within_threshold",
                    threshold_pct=2.0,
                    baseline_a=baseline_a["metrics"].get("water_flow_cfd_gph"),
                    baseline_b=baseline_b["metrics"].get("water_flow_cfd_gph"),
                    gate_class="hard",
                ),
                _relative_or_absolute_check(
                    "cd_mass_cfd_within_threshold",
                    threshold_pct=2.0,
                    threshold_abs=0.02,
                    baseline_a=baseline_a["metrics"].get("cd_mass_cfd"),
                    baseline_b=baseline_b["metrics"].get("cd_mass_cfd"),
                    gate_class="hard",
                ),
                _spray_angle_check(
                    baseline_a=baseline_a,
                    baseline_b=baseline_b,
                ),
                _review_threshold_check(
                    "air_core_area_ratio_within_threshold",
                    threshold_abs=0.05,
                    threshold_pct=10.0,
                    baseline_a=baseline_a["metrics"].get("air_core_area_ratio"),
                    baseline_b=baseline_b["metrics"].get("air_core_area_ratio"),
                ),
                _review_threshold_check(
                    "swirl_number_proxy_within_threshold",
                    threshold_abs=None,
                    threshold_pct=10.0,
                    baseline_a=baseline_a["metrics"].get("swirl_number_proxy"),
                    baseline_b=baseline_b["metrics"].get("swirl_number_proxy"),
                ),
                _review_threshold_check(
                    "recirculation_fraction_within_threshold",
                    threshold_abs=0.05,
                    threshold_pct=None,
                    baseline_a=baseline_a["metrics"].get("recirculation_fraction"),
                    baseline_b=baseline_b["metrics"].get("recirculation_fraction"),
                ),
            ]
        )

    status = _summarize_status(check["status"] for check in checks)
    return {
        "case_role": case_role,
        "case_id": expected_case_id,
        "status": status,
        "baseline_a": _baseline_summary(baseline_a),
        "baseline_b": _baseline_summary(baseline_b),
        "checks": checks,
    }


def _build_provenance_record(
    *,
    case_role: str,
    expected_case_id: str,
    baseline_a: Mapping[str, Any],
    baseline_b: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_role": case_role,
        "case_id": expected_case_id,
        "baselines": [
            _baseline_provenance_record("Baseline A", baseline_a),
            _baseline_provenance_record("Baseline B", baseline_b),
        ],
    }


def _load_case_inputs(
    bundle: AuthorityBundle,
    *,
    artifact_root: pathlib.Path,
    baseline_name: str,
    case_record: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_dir = pathlib.Path(str(case_record["artifact_dir"]))
    baseline_root = artifact_root / baseline_artifact_slug(baseline_name)
    case_meta = validate_case_meta(bundle, _read_json(artifact_dir / "case_meta.json"))
    stage_plan = validate_stage_plan(bundle, _read_json(artifact_dir / "stage_plan.json"))
    metrics_payload = _read_json(artifact_dir / "metrics.json")
    metric_values, metric_sources = _normalize_metrics(metrics_payload)
    return {
        "baseline": baseline_name,
        "baseline_slug": baseline_artifact_slug(baseline_name),
        "artifact_dir": artifact_dir.as_posix(),
        "case_meta": case_meta,
        "stage_plan": stage_plan,
        "build_fingerprint": _read_json(artifact_dir / "build_fingerprint.json"),
        "bundle_manifest": _read_json(artifact_dir / PHASE0_BUNDLE_MANIFEST_NAME),
        "baseline_verdict": _read_json(artifact_dir / PHASE0_VERDICT_NAME),
        "metrics_payload": metrics_payload,
        "metrics": metric_values,
        "metric_sources": metric_sources,
        "angle_source": metrics_payload.get("angle_source"),
        "time_windows": dict(metrics_payload.get("time_windows") or {}),
        "resolved_provenance": {
            field_name: _resolve_reference_path(
                case_meta.get("provenance", {}).get(field_name),
                case_dir=artifact_dir,
                baseline_root=baseline_root,
            ).as_posix()
            if _resolve_reference_path(
                case_meta.get("provenance", {}).get(field_name),
                case_dir=artifact_dir,
                baseline_root=baseline_root,
            )
            else None
            for field_name in ("probe_payload", "host_env", "manifest_refs")
        },
    }


def _bundle_status_check(
    check_id: str,
    baseline_payload: Mapping[str, Any],
    baseline_label: str,
) -> dict[str, Any]:
    status = str(baseline_payload["baseline_verdict"].get("status") or "").strip() or "fail"
    if status == "pass":
        check_status = "pass"
        details = None
    elif status == "review":
        check_status = "review"
        details = f"{baseline_label} baseline_verdict.json is review-only"
    else:
        check_status = "fail"
        details = f"{baseline_label} baseline_verdict.json reported {status!r}"
    return {
        "check_id": check_id,
        "gate_class": "hard",
        "status": check_status,
        "baseline_a": baseline_payload["baseline"] if baseline_label == "Baseline A" else None,
        "baseline_b": baseline_payload["baseline"] if baseline_label == "Baseline B" else None,
        "details": details,
    }


def _exact_check(
    check_id: str,
    *,
    expected: Any,
    baseline_a: Any,
    baseline_b: Any,
    gate_class: str,
) -> dict[str, Any]:
    passed = baseline_a == expected and baseline_b == expected
    return {
        "check_id": check_id,
        "gate_class": gate_class,
        "status": "pass" if passed else "fail",
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "details": None if passed else f"expected both baselines to equal {expected!r}",
    }


def _truthy_check(
    check_id: str,
    *,
    gate_class: str,
    observations: Sequence[Any],
    details: str | None,
) -> dict[str, Any]:
    observation_list = list(observations)
    passed = all(bool(observation) for observation in observation_list)
    return {
        "check_id": check_id,
        "gate_class": gate_class,
        "status": "pass" if passed else "fail",
        "baseline_a": observation_list[0] if observation_list else None,
        "baseline_b": observation_list[1] if len(observation_list) > 1 else None,
        "details": details if not passed else None,
    }


def _threshold_check(
    check_id: str,
    *,
    threshold: float,
    baseline_a: Any,
    baseline_b: Any,
    gate_class: str,
) -> dict[str, Any]:
    try:
        baseline_a_value = float(baseline_a)
        baseline_b_value = float(baseline_b)
    except (TypeError, ValueError):
        return {
            "check_id": check_id,
            "gate_class": gate_class,
            "status": "fail",
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "details": "missing numeric observation",
        }
    passed = baseline_a_value <= threshold and baseline_b_value <= threshold
    return {
        "check_id": check_id,
        "gate_class": gate_class,
        "status": "pass" if passed else "fail",
        "baseline_a": baseline_a_value,
        "baseline_b": baseline_b_value,
        "details": None if passed else f"expected both baselines <= {threshold}",
    }


def _relative_threshold_check(
    check_id: str,
    *,
    threshold_pct: float,
    baseline_a: Any,
    baseline_b: Any,
    gate_class: str,
) -> dict[str, Any]:
    try:
        baseline_a_value = float(baseline_a)
        baseline_b_value = float(baseline_b)
    except (TypeError, ValueError):
        return {
            "check_id": check_id,
            "gate_class": gate_class,
            "status": "fail",
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "details": "missing numeric observation",
        }
    relative_diff_pct = _relative_diff_pct(baseline_a_value, baseline_b_value)
    passed = relative_diff_pct <= threshold_pct
    return {
        "check_id": check_id,
        "gate_class": gate_class,
        "status": "pass" if passed else "fail",
        "baseline_a": baseline_a_value,
        "baseline_b": baseline_b_value,
        "relative_diff_pct": relative_diff_pct,
        "details": None if passed else f"relative diff {relative_diff_pct:.3f}% exceeds {threshold_pct:.3f}%",
    }


def _relative_or_absolute_check(
    check_id: str,
    *,
    threshold_pct: float,
    threshold_abs: float,
    baseline_a: Any,
    baseline_b: Any,
    gate_class: str,
) -> dict[str, Any]:
    try:
        baseline_a_value = float(baseline_a)
        baseline_b_value = float(baseline_b)
    except (TypeError, ValueError):
        return {
            "check_id": check_id,
            "gate_class": gate_class,
            "status": "fail",
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "details": "missing numeric observation",
        }
    absolute_diff = abs(baseline_a_value - baseline_b_value)
    relative_diff_pct = _relative_diff_pct(baseline_a_value, baseline_b_value)
    passed = relative_diff_pct <= threshold_pct or absolute_diff <= threshold_abs
    return {
        "check_id": check_id,
        "gate_class": gate_class,
        "status": "pass" if passed else "fail",
        "baseline_a": baseline_a_value,
        "baseline_b": baseline_b_value,
        "absolute_diff": absolute_diff,
        "relative_diff_pct": relative_diff_pct,
        "details": None
        if passed
        else (
            f"relative diff {relative_diff_pct:.3f}% exceeds {threshold_pct:.3f}% "
            f"and absolute diff {absolute_diff:.5f} exceeds {threshold_abs:.5f}"
        ),
    }


def _review_threshold_check(
    check_id: str,
    *,
    threshold_abs: float | None,
    threshold_pct: float | None,
    baseline_a: Any,
    baseline_b: Any,
) -> dict[str, Any]:
    try:
        baseline_a_value = float(baseline_a)
        baseline_b_value = float(baseline_b)
    except (TypeError, ValueError):
        return {
            "check_id": check_id,
            "gate_class": "review",
            "status": "review",
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "details": "missing numeric observation",
        }
    absolute_diff = abs(baseline_a_value - baseline_b_value)
    relative_diff_pct = _relative_diff_pct(baseline_a_value, baseline_b_value)
    abs_passed = threshold_abs is None or absolute_diff <= threshold_abs
    rel_passed = threshold_pct is None or relative_diff_pct <= threshold_pct
    passed = abs_passed and rel_passed
    details = None
    if not passed:
        if threshold_abs is not None and threshold_pct is not None:
            details = (
                f"absolute diff {absolute_diff:.5f} exceeds {threshold_abs:.5f} or "
                f"relative diff {relative_diff_pct:.3f}% exceeds {threshold_pct:.3f}%"
            )
        elif threshold_abs is not None:
            details = f"absolute diff {absolute_diff:.5f} exceeds {threshold_abs:.5f}"
        else:
            details = f"relative diff {relative_diff_pct:.3f}% exceeds {threshold_pct:.3f}%"
    return {
        "check_id": check_id,
        "gate_class": "review",
        "status": "pass" if passed else "review",
        "baseline_a": baseline_a_value,
        "baseline_b": baseline_b_value,
        "absolute_diff": absolute_diff,
        "relative_diff_pct": relative_diff_pct,
        "details": details,
    }


def _spray_angle_check(
    *,
    baseline_a: Mapping[str, Any],
    baseline_b: Mapping[str, Any],
) -> dict[str, Any]:
    source_a = str(baseline_a.get("angle_source") or "").strip()
    source_b = str(baseline_b.get("angle_source") or "").strip()
    if source_a != "geometric_a" or source_b != "geometric_a":
        return {
            "check_id": "spray_angle_cfd_deg_within_threshold",
            "gate_class": "review",
            "status": "review",
            "baseline_a": baseline_a["metrics"].get("spray_angle_cfd_deg"),
            "baseline_b": baseline_b["metrics"].get("spray_angle_cfd_deg"),
            "details": (
                "spray-angle comparison is review-only because sources are "
                f"{source_a or 'missing'} vs {source_b or 'missing'}"
            ),
        }
    return _review_threshold_check(
        "spray_angle_cfd_deg_within_threshold",
        threshold_abs=5.0,
        threshold_pct=None,
        baseline_a=baseline_a["metrics"].get("spray_angle_cfd_deg"),
        baseline_b=baseline_b["metrics"].get("spray_angle_cfd_deg"),
    )


def _baseline_summary(baseline_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": baseline_payload["baseline"],
        "baseline_slug": baseline_payload["baseline_slug"],
        "artifact_dir": baseline_payload["artifact_dir"],
        "bundle_status": baseline_payload["bundle_manifest"]["status"],
        "baseline_verdict_status": baseline_payload["baseline_verdict"]["status"],
    }


def _baseline_provenance_record(
    baseline_label: str,
    baseline_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "baseline": baseline_label,
        "baseline_slug": baseline_payload["baseline_slug"],
        "artifact_dir": baseline_payload["artifact_dir"],
        "runtime_base": baseline_payload["case_meta"].get("runtime_base"),
        "reviewed_source_tuple_id": baseline_payload["case_meta"].get("reviewed_source_tuple_id"),
        "resolved_vof_solver_exec": baseline_payload["case_meta"].get("resolved_vof_solver_exec"),
        "resolved_pressure_backend": baseline_payload["case_meta"].get("resolved_pressure_backend"),
        "provenance_refs": baseline_payload["resolved_provenance"],
        "metric_sources": baseline_payload["metric_sources"],
        "time_windows": baseline_payload["time_windows"],
    }


def _case_archive_record(case_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_role": case_record["case_role"],
        "case_id": case_record["case_id"],
        "artifact_dir": case_record["artifact_dir"],
        "bundle_manifest": case_record["bundle_manifest"],
        "baseline_verdict": case_record["baseline_verdict"],
        "status": case_record["status"],
    }


def _normalize_metrics(metrics_payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    raw_metrics = metrics_payload.get("metrics", metrics_payload)
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("metrics.json metrics payload must be an object")
    metric_values = {}
    metric_sources = {}
    for metric_name, metric_record in raw_metrics.items():
        if isinstance(metric_record, Mapping):
            metric_values[metric_name] = metric_record.get("value")
            source = metric_record.get("metric_source") or metric_record.get("source")
            if source is not None:
                metric_sources[metric_name] = str(source)
        else:
            metric_values[metric_name] = metric_record
    return metric_values, metric_sources


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
    return None


def _relative_diff_pct(baseline_a: float, baseline_b: float) -> float:
    if baseline_a == 0.0:
        return 0.0 if baseline_b == 0.0 else float("inf")
    return abs(baseline_b - baseline_a) / abs(baseline_a) * 100.0


def _summarize_status(statuses: Sequence[str] | Any) -> str:
    normalized = tuple(statuses)
    if any(status == "fail" for status in normalized):
        return "fail"
    if any(status != "pass" for status in normalized):
        return "review"
    return "pass"


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


if __name__ == "__main__":
    raise SystemExit(main())
