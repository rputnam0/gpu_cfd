from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any

from scripts.authority import reference_signoff
from scripts.authority import (
    PHASE0_ARCHIVE_INDEX_NAME,
    PHASE0_COMPARE_JSON_NAME,
    PHASE0_COMPARE_MARKDOWN_NAME,
    PHASE0_PROVENANCE_MANIFEST_NAME,
    baseline_artifact_slug,
    build_reference_io_normalization_payload,
    load_authority_bundle,
    publish_phase0_signoff_packet,
    reference_io_overlay_command,
    resolve_reference_case,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sample_case_meta_payload(
    bundle: Any,
    *,
    case_role: str,
    baseline: str,
    runtime_base: str,
    source_tuple_id: str,
    requested_vof_solver_mode: str = "vof_transient_preconditioned",
    resolved_vof_solver_exec: str = "incompressibleVoF",
    resolved_pressure_backend: str = "native_cpu",
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    baseline_slug = baseline_artifact_slug(baseline)
    return {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "ladder_position": resolved.ladder_position,
        "phase_gates": list(resolved.phase_gates),
        "baseline": baseline,
        "runtime_base": runtime_base,
        "reviewed_source_tuple_id": source_tuple_id,
        "requested_vof_solver_mode": requested_vof_solver_mode,
        "resolved_vof_solver_exec": resolved_vof_solver_exec,
        "resolved_pressure_backend": resolved_pressure_backend,
        "openfoam_bashrc_used": f"/envs/{baseline_slug}/etc/bashrc",
        "available_commands": {
            "foamRun": f"/envs/{baseline_slug}/bin/foamRun",
            "checkMesh": f"/envs/{baseline_slug}/bin/checkMesh",
            "setFields": f"/envs/{baseline_slug}/bin/setFields",
        },
        "mesh_full_360": 1,
        "mesh_resolution_scale": 2.0,
        "hydraulic_domain_mode": "internal_only" if case_role == "R1" else "full_domain",
        "near_field_radius_d": 10.0,
        "near_field_length_d": 20.0,
        "steady_end_time_iter": 10,
        "steady_write_interval_iter": 5,
        "steady_turbulence_model": "kOmegaSST",
        "vof_turbulence_model": "laminar",
        "delta_t_s": 1e-8,
        "write_interval_s": 1e-8,
        "end_time_s": 2e-8,
        "max_co": 0.05,
        "max_alpha_co": 0.01,
        "resolved_direct_slot_numerics": {
            "slot_count": 6,
            "nominal_slot_width_mm": 0.64,
        },
        "startup_fill_extension_d": 0.0,
        "air_core_seed_radius_d_requested": 1.0,
        "air_core_seed_radius_m_resolved": 0.0005,
        "air_core_seed_cap_applied": False,
        "fill_radius_m_resolved": 0.00075,
        "fill_z_start_m": -0.001,
        "fill_z_stop_m": 0.001,
        "DeltaP_Pa": 6894760.0,
        "DeltaP_effective_Pa": 6894760.0,
        "check_valve_loss_applied": False,
        "provenance": {
            "probe_payload": f"{baseline_slug}/probe.json",
            "host_env": f"{baseline_slug}/host_env.json",
            "manifest_refs": f"{baseline_slug}/manifest_refs.json",
            "reference_freeze_overlay": "reference_freeze_overlay.json",
        },
        "io_normalization": build_reference_io_normalization_payload(),
    }


def sample_stage_plan_payload(
    bundle: Any,
    *,
    case_role: str,
    baseline: str,
    runtime_base: str,
    source_tuple_id: str,
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    baseline_slug = baseline_artifact_slug(baseline)
    return {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "phase_gate": "Phase 0",
        "baseline": baseline,
        "runtime_base": runtime_base,
        "reviewed_source_tuple_id": source_tuple_id,
        "provenance": {
            "probe_payload": f"{baseline_slug}/probe.json",
            "host_env": f"{baseline_slug}/host_env.json",
            "manifest_refs": f"{baseline_slug}/manifest_refs.json",
        },
        "io_normalization": build_reference_io_normalization_payload(),
        "phase_gate_selection": {
            "selected_case_role": resolved.case_role,
            "available_case_roles": list(bundle.ladder.ordered_case_ids),
            "ordered_ladder": list(bundle.ladder.ordered_case_ids),
            "conditional_selection": False,
        },
        "stages": [
            {
                "name": "reference_io_normalization",
                "cmd": reference_io_overlay_command(json_out="reference_freeze_overlay.json"),
                "cwd": ".",
                "stage_kind": "compare-prep",
                "overlay_artifact": "reference_freeze_overlay.json",
            },
            {
                "name": "transient_run",
                "cmd": "foamRun -solver incompressibleVoF",
            },
        ],
    }


def sample_build_fingerprint(case_role: str) -> dict[str, Any]:
    patch_schema = [
        {"patch_name": "swirlInletA", "patch_type": "patch", "nFaces": 8, "startFace": 1},
        {"patch_name": "walls", "patch_type": "wall", "nFaces": 42, "startFace": 9},
    ]
    if case_role == "R0":
        patch_schema.append(
            {"patch_name": "ambient", "patch_type": "patch", "nFaces": 16, "startFace": 51}
        )
    return {
        "config_hashes": {"system/controlDict": "a" * 64},
        "mesh_hashes": {"constant/polyMesh/boundary": "b" * 64},
        "patch_schema": patch_schema,
        "mesh_counts": {
            "cells": 128,
            "faces": 256,
            "points": 512,
            "internal_faces": 196,
            "patches": len(patch_schema),
        },
        "bbox": {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0},
        "case_role": case_role,
        "patch_schema_exact": True,
    }


def sample_field_signatures() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "canonical_name": "field_signatures.json",
        "steady_precondition": {
            "available": True,
            "latest_time": "20",
            "fields": {
                "U": {
                    "field_kind": "vector",
                    "normalized_file_sha256": "e" * 64,
                    "sample_count": 2,
                    "stats": {"min_magnitude": 1.0, "max_magnitude": 2.0, "l2_magnitude": 3.0},
                }
            },
            "missing_optional_fields": [],
        },
        "transient_latest": {
            "available": True,
            "latest_time": "0.5",
            "fields": {
                "alpha.water": {
                    "field_kind": "scalar",
                    "normalized_file_sha256": "f" * 64,
                    "sample_count": 2,
                    "stats": {"min": 0.0, "max": 1.0, "sum": 1.0, "mean": 0.5},
                }
            },
            "missing_optional_fields": [],
        },
    }


def sample_metrics(
    case_role: str,
    *,
    baseline: str,
    angle_source: str = "geometric_a",
) -> dict[str, Any]:
    if case_role == "R1":
        if baseline == "Baseline A":
            metric_values = {
                "solver_ok": 1,
                "mass_imbalance_pct": 0.4,
                "patch_schema_exact": True,
                "startup_provenance_exact": True,
                "precondition_field_copy_provenance_exact": True,
                "water_flow_cfd_gph": 54.0,
                "cd_mass_cfd": 0.80,
                "air_core_area_ratio": 0.35,
                "swirl_number_proxy": 0.45,
                "recirculation_fraction": 0.10,
            }
        else:
            metric_values = {
                "solver_ok": 1,
                "mass_imbalance_pct": 0.5,
                "patch_schema_exact": True,
                "startup_provenance_exact": True,
                "precondition_field_copy_provenance_exact": True,
                "water_flow_cfd_gph": 55.0,
                "cd_mass_cfd": 0.81,
                "air_core_area_ratio": 0.36,
                "swirl_number_proxy": 0.46,
                "recirculation_fraction": 0.11,
            }
    else:
        if baseline == "Baseline A":
            metric_values = {
                "solver_ok": 1,
                "mass_imbalance_pct": 0.3,
                "patch_schema_exact": True,
                "startup_provenance_exact": True,
                "resolved_direct_slot_numerics_exact": True,
                "manufacturing_sanity_reasonable": True,
                "water_flow_cfd_gph": 54.2,
                "cd_mass_cfd": 0.82,
                "spray_angle_cfd_deg": 31.5,
                "air_core_area_ratio": 0.40,
                "swirl_number_proxy": 0.50,
                "recirculation_fraction": 0.15,
                "delta_p_rgh_mean_pa": 1250.0,
            }
        else:
            metric_values = {
                "solver_ok": 1,
                "mass_imbalance_pct": 0.4,
                "patch_schema_exact": True,
                "startup_provenance_exact": True,
                "resolved_direct_slot_numerics_exact": True,
                "manufacturing_sanity_reasonable": True,
                "water_flow_cfd_gph": 55.0,
                "cd_mass_cfd": 0.83,
                "spray_angle_cfd_deg": 33.0,
                "air_core_area_ratio": 0.42,
                "swirl_number_proxy": 0.54,
                "recirculation_fraction": 0.16,
                "delta_p_rgh_mean_pa": 1285.0,
            }

    metric_sources = {
        name: "extractor" if name not in {"patch_schema_exact", "startup_provenance_exact"} else "reference_freeze"
        for name in metric_values
        if name != "spray_angle_source"
    }
    return {
        "schema_version": "1.0.0",
        "canonical_name": "metrics.json",
        "metrics": {
            metric_name: {
                "value": metric_value,
                "metric_source": metric_sources[metric_name],
            }
            for metric_name, metric_value in metric_values.items()
        },
        "angle_source": angle_source if case_role == "R0" else None,
        "time_windows": {
            "transient_latest": {
                "start_time": "0.0",
                "end_time": "0.5",
                "latest_time": "0.5",
            }
        },
        "provenance": {"run_meta": "run_meta.json"},
    }


def write_case_artifact_dir(
    artifact_root: pathlib.Path,
    *,
    case_role: str,
    baseline: str,
    runtime_base: str,
    source_tuple_id: str = "SRC_CPU_PHASE0_ACCEPTED",
    angle_source: str = "geometric_a",
    requested_vof_solver_mode: str = "vof_transient_preconditioned",
    resolved_vof_solver_exec: str = "incompressibleVoF",
    resolved_pressure_backend: str = "native_cpu",
) -> pathlib.Path:
    bundle = load_authority_bundle(repo_root())
    resolved = resolve_reference_case(bundle, case_role=case_role)
    baseline_root = artifact_root / baseline_artifact_slug(baseline)
    case_dir = baseline_root / resolved.frozen_id
    case_dir.mkdir(parents=True, exist_ok=True)

    write_json(baseline_root / "probe.json", {"baseline": baseline, "status": "ok"})
    write_json(baseline_root / "host_env.json", {"runtime_base": runtime_base})
    write_json(
        baseline_root / "manifest_refs.json",
        {"case_role": case_role, "case_id": resolved.frozen_id},
    )
    write_json(case_dir / "probe.json", {"baseline": baseline, "status": "ok"})
    write_json(
        case_dir / "case_meta.json",
        sample_case_meta_payload(
            bundle,
            case_role=case_role,
            baseline=baseline,
            runtime_base=runtime_base,
            source_tuple_id=source_tuple_id,
            requested_vof_solver_mode=requested_vof_solver_mode,
            resolved_vof_solver_exec=resolved_vof_solver_exec,
            resolved_pressure_backend=resolved_pressure_backend,
        ),
    )
    write_json(
        case_dir / "stage_plan.json",
        sample_stage_plan_payload(
            bundle,
            case_role=case_role,
            baseline=baseline,
            runtime_base=runtime_base,
            source_tuple_id=source_tuple_id,
        ),
    )
    write_json(
        case_dir / "reference_freeze_overlay.json",
        build_reference_io_normalization_payload(),
    )
    write_json(case_dir / "build_fingerprint.json", sample_build_fingerprint(case_role))
    write_json(case_dir / "field_signatures.json", sample_field_signatures())
    write_json(
        case_dir / "metrics.json",
        sample_metrics(case_role, baseline=baseline, angle_source=angle_source),
    )
    logs_dir = case_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "transient_run.log").write_text("solver ok\n", encoding="utf-8")
    (logs_dir / "transient_run.time.txt").write_text("1.23\n", encoding="utf-8")
    return case_dir


class ReferenceSignoffTests(unittest.TestCase):
    def test_publish_phase0_signoff_packet_writes_compare_reports(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                    )

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

            compare_json = json.loads(
                (artifact_root / "comparisons" / PHASE0_COMPARE_JSON_NAME).read_text(encoding="utf-8")
            )
            archive_index = json.loads(
                (artifact_root / "comparisons" / PHASE0_ARCHIVE_INDEX_NAME).read_text(encoding="utf-8")
            )
            provenance_manifest = json.loads(
                (artifact_root / "comparisons" / PHASE0_PROVENANCE_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            compare_markdown = (
                artifact_root / "comparisons" / PHASE0_COMPARE_MARKDOWN_NAME
            ).read_text(encoding="utf-8")

        self.assertEqual(packet["status"], "pass")
        self.assertEqual(compare_json["status"], "pass")
        self.assertEqual(compare_json["compared_case_roles"], ["R1", "R0"])
        self.assertEqual([case["status"] for case in compare_json["cases"]], ["pass", "pass"])
        self.assertEqual(compare_json["human_signoff"]["status"], "pending")
        self.assertTrue(compare_json["outputs"]["compare_json"].endswith(PHASE0_COMPARE_JSON_NAME))
        self.assertTrue(compare_json["archive_index"].endswith(PHASE0_ARCHIVE_INDEX_NAME))
        self.assertTrue(archive_index["outputs"]["compare_json"].endswith(PHASE0_COMPARE_JSON_NAME))
        self.assertEqual(provenance_manifest["case_count"], 2)
        self.assertIn("`R1`", compare_markdown)
        self.assertIn("`R0`", compare_markdown)

    def test_publish_phase0_signoff_packet_marks_angle_source_mismatch_as_review(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                write_case_artifact_dir(
                    artifact_root,
                    case_role="R1",
                    baseline=baseline,
                    runtime_base=runtime_base,
                )
            write_case_artifact_dir(
                artifact_root,
                case_role="R0",
                baseline="Baseline A",
                runtime_base="OpenFOAM 12",
                angle_source="geometric_a",
            )
            write_case_artifact_dir(
                artifact_root,
                case_role="R0",
                baseline="Baseline B",
                runtime_base="SPUMA v2412",
                angle_source="velocity_proxy",
            )

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r0_case = next(case for case in packet["cases"] if case["case_role"] == "R0")
        spray_check = next(
            check for check in r0_case["checks"] if check["check_id"] == "spray_angle_cfd_deg_within_threshold"
        )
        self.assertEqual(packet["status"], "review")
        self.assertEqual(r0_case["status"], "review")
        self.assertEqual(spray_check["status"], "review")
        self.assertIn("velocity_proxy", spray_check["details"])

    def test_publish_phase0_signoff_packet_fails_when_r1_flow_exceeds_threshold(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                write_case_artifact_dir(
                    artifact_root,
                    case_role="R0",
                    baseline=baseline,
                    runtime_base=runtime_base,
                )
            write_case_artifact_dir(
                artifact_root,
                case_role="R1",
                baseline="Baseline A",
                runtime_base="OpenFOAM 12",
            )
            write_case_artifact_dir(
                artifact_root,
                case_role="R1",
                baseline="Baseline B",
                runtime_base="SPUMA v2412",
            )

            baseline_b_metrics_path = (
                artifact_root
                / "baseline_b"
                / resolve_reference_case(bundle, case_role="R1").frozen_id
                / "metrics.json"
            )
            baseline_b_metrics = json.loads(baseline_b_metrics_path.read_text(encoding="utf-8"))
            baseline_b_metrics["metrics"]["water_flow_cfd_gph"]["value"] = 60.0
            write_json(baseline_b_metrics_path, baseline_b_metrics)

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r1_case = next(case for case in packet["cases"] if case["case_role"] == "R1")
        flow_check = next(
            check for check in r1_case["checks"] if check["check_id"] == "water_flow_cfd_gph_within_threshold"
        )
        self.assertEqual(packet["status"], "fail")
        self.assertEqual(r1_case["status"], "fail")
        self.assertEqual(flow_check["status"], "fail")

    def test_publish_phase0_signoff_packet_fails_on_requested_solver_mode_mismatch(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                        requested_vof_solver_mode=(
                            "vof_transient_preconditioned"
                            if baseline == "Baseline A"
                            else "vof_shadow_variant"
                        )
                        if case_role == "R1"
                        else "vof_transient_preconditioned",
                    )

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r1_case = next(case for case in packet["cases"] if case["case_role"] == "R1")
        requested_mode_check = next(
            check for check in r1_case["checks"] if check["check_id"] == "requested_vof_solver_mode_exact"
        )
        self.assertEqual(packet["status"], "fail")
        self.assertEqual(r1_case["status"], "fail")
        self.assertEqual(requested_mode_check["status"], "fail")

    def test_publish_phase0_signoff_packet_marks_solver_family_split_as_review(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                        resolved_vof_solver_exec=(
                            "incompressibleVoF" if baseline == "Baseline A" else "interIsoFoam"
                        )
                        if case_role == "R0"
                        else "incompressibleVoF",
                    )

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r0_case = next(case for case in packet["cases"] if case["case_role"] == "R0")
        solver_family_check = next(
            check for check in r0_case["checks"] if check["check_id"] == "resolved_solver_family_exact"
        )
        self.assertEqual(packet["status"], "review")
        self.assertEqual(r0_case["status"], "review")
        self.assertEqual(solver_family_check["status"], "review")

    def test_publish_phase0_signoff_packet_marks_backend_mismatch_as_review(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                        resolved_pressure_backend=(
                            "native_cpu" if baseline == "Baseline A" else "native_cpu_fallback"
                        )
                        if case_role == "R1"
                        else "native_cpu",
                    )

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r1_case = next(case for case in packet["cases"] if case["case_role"] == "R1")
        backend_check = next(
            check for check in r1_case["checks"] if check["check_id"] == "resolved_pressure_backend_exact"
        )
        self.assertEqual(packet["status"], "review")
        self.assertEqual(r1_case["status"], "review")
        self.assertEqual(backend_check["status"], "review")

    def test_publish_phase0_signoff_packet_uses_or_semantics_for_air_core_review_gate(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                    )

            baseline_b_metrics_path = (
                artifact_root
                / "baseline_b"
                / resolve_reference_case(bundle, case_role="R1").frozen_id
                / "metrics.json"
            )
            baseline_b_metrics = json.loads(baseline_b_metrics_path.read_text(encoding="utf-8"))
            baseline_b_metrics["metrics"]["air_core_area_ratio"]["value"] = 0.39
            write_json(baseline_b_metrics_path, baseline_b_metrics)

            packet = publish_phase0_signoff_packet(bundle, artifact_root=artifact_root)

        r1_case = next(case for case in packet["cases"] if case["case_role"] == "R1")
        air_core_check = next(
            check for check in r1_case["checks"] if check["check_id"] == "air_core_area_ratio_within_threshold"
        )
        self.assertEqual(packet["status"], "pass")
        self.assertEqual(r1_case["status"], "pass")
        self.assertEqual(air_core_check["status"], "pass")

    def test_publish_phase0_signoff_packet_records_custom_output_paths(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for baseline, runtime_base in (("Baseline A", "OpenFOAM 12"), ("Baseline B", "SPUMA v2412")):
                for case_role in ("R1", "R0"):
                    write_case_artifact_dir(
                        artifact_root,
                        case_role=case_role,
                        baseline=baseline,
                        runtime_base=runtime_base,
                    )

            json_out = artifact_root / "reports" / "json" / "compare-output.json"
            markdown_out = artifact_root / "reports" / "markdown" / "compare-output.md"
            archive_index_out = artifact_root / "reports" / "meta" / "archive-output.json"
            provenance_manifest_out = artifact_root / "reports" / "meta" / "provenance-output.json"

            packet = publish_phase0_signoff_packet(
                bundle,
                artifact_root=artifact_root,
                json_out=json_out,
                markdown_out=markdown_out,
                archive_index_out=archive_index_out,
                provenance_manifest_out=provenance_manifest_out,
            )

            archive_index = json.loads(archive_index_out.read_text(encoding="utf-8"))
            self.assertTrue(json_out.exists())
            self.assertTrue(markdown_out.exists())
            self.assertTrue(archive_index_out.exists())
            self.assertTrue(provenance_manifest_out.exists())
            self.assertEqual(packet["outputs"]["compare_json"], json_out.as_posix())
            self.assertEqual(packet["outputs"]["compare_markdown"], markdown_out.as_posix())
            self.assertEqual(archive_index["outputs"]["compare_json"], json_out.as_posix())
            self.assertEqual(archive_index["outputs"]["compare_markdown"], markdown_out.as_posix())

    def test_review_threshold_check_reports_and_for_dual_threshold_failure(self) -> None:
        check = reference_signoff._review_threshold_check(
            "air_core_area_ratio_within_threshold",
            threshold_abs=0.05,
            threshold_pct=10.0,
            baseline_a=1.0,
            baseline_b=1.2,
        )

        self.assertEqual(check["status"], "review")
        self.assertIn("and relative diff", check["details"])
        self.assertNotIn("or relative diff", check["details"])

    def test_review_threshold_check_requires_at_least_one_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one review threshold"):
            reference_signoff._review_threshold_check(
                "air_core_area_ratio_within_threshold",
                threshold_abs=None,
                threshold_pct=None,
                baseline_a=1.0,
                baseline_b=1.1,
            )

    def test_write_case_artifact_dir_allows_reinvocation_for_same_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            write_case_artifact_dir(
                artifact_root,
                case_role="R1",
                baseline="Baseline A",
                runtime_base="OpenFOAM 12",
            )

            case_dir = write_case_artifact_dir(
                artifact_root,
                case_role="R1",
                baseline="Baseline A",
                runtime_base="OpenFOAM 12",
            )

            self.assertTrue((case_dir / "logs" / "transient_run.log").is_file())


if __name__ == "__main__":
    unittest.main()
