from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any

from scripts.authority import (
    build_reference_io_normalization_payload,
    freeze_case_artifact,
    load_authority_bundle,
    publish_baseline_control_index,
    reference_io_overlay_command,
    resolve_reference_case,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sample_case_meta_payload(
    bundle: Any,
    *,
    case_role: str,
    baseline: str = "Baseline A",
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    return {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "ladder_position": resolved.ladder_position,
        "phase_gates": list(resolved.phase_gates),
        "baseline": baseline,
        "runtime_base": "OpenFOAM 12",
        "reviewed_source_tuple_id": "SRC_OPENFOAM12_REFERENCE",
        "requested_vof_solver_mode": "vof_transient_preconditioned",
        "resolved_vof_solver_exec": "incompressibleVoF",
        "resolved_pressure_backend": "native_cpu",
        "openfoam_bashrc_used": "/envs/baseline-a/etc/bashrc",
        "available_commands": {
            "foamRun": "/envs/baseline-a/bin/foamRun",
            "checkMesh": "/envs/baseline-a/bin/checkMesh",
            "setFields": "/envs/baseline-a/bin/setFields",
        },
        "mesh_full_360": 1,
        "mesh_resolution_scale": 2.0,
        "hydraulic_domain_mode": "internal_only",
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
            "probe_payload": "baseline_a/probe.json",
            "host_env": "baseline_a/host_env.json",
            "manifest_refs": "baseline_a/manifest_refs.json",
            "reference_freeze_overlay": "reference_freeze_overlay.json",
        },
        "io_normalization": build_reference_io_normalization_payload(),
    }


def sample_stage_plan_payload(
    bundle: Any,
    *,
    case_role: str,
    baseline: str = "Baseline A",
) -> dict[str, Any]:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    return {
        "schema_version": "1.0.0",
        "case_id": resolved.frozen_id,
        "case_role": resolved.case_role,
        "phase_gate": "Phase 0",
        "baseline": baseline,
        "runtime_base": "OpenFOAM 12",
        "reviewed_source_tuple_id": "SRC_OPENFOAM12_REFERENCE",
        "provenance": {
            "probe_payload": "baseline_a/probe.json",
            "host_env": "baseline_a/host_env.json",
            "manifest_refs": "baseline_a/manifest_refs.json",
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


def sample_metrics(case_role: str) -> dict[str, Any]:
    if case_role == "R2":
        return {
            "solver_ok": {"value": 1, "unit": "flag", "source": "solver_log"},
            "alpha_min": {"value": 0.0, "unit": "", "source": "alpha.water"},
            "alpha_max": {"value": 1.0, "unit": "", "source": "alpha.water"},
            "alpha_integral_change_pct": {"value": 0.1, "unit": "%", "source": "alpha.water"},
            "same_baseline_field_hash_exact": {
                "value": True,
                "unit": "flag",
                "source": "rerun_compare",
            },
        }
    if case_role == "R1-core":
        return {
            "solver_ok": {"value": 1, "unit": "flag", "source": "solver_log"},
            "generic_phase5_subset_exact": {
                "value": True,
                "unit": "flag",
                "source": "support_scan",
            },
            "patch_schema_exact": {"value": True, "unit": "flag", "source": "build_fingerprint"},
            "fingerprint_repeatable": {
                "value": True,
                "unit": "flag",
                "source": "rerun_compare",
            },
        }
    if case_role == "R1":
        return {
            "solver_ok": {"value": 1, "unit": "flag", "source": "solver_log"},
            "mass_imbalance_pct": {"value": 0.4, "unit": "%", "source": "extractor"},
            "patch_schema_exact": {"value": True, "unit": "flag", "source": "build_fingerprint"},
            "startup_provenance_exact": {"value": True, "unit": "flag", "source": "startup"},
            "precondition_field_copy_provenance_exact": {
                "value": True,
                "unit": "flag",
                "source": "steady_precondition",
            },
        }
    return {
        "solver_ok": {"value": 1, "unit": "flag", "source": "solver_log"},
        "mass_imbalance_pct": {"value": 0.3, "unit": "%", "source": "extractor"},
        "patch_schema_exact": {"value": True, "unit": "flag", "source": "build_fingerprint"},
        "startup_provenance_exact": {"value": True, "unit": "flag", "source": "startup"},
        "resolved_direct_slot_numerics_exact": {
            "value": True,
            "unit": "flag",
            "source": "case_meta",
        },
        "manufacturing_sanity_reasonable": {
            "value": True,
            "unit": "flag",
            "source": "review_packet",
        },
    }


def sample_build_fingerprint(case_role: str) -> dict[str, Any]:
    return {
        "config_hashes": {"system/controlDict": "a" * 64},
        "mesh_hashes": {"constant/polyMesh/boundary": "b" * 64},
        "patch_schema": [
            {"patch_name": "swirlInletA", "patch_type": "patch", "nFaces": 8, "startFace": 1},
            {"patch_name": "walls", "patch_type": "wall", "nFaces": 42, "startFace": 9},
        ],
        "mesh_counts": {"cells": 128, "faces": 256, "points": 512, "internal_faces": 196, "patches": 2},
        "bbox": {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0},
        "case_role": case_role,
        "patch_schema_exact": True,
    }


def sample_field_signatures() -> list[dict[str, Any]]:
    return [
        {
            "time_dir": "latestTime",
            "field_name": "alpha.water",
            "file_sha256": "c" * 64,
            "stats": {"min": 0.0, "max": 1.0, "sum": 42.0, "mean": 0.5},
        }
    ]


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_case_artifact_dir(root: pathlib.Path, *, case_role: str) -> pathlib.Path:
    bundle = load_authority_bundle(repo_root())
    case_meta = sample_case_meta_payload(bundle, case_role=case_role)
    stage_plan = sample_stage_plan_payload(bundle, case_role=case_role)
    artifact_dir = root / "baseline_a" / case_meta["case_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "probe.json", {"baseline": "Baseline A", "status": "ok"})
    write_json(artifact_dir / "case_meta.json", case_meta)
    write_json(artifact_dir / "stage_plan.json", stage_plan)
    write_json(
        artifact_dir / "reference_freeze_overlay.json",
        build_reference_io_normalization_payload(),
    )
    write_json(artifact_dir / "build_fingerprint.json", sample_build_fingerprint(case_role))
    write_json(artifact_dir / "field_signatures.json", sample_field_signatures())
    write_json(artifact_dir / "metrics.json", sample_metrics(case_role))
    logs_dir = artifact_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "transient_run.log").write_text("solver ok\n", encoding="utf-8")
    (logs_dir / "transient_run.time.txt").write_text("1.23\n", encoding="utf-8")
    return artifact_dir


class ReferenceFreezeTests(unittest.TestCase):
    def test_freeze_case_artifact_writes_bundle_manifest_and_verdict(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = write_case_artifact_dir(pathlib.Path(temp_dir), case_role="R2")

            manifest = freeze_case_artifact(bundle, artifact_dir=artifact_dir, baseline_name="Baseline A")

            written_verdict = json.loads(
                (artifact_dir / "baseline_verdict.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["case_role"], "R2")
        self.assertEqual(manifest["baseline"], "Baseline A")
        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(written_verdict["case_role"], "R2")
        self.assertEqual(written_verdict["status"], "pass")
        self.assertIn("build_fingerprint.json", manifest["artifacts"])
        self.assertIn("field_signatures.json", manifest["artifacts"])

    def test_publish_baseline_control_index_preserves_phase0_ladder_order(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_root = pathlib.Path(temp_dir)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                write_case_artifact_dir(artifact_root, case_role=case_role)

            control_index = publish_baseline_control_index(
                bundle,
                artifact_root=artifact_root,
                baseline_name="Baseline A",
            )
            written = json.loads(
                (artifact_root / "baseline_a" / "control_index.json").read_text(encoding="utf-8")
            )

        self.assertEqual(control_index, written)
        self.assertEqual(control_index["baseline"], "Baseline A")
        self.assertEqual(
            [item["case_role"] for item in control_index["cases"]],
            ["R2", "R1-core", "R1", "R0"],
        )
        self.assertEqual(control_index["status"], "pass")
        self.assertEqual(
            [item["status"] for item in control_index["cases"]],
            ["pass", "pass", "pass", "pass"],
        )

    def test_freeze_case_artifact_requires_complete_bundle_inputs(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = write_case_artifact_dir(pathlib.Path(temp_dir), case_role="R1")
            (artifact_dir / "field_signatures.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "field_signatures.json"):
                freeze_case_artifact(bundle, artifact_dir=artifact_dir, baseline_name="Baseline A")


if __name__ == "__main__":
    unittest.main()
