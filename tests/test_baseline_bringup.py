from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from typing import Any

from scripts.authority import (
    BaselineProbeRequest,
    baseline_artifact_slug,
    build_case_meta_payload,
    build_stage_plan_payload,
    emit_case_bundle,
    load_authority_bundle,
    probe_openfoam_baselines,
    resolve_reference_case,
    resolve_stage_runner_context,
)
from scripts.authority.baseline_bringup import (
    BRINGUP_PACKET_NAME,
    BRINGUP_SUMMARY_NAME,
    publish_baseline_bringup_packet,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sample_host_observations(*, lane: str = "experimental") -> dict[str, str]:
    nvcc_version = "Cuda compilation tools, release 13.2, V13.2.0"
    if lane == "primary":
        nvcc_version = "Cuda compilation tools, release 12.9, V12.9.1"
    return {
        "gpu_query": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
        "nvcc_version": nvcc_version,
        "gcc_version": "gcc (Ubuntu 14.2.0) 14.2.0",
        "nsys_version": "NVIDIA Nsight Systems version 2025.2",
        "ncu_version": "NVIDIA Nsight Compute version 2025.3",
        "compute_sanitizer_version": "Compute Sanitizer version 2025.1",
        "compiler_version": "gcc (Ubuntu 14.2.0) 14.2.0",
        "os_release": "Ubuntu 24.04.2 LTS",
        "kernel": "6.8.0-60-generic",
    }


def sample_local_mirror_refs() -> dict[str, str]:
    return {
        "SPUMA runtime base": "3d1d7bf598ec8a66e099d8688b8597422c361960",
        "SPUMA support-policy snapshot": "ad2a385e44f2c01b7d1df44c5bc51d7996c95554",
        "External solver bridge": "4c764d027f8f124a1cc0b6df0520eb63593c2a2b",
        "AmgX backend": "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47",
    }


def sample_commands() -> dict[str, str]:
    return {
        "foamRun": "/opt/openfoam-v2412/bin/foamRun",
        "incompressibleVoF": "/opt/openfoam-v2412/bin/incompressibleVoF",
        "interIsoFoam": "/opt/openfoam-v2412/bin/interIsoFoam",
        "interFoam": "/opt/openfoam-v2412/bin/interFoam",
        "setFields": "/opt/openfoam-v2412/bin/setFields",
        "checkMesh": "/opt/openfoam-v2412/bin/checkMesh",
        "potentialFoam": "/opt/openfoam-v2412/bin/potentialFoam",
        "foamListTimes": "/opt/openfoam-v2412/bin/foamListTimes",
        "decomposePar": "/opt/openfoam-v2412/bin/decomposePar",
    }


def write_bashrc(root: pathlib.Path, *, name: str = "baseline_b.sh") -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("# baseline b test bashrc\n", encoding="utf-8")
    return path


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_context(temp_root: pathlib.Path):
    bundle = load_authority_bundle(repo_root())
    bashrc_path = write_bashrc(temp_root)
    report = probe_openfoam_baselines(
        bundle,
        output_dir=temp_root,
        baselines={
            "Baseline B": BaselineProbeRequest(
                lane="experimental",
                runtime_base="exaFOAM/SPUMA 0.1-v2412",
                bashrc_path=str(bashrc_path),
                host_observations=sample_host_observations(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="def456abc123",
                command_paths=sample_commands(),
                openfoam_env={
                    "WM_PROJECT": "OpenFOAM",
                    "WM_PROJECT_VERSION": "v2412",
                    "WM_OPTIONS": "linux64ClangDPInt32Opt",
                },
            ),
        },
    )
    context = resolve_stage_runner_context(
        temp_root,
        probe_report=report,
        baseline_name="Baseline B",
    )
    return bundle, context


def emit_case_bundle_for_role(
    bundle: Any,
    context: Any,
    temp_root: pathlib.Path,
    *,
    case_role: str,
    resolved_pressure_backend: str = "native_cpu",
) -> pathlib.Path:
    resolved = resolve_reference_case(bundle, case_role=case_role)
    output_dir = temp_root / baseline_artifact_slug("Baseline B") / resolved.frozen_id
    case_meta = build_case_meta_payload(
        bundle,
        context=context,
        case_role=case_role,
        requested_vof_solver_mode="vof_transient_preconditioned",
        resolved_vof_solver_exec="incompressibleVoF",
        resolved_pressure_backend=resolved_pressure_backend,
        mesh={
            "mesh_full_360": 1,
            "mesh_resolution_scale": 2.0,
            "hydraulic_domain_mode": "internal_only" if case_role != "R0" else "full_360",
            "near_field_radius_d": 10.0,
            "near_field_length_d": 20.0,
            "steady_end_time_iter": 10,
            "steady_write_interval_iter": 5,
            "steady_turbulence_model": "kOmegaSST",
            "vof_turbulence_model": "laminar",
        },
        numerics={
            "delta_t_s": 1e-8,
            "write_interval_s": 1e-8,
            "end_time_s": 2e-8,
            "max_co": 0.05,
            "max_alpha_co": 0.01,
            "resolved_direct_slot_numerics": {
                "slot_count": 6,
                "nominal_slot_width_mm": 0.64,
            },
        },
        startup={
            "startup_fill_extension_d": 0.0,
            "air_core_seed_radius_d_requested": 1.0,
            "air_core_seed_radius_m_resolved": 0.0005,
            "air_core_seed_cap_applied": False,
            "fill_radius_m_resolved": 0.00075,
            "fill_z_start_m": -0.001,
            "fill_z_stop_m": 0.001,
        },
        pressure={
            "DeltaP_Pa": 6894760.0,
            "DeltaP_effective_Pa": 6894760.0,
            "check_valve_loss_applied": False,
        },
    )
    stage_plan = build_stage_plan_payload(
        bundle,
        context=context,
        case_role=case_role,
        phase_gate="Phase 0",
        stages=[
            {"name": "checkMesh_build", "cmd": "checkMesh"},
            {"name": "transient_run", "cmd": "foamRun -solver incompressibleVoF"},
        ],
    )
    emit_case_bundle(
        bundle,
        output_dir=output_dir,
        case_meta=case_meta,
        stage_plan=stage_plan,
    )
    return output_dir


def write_r2_metrics(case_dir: pathlib.Path) -> None:
    write_json(
        case_dir / "metrics.json",
        {
            "metrics": {
                "solver_ok": {"value": 1, "unit": "flag", "metric_source": "solver_log"},
                "alpha_min": {"value": 0.0, "unit": "", "metric_source": "alpha.water"},
                "alpha_max": {"value": 1.0, "unit": "", "metric_source": "alpha.water"},
                "alpha_integral_change_pct": {
                    "value": 0.1,
                    "unit": "%",
                    "metric_source": "alpha.water",
                },
            }
        },
    )


class BaselineBringupPacketTests(unittest.TestCase):
    def test_publish_baseline_bringup_packet_marks_go_for_clean_builds_and_r2_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

            baseline_root = temp_root / baseline_artifact_slug("Baseline B")
            self.assertEqual(packet["decision"]["disposition"], "go")
            self.assertTrue(packet["decision"]["proceed_to_nozzle_freeze"])
            self.assertTrue(packet["build_summary"]["build_ready"])
            self.assertEqual(packet["r2_smoke"]["status"], "pass")
            self.assertEqual(packet["legacy_path_audit"]["hit_count"], 0)
            self.assertEqual(
                packet["tuple_traceability"]["reviewed_source_tuple_ids"],
                ["SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0"],
            )
            self.assertTrue(
                packet["backend_policy"][
                    "native_pressure_required_baseline_on_every_accepted_case"
                ]
            )
            self.assertTrue((baseline_root / BRINGUP_PACKET_NAME).exists())
            self.assertTrue((baseline_root / BRINGUP_SUMMARY_NAME).exists())

            rerun_packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )
            self.assertEqual(rerun_packet["legacy_path_audit"]["hit_count"], 0)

    def test_publish_baseline_bringup_packet_detects_legacy_of12_reference_and_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                    case_meta_path = case_dir / "case_meta.json"
                    case_meta = json.loads(case_meta_path.read_text(encoding="utf-8"))
                    case_meta["openfoam_bashrc_used"] = "/opt/openfoam12/etc/bashrc"
                    write_json(case_meta_path, case_meta)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["decision"]["disposition"], "blocked")
        self.assertFalse(packet["decision"]["proceed_to_nozzle_freeze"])
        self.assertGreater(packet["legacy_path_audit"]["hit_count"], 0)
        self.assertIn(
            "legacy /opt/openfoam12 references remain in Baseline B artifacts",
            packet["decision"]["reasons"],
        )

    def test_publish_baseline_bringup_packet_marks_backend_fallback_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                    write_json(
                        case_dir / "pressure_backend_patch.json",
                        {
                            "original_backend": "amgx",
                            "resolved_backend": "native_cpu",
                            "reason": "external backend unavailable on Baseline B",
                        },
                    )

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["r2_smoke"]["status"], "review")
        self.assertTrue(packet["r2_smoke"]["fallback_applied"])
        self.assertEqual(packet["decision"]["disposition"], "review")
        self.assertFalse(packet["decision"]["proceed_to_nozzle_freeze"])
        self.assertIn(
            "pressure backend fallback was applied",
            packet["decision"]["reasons"],
        )

    def test_publish_baseline_bringup_packet_blocks_failed_smoke_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                    write_json(case_dir / "baseline_verdict.json", {"status": "fail"})

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["r2_smoke"]["status"], "blocked")
        self.assertEqual(packet["decision"]["disposition"], "blocked")
        self.assertFalse(packet["decision"]["proceed_to_nozzle_freeze"])
        self.assertIn(
            "baseline_verdict.json reported status 'fail'",
            packet["decision"]["reasons"],
        )

    def test_publish_baseline_bringup_packet_marks_mixed_runtime_as_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                if case_role == "R1":
                    for filename in ("case_meta.json", "stage_plan.json"):
                        path = case_dir / filename
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        payload["runtime_base"] = "OpenFOAM v2412"
                        write_json(path, payload)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertTrue(packet["runtime_summary"]["contingency_runtime_used"])
        self.assertEqual(packet["decision"]["disposition"], "review")
        self.assertIn(
            "Baseline B used a contingency runtime instead of the canonical SPUMA line",
            packet["decision"]["reasons"],
        )

    def test_publish_baseline_bringup_packet_blocks_mismatched_case_bundle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                    stage_plan_path = case_dir / "stage_plan.json"
                    stage_plan = json.loads(stage_plan_path.read_text(encoding="utf-8"))
                    stage_plan["runtime_base"] = "OpenFOAM v2412"
                    write_json(stage_plan_path, stage_plan)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["decision"]["disposition"], "blocked")
        self.assertIn("R2", packet["build_summary"]["blocked_case_roles"])
        self.assertIn(
            "case bundle runtime_base mismatch between case_meta.json and stage_plan.json",
            packet["cases"][0]["notes"],
        )

    def test_publish_baseline_bringup_packet_blocks_missing_runtime_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)
                for filename in ("case_meta.json", "stage_plan.json"):
                    path = case_dir / filename
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["runtime_base"] = None
                    write_json(path, payload)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["decision"]["disposition"], "blocked")
        self.assertEqual(packet["runtime_summary"]["runtime_bases"], [])
        self.assertEqual(
            packet["runtime_summary"]["missing_runtime_case_roles"],
            ["R0", "R1", "R1-core", "R2"],
        )
        self.assertIn(
            "runtime base traceability is missing for: R0, R1, R1-core, R2",
            packet["decision"]["reasons"],
        )

    def test_publish_baseline_bringup_packet_ignores_stale_default_outputs_in_legacy_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_r2_metrics(case_dir)

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )
            self.assertEqual(packet["legacy_path_audit"]["hit_count"], 0)

            baseline_root = temp_root / baseline_artifact_slug("Baseline B")
            stale_packet_path = baseline_root / BRINGUP_PACKET_NAME
            stale_packet_path.write_text(
                "{\"legacy_literal\": \"/opt/openfoam12/etc/bashrc\"}\n",
                encoding="utf-8",
            )

            rerun_packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
                json_out=temp_root / "custom" / BRINGUP_PACKET_NAME,
                markdown_out=temp_root / "custom" / BRINGUP_SUMMARY_NAME,
            )

        self.assertEqual(rerun_packet["decision"]["disposition"], "go")
        self.assertEqual(rerun_packet["legacy_path_audit"]["hit_count"], 0)

    def test_publish_baseline_bringup_packet_preserves_review_notes_from_baseline_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bundle, context = build_context(temp_root)
            for case_role in ("R2", "R1-core", "R1", "R0"):
                case_dir = emit_case_bundle_for_role(bundle, context, temp_root, case_role=case_role)
                if case_role == "R2":
                    write_json(
                        case_dir / "baseline_verdict.json",
                        {
                            "status": "review",
                            "checks": [
                                {
                                    "check_id": "alpha_bounds",
                                    "gate_class": "review",
                                    "status": "fail",
                                    "details": "alpha range requires manual review",
                                }
                            ],
                        },
                    )

            packet = publish_baseline_bringup_packet(
                bundle,
                artifact_root=temp_root,
                baseline_name="Baseline B",
            )

        self.assertEqual(packet["r2_smoke"]["status"], "review")
        self.assertIn(
            "alpha_bounds: alpha range requires manual review",
            packet["r2_smoke"]["notes"],
        )
        self.assertIn(
            "alpha_bounds: alpha range requires manual review",
            packet["decision"]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
