from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import (
    BaselineProbeRequest,
    build_case_meta_payload,
    build_stage_plan_payload,
    emit_case_bundle,
    emit_reference_problem_artifacts,
    load_authority_bundle,
    probe_openfoam_baselines,
    resolve_stage_runner_context,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sample_host_observations(*, lane: str = "primary") -> dict[str, str]:
    nvcc_version = "Cuda compilation tools, release 12.9, V12.9.1"
    if lane == "experimental":
        nvcc_version = "Cuda compilation tools, release 13.2, V13.2.0"
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


def write_bashrc(root: pathlib.Path, *, name: str = "baseline.sh") -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("# reference artifact test bashrc\n", encoding="utf-8")
    return path


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_poly_mesh(case_dir: pathlib.Path) -> None:
    poly_mesh = case_dir / "constant" / "polyMesh"
    write_text(
        poly_mesh / "points",
        """FoamFile
{
    version 2.0;
}
4
(
(0 0 0)
(1 0 0)
(0 2 0)
(0 0 3)
)
// ************************************************************************* //
""",
    )
    write_text(
        poly_mesh / "faces",
        """FoamFile
{
    version 2.0;
}
3
(
4(0 1 2 3)
3(0 1 2)
3(1 2 3)
)
// ************************************************************************* //
""",
    )
    write_text(
        poly_mesh / "owner",
        """FoamFile
{
    version 2.0;
}
3
(
0
0
1
)
// ************************************************************************* //
""",
    )
    write_text(
        poly_mesh / "neighbour",
        """FoamFile
{
    version 2.0;
}
1
(
1
)
// ************************************************************************* //
""",
    )
    write_text(
        poly_mesh / "boundary",
        """FoamFile
{
    version 2.0;
}
2
(
inlet
{
    type patch;
    nFaces 1;
    startFace 1;
}
outlet
{
    type wall;
    nFaces 1;
    startFace 2;
}
)
// ************************************************************************* //
""",
    )


def write_scalar_field(path: pathlib.Path, object_name: str, values: list[float]) -> None:
    body = "\n".join(str(value) for value in values)
    write_text(
        path,
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {object_name};
}}
dimensions [0 0 0 0 0 0 0];
internalField nonuniform List<scalar>
{len(values)}
(
{body}
)
;
boundaryField
{{
}}
""",
    )


def write_vector_field(
    path: pathlib.Path,
    object_name: str,
    values: list[tuple[float, float, float]],
) -> None:
    body = "\n".join(f"({x} {y} {z})" for x, y, z in values)
    write_text(
        path,
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object {object_name};
}}
dimensions [0 0 0 0 0 0 0];
internalField nonuniform List<vector>
{len(values)}
(
{body}
)
;
boundaryField
{{
}}
""",
    )


def write_uniform_scalar_field(path: pathlib.Path, object_name: str, value: float) -> None:
    write_text(
        path,
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object {object_name};
}}
dimensions [0 0 0 0 0 0 0];
internalField uniform {value};
boundaryField
{{
}}
""",
    )


def write_uniform_vector_field(
    path: pathlib.Path,
    object_name: str,
    value: tuple[float, float, float],
) -> None:
    write_text(
        path,
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object {object_name};
}}
dimensions [0 0 0 0 0 0 0];
internalField uniform ({value[0]} {value[1]} {value[2]});
boundaryField
{{
}}
""",
    )

def build_case_bundle(
    temp_root: pathlib.Path,
    *,
    case_role: str = "R1",
    phase_gate: str = "Phase 2",
    conditional_reason: str | None = "patch-manifest coverage under test",
    reference_artifacts: dict[str, object] | None = None,
):
    bundle = load_authority_bundle(repo_root())
    bashrc_path = write_bashrc(temp_root, name="baseline_b_reference.sh")
    report = probe_openfoam_baselines(
        bundle,
        output_dir=temp_root,
        baselines={
            "Baseline B": BaselineProbeRequest(
                lane="experimental",
                runtime_base="exaFOAM/SPUMA 0.1-v2412",
                bashrc_path=str(bashrc_path),
                host_observations=sample_host_observations(lane="experimental"),
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
    case_meta = build_case_meta_payload(
        bundle,
        context=context,
        case_role=case_role,
        requested_vof_solver_mode="vof_transient_preconditioned",
        resolved_vof_solver_exec="incompressibleVoF",
        resolved_pressure_backend="amgx",
        mesh={
            "mesh_full_360": 1,
            "mesh_resolution_scale": 2.0,
            "hydraulic_domain_mode": "internal_only",
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
        phase_gate=phase_gate,
        conditional_reason=conditional_reason,
        stages=[
            {"name": "checkMesh_build", "cmd": "checkMesh"},
            {"name": "transient_run", "cmd": "foamRun -solver incompressibleVoF"},
        ],
    )
    emitted = emit_case_bundle(
        bundle,
        output_dir=temp_root / "case_bundle",
        case_meta=case_meta,
        stage_plan=stage_plan,
        reference_artifacts=reference_artifacts,
    )
    return emitted


class ReferenceProblemArtifactTests(unittest.TestCase):
    def test_emit_reference_problem_artifacts_writes_comparison_ready_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            case_dir = temp_root / "case"
            write_poly_mesh(case_dir)
            steady_root = temp_root / "normalized" / "steady" / "20"
            transient_root = temp_root / "normalized" / "transient" / "0.5"
            write_vector_field(
                steady_root / "U",
                "U",
                [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)],
            )
            write_scalar_field(steady_root / "k", "k", [0.1, 0.2, 0.3])
            write_scalar_field(steady_root / "omega", "omega", [1.0, 1.5, 2.0])
            write_vector_field(
                transient_root / "U",
                "U",
                [(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 6.0)],
            )
            write_scalar_field(
                transient_root / "alpha.water",
                "alpha.water",
                [0.0, 0.5, 1.0],
            )
            write_scalar_field(transient_root / "p_rgh", "p_rgh", [10.0, 20.0, 30.0])
            write_scalar_field(transient_root / "rho", "rho", [998.0, 999.0, 1000.0])
            write_text(
                temp_root / "run_meta.json",
                json.dumps({"solver_log": "logs/solver.log"}, indent=2, sort_keys=True) + "\n",
            )
            emitted = build_case_bundle(
                temp_root,
                reference_artifacts={
                    "case_dir": case_dir,
                    "artifact_root": temp_root / "artifacts",
                    "normalized_steady_root": temp_root / "normalized" / "steady",
                    "normalized_transient_root": temp_root / "normalized" / "transient",
                    "run_meta_path": temp_root / "run_meta.json",
                    "metrics": {
                        "water_flow_cfd_gph": 54.2,
                        "mass_imbalance_pct": 0.1,
                        "spray_angle_cfd_deg": 31.5,
                        "spray_angle_source": "geometric_a",
                    },
                    "metric_sources": {
                        "water_flow_cfd_gph": "continuity_integral",
                        "mass_imbalance_pct": "continuity_log",
                        "spray_angle_cfd_deg": "geometric_fit",
                    },
                    "time_windows": {
                        "transient_latest": {
                            "start_time": "0.0",
                            "end_time": "0.5",
                            "latest_time": "0.5",
                        }
                    },
                },
            )

            build_fingerprint = json.loads(emitted.build_fingerprint_path.read_text(encoding="utf-8"))
            field_signatures = json.loads(emitted.field_signatures_path.read_text(encoding="utf-8"))
            metrics_payload = json.loads(emitted.metrics_path.read_text(encoding="utf-8"))
            case_meta_path = emitted.case_meta_path

        self.assertEqual(build_fingerprint["canonical_name"], "build_fingerprint.json")
        self.assertEqual(build_fingerprint["case_id"], "phase0_r1_57_28_1000_internal_v1")
        self.assertEqual(build_fingerprint["case_role"], "R1")
        self.assertEqual(build_fingerprint["mesh_counts"]["points"], 4)
        self.assertEqual(build_fingerprint["mesh_counts"]["faces"], 3)
        self.assertEqual(build_fingerprint["mesh_counts"]["internal_faces"], 1)
        self.assertEqual(build_fingerprint["mesh_counts"]["cells"], 2)
        self.assertEqual(build_fingerprint["mesh_counts"]["patches"], 2)
        self.assertEqual(build_fingerprint["bounding_box"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(build_fingerprint["bounding_box"]["max"], [1.0, 2.0, 3.0])
        self.assertEqual(build_fingerprint["semantic_patches"][0]["name"], "inlet")
        self.assertEqual(build_fingerprint["semantic_patches"][1]["type"], "wall")
        self.assertEqual(
            build_fingerprint["provenance"]["case_meta"],
            case_meta_path.as_posix(),
        )

        self.assertEqual(field_signatures["canonical_name"], "field_signatures.json")
        self.assertEqual(field_signatures["steady_precondition"]["latest_time"], "20")
        self.assertEqual(field_signatures["transient_latest"]["latest_time"], "0.5")
        self.assertEqual(
            field_signatures["steady_precondition"]["fields"]["U"]["stats"]["min_magnitude"],
            1.0,
        )
        self.assertEqual(
            field_signatures["steady_precondition"]["fields"]["U"]["stats"]["l2_magnitude"],
            3.0,
        )
        self.assertEqual(
            field_signatures["transient_latest"]["fields"]["alpha.water"]["stats"]["mean"],
            0.5,
        )
        self.assertEqual(
            field_signatures["transient_latest"]["fields"]["p_rgh"]["stats"]["sum"],
            60.0,
        )
        self.assertEqual(
            field_signatures["transient_latest"]["missing_optional_fields"],
            ["phi"],
        )
        self.assertEqual(
            field_signatures["provenance"]["normalized_transient_root"],
            (temp_root / "normalized" / "transient").as_posix(),
        )

        self.assertEqual(metrics_payload["canonical_name"], "metrics.json")
        self.assertEqual(metrics_payload["angle_source"], "geometric_a")
        self.assertNotIn("spray_angle_source", metrics_payload["metrics"])
        self.assertEqual(
            metrics_payload["metrics"]["water_flow_cfd_gph"]["metric_source"],
            "continuity_integral",
        )
        self.assertEqual(
            metrics_payload["metrics"]["spray_angle_cfd_deg"]["value"],
            31.5,
        )
        self.assertEqual(
            metrics_payload["time_windows"]["transient_latest"]["latest_time"],
            "0.5",
        )
        self.assertEqual(
            metrics_payload["provenance"]["run_meta"],
            (temp_root / "run_meta.json").as_posix(),
        )

    def test_emit_reference_problem_artifacts_fails_when_required_field_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            case_dir = temp_root / "case"
            write_poly_mesh(case_dir)
            steady_root = temp_root / "normalized" / "steady" / "20"
            transient_root = temp_root / "normalized" / "transient" / "0.5"
            write_vector_field(steady_root / "U", "U", [(1.0, 0.0, 0.0)])
            write_vector_field(transient_root / "U", "U", [(2.0, 0.0, 0.0)])
            write_scalar_field(
                transient_root / "alpha.water",
                "alpha.water",
                [0.25],
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing required transient field\\(s\\): p_rgh",
            ):
                build_case_bundle(
                    temp_root,
                    reference_artifacts={
                        "case_dir": case_dir,
                        "artifact_root": temp_root / "artifacts",
                        "normalized_steady_root": temp_root / "normalized" / "steady",
                        "normalized_transient_root": temp_root / "normalized" / "transient",
                    },
                )

    def test_emit_reference_problem_artifacts_allows_transient_only_r2(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted_bundle = build_case_bundle(
                temp_root,
                case_role="R2",
                phase_gate="Phase 0",
                conditional_reason=None,
            )
            case_dir = temp_root / "case"
            write_poly_mesh(case_dir)
            transient_root = temp_root / "normalized" / "transient" / "0.5"
            write_uniform_vector_field(transient_root / "U", "U", (2.0, 0.0, 0.0))
            write_scalar_field(transient_root / "alpha.water", "alpha.water", [0.25])
            write_uniform_scalar_field(transient_root / "p_rgh", "p_rgh", 10.0)

            emitted = emit_reference_problem_artifacts(
                bundle,
                case_dir=case_dir,
                artifact_root=temp_root / "artifacts",
                case_meta_path=emitted_bundle.case_meta_path,
                stage_plan_path=emitted_bundle.stage_plan_path,
                normalized_steady_root=temp_root / "normalized" / "steady",
                normalized_transient_root=temp_root / "normalized" / "transient",
            )
            field_signatures = json.loads(emitted.field_signatures_path.read_text(encoding="utf-8"))

        self.assertFalse(field_signatures["steady_precondition"]["available"])
        self.assertIsNone(field_signatures["steady_precondition"]["latest_time"])
        self.assertEqual(field_signatures["transient_latest"]["latest_time"], "0.5")
        self.assertEqual(field_signatures["case_role"], "R2")
        self.assertEqual(
            field_signatures["transient_latest"]["fields"]["p_rgh"]["sample_count"],
            2,
        )
        self.assertEqual(
            field_signatures["transient_latest"]["fields"]["p_rgh"]["stats"]["sum"],
            20.0,
        )
        self.assertAlmostEqual(
            field_signatures["transient_latest"]["fields"]["U"]["stats"]["l2_magnitude"],
            2.8284271247461903,
        )

    def test_emit_reference_problem_artifacts_fails_before_writing_on_mismatched_bundle(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            r1_bundle = build_case_bundle(temp_root / "r1")
            r2_bundle = build_case_bundle(
                temp_root / "r2",
                case_role="R2",
                phase_gate="Phase 0",
                conditional_reason=None,
            )
            case_dir = temp_root / "case"
            write_poly_mesh(case_dir)
            transient_root = temp_root / "normalized" / "transient" / "0.5"
            write_vector_field(transient_root / "U", "U", [(2.0, 0.0, 0.0)])
            write_scalar_field(transient_root / "alpha.water", "alpha.water", [0.25])
            write_scalar_field(transient_root / "p_rgh", "p_rgh", [10.0])
            artifact_root = temp_root / "artifacts"

            with self.assertRaisesRegex(
                ValueError,
                "matching case_id values",
            ):
                emit_reference_problem_artifacts(
                    bundle,
                    case_dir=case_dir,
                    artifact_root=artifact_root,
                    case_meta_path=r1_bundle.case_meta_path,
                    stage_plan_path=r2_bundle.stage_plan_path,
                    normalized_steady_root=temp_root / "normalized" / "steady",
                    normalized_transient_root=temp_root / "normalized" / "transient",
                )

        self.assertFalse(artifact_root.exists())


if __name__ == "__main__":
    unittest.main()
