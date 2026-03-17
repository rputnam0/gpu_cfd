from __future__ import annotations

import pathlib
import os
import subprocess
import sys
import tempfile
import unittest
import json

from scripts.authority import (
    AuthoritySelectionError,
    BaselineProbeRequest,
    build_case_meta_payload,
    build_stage_plan_payload,
    emit_case_bundle,
    load_authority_bundle,
    probe_openfoam_baselines,
    render_stage_runner_log_context,
    resolve_stage_runner_context,
    wrap_stage_command,
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
    path = root / name
    path.write_text("# stage runner test bashrc\n", encoding="utf-8")
    return path


class StageRunnerTests(unittest.TestCase):
    def test_stage_runner_context_consumes_probe_and_manifest_artifacts(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            bashrc_path = write_bashrc(pathlib.Path(temp_dir), name="baseline_b.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                pathlib.Path(temp_dir),
                probe_report=report,
                baseline_name="Baseline B",
            )

        self.assertEqual(context.baseline_name, "Baseline B")
        self.assertEqual(context.bashrc_path, bashrc_path.as_posix())
        self.assertEqual(context.runtime_base, "exaFOAM/SPUMA 0.1-v2412")
        self.assertEqual(context.reviewed_source_tuple_id, "SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0")
        self.assertTrue(context.probe_payload_path.name.endswith("probe.json"))
        self.assertTrue(context.host_env_path.name.endswith("host_env.json"))
        self.assertTrue(context.manifest_refs_path.name.endswith("manifest_refs.json"))

    def test_stage_runner_context_fails_when_baseline_activation_is_missing(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "missing baseline environment activation for Baseline A",
            ):
                resolve_stage_runner_context(
                    pathlib.Path(temp_dir),
                    probe_report=report,
                    baseline_name="Baseline A",
                )

    def test_wrap_stage_command_sources_the_resolved_bashrc(self) -> None:
        bundle = load_authority_bundle(repo_root())
        case_dir = pathlib.Path("/tmp/cases/r1")

        with tempfile.TemporaryDirectory() as temp_dir:
            bashrc_path = write_bashrc(pathlib.Path(temp_dir), name="baseline_b_wrap.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                pathlib.Path(temp_dir),
                probe_report=report,
                baseline_name="Baseline B",
            )

        wrapped = wrap_stage_command(
            context,
            case_dir=case_dir,
            stage={
                "name": "transient_run",
                "cmd": "python -c 'print(\"quoted\")'",
                "cwd": "run",
                "env_prefix": "FOAM_SIGFPE=1",
            },
        )

        self.assertIn("env -i", wrapped)
        self.assertIn("PATH=/usr/bin:/bin", wrapped)
        self.assertIn("bash --noprofile --norc -c", wrapped)
        self.assertIn(f". {bashrc_path.as_posix()}", wrapped)
        self.assertIn("cd /tmp/cases/r1/run", wrapped)
        self.assertIn("FOAM_SIGFPE=1 python -c", wrapped)
        self.assertIn('print("quoted")', wrapped)
        self.assertIn("GPU_CFD_REVIEWED_SOURCE_TUPLE_ID", wrapped)
        self.assertNotIn("/opt/openfoam12", wrapped)

    def test_stage_runner_context_allows_explicit_manifest_overrides(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = write_bashrc(temp_root, name="baseline_b_override.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
            override_bashrc = write_bashrc(temp_root, name="override.sh")
            context = resolve_stage_runner_context(
                temp_root,
                probe_report=report,
                baseline_name="Baseline B",
                bashrc_path=str(override_bashrc),
                probe_payload_path=temp_root / "baseline_b" / "probe.json",
                host_env_path=temp_root / "baseline_b" / "host_env.json",
                manifest_refs_path=temp_root / "baseline_b" / "manifest_refs.json",
            )

        self.assertEqual(context.bashrc_path, override_bashrc.as_posix())
        self.assertEqual(context.host_env_path.name, "host_env.json")
        self.assertEqual(context.manifest_refs_path.name, "manifest_refs.json")

    def test_wrap_stage_command_executes_single_quoted_commands(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = temp_root / "baseline.sh"
            bashrc_path.write_text("export FROM_STAGE_BASHRC=1\n", encoding="utf-8")
            case_dir = temp_root / "case"
            case_dir.mkdir()

            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        runtime_base="OpenFOAM 12",
                        bashrc_path=str(bashrc_path),
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )
            context = resolve_stage_runner_context(
                temp_root,
                probe_report=report,
                baseline_name="Baseline A",
            )
            wrapped = wrap_stage_command(
                context,
                case_dir=case_dir,
                stage={
                    "name": "quoted_python",
                    "cmd": f"{sys.executable} -c 'import os; print(os.environ[\"FROM_STAGE_BASHRC\"])'",
                },
            )
            completed = subprocess.run(
                wrapped,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "1")

    def test_wrap_stage_command_scrubs_inherited_openfoam_environment(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = temp_root / "baseline.sh"
            bashrc_path.write_text("", encoding="utf-8")
            case_dir = temp_root / "case"
            case_dir.mkdir()

            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        runtime_base="OpenFOAM 12",
                        bashrc_path=str(bashrc_path),
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )
            context = resolve_stage_runner_context(
                temp_root,
                probe_report=report,
                baseline_name="Baseline A",
            )
            wrapped = wrap_stage_command(
                context,
                case_dir=case_dir,
                stage={
                    "name": "check_clean_env",
                    "cmd": 'if [ -n "${WM_PROJECT_VERSION:-}" ]; then printf "%s" "$WM_PROJECT_VERSION"; else printf missing; fi',
                },
            )
            completed = subprocess.run(
                wrapped,
                shell=True,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "WM_PROJECT_VERSION": "leaked"},
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "missing")

    def test_stage_runner_context_rejects_diagnostic_probe_payloads(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            bashrc_path = write_bashrc(pathlib.Path(temp_dir), name="baseline_b_diag.sh")
            command_paths = sample_commands()
            command_paths["foamRun"] = ""

            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline B": BaselineProbeRequest(
                        lane="experimental",
                        runtime_base="exaFOAM/SPUMA 0.1-v2412",
                        bashrc_path=str(bashrc_path),
                        host_observations=sample_host_observations(lane="experimental"),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="def456abc123",
                        command_paths=command_paths,
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "v2412",
                            "WM_OPTIONS": "linux64ClangDPInt32Opt",
                        },
                    ),
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "probe payload for Baseline B is not runnable; status=diagnostic",
            ):
                resolve_stage_runner_context(
                    pathlib.Path(temp_dir),
                    probe_report=report,
                    baseline_name="Baseline B",
                )

    def test_stage_runner_context_resolves_relative_artifacts_to_absolute_paths(self) -> None:
        bundle = load_authority_bundle(repo_root())
        repo = repo_root()

        with tempfile.TemporaryDirectory(dir=repo) as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            relative_root = temp_root.relative_to(repo)
            bashrc_path = write_bashrc(temp_root, name="baseline_a_relative.sh")

            report = probe_openfoam_baselines(
                bundle,
                output_dir=relative_root,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        runtime_base="OpenFOAM 12",
                        bashrc_path=str(bashrc_path),
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )
            context = resolve_stage_runner_context(
                relative_root,
                probe_report=report,
                baseline_name="Baseline A",
            )

        self.assertTrue(context.probe_payload_path.is_absolute())

    def test_build_and_emit_case_bundle_uses_stage_runner_context(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = write_bashrc(temp_root, name="baseline_b_emit.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                case_role="R1",
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
                case_role="R1",
                phase_gate="Phase 2",
                conditional_reason="patch-manifest coverage under test",
                stages=[
                    {"name": "checkMesh_build", "cmd": "checkMesh"},
                    {
                        "name": "transient_run",
                        "cmd": "foamRun -solver incompressibleVoF",
                        "cwd": "run",
                    },
                ],
            )
            emitted = emit_case_bundle(
                bundle,
                output_dir=temp_root / "case_artifacts",
                case_meta=case_meta,
                stage_plan=stage_plan,
            )

            emitted_case_meta = json.loads(emitted.case_meta_path.read_text(encoding="utf-8"))
            emitted_stage_plan = json.loads(emitted.stage_plan_path.read_text(encoding="utf-8"))

        self.assertEqual(emitted_case_meta["baseline"], "Baseline B")
        self.assertEqual(
            emitted_case_meta["openfoam_bashrc_used"],
            bashrc_path.as_posix(),
        )
        self.assertEqual(
            emitted_case_meta["available_commands"]["foamRun"],
            sample_commands()["foamRun"],
        )
        self.assertEqual(
            emitted_case_meta["provenance"]["probe_payload"],
            context.probe_payload_path.as_posix(),
        )
        self.assertEqual(emitted_stage_plan["phase_gate"], "Phase 2")
        self.assertEqual(
            emitted_stage_plan["phase_gate_selection"]["available_case_roles"],
            ["R2", "R1-core", "R1"],
        )
        self.assertTrue(emitted_stage_plan["phase_gate_selection"]["conditional_selection"])
        self.assertEqual(
            emitted_stage_plan["provenance"]["manifest_refs"],
            context.manifest_refs_path.as_posix(),
        )

    def test_build_payload_helpers_preserve_context_owned_provenance_refs(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = write_bashrc(temp_root, name="baseline_b_reserved.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                case_role="R1",
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
                provenance={
                    "probe_payload": "wrong/probe.json",
                    "host_env": "wrong/host_env.json",
                    "manifest_refs": "wrong/manifest_refs.json",
                    "reference_freeze_overlay": "reference_freeze_overlay.json",
                },
            )

        self.assertEqual(
            case_meta["provenance"]["probe_payload"],
            context.probe_payload_path.as_posix(),
        )
        self.assertEqual(
            case_meta["provenance"]["host_env"],
            context.host_env_path.as_posix(),
        )
        self.assertEqual(
            case_meta["provenance"]["manifest_refs"],
            context.manifest_refs_path.as_posix(),
        )
        self.assertEqual(
            case_meta["provenance"]["reference_freeze_overlay"],
            "reference_freeze_overlay.json",
        )

    def test_emit_case_bundle_allows_case_specific_provenance_fields(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = write_bashrc(temp_root, name="baseline_b_case_specific.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                case_role="R1",
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
                provenance={"reference_freeze_overlay": "reference_freeze_overlay.json"},
            )
            stage_plan = build_stage_plan_payload(
                bundle,
                context=context,
                case_role="R1",
                phase_gate="Phase 2",
                conditional_reason="patch-manifest coverage under test",
                stages=[{"name": "transient_run", "cmd": "foamRun -solver incompressibleVoF"}],
            )
            emitted = emit_case_bundle(
                bundle,
                output_dir=temp_root / "case_artifacts",
                case_meta=case_meta,
                stage_plan=stage_plan,
            )
            emitted_case_meta = json.loads(emitted.case_meta_path.read_text(encoding="utf-8"))

        self.assertEqual(
            emitted_case_meta["provenance"]["reference_freeze_overlay"],
            "reference_freeze_overlay.json",
        )

    def test_emit_case_bundle_rejects_runtime_and_shared_provenance_mismatch(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            bashrc_path = write_bashrc(temp_root, name="baseline_b_mismatch.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
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
                case_role="R1",
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
                case_role="R1",
                phase_gate="Phase 2",
                conditional_reason="patch-manifest coverage under test",
                stages=[{"name": "transient_run", "cmd": "foamRun -solver incompressibleVoF"}],
            )
            runtime_mismatch_stage_plan = dict(stage_plan)
            runtime_mismatch_stage_plan["runtime_base"] = "OpenFOAM 12"

            with self.assertRaisesRegex(
                AuthoritySelectionError,
                "case bundle runtime_base mismatch",
            ):
                emit_case_bundle(
                    bundle,
                    output_dir=temp_root / "case_artifacts_runtime",
                    case_meta=case_meta,
                    stage_plan=runtime_mismatch_stage_plan,
                )

            provenance_mismatch_stage_plan = dict(stage_plan)
            provenance_mismatch_stage_plan["provenance"] = dict(stage_plan["provenance"])
            provenance_mismatch_stage_plan["provenance"]["manifest_refs"] = "stale/manifest_refs.json"

            with self.assertRaisesRegex(
                AuthoritySelectionError,
                "case bundle provenance manifest_refs mismatch",
            ):
                emit_case_bundle(
                    bundle,
                    output_dir=temp_root / "case_artifacts_provenance",
                    case_meta=case_meta,
                    stage_plan=provenance_mismatch_stage_plan,
                )
        self.assertTrue(context.host_env_path.is_absolute())
        self.assertTrue(context.manifest_refs_path.is_absolute())

    def test_stage_runner_context_rejects_missing_bashrc_file(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        runtime_base="OpenFOAM 12",
                        bashrc_path="/tmp/definitely-missing-bashrc-for-stage-runner-test",
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "baseline bashrc for Baseline A does not exist",
            ):
                resolve_stage_runner_context(
                    pathlib.Path(temp_dir),
                    probe_report=report,
                    baseline_name="Baseline A",
                )

    def test_stage_runner_log_context_includes_baseline_and_manifest_provenance(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            bashrc_path = write_bashrc(pathlib.Path(temp_dir), name="baseline_a_log.sh")
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        runtime_base="OpenFOAM 12",
                        bashrc_path=str(bashrc_path),
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                },
            )
            context = resolve_stage_runner_context(
                pathlib.Path(temp_dir),
                probe_report=report,
                baseline_name="Baseline A",
            )

        log_context = render_stage_runner_log_context(context, stage_name="checkMesh")

        self.assertIn("stage=checkMesh", log_context)
        self.assertIn("baseline=Baseline A", log_context)
        self.assertIn(f"bashrc_path={bashrc_path.as_posix()}", log_context)
        self.assertIn("runtime_base=OpenFOAM 12", log_context)
        self.assertIn("host_env_manifest=", log_context)
        self.assertIn("manifest_refs=", log_context)
        self.assertIn("probe_payload=", log_context)


if __name__ == "__main__":
    unittest.main()
