from __future__ import annotations

import pathlib
import tempfile
import unittest

from scripts.authority import BaselineProbeRequest, load_authority_bundle, probe_openfoam_baselines


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


class OpenFOAMBaselineProbeTests(unittest.TestCase):
    def test_probe_openfoam_baselines_emits_baseline_records_and_canonical_artifacts(
        self,
    ) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline A": BaselineProbeRequest(
                        lane="primary",
                        bashrc_path="/envs/baseline-a/etc/bashrc",
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="abc123def456",
                        command_paths=sample_commands(),
                        library_hints={
                            "libpetscFoam.so": "/envs/baseline-a/lib/libpetscFoam.so",
                            "amgx": "/envs/baseline-a/amgx/system",
                        },
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "12",
                            "WM_OPTIONS": "linux64GccDPInt32Opt",
                        },
                    ),
                    "Baseline B": BaselineProbeRequest(
                        lane="experimental",
                        bashrc_path="/envs/baseline-b/etc/bashrc",
                        host_observations=sample_host_observations(lane="experimental"),
                        local_mirror_refs=sample_local_mirror_refs(),
                        repo_commit="def456abc123",
                        command_paths={
                            **sample_commands(),
                            "foamRun": "/envs/baseline-b/bin/foamRun",
                        },
                        library_hints={
                            "libpetscFoam.so": "/envs/baseline-b/lib/libpetscFoam.so",
                            "amgx": "/envs/baseline-b/amgx/system",
                        },
                        openfoam_env={
                            "WM_PROJECT": "OpenFOAM",
                            "WM_PROJECT_VERSION": "v2412",
                            "WM_OPTIONS": "linux64ClangDPInt32Opt",
                        },
                    ),
                },
            )

        self.assertEqual(
            report.compatibility_aliases,
            {"env.json": "host_env.json"},
        )
        self.assertEqual(set(report.records), {"Baseline A", "Baseline B"})

        baseline_a = report.records["Baseline A"]
        baseline_b = report.records["Baseline B"]

        self.assertEqual(baseline_a["status"], "ok")
        self.assertEqual(baseline_b["status"], "ok")
        self.assertEqual(baseline_a["activation"]["bashrc_path"], "/envs/baseline-a/etc/bashrc")
        self.assertEqual(
            baseline_b["host_env"]["toolkit"]["selected_lane_value"],
            "CUDA 13.2",
        )
        self.assertEqual(
            baseline_a["artifacts"]["host_env"],
            "baseline_a/host_env.json",
        )
        self.assertEqual(
            baseline_b["artifacts"]["compatibility_aliases"]["env.json"],
            "baseline_b/env.json",
        )
        self.assertTrue(baseline_a["commands"]["foamRun"]["found"])
        self.assertEqual(
            baseline_b["library_hints"]["libpetscFoam.so"],
            "/envs/baseline-b/lib/libpetscFoam.so",
        )
        self.assertEqual(
            baseline_b["openfoam_env"]["WM_PROJECT_VERSION"],
            "v2412",
        )

    def test_probe_openfoam_baselines_records_missing_activation_diagnostic(self) -> None:
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
                    ),
                },
            )

        baseline_a = report.records["Baseline A"]
        self.assertEqual(baseline_a["status"], "diagnostic")
        self.assertTrue(
            any(
                "missing baseline environment activation" in diagnostic
                for diagnostic in baseline_a["diagnostics"]
            )
        )
        self.assertEqual(baseline_a["artifacts"], {})

    def test_probe_openfoam_baselines_records_unresolved_toolkit_lane(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline B": BaselineProbeRequest(
                        lane="nightly",
                        bashrc_path="/envs/baseline-b/etc/bashrc",
                        host_observations=sample_host_observations(),
                        local_mirror_refs=sample_local_mirror_refs(),
                        command_paths=sample_commands(),
                    ),
                },
            )

        baseline_b = report.records["Baseline B"]
        self.assertEqual(baseline_b["status"], "diagnostic")
        self.assertTrue(
            any("unresolved toolkit lane" in diagnostic for diagnostic in baseline_b["diagnostics"])
        )
        self.assertEqual(baseline_b["artifacts"], {})

    def test_probe_openfoam_baselines_notes_missing_commands(self) -> None:
        bundle = load_authority_bundle(repo_root())
        command_paths = sample_commands()
        command_paths["interFoam"] = ""
        command_paths["decomposePar"] = ""

        with tempfile.TemporaryDirectory() as temp_dir:
            report = probe_openfoam_baselines(
                bundle,
                output_dir=temp_dir,
                baselines={
                    "Baseline B": BaselineProbeRequest(
                        lane="experimental",
                        bashrc_path="/envs/baseline-b/etc/bashrc",
                        host_observations=sample_host_observations(lane="experimental"),
                        local_mirror_refs=sample_local_mirror_refs(),
                        command_paths=command_paths,
                    ),
                },
            )

        baseline_b = report.records["Baseline B"]
        self.assertEqual(baseline_b["status"], "diagnostic")
        self.assertFalse(baseline_b["commands"]["interFoam"]["found"])
        self.assertFalse(baseline_b["commands"]["decomposePar"]["found"])
        self.assertTrue(
            any("missing OpenFOAM command(s)" in diagnostic for diagnostic in baseline_b["diagnostics"])
        )
        self.assertIn("host_env", baseline_b)

    def test_scripts_do_not_embed_openfoam12_path(self) -> None:
        repo = repo_root()
        hits: list[str] = []

        for path in sorted((repo / "scripts").rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "/opt/openfoam12" in text:
                hits.append(str(path.relative_to(repo)))

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
