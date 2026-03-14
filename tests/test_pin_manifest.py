from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import (
    AuthorityConflictError,
    emit_environment_manifests,
    load_authority_bundle,
    resolve_consumer_pin_manifest,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class PinManifestResolutionTests(unittest.TestCase):
    def test_resolves_frozen_pin_manifest_and_emits_canonical_artifacts(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = pathlib.Path(temp_dir)
            emitted = emit_environment_manifests(
                bundle,
                consumer="build",
                output_dir=output_dir,
                host_observations={
                    "gpu_query": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
                    "nvcc_version": "Cuda compilation tools, release 12.9, V12.9.1",
                    "gcc_version": "gcc (Ubuntu 14.2.0) 14.2.0",
                    "nsys_version": "NVIDIA Nsight Systems version 2025.2",
                    "ncu_version": "NVIDIA Nsight Compute version 2025.3",
                    "compute_sanitizer_version": "Compute Sanitizer version 2025.1",
                    "compiler_version": "gcc (Ubuntu 14.2.0) 14.2.0",
                },
                local_mirror_refs={
                    "SPUMA runtime base": "3d1d7bf598ec8a66e099d8688b8597422c361960",
                    "SPUMA support-policy snapshot": "ad2a385e44f2c01b7d1df44c5bc51d7996c95554",
                    "External solver bridge": "4c764d027f8f124a1cc0b6df0520eb63593c2a2b",
                    "AmgX backend": "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47",
                },
                repo_commit="abc123def456",
            )

            self.assertEqual(emitted.host_env_path.name, "host_env.json")
            self.assertEqual(emitted.manifest_refs_path.name, "manifest_refs.json")
            self.assertEqual(emitted.alias_paths["env.json"].name, "env.json")

            host_env = json.loads(emitted.host_env_path.read_text(encoding="utf-8"))
            manifest_refs = json.loads(emitted.manifest_refs_path.read_text(encoding="utf-8"))
            env_alias = json.loads(emitted.alias_paths["env.json"].read_text(encoding="utf-8"))

        self.assertEqual(host_env["consumer"], "build")
        self.assertEqual(host_env["runtime_base"], "exaFOAM/SPUMA 0.1-v2412")
        self.assertEqual(host_env["toolkit"]["primary_lane"], "CUDA 12.9.1")
        self.assertEqual(host_env["toolkit"]["experimental_lane"], "CUDA 13.2")
        self.assertEqual(host_env["toolkit"]["selected_lane"], "primary")
        self.assertEqual(host_env["toolkit"]["driver_floor"], ">=595.45.04")
        self.assertEqual(host_env["gpu_target"], "`NVARCH=120`, native `sm_120` plus PTX")
        self.assertEqual(host_env["instrumentation"], "NVTX3")
        self.assertEqual(host_env["profilers"]["nsight_systems"], "2025.2")
        self.assertEqual(host_env["profilers"]["nsight_compute"], "2025.3")
        self.assertEqual(host_env["profilers"]["compute_sanitizer"], "2025.1")
        self.assertEqual(
            host_env["compatibility_aliases"],
            {"env.json": "host_env.json"},
        )
        self.assertEqual(env_alias["canonical_name"], "host_env.json")

        self.assertEqual(
            manifest_refs["reviewed_source_tuple_id"],
            "SRC_SPUMA_V2412_AD2A_FES_MAIN_AMGX_2_5_0",
        )
        self.assertEqual(
            manifest_refs["source_components"]["AmgX backend"]["resolved_commit"],
            "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47",
        )
        self.assertEqual(
            manifest_refs["source_components"]["External solver bridge"]["frozen_ref"],
            "main",
        )
        self.assertEqual(manifest_refs["repo"]["git_commit"], "abc123def456")
        self.assertIn("master_pin_manifest", manifest_refs["authority_revisions"])

    def test_conflicting_local_override_fails_fast(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with self.assertRaisesRegex(
            AuthorityConflictError,
            "primary_toolkit_lane.*CUDA 12.9.1.*CUDA 12.8",
        ):
            resolve_consumer_pin_manifest(
                bundle,
                consumer="run",
                overrides={"primary_toolkit_lane": "CUDA 12.8"},
            )

    def test_build_run_and_profiling_consumers_share_same_frozen_resolution(self) -> None:
        bundle = load_authority_bundle(repo_root())

        build_resolution = resolve_consumer_pin_manifest(bundle, consumer="build")
        run_resolution = resolve_consumer_pin_manifest(bundle, consumer="run")
        profiling_resolution = resolve_consumer_pin_manifest(bundle, consumer="profiling")

        self.assertEqual(
            build_resolution.shared_resolution_key,
            run_resolution.shared_resolution_key,
        )
        self.assertEqual(
            run_resolution.shared_resolution_key,
            profiling_resolution.shared_resolution_key,
        )
        self.assertEqual(build_resolution.host_env["toolkit"]["primary_lane"], "CUDA 12.9.1")
        self.assertEqual(run_resolution.host_env["toolkit"]["primary_lane"], "CUDA 12.9.1")
        self.assertEqual(
            profiling_resolution.host_env["profilers"]["compute_sanitizer"],
            "2025.1",
        )


if __name__ == "__main__":
    unittest.main()
