from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.authority import emit_phase1_discovery_artifacts, load_authority_bundle
from scripts.authority.phase1_build import (
    Phase1BuildError,
    plan_phase1_build,
    render_phase1_env_exports,
    run_phase1_build,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def sample_host_observations(*, lane: str = "primary") -> dict[str, str]:
    nvcc_version = "Cuda compilation tools, release 12.9, V12.9.1"
    if lane == "experimental":
        nvcc_version = "Cuda compilation tools, release 13.2, V13.2.0"
    return {
        "hostname": "ws-rtx5080-01",
        "gpu_query": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
        "nvcc_version": nvcc_version,
        "gcc_version": "gcc (Ubuntu 14.2.0) 14.2.0",
        "nsys_version": "NVIDIA Nsight Systems version 2025.2",
        "ncu_version": "NVIDIA Nsight Compute version 2025.3",
        "compute_sanitizer_version": "Compute Sanitizer version 2025.1",
        "compiler_version": "gcc (Ubuntu 14.2.0) 14.2.0",
        "os_release": "Ubuntu 24.04.2 LTS",
        "kernel": "6.8.0-60-generic",
        "nvcc_path": "/opt/cuda/bin/nvcc",
        "nsys_path": "/opt/nsight-systems/bin/nsys",
        "ncu_path": "/opt/nsight-compute/bin/ncu",
        "compute_sanitizer_path": "/opt/cuda/bin/compute-sanitizer",
    }


def sample_local_mirror_refs() -> dict[str, str]:
    return {
        "SPUMA runtime base": "3d1d7bf598ec8a66e099d8688b8597422c361960",
        "SPUMA support-policy snapshot": "ad2a385e44f2c01b7d1df44c5bc51d7996c95554",
        "External solver bridge": "4c764d027f8f124a1cc0b6df0520eb63593c2a2b",
        "AmgX backend": "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47",
    }


def sample_cuda_probe_payload() -> dict[str, object]:
    return {
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5080",
        "cc_major": 12,
        "cc_minor": 0,
        "total_global_mem_bytes": 17179869184,
        "managed_memory": True,
        "concurrent_managed_access": True,
        "unified_addressing": True,
        "native_kernel_ok": True,
        "managed_memory_probe_ok": True,
    }


class Phase1BuildTests(unittest.TestCase):
    def test_plan_phase1_build_emits_build_manifests_metadata_and_env(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_root / "discovery",
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            source_root.mkdir()
            allwmake = source_root / "Allwmake"
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )

            build_host_env = json.loads(plan.host_env_path.read_text(encoding="utf-8"))
            build_manifest_refs = json.loads(
                plan.manifest_refs_path.read_text(encoding="utf-8")
            )
            metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(plan.env_exports["have_cuda"], "true")
        self.assertEqual(plan.env_exports["NVARCH"], "120")
        self.assertEqual(plan.env_exports["CUDA_HOME"], "/usr/local/cuda-12.9")
        self.assertEqual(plan.env_exports["SPUMA_ENABLE_NVTX"], "1")
        self.assertIn(
            "-gencode=arch=compute_120,code=sm_120",
            plan.env_exports["SPUMA_EXTRA_NVCC_FLAGS"],
        )
        self.assertIn(
            "-gencode=arch=compute_120,code=compute_120",
            plan.env_exports["SPUMA_EXTRA_NVCC_FLAGS"],
        )
        self.assertNotIn("-use_fast_math", plan.env_exports["SPUMA_EXTRA_NVCC_FLAGS"])
        self.assertEqual(build_host_env["consumer"], "build")
        self.assertEqual(build_host_env["toolkit"]["selected_lane"], "primary")
        self.assertEqual(build_manifest_refs["consumer"], "build")
        self.assertEqual(metadata["lane"], "primary")
        self.assertEqual(metadata["mode"], "relwithdebinfo")
        self.assertEqual(metadata["build_entrypoint"], "./Allwmake")
        self.assertTrue(metadata["ptx_retention_required"])
        self.assertEqual(metadata["have_cuda"], True)
        self.assertEqual(metadata["nvarch"], 120)
        self.assertEqual(metadata["instrumentation"], "NVTX3")

    def test_plan_phase1_build_rejects_nvtx2_includes(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_root / "discovery",
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            source_root.mkdir()
            bad_header = source_root / "GpuTrace.C"
            bad_header.write_text(
                "#include <nvToolsExt.h>\nint main() { return 0; }\n",
                encoding="utf-8",
            )
            allwmake = source_root / "Allwmake"
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)

            with self.assertRaisesRegex(Phase1BuildError, "NVTX2|nvToolsExt"):
                plan_phase1_build(
                    bundle,
                    source_root=source_root,
                    output_dir=temp_root / "build",
                    discovery_host_env_path=emitted.host_env_path,
                    discovery_manifest_refs_path=emitted.manifest_refs_path,
                    cuda_probe_path=emitted.cuda_probe_path,
                )

    def test_render_phase1_env_exports_emits_sourceable_exports(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_root / "discovery",
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            source_root.mkdir()
            allwmake = source_root / "Allwmake"
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )

        exports = render_phase1_env_exports(plan)
        self.assertIn("export have_cuda=true", exports)
        self.assertIn("export NVARCH=120", exports)
        self.assertIn("export CUDA_HOME=/usr/local/cuda-12.9", exports)
        self.assertIn("export SPUMA_ENABLE_NVTX=1", exports)

    def test_plan_phase1_build_requires_repo_native_build_entrypoint(self) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_root / "discovery",
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            source_root.mkdir()

            with self.assertRaisesRegex(Phase1BuildError, "repo-native build entrypoint"):
                plan_phase1_build(
                    bundle,
                    source_root=source_root,
                    output_dir=temp_root / "build",
                    discovery_host_env_path=emitted.host_env_path,
                    discovery_manifest_refs_path=emitted.manifest_refs_path,
                    cuda_probe_path=emitted.cuda_probe_path,
                )

    def test_run_phase1_build_executes_repo_native_entrypoint_and_captures_log(
        self,
    ) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_root / "discovery",
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            source_root.mkdir()
            allwmake = source_root / "Allwmake"
            allwmake.write_text(
                (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "echo \"have_cuda=${have_cuda}\"\n"
                    "echo \"NVARCH=${NVARCH}\"\n"
                    "echo \"CUDA_HOME=${CUDA_HOME}\"\n"
                    "echo \"SPUMA_ENABLE_NVTX=${SPUMA_ENABLE_NVTX}\"\n"
                ),
                encoding="utf-8",
            )
            allwmake.chmod(0o755)

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )
            result = run_phase1_build(plan)
            metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
            build_log = plan.log_path.read_text(encoding="utf-8")

        self.assertTrue(result.succeeded)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(metadata["returncode"], 0)
        self.assertTrue(metadata["succeeded"])
        self.assertIn("have_cuda=true", build_log)
        self.assertIn("NVARCH=120", build_log)
        self.assertIn("CUDA_HOME=/usr/local/cuda-12.9", build_log)
        self.assertIn("SPUMA_ENABLE_NVTX=1", build_log)


if __name__ == "__main__":
    unittest.main()
