from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest import mock

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

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_executes_repo_native_entrypoint_and_captures_log(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = phase1_build_subprocess_side_effect(
            build_stdout=(
                "have_cuda=true\n"
                "NVARCH=120\n"
                "CUDA_HOME=/usr/local/cuda-12.9\n"
                "SPUMA_ENABLE_NVTX=1\n"
            )
        )

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
        self.assertEqual(run_subprocess.call_count, 2)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_temporarily_wires_extra_nvcc_flags_into_cuda_rules(
        self,
        run_subprocess: mock.Mock,
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
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)
            cuda_rules = source_root / "wmake" / "rules" / "General" / "cuda"
            cuda_rules.parent.mkdir(parents=True)
            original_cuda_rules = (
                "cuFLAGS    = \\\n"
                "    $(cuARCH) $(GFLAGS) $(cuWARN) $(cuOPT) $(cuDBUG) $(ptFLAGS) \\\n"
                "    $(FOAM_EXTRA_CXXFLAGS) $(CU_LIB_HEADER_DIRS)\n"
            )
            cuda_rules.write_text(original_cuda_rules, encoding="utf-8")
            probe_calls = 0

            def side_effect(*args, **kwargs):
                nonlocal probe_calls
                probe_calls += 1
                if probe_calls == 1:
                    return subprocess_completed(
                        stdout=sample_repo_native_probe_output(),
                        returncode=0,
                    )
                patched_rules = cuda_rules.read_text(encoding="utf-8")
                self.assertIn("$(SPUMA_EXTRA_NVCC_FLAGS)", patched_rules)
                stdout_handle = kwargs["stdout"]
                stdout_handle.write("Allwmake ok\n")
                stdout_handle.flush()
                return subprocess_completed(returncode=0)

            run_subprocess.side_effect = side_effect

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )
            run_phase1_build(plan)
            restored_cuda_rules = cuda_rules.read_text(encoding="utf-8")

        self.assertEqual(restored_cuda_rules, original_cuda_rules)
        self.assertEqual(run_subprocess.call_count, 2)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_maps_host_flags_to_repo_native_cxx_env(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        observed_shell_command: list[str] = []
        probe_calls = 0

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

            def side_effect(*args, **kwargs):
                nonlocal probe_calls
                probe_calls += 1
                if probe_calls == 1:
                    return subprocess_completed(
                        stdout=sample_repo_native_probe_output(),
                        returncode=0,
                    )
                observed_shell_command.extend(args[0])
                stdout_handle = kwargs["stdout"]
                stdout_handle.write("Allwmake ok\n")
                stdout_handle.flush()
                return subprocess_completed(returncode=0)

            run_subprocess.side_effect = side_effect

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )
            run_phase1_build(plan)

        self.assertEqual(run_subprocess.call_count, 2)
        self.assertIn(
            'export FOAM_EXTRA_CXXFLAGS="${FOAM_EXTRA_CXXFLAGS:+${FOAM_EXTRA_CXXFLAGS} }${SPUMA_EXTRA_CXX_FLAGS}"',
            observed_shell_command[-1],
        )

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_sources_repo_native_bashrc_when_present(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = phase1_build_subprocess_side_effect(
            build_stdout="SPUMA_BASHRC_MARKER=enabled\n"
        )

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
            etc_dir = source_root / "etc"
            etc_dir.mkdir()
            bashrc = etc_dir / "bashrc"
            bashrc.write_text(
                "export SPUMA_BASHRC_MARKER=enabled\n",
                encoding="utf-8",
            )
            allwmake = source_root / "Allwmake"
            allwmake.write_text(
                (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "echo \"SPUMA_BASHRC_MARKER=${SPUMA_BASHRC_MARKER:-missing}\"\n"
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
            run_phase1_build(plan)
            build_log = plan.log_path.read_text(encoding="utf-8")

        self.assertIn("SPUMA_BASHRC_MARKER=enabled", build_log)
        self.assertEqual(run_subprocess.call_count, 2)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_preserves_bashrc_path_updates(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = phase1_build_subprocess_side_effect(
            build_stdout="bashrc-helper-ok\n"
        )

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
            mockbin = source_root / "mockbin"
            mockbin.mkdir()
            helper = mockbin / "bashrc-helper"
            helper.write_text(
                "#!/usr/bin/env bash\necho \"bashrc-helper-ok\"\n",
                encoding="utf-8",
            )
            helper.chmod(0o755)
            etc_dir = source_root / "etc"
            etc_dir.mkdir()
            bashrc = etc_dir / "bashrc"
            bashrc.write_text(
                f"export PATH={mockbin.as_posix()}:$PATH\n",
                encoding="utf-8",
            )
            allwmake = source_root / "Allwmake"
            allwmake.write_text(
                (
                    "#!/usr/bin/env bash\n"
                    "set -euo pipefail\n"
                    "bashrc-helper\n"
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
            run_phase1_build(plan)
            build_log = plan.log_path.read_text(encoding="utf-8")

        self.assertIn("bashrc-helper-ok", build_log)
        self.assertEqual(run_subprocess.call_count, 2)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_fast_when_nvc_is_missing(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = [
            subprocess_completed(stdout="WM_COMPILER=Nvidia\nNVC=\n", returncode=0),
        ]

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
            etc_dir = source_root / "etc"
            etc_dir.mkdir()
            bashrc = etc_dir / "bashrc"
            bashrc.write_text("export WM_COMPILER=Nvidia\n", encoding="utf-8")
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
            plan.log_path.write_text("stale prior run output\n", encoding="utf-8")

            with self.assertRaisesRegex(Phase1BuildError, "nvc|Nvidia HPC"):
                run_phase1_build(plan)

            metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
            build_log = plan.log_path.read_text(encoding="utf-8")

        self.assertEqual(run_subprocess.call_count, 1)
        self.assertFalse(metadata["succeeded"])
        self.assertIsNone(metadata["returncode"])
        self.assertEqual(metadata["failure_stage"], "preflight")
        self.assertIn("nvc is unavailable", metadata["failure_reason"])
        self.assertIn("nvc is unavailable", build_log)
        self.assertNotIn("stale prior run output", build_log)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_fast_when_nvcc_version_mismatches_frozen_lane(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = [
            subprocess_completed(
                stdout=(
                    "WM_COMPILER=Gcc\n"
                    "NVC=\n"
                    "NVCC=/usr/bin/nvcc\n"
                    "NVCC_VERSION=Cuda compilation tools, release 12.0, V12.0.140\n"
                    "NSYS=/usr/bin/nsys\n"
                    "NSYS_VERSION=NVIDIA Nsight Systems version 2025.2\n"
                    "COMPUTE_SANITIZER=/usr/bin/compute-sanitizer\n"
                    "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2025.1\n"
                    "NVIDIA_SMI=/usr/bin/nvidia-smi\n"
                    "NVIDIA_SMI_GPU=NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n"
                ),
                returncode=0,
            ),
        ]

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

            with self.assertRaisesRegex(Phase1BuildError, "nvcc|12.9.1|12.0"):
                run_phase1_build(plan)

        self.assertEqual(run_subprocess.call_count, 1)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_fast_when_nsys_version_mismatches_frozen_lane(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = [
            subprocess_completed(
                stdout=(
                    "WM_COMPILER=Gcc\n"
                    "NVC=\n"
                    "NVCC=/usr/bin/nvcc\n"
                    "NVCC_VERSION=Cuda compilation tools, release 12.9, V12.9.1\n"
                    "NSYS=/usr/bin/nsys\n"
                    "NSYS_VERSION=NVIDIA Nsight Systems version 2022.4\n"
                    "COMPUTE_SANITIZER=/usr/bin/compute-sanitizer\n"
                    "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2025.1\n"
                    "NVIDIA_SMI=/usr/bin/nvidia-smi\n"
                    "NVIDIA_SMI_GPU=NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n"
                ),
                returncode=0,
            ),
        ]

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

            with self.assertRaisesRegex(Phase1BuildError, "nsys|2025.2|2022.4"):
                run_phase1_build(plan)

        self.assertEqual(run_subprocess.call_count, 1)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_fast_when_compute_sanitizer_version_mismatches_frozen_lane(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = [
            subprocess_completed(
                stdout=(
                    "WM_COMPILER=Gcc\n"
                    "NVC=\n"
                    "NVCC=/usr/bin/nvcc\n"
                    "NVCC_VERSION=Cuda compilation tools, release 12.9, V12.9.1\n"
                    "NSYS=/usr/bin/nsys\n"
                    "NSYS_VERSION=NVIDIA Nsight Systems version 2025.2\n"
                    "COMPUTE_SANITIZER=/usr/bin/compute-sanitizer\n"
                    "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2022.4\n"
                    "NVIDIA_SMI=/usr/bin/nvidia-smi\n"
                    "NVIDIA_SMI_GPU=NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n"
                ),
                returncode=0,
            ),
        ]

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

            with self.assertRaisesRegex(
                Phase1BuildError,
                "compute-sanitizer|2025.1|2022.4",
            ):
                run_phase1_build(plan)

        self.assertEqual(run_subprocess.call_count, 1)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_fast_when_nvidia_smi_is_missing(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = [
            subprocess_completed(
                stdout=(
                    "WM_COMPILER=Gcc\n"
                    "NVC=\n"
                    "NVCC=/usr/bin/nvcc\n"
                    "NVCC_VERSION=Cuda compilation tools, release 12.9, V12.9.1\n"
                    "NSYS=/usr/bin/nsys\n"
                    "NSYS_VERSION=NVIDIA Nsight Systems version 2025.2\n"
                    "COMPUTE_SANITIZER=/usr/bin/compute-sanitizer\n"
                    "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2025.1\n"
                    "NVIDIA_SMI=\n"
                    "NVIDIA_SMI_GPU=\n"
                ),
                returncode=0,
            ),
        ]

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

            with self.assertRaisesRegex(Phase1BuildError, "nvidia-smi|gpu_csv"):
                run_phase1_build(plan)

        self.assertEqual(run_subprocess.call_count, 1)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_records_build_command_failures(
        self,
        run_subprocess: mock.Mock,
    ) -> None:
        bundle = load_authority_bundle(repo_root())
        run_subprocess.side_effect = phase1_build_subprocess_side_effect(
            build_stdout="Allwmake failed\n",
            build_returncode=5,
        )

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
            allwmake.write_text("#!/usr/bin/env bash\nexit 5\n", encoding="utf-8")
            allwmake.chmod(0o755)

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "build",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )

            with self.assertRaisesRegex(Phase1BuildError, "exit code 5"):
                run_phase1_build(plan)

            metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
            build_log = plan.log_path.read_text(encoding="utf-8")

        self.assertEqual(run_subprocess.call_count, 2)
        self.assertFalse(metadata["succeeded"])
        self.assertEqual(metadata["returncode"], 5)
        self.assertEqual(metadata["failure_stage"], "build")
        self.assertIn("exit code 5", metadata["failure_reason"])
        self.assertIn("Allwmake failed", build_log)


def subprocess_completed(*, stdout: str = "", returncode: int = 0):
    return mock.Mock(stdout=stdout, returncode=returncode)


def phase1_build_subprocess_side_effect(*, build_stdout: str, build_returncode: int = 0):
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess_completed(
                stdout=sample_repo_native_probe_output(),
                returncode=0,
            )
        stdout_handle = kwargs["stdout"]
        stdout_handle.write(build_stdout)
        stdout_handle.flush()
        return subprocess_completed(returncode=build_returncode)

    return side_effect


def sample_repo_native_probe_output() -> str:
    return (
        "WM_COMPILER=Gcc\n"
        "NVC=\n"
        "NVCC=/usr/bin/nvcc\n"
        "NVCC_VERSION=Cuda compilation tools, release 12.9, V12.9.1\n"
        "NSYS=/usr/bin/nsys\n"
        "NSYS_VERSION=NVIDIA Nsight Systems version 2025.2\n"
        "COMPUTE_SANITIZER=/usr/bin/compute-sanitizer\n"
        "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2025.1\n"
        "NVIDIA_SMI=/usr/bin/nvidia-smi\n"
        "NVIDIA_SMI_GPU=NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n"
    )


if __name__ == "__main__":
    unittest.main()
