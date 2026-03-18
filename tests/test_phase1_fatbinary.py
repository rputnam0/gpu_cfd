from __future__ import annotations

import json
import pathlib
import socket
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts.authority import emit_phase1_discovery_artifacts, load_authority_bundle
from scripts.authority.phase1_build import (
    Phase1BuildError,
    inspect_phase1_build_fatbinaries,
    plan_phase1_build,
    run_phase1_build,
)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def materialize_sample_toolchain(root: pathlib.Path) -> dict[str, str]:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for tool_name in (
        "nvc",
        "nvcc",
        "nsys",
        "ncu",
        "compute-sanitizer",
        "nvidia-smi",
        "cuobjdump",
    ):
        tool_path = bin_dir / tool_name
        tool_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        tool_path.chmod(0o755)
        paths[tool_name.replace("-", "_")] = tool_path.as_posix()
    return paths


def sample_host_observations(*, tool_root: pathlib.Path) -> dict[str, str]:
    tool_paths = materialize_sample_toolchain(tool_root)
    return {
        "hostname": socket.gethostname().strip() or "phase1-fatbinary-test-host",
        "gpu_csv": "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
        "nvcc_version": "Cuda compilation tools, release 12.9, V12.9.86",
        "gcc_version": "gcc (Ubuntu 14.2.0) 14.2.0",
        "nsys_version": "NVIDIA Nsight Systems version 2025.2.1.130-252135690618v0",
        "ncu_version": "NVIDIA Nsight Compute version 2025.3.1.0",
        "compute_sanitizer_version": "Compute Sanitizer version 2025.1.0.0",
        "os_release": "Ubuntu 24.04.2 LTS",
        "kernel": "6.8.0-60-generic",
        "nvidia_smi_path": tool_paths["nvidia_smi"],
        "nvc_path": tool_paths["nvc"],
        "nvcc_path": tool_paths["nvcc"],
        "nsys_path": tool_paths["nsys"],
        "ncu_path": tool_paths["ncu"],
        "compute_sanitizer_path": tool_paths["compute_sanitizer"],
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


def subprocess_completed(*, stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=(), returncode=returncode, stdout=stdout, stderr=stderr)


def cuobjdump_side_effect(*, sass_stdout: str, ptx_stdout: str):
    def side_effect(command, *args, **kwargs):
        if "-sass" in command:
            return subprocess_completed(stdout=sass_stdout)
        if "-ptx" in command:
            return subprocess_completed(stdout=ptx_stdout)
        raise AssertionError(f"unexpected subprocess command: {command!r}")

    return side_effect


class Phase1FatbinaryTests(unittest.TestCase):
    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_inspect_phase1_build_fatbinaries_emits_report_and_dump_artifacts(
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
                host_observations=sample_host_observations(tool_root=temp_root / "toolchain"),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            allwmake = source_root / "Allwmake"
            source_root.mkdir()
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)
            target = source_root / "platforms" / "linux64GccDPInt32Opt" / "bin" / "pimpleFoam"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("binary", encoding="utf-8")
            target.chmod(0o755)
            run_subprocess.side_effect = cuobjdump_side_effect(
                sass_stdout="Fatbin elf code:\n================\narch = sm_120\ncode for sm_120\n",
                ptx_stdout=".version 8.0\n.target sm_120\n.address_size 64\n",
            )

            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "inspection",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )
            result = inspect_phase1_build_fatbinaries(plan)

            report = json.loads(plan.fatbinary_report_path.read_text(encoding="utf-8"))
            ptx_dump = plan.ptx_dump_path.read_text(encoding="utf-8")
            sass_dump = plan.sass_dump_path.read_text(encoding="utf-8")

        self.assertTrue(result.smoke_gate_ready)
        self.assertEqual(report["native_sm_present"], [120])
        self.assertEqual(report["ptx_targets"], [120])
        self.assertTrue(report["ptx_present"])
        self.assertTrue(report["required_native_sm_found"])
        self.assertEqual(report["inspected_binary_count"], 1)
        self.assertEqual(
            report["inspection_targets"][0]["path"],
            "platforms/linux64GccDPInt32Opt/bin/pimpleFoam",
        )
        self.assertIn(".target sm_120", ptx_dump)
        self.assertIn("arch = sm_120", sass_dump)

    @mock.patch("scripts.authority.phase1_build.subprocess.run")
    def test_run_phase1_build_fails_after_build_when_ptx_is_missing(
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
                host_observations=sample_host_observations(tool_root=temp_root / "toolchain"),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )
            source_root = temp_root / "spuma"
            allwmake = source_root / "Allwmake"
            source_root.mkdir()
            allwmake.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            allwmake.chmod(0o755)
            target = source_root / "platforms" / "linux64GccDPInt32Opt" / "bin" / "simpleFoam"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("binary", encoding="utf-8")
            target.chmod(0o755)
            plan = plan_phase1_build(
                bundle,
                source_root=source_root,
                output_dir=temp_root / "inspection",
                discovery_host_env_path=emitted.host_env_path,
                discovery_manifest_refs_path=emitted.manifest_refs_path,
                cuda_probe_path=emitted.cuda_probe_path,
            )

            def side_effect(command, *args, **kwargs):
                if "-sass" in command:
                    return subprocess_completed(
                        stdout="Fatbin elf code:\n================\narch = sm_120\n"
                    )
                if "-ptx" in command:
                    return subprocess_completed(stdout="")
                if kwargs.get("stdout") is not None:
                    handle = kwargs["stdout"]
                    handle.write("build ok\n")
                    handle.flush()
                    return subprocess_completed(returncode=0)
                return subprocess_completed(
                    stdout=sample_repo_native_probe_output(tool_root=temp_root / "toolchain"),
                    returncode=0,
                )

            run_subprocess.side_effect = side_effect

            with self.assertRaisesRegex(Phase1BuildError, "PTX"):
                run_phase1_build(plan)

            report = json.loads(plan.fatbinary_report_path.read_text(encoding="utf-8"))
            metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))

        self.assertFalse(report["smoke_gate_ready"])
        self.assertFalse(report["ptx_present"])
        self.assertEqual(report["native_sm_present"], [120])
        self.assertIn("PTX", report["failure_reason"])
        self.assertFalse(metadata["succeeded"])
        self.assertEqual(metadata["failure_stage"], "inspection")

    def test_inspect_fatbinary_wrapper_exists_and_invokes_phase1_build_module(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "build" / "inspect_fatbinary.sh"

        self.assertTrue(wrapper_path.is_file())

        wrapper = wrapper_path.read_text(encoding="utf-8")
        self.assertIn("scripts/authority/phase1_build.py inspect", wrapper)
        self.assertIn("GPU_CFD_HOST_ENV", wrapper)
        self.assertIn("GPU_CFD_MANIFEST_REFS", wrapper)
        self.assertIn("GPU_CFD_CUDA_PROBE", wrapper)


def sample_repo_native_probe_output(*, tool_root: pathlib.Path) -> str:
    tool_paths = materialize_sample_toolchain(tool_root)
    return (
        "WM_COMPILER=Gcc\n"
        f"NVC={tool_paths['nvc']}\n"
        f"NVCC={tool_paths['nvcc']}\n"
        "NVCC_VERSION=Cuda compilation tools, release 12.9, V12.9.86\n"
        f"NSYS={tool_paths['nsys']}\n"
        "NSYS_VERSION=NVIDIA Nsight Systems version 2025.2.1.130-252135690618v0\n"
        f"COMPUTE_SANITIZER={tool_paths['compute_sanitizer']}\n"
        "COMPUTE_SANITIZER_VERSION=Compute Sanitizer version 2025.1.0.0\n"
        f"NVIDIA_SMI={tool_paths['nvidia_smi']}\n"
        "NVIDIA_SMI_GPU=NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n"
    )


if __name__ == "__main__":
    unittest.main()
