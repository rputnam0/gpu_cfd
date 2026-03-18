from __future__ import annotations

import json
import io
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from scripts.authority import (
    AuthorityConflictError,
    emit_phase1_discovery_artifacts,
    load_authority_bundle,
)
from scripts.authority.phase1_discovery import collect_host_observations, main


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


class Phase1DiscoveryTests(unittest.TestCase):
    def test_emit_phase1_discovery_artifacts_emits_canonical_host_and_cuda_probe_json(
        self,
    ) -> None:
        bundle = load_authority_bundle(repo_root())

        with tempfile.TemporaryDirectory() as temp_dir:
            emitted = emit_phase1_discovery_artifacts(
                bundle,
                output_dir=temp_dir,
                lane="primary",
                host_observations=sample_host_observations(),
                cuda_probe=sample_cuda_probe_payload(),
                local_mirror_refs=sample_local_mirror_refs(),
                repo_commit="abc123def456",
            )

            host_env = json.loads(emitted.host_env_path.read_text(encoding="utf-8"))
            manifest_refs = json.loads(
                emitted.manifest_refs_path.read_text(encoding="utf-8")
            )
            cuda_probe = json.loads(emitted.cuda_probe_path.read_text(encoding="utf-8"))

        self.assertEqual(emitted.host_env_path.name, "host_env.json")
        self.assertEqual(emitted.manifest_refs_path.name, "manifest_refs.json")
        self.assertEqual(emitted.cuda_probe_path.name, "cuda_probe.json")
        self.assertEqual(host_env["canonical_name"], "host_env.json")
        self.assertEqual(host_env["host_observations"]["hostname"], "ws-rtx5080-01")
        self.assertEqual(
            manifest_refs["invoked_tool_paths"]["compute_sanitizer_path"],
            "/opt/cuda/bin/compute-sanitizer",
        )
        self.assertEqual(cuda_probe["canonical_name"], "cuda_probe.json")
        self.assertEqual(cuda_probe["toolkit"]["selected_lane"], "primary")
        self.assertEqual(cuda_probe["device_name"], "NVIDIA GeForce RTX 5080")
        self.assertEqual(cuda_probe["cc_major"], 12)
        self.assertEqual(cuda_probe["cc_minor"], 0)
        self.assertTrue(cuda_probe["native_kernel_ok"])
        self.assertTrue(cuda_probe["managed_memory_probe_ok"])
        self.assertEqual(cuda_probe["host_env"], "host_env.json")
        self.assertEqual(cuda_probe["manifest_refs"], "manifest_refs.json")

    def test_emit_phase1_discovery_artifacts_rejects_wrong_compute_capability(self) -> None:
        bundle = load_authority_bundle(repo_root())
        cuda_probe = sample_cuda_probe_payload()
        cuda_probe["cc_major"] = 8
        cuda_probe["cc_minor"] = 9

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(AuthorityConflictError, "compute capability"):
                emit_phase1_discovery_artifacts(
                    bundle,
                    output_dir=temp_dir,
                    lane="primary",
                    host_observations=sample_host_observations(),
                    cuda_probe=cuda_probe,
                    local_mirror_refs=sample_local_mirror_refs(),
                    repo_commit="abc123def456",
                )

    def test_emit_phase1_discovery_artifacts_rejects_failed_managed_memory_probe(self) -> None:
        bundle = load_authority_bundle(repo_root())
        cuda_probe = sample_cuda_probe_payload()
        cuda_probe["managed_memory_probe_ok"] = False
        cuda_probe["managed_memory_failure_reason"] = "nvidia-uvm HMM init failed"

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(AuthorityConflictError, "managed-memory"):
                emit_phase1_discovery_artifacts(
                    bundle,
                    output_dir=temp_dir,
                    lane="primary",
                    host_observations=sample_host_observations(),
                    cuda_probe=cuda_probe,
                    local_mirror_refs=sample_local_mirror_refs(),
                    repo_commit="abc123def456",
                )

    @mock.patch("scripts.authority.phase1_discovery.shutil.which")
    def test_collect_host_observations_reads_required_host_fields(
        self,
        which: mock.Mock,
    ) -> None:
        which.side_effect = lambda tool: f"/mock/bin/{tool}"
        responses = {
            ("hostname",): "ws-rtx5080-01\n",
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ): "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB\n",
            ("nvcc", "--version"): "Cuda compilation tools, release 12.9, V12.9.1\n",
            ("gcc", "--version"): "gcc (Ubuntu 14.2.0) 14.2.0\nCopyright ...\n",
            ("nsys", "--version"): "NVIDIA Nsight Systems version 2025.2\n",
            ("ncu", "--version"): "NVIDIA Nsight Compute version 2025.3\n",
            ("compute-sanitizer", "--version"): "Compute Sanitizer version 2025.1\n",
            ("uname", "-r"): "6.8.0-60-generic\n",
        }

        def runner(args: list[str]) -> str:
            return responses[tuple(args)]

        with tempfile.TemporaryDirectory() as temp_dir:
            os_release = pathlib.Path(temp_dir) / "os-release"
            os_release.write_text('PRETTY_NAME="Ubuntu 24.04.2 LTS"\n', encoding="utf-8")
            observations = collect_host_observations(
                command_runner=runner,
                os_release_path=os_release,
            )

        self.assertEqual(observations["hostname"], "ws-rtx5080-01")
        self.assertEqual(
            observations["gpu_csv"],
            "NVIDIA GeForce RTX 5080, 595.50.00, 16384 MiB",
        )
        self.assertEqual(observations["nvcc_path"], "/mock/bin/nvcc")
        self.assertEqual(
            observations["compute_sanitizer_path"],
            "/mock/bin/compute-sanitizer",
        )

    def test_main_emits_phase1_discovery_artifacts_from_json_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            host_observations_path = temp_root / "host_observations.json"
            host_observations_path.write_text(
                json.dumps(sample_host_observations(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cuda_probe_path = temp_root / "raw_cuda_probe.json"
            cuda_probe_path.write_text(
                json.dumps(sample_cuda_probe_payload(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_dir = temp_root / "artifacts"
            stdout = io.StringIO()
            argv = [
                "--root",
                str(repo_root()),
                "--output-dir",
                str(output_dir),
                "--host-observations-json",
                str(host_observations_path),
                "--cuda-probe-json",
                str(cuda_probe_path),
                "--repo-commit",
                "abc123def456",
            ]
            for component, commit in sample_local_mirror_refs().items():
                argv.extend(["--local-mirror-ref", f"{component}={commit}"])

            with redirect_stdout(stdout):
                exit_code = main(argv)

            result = json.loads(stdout.getvalue())
            host_env_exists = (output_dir / "host_env.json").exists()
            manifest_refs_exists = (output_dir / "manifest_refs.json").exists()
            cuda_probe_exists = (output_dir / "cuda_probe.json").exists()

        self.assertEqual(exit_code, 0)
        self.assertTrue(host_env_exists)
        self.assertTrue(manifest_refs_exists)
        self.assertTrue(cuda_probe_exists)
        self.assertTrue(result["host_env"].endswith("host_env.json"))
        self.assertTrue(result["cuda_probe"].endswith("cuda_probe.json"))


if __name__ == "__main__":
    unittest.main()
