from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class Phase1ProbeAssetTests(unittest.TestCase):
    def test_cuda_runtime_probe_assets_exist(self) -> None:
        source = repo_root() / "tools" / "bringup" / "src" / "validate_cuda_runtime.cu"
        wrapper = repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        host_wrapper = repo_root() / "tools" / "bringup" / "env" / "check_host_env.sh"

        self.assertTrue(source.is_file())
        self.assertTrue(wrapper.is_file())
        self.assertTrue(host_wrapper.is_file())

    def test_cuda_runtime_probe_wrapper_compiles_expected_source(self) -> None:
        wrapper = (
            repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("nvcc", wrapper)
        self.assertIn("validate_cuda_runtime.cu", wrapper)
        self.assertIn("-ccbin", wrapper)
        self.assertIn("compute_120", wrapper)
        self.assertIn("sm_120", wrapper)

    def test_cuda_runtime_probe_wrapper_prefers_wsl_driver_libs(self) -> None:
        wrapper = (
            repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("/usr/lib/wsl/lib", wrapper)
        self.assertIn("LD_LIBRARY_PATH", wrapper)
        self.assertIn("/usr/lib/x86_64-linux-gnu/libcuda.so.1", wrapper)
        self.assertIn("Linux display driver", wrapper)
        self.assertIn("dpkg-query", wrapper)
        self.assertIn("nvidia-cuda-toolkit", wrapper)
        self.assertIn("libnvidia-compute-", wrapper)
        self.assertIn("libnvidia-ptxjitcompiler", wrapper)
        self.assertIn("libnvidia-ml", wrapper)
        self.assertIn("sudo apt remove --purge", wrapper)
        self.assertIn("driver owner packages", wrapper)
        self.assertIn("-server", wrapper)
        self.assertIn("apt-get", wrapper)
        self.assertIn("Simulated apt fallout", wrapper)
        self.assertIn("cuda-toolkit-12-x", wrapper)
        self.assertIn("cuda-drivers", wrapper)
        self.assertIn("Usage: run_cuda_probe.sh", wrapper)
        self.assertNotIn('&& -e "${native_libcuda}"', wrapper)
        self.assertIn('if [[ -e "${wsl_lib_dir}/libcuda.so.1" ]]; then', wrapper)

    def test_cuda_runtime_probe_wrapper_renders_help_without_running_probe(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [str(wrapper_path), "--help"],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage: run_cuda_probe.sh", completed.stdout)
        self.assertIn("<output-json>", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_cuda_runtime_probe_wrapper_requires_output_path_argument(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [str(wrapper_path)],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage: run_cuda_probe.sh", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_host_env_wrapper_runs_probe_and_discovery(self) -> None:
        wrapper = (
            repo_root() / "tools" / "bringup" / "env" / "check_host_env.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("run_cuda_probe.sh", wrapper)
        self.assertIn("phase1_discovery.py", wrapper)
        self.assertIn("host_env.json", wrapper)
        self.assertIn("manifest_refs.json", wrapper)
        self.assertIn("cuda_probe.json", wrapper)
        self.assertIn("raw_cuda_probe.json", wrapper)
        self.assertIn("check_host_env.log", wrapper)
        self.assertIn("nvidia_runtime_snapshot.txt", wrapper)
        self.assertIn("nvidia-smi -L", wrapper)
        self.assertIn("/proc/cmdline", wrapper)
        self.assertIn("/dev/dxg", wrapper)
        self.assertIn("ldconfig -p", wrapper)
        self.assertIn("libnvidia-ptxjitcompiler", wrapper)
        self.assertIn("dpkg -S", wrapper)
        self.assertIn("dpkg-query -W", wrapper)
        self.assertIn("apt-cache policy", wrapper)
        self.assertIn("apt-mark showmanual", wrapper)
        self.assertIn("realpath", wrapper)
        self.assertIn("tee -a", wrapper)
        self.assertIn("repo_commit=", wrapper)
        self.assertIn("output_dir=", wrapper)
        self.assertIn("lane=", wrapper)
        self.assertIn("command=", wrapper)
        self.assertIn("/lib/x86_64-linux-gnu", wrapper)
        self.assertIn("/usr/lib/x86_64-linux-gnu", wrapper)
        self.assertIn("--output-dir", wrapper)
        self.assertIn("Usage: check_host_env.sh", wrapper)

    def test_host_env_wrapper_renders_help_without_running_probe(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "check_host_env.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [str(wrapper_path), "--help"],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Usage: check_host_env.sh", completed.stdout)
        self.assertIn("<output-dir-or-host-env-json>", completed.stdout)
        self.assertEqual(completed.stderr, "")

    def test_host_env_wrapper_requires_output_path_argument(self) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "check_host_env.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [str(wrapper_path)],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Usage: check_host_env.sh", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_cuda_runtime_probe_source_mentions_required_probe_fields(self) -> None:
        source = (
            repo_root() / "tools" / "bringup" / "src" / "validate_cuda_runtime.cu"
        ).read_text(encoding="utf-8")

        for token in (
            "device_index",
            "device_name",
            "cc_major",
            "cc_minor",
            "total_global_mem_bytes",
            "native_kernel_ok",
            "managed_memory_probe_ok",
            "cudaGetDeviceProperties",
            "cudaMallocManaged",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
