from __future__ import annotations

import pathlib
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
        self.assertNotIn('&& -e "${native_libcuda}"', wrapper)
        self.assertIn('if [[ -e "${wsl_lib_dir}/libcuda.so.1" ]]; then', wrapper)

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
        self.assertIn("nvidia_runtime_snapshot.txt", wrapper)
        self.assertIn("nvidia-smi -L", wrapper)
        self.assertIn("/proc/cmdline", wrapper)
        self.assertIn("/dev/dxg", wrapper)
        self.assertIn("ldconfig -p", wrapper)
        self.assertIn("libnvidia-ptxjitcompiler", wrapper)
        self.assertIn("dpkg -S", wrapper)
        self.assertIn("dpkg-query -W", wrapper)
        self.assertIn("realpath", wrapper)
        self.assertIn("/lib/x86_64-linux-gnu", wrapper)
        self.assertIn("/usr/lib/x86_64-linux-gnu", wrapper)
        self.assertIn("--output-dir", wrapper)

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
