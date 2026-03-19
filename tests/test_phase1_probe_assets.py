from __future__ import annotations

import pathlib
import unittest


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


class Phase1ProbeAssetTests(unittest.TestCase):
    def test_cuda_runtime_probe_assets_exist(self) -> None:
        source = repo_root() / "tools" / "bringup" / "src" / "validate_cuda_runtime.cu"
        wrapper = repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"

        self.assertTrue(source.is_file())
        self.assertTrue(wrapper.is_file())

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
