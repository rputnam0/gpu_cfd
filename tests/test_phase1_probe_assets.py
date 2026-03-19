from __future__ import annotations

import pathlib
import os
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
        self.assertIn("GPU_CFD_NATIVE_LIBCUDA", wrapper)
        self.assertIn("/usr/lib/x86_64-linux-gnu", wrapper)
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

    def test_cuda_runtime_probe_wrapper_reports_wsl_driver_conflict_before_compile(
        self,
    ) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "run_cuda_probe.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            fake_bin = temp_root / "bin"
            fake_bin.mkdir()
            fake_wsl = temp_root / "wsl"
            fake_wsl.mkdir()
            fake_native = temp_root / "native"
            fake_native.mkdir()
            (fake_wsl / "libcuda.so.1").write_text("", encoding="utf-8")
            for library_name in (
                "libcuda.so.1",
                "libnvidia-ml.so.1",
                "libnvidia-ptxjitcompiler.so.1",
            ):
                (fake_native / library_name).write_text("", encoding="utf-8")

            for tool_name in ("nvcc", "g++", "g++-12"):
                tool_path = fake_bin / tool_name
                tool_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                tool_path.chmod(0o755)

            dpkg_query_path = fake_bin / "dpkg-query"
            dpkg_query_path.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == \"-S\" ]]; then\n"
                "  shift\n"
                "  for arg in \"$@\"; do\n"
                "    echo \"libnvidia-compute-535:amd64: ${arg}\"\n"
                "  done\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == \"-W\" ]]; then\n"
                "  cat <<'EOF'\n"
                "ii libcudart12:amd64\n"
                "ii libnvidia-compute-535:amd64\n"
                "ii nvidia-cuda-toolkit\n"
                "EOF\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            dpkg_query_path.chmod(0o755)

            apt_get_path = fake_bin / "apt-get"
            apt_get_path.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<'EOF'\n"
                "The following packages will be REMOVED:\n"
                "  libnvidia-compute-535* nvidia-cuda-toolkit nsight-systems\n"
                "0 upgraded, 0 newly installed, 3 to remove and 0 not upgraded.\n"
                "EOF\n",
                encoding="utf-8",
            )
            apt_get_path.chmod(0o755)

            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["GPU_CFD_WSL_LIB_DIR"] = str(fake_wsl)
            env["GPU_CFD_NATIVE_DRIVER_ROOT"] = str(fake_native)
            env["GPU_CFD_NATIVE_LIBCUDA"] = str(fake_native / "libcuda.so.1")

            completed = subprocess.run(
                [str(wrapper_path), str(temp_root / "raw_cuda_probe.json")],
                cwd=temp_root,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("WSL host should not expose Linux display driver libraries", completed.stderr)
        self.assertIn(str(fake_native / "libcuda.so.1"), completed.stderr)
        self.assertIn("Example cleanup command: sudo apt remove --purge", completed.stderr)
        self.assertIn("libnvidia-compute-535-server", completed.stderr)
        self.assertIn("Simulated apt fallout: nvidia-cuda-toolkit, nsight-systems", completed.stderr)

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

    def test_host_env_wrapper_preserves_log_and_snapshot_when_probe_guard_fails(
        self,
    ) -> None:
        wrapper_path = repo_root() / "tools" / "bringup" / "env" / "check_host_env.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = pathlib.Path(temp_dir)
            fake_bin = temp_root / "bin"
            fake_bin.mkdir()
            fake_wsl = temp_root / "wsl"
            fake_wsl.mkdir()
            fake_native = temp_root / "native"
            fake_native.mkdir()
            (fake_wsl / "libcuda.so.1").write_text("", encoding="utf-8")
            for library_name in (
                "libcuda.so.1",
                "libnvidia-ml.so.1",
                "libnvidia-ptxjitcompiler.so.1",
            ):
                (fake_native / library_name).write_text("", encoding="utf-8")

            for tool_name in ("nvcc", "g++", "g++-12"):
                tool_path = fake_bin / tool_name
                tool_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
                tool_path.chmod(0o755)

            dpkg_query_path = fake_bin / "dpkg-query"
            dpkg_query_path.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$1\" == \"-S\" ]]; then\n"
                "  shift\n"
                "  for arg in \"$@\"; do\n"
                "    echo \"libnvidia-compute-535:amd64: ${arg}\"\n"
                "  done\n"
                "  exit 0\n"
                "fi\n"
                "if [[ \"$1\" == \"-W\" ]]; then\n"
                "  cat <<'EOF'\n"
                "ii libcudart12:amd64\n"
                "ii libnvidia-compute-535:amd64\n"
                "ii nvidia-cuda-toolkit\n"
                "EOF\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            dpkg_query_path.chmod(0o755)

            apt_get_path = fake_bin / "apt-get"
            apt_get_path.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<'EOF'\n"
                "The following packages will be REMOVED:\n"
                "  libnvidia-compute-535* nvidia-cuda-toolkit nsight-systems\n"
                "0 upgraded, 0 newly installed, 3 to remove and 0 not upgraded.\n"
                "EOF\n",
                encoding="utf-8",
            )
            apt_get_path.chmod(0o755)

            output_dir = temp_root / "discovery"
            log_path = output_dir / "check_host_env.log"
            snapshot_path = output_dir / "nvidia_runtime_snapshot.txt"

            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["GPU_CFD_WSL_LIB_DIR"] = str(fake_wsl)
            env["GPU_CFD_NATIVE_DRIVER_ROOT"] = str(fake_native)
            env["GPU_CFD_NATIVE_LIBCUDA"] = str(fake_native / "libcuda.so.1")

            completed = subprocess.run(
                [str(wrapper_path), str(output_dir), "primary"],
                cwd=temp_root,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stderr, "")
            self.assertIn(
                "WSL host should not expose Linux display driver libraries",
                completed.stdout,
            )
            self.assertTrue(log_path.is_file())
            self.assertTrue(snapshot_path.is_file())
            log_body = log_path.read_text(encoding="utf-8")
            snapshot_body = snapshot_path.read_text(encoding="utf-8")
            self.assertIn("output_dir=", log_body)
            self.assertIn("lane=primary", log_body)
            self.assertIn(
                "Simulated apt fallout: nvidia-cuda-toolkit, nsight-systems",
                log_body,
            )
            self.assertIn("# /dev/dxg", snapshot_body)
            self.assertIn("# ldconfig -p (CUDA driver libraries)", snapshot_body)

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
