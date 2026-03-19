#!/usr/bin/env bash
set -euo pipefail

output_json="${1:?usage: run_cuda_probe.sh <output-json>}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source_path="${repo_root}/tools/bringup/src/validate_cuda_runtime.cu"

nvcc_bin="${NVCC_BIN:-$(command -v nvcc || true)}"
if [[ -z "${nvcc_bin}" ]]; then
  echo "nvcc is required to compile validate_cuda_runtime.cu" >&2
  exit 1
fi
host_cxx="${NVCC_HOST_COMPILER:-$(command -v g++-12 || command -v g++ || true)}"
if [[ -z "${host_cxx}" ]]; then
  echo "a supported host C++ compiler is required to compile validate_cuda_runtime.cu" >&2
  exit 1
fi

build_dir="${GPU_CFD_PHASE1_PROBE_BUILD_DIR:-${TMPDIR:-/tmp}/gpu_cfd_phase1_probe}"
mkdir -p "${build_dir}" "$(dirname "${output_json}")"

binary_path="${build_dir}/validate_cuda_runtime"
cuda_home="${CUDA_HOME:-$(cd "$(dirname "${nvcc_bin}")/.." && pwd)}"
wsl_lib_dir="/usr/lib/wsl/lib"
native_libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so.1"

if [[ -e "${wsl_lib_dir}/libcuda.so.1" && -e "${native_libcuda}" ]]; then
  conflicting_packages="$(
    dpkg-query -W -f='${binary:Package}\n' \
      'libnvidia-compute-*' \
      'libcudart*' \
      'nvidia-cuda-dev' \
      'nvidia-cuda-toolkit' 2>/dev/null | sort -u || true
  )"
  echo "WSL host should not expose Linux display driver libraries at ${native_libcuda}." >&2
  echo "Remove the Linux display driver packages from WSL and rely on ${wsl_lib_dir}." >&2
  if [[ -n "${conflicting_packages}" ]]; then
    echo "Installed Linux-side CUDA/NVIDIA packages: ${conflicting_packages//$'\n'/, }" >&2
    echo "Example cleanup command: sudo apt remove --purge ${conflicting_packages//$'\n'/ }" >&2
  fi
  exit 1
fi

"${nvcc_bin}" \
  -ccbin "${host_cxx}" \
  -std=c++17 \
  -O2 \
  -lineinfo \
  -gencode=arch=compute_120,code=sm_120 \
  -gencode=arch=compute_120,code=compute_120 \
  -o "${binary_path}" \
  "${source_path}"

if [[ -d "${wsl_lib_dir}" ]]; then
  export LD_LIBRARY_PATH="${wsl_lib_dir}:${cuda_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${cuda_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"${binary_path}" "${output_json}"
