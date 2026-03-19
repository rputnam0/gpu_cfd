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
