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
  native_libcuda_real="$(readlink -f "${native_libcuda}" 2>/dev/null || printf '%s\n' "${native_libcuda}")"
  owner_packages="$(
    {
      dpkg-query -S "${native_libcuda}" 2>/dev/null || true
      dpkg-query -S "${native_libcuda_real}" 2>/dev/null || true
    } | cut -d: -f1 | awk 'NF' | sort -u || true
  )"
  conflicting_packages="$(
    dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' \
      'libnvidia-compute-*' \
      'libcudart*' \
      'nvidia-cuda-dev' \
      'nvidia-cuda-toolkit' 2>/dev/null | awk '$1 == "ii" { print $2 }' | sort -u || true
  )"
  echo "WSL host should not expose Linux display driver libraries at ${native_libcuda}." >&2
  echo "Remove the Linux display driver packages from WSL and rely on ${wsl_lib_dir}." >&2
  if [[ -n "${owner_packages}" ]]; then
    echo "Installed Linux-side libcuda owner packages: ${owner_packages//$'\n'/, }" >&2
    echo "Example cleanup command: sudo apt remove --purge ${owner_packages//$'\n'/ }" >&2
  fi
  if [[ -n "${conflicting_packages}" ]]; then
    related_packages="$(
      printf '%s\n' "${conflicting_packages}" | awk '
        NR == FNR { owners[$1] = 1; next }
        {
          pkg = $1
          base = pkg
          sub(/:.*/, "", base)
          if (!owners[base]) {
            print pkg
          }
        }
      ' <(printf '%s\n' "${owner_packages}") - | awk 'NF' || true
    )"
    if [[ -n "${related_packages}" ]]; then
      echo "Installed related CUDA toolkit packages: ${related_packages//$'\n'/, }" >&2
    fi
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
