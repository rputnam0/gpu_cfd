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
  expand_cleanup_targets() {
    local package_name
    local base_name
    local -a expanded=()
    for package_name in "$@"; do
      [[ -n "${package_name}" ]] || continue
      expanded+=("${package_name}")
      base_name="${package_name%%:*}"
      if [[ "${base_name}" == libnvidia-compute-* && "${base_name}" != *-server ]]; then
        expanded+=("${base_name}-server")
      fi
    done
    printf '%s\n' "${expanded[@]}" | awk 'NF && !seen[$0]++'
  }

  simulate_cleanup_fallout() {
    command -v apt-get >/dev/null 2>&1 || return 0
    (($# > 0)) || return 0
    local apt_output
    apt_output="$(apt-get -s remove --purge "$@" 2>/dev/null || true)"
    [[ -n "${apt_output}" ]] || return 0
    printf '%s\n' "${apt_output}" | awk -v cleanup_targets="$(printf '%s\n' "$@")" '
      BEGIN {
        count = split(cleanup_targets, cleanup_lines, "\n")
        for (i = 1; i <= count; ++i) {
          target = cleanup_lines[i]
          sub(/:.*/, "", target)
          if (target != "") {
            cleanup[target] = 1
          }
        }
      }
      /^The following packages will be REMOVED:/ {
        collect = 1
        next
      }
      collect && /^[0-9]+ upgraded,/ {
        exit
      }
      collect {
        for (i = 1; i <= NF; ++i) {
          pkg = $i
          sub(/\*$/, "", pkg)
          base = pkg
          sub(/:.*/, "", base)
          if (pkg != "" && !(base in cleanup) && !seen[pkg]++) {
            print pkg
          }
        }
      }
    '
  }

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
    mapfile -t cleanup_targets < <(expand_cleanup_targets ${owner_packages//$'\n'/ })
    echo "Installed Linux-side libcuda owner packages: ${owner_packages//$'\n'/, }" >&2
    echo "Example cleanup command: sudo apt remove --purge ${cleanup_targets[*]}" >&2
    cleanup_fallout="$(simulate_cleanup_fallout "${cleanup_targets[@]}")"
    if [[ -n "${cleanup_fallout}" ]]; then
      echo "Simulated apt fallout: ${cleanup_fallout//$'\n'/, }" >&2
    fi
    echo "If you need to restore the CUDA toolkit in WSL afterward, use NVIDIA's WSL-Ubuntu installer path or the cuda-toolkit-12-x meta-package only; do not install cuda, cuda-12-x, or cuda-drivers under WSL." >&2
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
