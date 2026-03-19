#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check_host_env.sh <output-dir-or-host-env-json> [primary|experimental]

Runs the Phase 1 CUDA probe plus discovery flow and emits:
- host_env.json
- manifest_refs.json
- cuda_probe.json
- raw_cuda_probe.json
- check_host_env.log
- nvidia_runtime_snapshot.txt
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

output_arg="${1}"
lane="${2:-primary}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
snapshot_path=""

snapshot_policy_targets() {
  local package_name
  local base_name
  local -a policy_targets=()
  mapfile -t policy_targets < <(
    dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' \
      'libcudart*' \
      'libnvidia-compute-*' \
      'nvidia-cuda-dev' \
      'nvidia-cuda-toolkit' 2>/dev/null | awk '
        $1 == "ii" {
          package_name = $2
          sub(/:.*/, "", package_name)
          if (package_name != "") {
            print package_name
            if (package_name ~ /^libnvidia-compute-/ && package_name !~ /-server$/) {
              print package_name "-server"
            }
          }
        }
      ' | awk 'NF && !seen[$0]++'
  )
  printf '%s\n' "${policy_targets[@]}"
}

write_runtime_snapshot() {
  [[ -n "${snapshot_path}" ]] || return 0
  {
    local -a policy_targets=()
    echo "# /dev/dxg"
    ls -l /dev/dxg 2>/dev/null || true
    echo
    echo "# /proc/cmdline"
    cat /proc/cmdline 2>/dev/null || true
    echo
    echo "# nvidia-smi -L"
    nvidia-smi -L 2>/dev/null || true
    echo
    echo "# nvidia-smi --query-gpu"
    nvidia-smi --query-gpu=name,driver_version,compute_mode,pstate,persistence_mode --format=csv,noheader 2>/dev/null || true
    echo
    echo "# nvidia-smi -q (selected)"
    nvidia-smi -q 2>/dev/null | rg -n 'Driver Version|CUDA Version|GPU Virtualization Mode|Persistence Mode|Compute Mode|MIG Mode|Attached GPUs|Minor Number|BAR1 Memory Usage|Processes|Product Name' || true
    echo
    echo "# command -v nvcc"
    command -v nvcc 2>/dev/null || true
    echo
    echo "# nvcc --version"
    nvcc --version 2>/dev/null || true
    echo
    echo "# environment"
    printf 'CUDA_HOME=%s\n' "${CUDA_HOME:-}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
    printf 'PATH=%s\n' "${PATH:-}"
    echo
    echo "# ldconfig -p (CUDA driver libraries)"
    ldconfig -p 2>/dev/null | rg 'libcuda\.so|libnvidia-ml\.so|libnvidia-ptxjitcompiler\.so' || true
    echo
    echo "# native driver library paths"
    find /usr/lib/x86_64-linux-gnu \
      -maxdepth 1 \
      \( -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' -o -name 'libnvidia-ptxjitcompiler.so*' \) \
      -print 2>/dev/null | sort || true
    echo
    echo "# canonical native driver realpaths"
    realpath \
      /lib/x86_64-linux-gnu/libcuda.so.1 \
      /lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
      /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
      /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
      /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
      /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
      2>/dev/null || true
    echo
    echo "# WSL shim library paths"
    find /usr/lib/wsl/lib \
      -maxdepth 1 \
      \( -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' -o -name 'libnvidia-ptxjitcompiler.so*' \) \
      -print 2>/dev/null | sort || true
    echo
    echo "# dpkg -S (native driver library owners)"
    dpkg -S \
      /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
      /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
      /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 \
      2>/dev/null || true
    echo
    echo "# dpkg-query -W (installed NVIDIA/CUDA packages)"
    dpkg-query -W -f='${db:Status-Abbrev} ${binary:Package}\n' \
      'libcudart*' \
      'libnvidia-compute-*' \
      'nvidia-cuda-dev' \
      'nvidia-cuda-toolkit' \
      2>/dev/null | rg '^ii ' || true
    echo
    echo "# apt-cache policy (driver/toolkit packages)"
    mapfile -t policy_targets < <(snapshot_policy_targets)
    if ((${#policy_targets[@]} > 0)); then
      apt-cache policy "${policy_targets[@]}" 2>/dev/null || true
    fi
    echo
    echo "# apt-mark showmanual (relevant packages)"
    apt-mark showmanual 2>/dev/null | rg 'libnvidia-compute-|nvidia-cuda-toolkit|nvidia-cuda-dev' || true
    echo
    echo "# apt-cache depends (toolkit anchor)"
    apt-cache depends nvidia-cuda-toolkit nvidia-cuda-dev 2>/dev/null | rg '^(nvidia-cuda-toolkit|nvidia-cuda-dev)|Depends: nvidia-cuda-dev|Depends: libnvidia-compute-' || true
    echo
    echo "# dmesg dxg tail"
    dmesg 2>/dev/null | rg -i 'dxgkio_query_adapter_info|dxgkio_is_feature_enabled|uvm|hmm|kaslr' | tail -n 40 || true
  } > "${snapshot_path}" 2>/dev/null || true
}

# Emits host_env.json, manifest_refs.json, cuda_probe.json, and raw_cuda_probe.json.
if [[ "${output_arg}" == *.json ]]; then
  output_dir="$(dirname "${output_arg}")"
else
  output_dir="${output_arg}"
fi

mkdir -p "${output_dir}"

raw_cuda_probe_json="${output_dir}/raw_cuda_probe.json"
wrapper_log_path="${output_dir}/check_host_env.log"
snapshot_path="${output_dir}/nvidia_runtime_snapshot.txt"
repo_commit="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || echo unknown)"

trap 'write_runtime_snapshot' ERR

exec > >(tee -a "${wrapper_log_path}") 2>&1

printf '# check_host_env invocation\n'
printf 'utc_time=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repo_root=%s\n' "${repo_root}"
printf 'repo_commit=%s\n' "${repo_commit}"
printf 'output_dir=%s\n' "${output_dir}"
printf 'lane=%s\n' "${lane}"
printf 'command=%q %q %q\n' "${BASH_SOURCE[0]}" "${output_arg}" "${lane}"
printf '\n'

"${script_dir}/run_cuda_probe.sh" "${raw_cuda_probe_json}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"

cd "${repo_root}"
uv run python scripts/authority/phase1_discovery.py \
  --root "${repo_root}" \
  --output-dir "${output_dir}" \
  --lane "${lane}" \
  --cuda-probe-json "${raw_cuda_probe_json}"

trap - ERR
