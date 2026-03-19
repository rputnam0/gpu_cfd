#!/usr/bin/env bash
set -euo pipefail

output_arg="${1:-host_env.json}"
lane="${2:-primary}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
snapshot_path=""

write_runtime_snapshot() {
  [[ -n "${snapshot_path}" ]] || return 0
  {
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
snapshot_path="${output_dir}/nvidia_runtime_snapshot.txt"

trap 'write_runtime_snapshot' ERR

"${script_dir}/run_cuda_probe.sh" "${raw_cuda_probe_json}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"

cd "${repo_root}"
uv run python scripts/authority/phase1_discovery.py \
  --root "${repo_root}" \
  --output-dir "${output_dir}" \
  --lane "${lane}" \
  --cuda-probe-json "${raw_cuda_probe_json}"

trap - ERR
