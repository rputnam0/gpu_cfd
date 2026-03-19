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
