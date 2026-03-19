#!/usr/bin/env bash
set -euo pipefail

output_arg="${1:-host_env.json}"
lane="${2:-primary}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

# Emits host_env.json, manifest_refs.json, cuda_probe.json, and raw_cuda_probe.json.
if [[ "${output_arg}" == *.json ]]; then
  output_dir="$(dirname "${output_arg}")"
else
  output_dir="${output_arg}"
fi

mkdir -p "${output_dir}"

raw_cuda_probe_json="${output_dir}/raw_cuda_probe.json"

"${script_dir}/run_cuda_probe.sh" "${raw_cuda_probe_json}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.uv-cache}"

cd "${repo_root}"
uv run python scripts/authority/phase1_discovery.py \
  --root "${repo_root}" \
  --output-dir "${output_dir}" \
  --lane "${lane}" \
  --cuda-probe-json "${raw_cuda_probe_json}"
