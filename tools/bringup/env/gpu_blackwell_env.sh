#!/usr/bin/env bash
set -euo pipefail

lane="${1:-primary}"
mode="${2:-relwithdebinfo}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
source_root="${SPUMA_SOURCE_ROOT:-$PWD}"
output_dir="${GPU_CFD_PHASE1_BUILD_ARTIFACTS:-${source_root}/artifacts/phase1_build}"

host_env_json="${GPU_CFD_HOST_ENV:?set GPU_CFD_HOST_ENV to the canonical P1-01 host_env.json path}"
manifest_refs_json="${GPU_CFD_MANIFEST_REFS:?set GPU_CFD_MANIFEST_REFS to the canonical P1-01 manifest_refs.json path}"
cuda_probe_json="${GPU_CFD_CUDA_PROBE:?set GPU_CFD_CUDA_PROBE to the canonical P1-01 cuda_probe.json path}"

if env_exports="$(
  cd "${repo_root}"
  uv run python scripts/authority/phase1_build.py env \
    --root "${repo_root}" \
    --source-root "${source_root}" \
    --output-dir "${output_dir}" \
    --host-env-json "${host_env_json}" \
    --manifest-refs-json "${manifest_refs_json}" \
    --cuda-probe-json "${cuda_probe_json}" \
    --lane "${lane}" \
    --mode "${mode}"
)"; then
  eval "${env_exports}"
else
  status=$?
  return "${status}" 2>/dev/null || exit "${status}"
fi
