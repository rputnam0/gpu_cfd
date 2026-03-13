#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
symphony_root="${SYMPHONY_ROOT:-$HOME/projects/symphony}"
symphony_elixir_root="$symphony_root/elixir"
workflow_path="$repo_root/WORKFLOW.md"
env_file="${SYMPHONY_ENV_FILE:-$symphony_root/.env}"
mise_bin="${MISE_BIN:-}"
logs_root="${SYMPHONY_LOGS_ROOT:-$HOME/projects/symphony-logs/gpu_cfd}"
telemetry_root="${GPU_CFD_TELEMETRY_ROOT:-$logs_root}"
telemetry_script="$repo_root/scripts/symphony/telemetry.py"

emit_telemetry() {
  if [[ -f "$telemetry_script" ]]; then
    python3 "$telemetry_script" event "$@" >/dev/null 2>&1 || true
  fi
}

on_exit() {
  local status=$?
  if [[ $status -eq 0 ]]; then
    emit_telemetry \
      --event-type symphony_launcher_exit \
      --message "Symphony launcher exited cleanly" \
      --detail exit_code="$status" \
      --detail logs_root="$logs_root"
  else
    emit_telemetry \
      --event-type symphony_launcher_exit \
      --message "Symphony launcher exited with failure" \
      --detail exit_code="$status" \
      --detail logs_root="$logs_root"
  fi
}

trap on_exit EXIT

if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

: "${LINEAR_API_KEY:?LINEAR_API_KEY must be set, or defined in $env_file}"
: "${SYMPHONY_WORKSPACE_ROOT:=$HOME/projects/symphony-workspaces/gpu_cfd}"
: "${GPU_CFD_SOURCE_REPO_URL:=https://github.com/rputnam0/gpu_cfd.git}"
: "${GPU_CFD_BOOTSTRAP_REF:=}"

export LINEAR_API_KEY
export SYMPHONY_WORKSPACE_ROOT
export GPU_CFD_SOURCE_REPO_URL
export GPU_CFD_BOOTSTRAP_REF
export SYMPHONY_LOGS_ROOT="$logs_root"
export GPU_CFD_TELEMETRY_ROOT="$telemetry_root"

emit_telemetry \
  --event-type symphony_launcher_invoked \
  --message "Launcher invoked" \
  --detail env_file="$env_file" \
  --detail workflow_path="$workflow_path" \
  --detail symphony_root="$symphony_root"

python3 "$repo_root/scripts/symphony/preflight.py" --mode runtime

if [[ ! -d "$symphony_elixir_root" ]]; then
  echo "Missing Symphony checkout at $symphony_elixir_root" >&2
  exit 1
fi

if [[ -z "$mise_bin" ]]; then
  if command -v mise >/dev/null 2>&1; then
    mise_bin="$(command -v mise)"
  elif [[ -x "$HOME/.local/bin/mise" ]]; then
    mise_bin="$HOME/.local/bin/mise"
  fi
fi

if [[ -z "$mise_bin" ]]; then
  echo "Missing mise on PATH. Install mise or add it to PATH before launching Symphony." >&2
  exit 1
fi

cd "$symphony_elixir_root"
emit_telemetry \
  --event-type symphony_launcher_starting_process \
  --message "Launching Symphony via the sanctioned WSL entrypoint" \
  --detail logs_root="$logs_root" \
  --detail workspace_root="$SYMPHONY_WORKSPACE_ROOT"

"$mise_bin" exec -- ./bin/symphony \
  --i-understand-that-this-will-be-running-without-the-usual-guardrails \
  "$workflow_path" \
  --logs-root "$logs_root"
