#!/usr/bin/env bash
set -euo pipefail

SSH_DIR="${HOME}/.ssh"
KEY_PATH="${SSH_DIR}/symphony_live_e2e_ed25519"
CONFIG_PATH="${SSH_DIR}/symphony_live_e2e_config"
AUTHORIZED_KEYS="${SSH_DIR}/authorized_keys"

mkdir -p "${SSH_DIR}"
chmod 700 "${SSH_DIR}"

if [[ ! -f "${KEY_PATH}" ]]; then
  ssh-keygen -t ed25519 -N "" -f "${KEY_PATH}" >/dev/null
fi

touch "${AUTHORIZED_KEYS}"
chmod 600 "${AUTHORIZED_KEYS}"

PUB_KEY="$(cat "${KEY_PATH}.pub")"
grep -qxF "${PUB_KEY}" "${AUTHORIZED_KEYS}" || printf "%s\n" "${PUB_KEY}" >> "${AUTHORIZED_KEYS}"

cat > "${CONFIG_PATH}" <<EOF
Host symphony-e2e-1 symphony-e2e-2
  HostName localhost
  User ${USER}
  IdentityFile ${KEY_PATH}
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF

chmod 600 "${CONFIG_PATH}"

ssh -F "${CONFIG_PATH}" -o BatchMode=yes symphony-e2e-1 "echo ok" >/dev/null

cat <<EOF
export SYMPHONY_LIVE_SSH_WORKER_HOSTS=symphony-e2e-1,symphony-e2e-2
export SYMPHONY_SSH_CONFIG=${CONFIG_PATH}
EOF
