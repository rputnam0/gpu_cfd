#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-$HOME/projects/symphony/elixir}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/patches/symphony-thread-resume-v2.patch"

if [[ ! -d "$TARGET_DIR/.git" ]]; then
  echo "Target is not a git checkout: $TARGET_DIR" >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not found: $PATCH_FILE" >&2
  exit 1
fi

if git -C "$TARGET_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  echo "Symphony runtime patch is already applied at $TARGET_DIR"
  exit 0
fi

git -C "$TARGET_DIR" apply --check "$PATCH_FILE"
git -C "$TARGET_DIR" apply "$PATCH_FILE"
echo "Applied Symphony runtime patch to $TARGET_DIR"
