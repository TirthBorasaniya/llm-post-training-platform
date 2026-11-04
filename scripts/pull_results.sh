#!/usr/bin/env bash
# Fetch results and logs produced on the remote machine.
# Reads REMOTE_HOST and REMOTE_DIR like scripts/sync.sh. Set DRY_RUN=1 to print commands only.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
: "${REMOTE_HOST:?REMOTE_HOST is not set}"
: "${REMOTE_DIR:?REMOTE_DIR is not set}"

mkdir -p results logs
for name in results logs; do
  CMD=(rsync -az "$REMOTE_HOST:$REMOTE_DIR/$name/" "$name/")
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "${CMD[*]}"
  else
    "${CMD[@]}"
  fi
done
