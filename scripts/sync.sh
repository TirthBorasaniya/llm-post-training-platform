#!/usr/bin/env bash
# Copy the working tree to the remote machine, recording the code version first.
# Reads REMOTE_HOST and REMOTE_DIR from the environment or from the file named by ENV_FILE
# (default .env). Set DRY_RUN=1 to print the command without running it.
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

SHA="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  SHA="$SHA-dirty"
fi
printf '%s\n' "$SHA" >.sync_sha

# --update keeps newer files on the remote, so results produced there are never overwritten
CMD=(rsync -az --update --exclude-from=.syncignore
  --rsync-path="mkdir -p '$REMOTE_DIR' && rsync"
  ./ "$REMOTE_HOST:$REMOTE_DIR/")
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "${CMD[*]}"
  exit 0
fi
"${CMD[@]}"
echo "synced $SHA"
