#!/usr/bin/env bash
# Run the comparison demo locally in its own environment built from space/requirements.txt.
# Usage: GGUF_REPO=owner/name scripts/demo.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
: "${GGUF_REPO:?set GGUF_REPO to the Hub repository holding the GGUF files}"

ENV_DIR="$REPO_DIR/.cache/space-env"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  uv venv --python 3.11 "$ENV_DIR"
  uv pip install --python "$ENV_DIR/bin/python" -r space/requirements.txt
fi
cd space
exec "$ENV_DIR/bin/python" app.py
