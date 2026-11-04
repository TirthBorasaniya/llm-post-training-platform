#!/usr/bin/env bash
# Small shell helpers shared by submit wrappers and job bodies; defines functions only.

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# print the Python interpreter that can import the platform package and read configs
bootstrap_python() {
  local root="${PLATFORM_ROOT:-$HOME/llmplatform}"
  if [ -x "$root/envs/core/bin/python" ]; then
    echo "$root/envs/core/bin/python"
  elif [ -x "$REPO_DIR/.venv/bin/python" ]; then
    echo "$REPO_DIR/.venv/bin/python"
  else
    command -v python3.11 || command -v python3 || die "no python interpreter found"
  fi
}

# run a platform config query with the bootstrap interpreter
compute_query() {
  PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$(bootstrap_python)" -m llmplatform.common.compute "$@"
}
