#!/usr/bin/env bash
# Regenerate the pinned requirement files in envs/ from their .in sources.
# Usage: scripts/compile_envs.sh [env ...]   (default: every .in file)
# Resolution targets Linux x86_64 with Python 3.11, the platform of the training and serving hosts.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR/envs"

DEFAULT_PLATFORM="x86_64-manylinux_2_28"
# SGLang publishes wheels only for newer glibc baselines
platform_for() {
  case "$1" in
    serve-sglang) echo "x86_64-manylinux_2_34" ;;
    *) echo "$DEFAULT_PLATFORM" ;;
  esac
}

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
  for file in *.in; do
    TARGETS+=("${file%.in}")
  done
fi

for name in "${TARGETS[@]}"; do
  echo "compiling $name"
  uv pip compile "$name.in" --python-version 3.11 --python-platform "$(platform_for "$name")" \
    --no-header --no-annotate -o "$name.txt"
done
