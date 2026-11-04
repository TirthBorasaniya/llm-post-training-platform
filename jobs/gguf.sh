#!/usr/bin/env bash
# Convert the teacher, distilled student, and base student to Q4_K_M GGUF and verify each one.
# Usage: jobs/gguf.sh
# Builds the pinned llama.cpp release on first use into $LLAMA_CPP_DIR (default $PLATFORM_ROOT/llama.cpp).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PLATFORM_ROOT/llama.cpp}"
export LLAMA_CPP_DIR
TAG="$(run_in_env quantize python scripts/quantize.py list llama-cpp-tag)"

if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama" ] && [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
  if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
  cmake --build "$LLAMA_CPP_DIR/build" -j "$(nproc 2>/dev/null || echo 4)"
fi

while IFS= read -r model; do
  run_in_env quantize python scripts/quantize.py gguf --model "$model" --llama-cpp-dir "$LLAMA_CPP_DIR"
done < <(run_in_env quantize python scripts/quantize.py list gguf-models)
