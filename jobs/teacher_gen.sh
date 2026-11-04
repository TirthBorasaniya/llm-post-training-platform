#!/usr/bin/env bash
# Generate teacher completions for the distillation prompts with a local SGLang server.
# Usage: jobs/teacher_gen.sh [--limit N]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

log_hardware
TEACHER_PATH="$(PYTHONPATH="$REPO_DIR/src" "$(bootstrap_python)" -m llmplatform.common.models resolve teacher)"
run_in_env serve-sglang python -m llmplatform.serve.launch --engine sglang \
  --model "$TEACHER_PATH" --served-model-name teacher --log logs/teacher_gen_server.log \
  -- python scripts/distill.py teacher-gen --base-url '{openai_url}' \
  --served-model-name teacher --gpus "$NUM_GPUS" "$@"
