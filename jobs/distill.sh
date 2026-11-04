#!/usr/bin/env bash
# On-policy distillation of the teacher into the student, after the mandatory vocabulary check.
# Usage: jobs/distill.sh [--alternative-lr]
# Resumes from the newest checkpoint, so chained submissions continue an unfinished run.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

log_hardware
ENV_NAME="$(pick_env distill-vllm distill)"
log "distillation environment: $ENV_NAME"
run_in_env "$ENV_NAME" python scripts/check_vocab.py
if [ "$NUM_GPUS" -gt 1 ]; then
  run_in_env "$ENV_NAME" accelerate launch --multi_gpu --num_processes "$NUM_GPUS" \
    scripts/distill.py opd "$@"
else
  run_in_env "$ENV_NAME" python scripts/distill.py opd "$@"
fi
