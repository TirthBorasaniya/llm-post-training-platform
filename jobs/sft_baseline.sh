#!/usr/bin/env bash
# Student SFT on teacher completions, stopped at the GPU-hours the distillation run used.
# Usage: jobs/sft_baseline.sh [--budget-gpu-hours H]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware
run_in_env "$(pick_env distill distill-vllm)" python scripts/distill.py sft-baseline "$@"
