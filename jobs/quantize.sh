#!/usr/bin/env bash
# GPU quantization: AWQ 4-bit weights on GPUs without FP8, nothing to write on FP8 GPUs.
# Usage: jobs/quantize.sh
# FP8 is applied online by the server (--quantization fp8), so Ada and Hopper GPUs need no files.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware
METHOD="$(run_in_env quantize python scripts/quantize.py plan --method-only)"
log "quantization method for this GPU: $METHOD"
case "$METHOD" in
  awq)
    while IFS= read -r model; do
      run_in_env quantize python scripts/quantize.py awq --model "$model"
    done < <(run_in_env quantize python scripts/quantize.py list gpu-models)
    ;;
  fp8) log "FP8 is served online; no checkpoint is written" ;;
  *) die "no CUDA device visible; run this job on a GPU" ;;
esac
