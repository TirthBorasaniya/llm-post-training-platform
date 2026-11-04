#!/usr/bin/env bash
# Speculative decoding of the teacher: baseline, then each draft at each draft length.
# Usage: jobs/speculative.sh
# FP8 runs are added only on GPUs that support it. A greedy-output mismatch fails the experiment.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware

CKPT="$CKPT_ROOT"
TARGET="$(PYTHONPATH="$REPO_DIR/src" "$(bootstrap_python)" -m llmplatform.common.models resolve teacher)"
METHOD="$(run_in_env quantize python scripts/quantize.py plan --method-only)"
PRECISIONS=(bf16)
[ "$METHOD" = "fp8" ] && PRECISIONS+=(fp8)

DRAFTS=(base_1_5b distilled_1_5b)
TOKENS=(3 5 8)
for draft in "${DRAFTS[@]}"; do
  run_in_env serve-sglang python scripts/export_draft.py --draft "$draft"
done

for precision in "${PRECISIONS[@]}"; do
  QUANT=()
  [ "$precision" = "fp8" ] && QUANT=(--quantization fp8)
  # baseline without speculation
  run_in_env serve-sglang python -m llmplatform.serve.launch --engine sglang --model "$TARGET" \
    --served-model-name teacher --max-model-len 4096 --mem-fraction 0.8 "${QUANT[@]}" \
    --log "logs/spec_baseline_${precision}.log" -- \
    python scripts/speculative.py run --precision "$precision" --served-model-name teacher
  for draft in "${DRAFTS[@]}"; do
    for tokens in "${TOKENS[@]}"; do
      run_in_env serve-sglang python -m llmplatform.serve.launch --engine sglang --model "$TARGET" \
        --served-model-name teacher --max-model-len 4096 --mem-fraction 0.8 "${QUANT[@]}" \
        --draft-model "$CKPT/speculative/${draft}_padded" --spec-steps "$((tokens - 1))" \
        --spec-topk 1 --spec-draft-tokens "$tokens" \
        --log "logs/spec_${draft}_k${tokens}_${precision}.log" -- \
        python scripts/speculative.py run --precision "$precision" --draft "$draft" \
        --draft-tokens "$tokens" --served-model-name teacher
    done
  done
done
run_in_env core python scripts/speculative.py e6
