#!/usr/bin/env bash
# Agent-style traffic with prefix caching on and off, for the teacher and the student.
# Usage: jobs/agent_bench.sh [--models teacher,distilled_1_5b]
# Requires eval/agent_traces/traces.jsonl (make agent-traces).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

MODELS="teacher,distilled_1_5b"
if [ "${1:-}" = "--models" ]; then
  MODELS="$2"
  shift 2
fi
[ -f eval/agent_traces/traces.jsonl ] || die "eval/agent_traces/traces.jsonl is missing; run make agent-traces"

log_hardware
CONFIGS=(sglang_cache_on sglang_cache_off vllm_cache_on vllm_cache_off)
for model in ${MODELS//,/ }; do
  MODEL_PATH="$(PYTHONPATH="$REPO_DIR/src" "$(bootstrap_python)" -m llmplatform.common.models resolve "$model")"
  for name in "${CONFIGS[@]}"; do
    engine="${name%%_*}"
    PORT=8000
    [ "$engine" = "sglang" ] && PORT=30000
    FLAGS=()
    case "$name" in *_cache_off) FLAGS+=(--no-prefix-cache) ;; esac
    [ "$engine" = "vllm" ] && FLAGS+=(--env VLLM_SERVER_DEV_MODE=1)
    run_in_env "serve-$engine" python -m llmplatform.serve.launch --engine "$engine" \
      --model "$MODEL_PATH" --served-model-name "$model" "${FLAGS[@]}" \
      --log "logs/agent_${name}_${model}.log" -- \
      python scripts/bench.py agent --config-name "$name" --model "$model" \
      --served-model-name "$model" --port "$PORT"
  done
done
run_in_env core python scripts/bench.py e8
