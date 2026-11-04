#!/usr/bin/env bash
# Serving benchmark grid: for each model and engine, start a server, run the grid, stop the server.
# Usage: jobs/serve_bench.sh [--engines sglang,vllm,hf] [--models teacher_bf16,...]
# Finished cells are skipped on rerun, so a chained job continues where it stopped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

ENGINES="sglang,vllm,hf"
MODELS=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --engines) ENGINES="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done

log_hardware
METHOD="$(run_in_env quantize python scripts/quantize.py plan --method-only)"
[ "$METHOD" = "none" ] && METHOD=""
if [ -z "$MODELS" ]; then
  MODELS="$(run_in_env core python -m llmplatform.bench.models list | paste -sd, -)"
fi

for engine in ${ENGINES//,/ }; do
  for model in ${MODELS//,/ }; do
    case "$engine" in
      hf)
        case "$model" in
          *_quant) continue ;;
        esac
        run_in_env train python scripts/hf_generate_bench.py --model-name "$model"
        ;;
      sglang | vllm)
        METHOD_ARG=()
        case "$model" in
          *_quant) [ -n "$METHOD" ] || die "quantized variants need a GPU"; METHOD_ARG=(--method "$METHOD") ;;
        esac
        eval "LAUNCH=($(run_in_env core python -m llmplatform.bench.models launch-args "$model" "$engine" "${METHOD_ARG[@]}"))"
        run_in_env "serve-$engine" python -m llmplatform.serve.launch "${LAUNCH[@]}" \
          --log "logs/bench_${engine}_${model}.log" -- \
          python scripts/bench.py grid --engine "$engine" --model-name "$model" \
          --base-url '{base_url}'
        ;;
    esac
  done
done
run_in_env core python scripts/bench.py e4
