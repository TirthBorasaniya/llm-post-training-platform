#!/usr/bin/env bash
# Evaluate one quantized variant: serve it, run its benchmark tasks, and generate judge responses.
# Usage: jobs/eval_variant.sh --checkpoint NAME [--limit N]
# GGUF variants need LLAMA_CPP_DIR to point at the built llama.cpp checkout.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

[ "${1:-}" = "--checkpoint" ] || die "usage: jobs/eval_variant.sh --checkpoint NAME"
NAME="${2:?checkpoint name required}"
shift 2
LIMIT_ARGS=("$@")

log_hardware
eval "LAUNCH_ARGS=($(run_in_env eval python -m llmplatform.eval.variants launch-args "$NAME"))"
if [[ " ${LAUNCH_ARGS[*]} " == *" llamacpp "* ]]; then
  SERVER_ENV="core"
else
  SERVER_ENV="serve-sglang"
fi
EVAL_PY="$ENV_ROOT/eval/bin/python"
[ -x "$EVAL_PY" ] || die "the eval environment is required as a virtual environment at $EVAL_PY"

run_in_env "$SERVER_ENV" python -m llmplatform.serve.launch "${LAUNCH_ARGS[@]}" \
  --log "logs/variant_$NAME.log" -- bash -c \
  "'$EVAL_PY' scripts/eval.py benchmarks --checkpoint $NAME --base-url {openai_url} ${LIMIT_ARGS[*]:-} \
   && '$EVAL_PY' scripts/eval.py generate --checkpoint $NAME --base-url {openai_url}"
