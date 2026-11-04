#!/usr/bin/env bash
# Train one alignment stage with DeepSpeed ZeRO-2 and merge the adapter.
# Usage: jobs/train.sh --stage sft|dpo|orpo|simpo [train.py options, for example --set optim.max_steps=60]
# Resumes from the newest checkpoint, so chained submissions continue an unfinished run.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

STAGE=""
if [ "${1:-}" = "--stage" ]; then
  STAGE="${2:?stage name required}"
  shift 2
fi
[ -n "$STAGE" ] || die "usage: jobs/train.sh --stage sft|dpo|orpo|simpo [options]"
CONFIG="configs/train/$STAGE.yaml"
[ -f "$CONFIG" ] || die "no config $CONFIG"

log_hardware
run_in_env train accelerate launch --config_file configs/accelerate/zero2.yaml \
  --num_processes "$NUM_GPUS" scripts/train.py fit --config "$CONFIG" "$@"
run_in_env train python scripts/train.py merge --config "$CONFIG" "$@"
