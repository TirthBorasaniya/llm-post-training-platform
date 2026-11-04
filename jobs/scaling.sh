#!/usr/bin/env bash
# Measure data-parallel throughput: 60 DPO optimizer steps on the allocated GPU count.
# Usage: scripts/run_job.sh scaling --gpus N   (N = 1, 2, 4, 8 on one node)
# The report aggregates every finished GPU count, so rerunning after each point keeps it current.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"

OUT="$CKPT_ROOT/scaling/gpus_$NUM_GPUS"
rm -rf "$OUT"
log_hardware
run_in_env train accelerate launch --config_file configs/accelerate/zero2.yaml \
  --num_processes "$NUM_GPUS" scripts/train.py fit --config configs/train/dpo.yaml \
  --output-dir "$OUT" --no-resume \
  --set optim.max_steps=60 --set checkpointing.save_steps=100000 --set tracking.enabled=false
run_in_env train python scripts/train.py scaling-report
