#!/usr/bin/env bash
# Disaggregated versus colocated serving on the same two GPUs under mixed traffic.
# Usage: scripts/run_job.sh disagg --gpus 2
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
[ "$NUM_GPUS" -ge 2 ] || die "this job needs two GPUs on one node"
log_hardware
for setup in disaggregated colocated; do
  run_in_env serve-sglang python scripts/disagg.py run --setup "$setup"
done
run_in_env core python scripts/disagg.py e7
