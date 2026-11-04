#!/usr/bin/env bash
# Run the benchmark suite on one checkpoint.
# Usage: jobs/eval.sh --checkpoint NAME [--limit N]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware
ENV_NAME="$(pick_env eval-vllm eval)"
run_in_env "$ENV_NAME" python scripts/eval.py benchmarks "$@"
