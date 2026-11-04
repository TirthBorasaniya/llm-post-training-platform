#!/usr/bin/env bash
# Write the distillation summary from the checkpoint result files, the ledgers, and the generation summary.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
run_in_env core python scripts/distill.py e3
