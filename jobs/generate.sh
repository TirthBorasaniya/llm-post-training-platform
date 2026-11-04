#!/usr/bin/env bash
# Generate responses to the judge prompts for one checkpoint (greedy, 1,024 new tokens).
# Usage: jobs/generate.sh --checkpoint NAME [--base-url URL --served-model-name NAME]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
log_hardware
run_in_env eval python scripts/eval.py generate "$@"
