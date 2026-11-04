#!/usr/bin/env bash
# Download pinned datasets, decontaminate them against evaluation questions, and write the manifest.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
run_in_env train python scripts/prepare_data.py "$@"
