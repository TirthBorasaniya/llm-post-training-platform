#!/usr/bin/env bash
# Print node, GPU, driver, and CUDA details so a new machine can be checked before real jobs.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
log_hardware
compute_query validate
