#!/usr/bin/env bash
# Merge the adapter of a trained stage into its base model.
# Usage: jobs/merge.sh --stage sft|dpo|orpo|simpo
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
[ "${1:-}" = "--stage" ] || die "usage: jobs/merge.sh --stage NAME"
STAGE="${2:?stage name required}"
shift 2
run_in_env train python scripts/train.py merge --config "configs/train/$STAGE.yaml" "$@"
