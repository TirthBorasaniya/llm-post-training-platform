#!/usr/bin/env bash
# Upload a model folder or the GGUF files with a generated card; repositories are created private.
# Usage: jobs/publish.sh --repo teacher|student|gguf [--dry-run]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
case " $* " in *" --dry-run "*) ;; *) require_secret HF_TOKEN ;; esac
require_secret HF_NAMESPACE
run_in_env quantize python scripts/publish_hub.py "$@"
