#!/usr/bin/env bash
# Judge a candidate against the reference with the Groq-hosted judge; needs outbound HTTPS, no GPU.
# Usage: jobs/judge.sh --candidate NAME [--reference NAME] [--limit N]
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_DIR"
require_secret GROQ_API_KEY
run_in_env core python scripts/eval.py judge "$@"
