#!/usr/bin/env bash
# Run a job body on whichever backend configs/compute.yaml selects.
# Usage: scripts/run_job.sh JOB [--gpus N] [--smoke] [--chain K] [--time HH:MM:SS] [-- args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=jobs/lib.sh
. "$REPO_DIR/jobs/lib.sh"

case "$(compute_query backend)" in
  slurm) exec "$REPO_DIR/slurm/submit.sh" "$@" ;;
  local) exec "$REPO_DIR/local/run.sh" "$@" ;;
  *) die "unsupported backend" ;;
esac
