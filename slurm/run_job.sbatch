#!/usr/bin/env bash
# Generic SLURM entry point. Resources come from sbatch options built by slurm/submit.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?must run under sbatch}"
JOB="${1:?job name required}"
shift
exec bash "jobs/${JOB}.sh" "$@"
