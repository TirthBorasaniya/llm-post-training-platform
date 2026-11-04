#!/usr/bin/env bash
# Sourced by every job body: storage paths, secrets, environment selection, and helpers.
# Works unchanged under a SLURM allocation and on a single machine.

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$JOBS_DIR/.." && pwd)"
export REPO_DIR
# shellcheck source=jobs/lib.sh
. "$JOBS_DIR/lib.sh"
set -euo pipefail

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# secrets live outside the repository and are never synced
PLATFORM_ENV_FILE="${PLATFORM_ENV_FILE:-$HOME/.llmplatform.env}"
if [ -f "$PLATFORM_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PLATFORM_ENV_FILE"
  set +a
fi

eval "$(compute_query shell-env)"
export PLATFORM_ROOT ENV_ROOT HF_HOME DATA_ROOT CKPT_ROOT MODEL_ROOT
mkdir -p "$ENV_ROOT" "$HF_HOME" "$DATA_ROOT" "$CKPT_ROOT" "$MODEL_ROOT" "$REPO_DIR/logs"

export TOKENIZERS_PARALLELISM=false
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:$REPO_DIR/mlruns}"
export NUM_GPUS="${LLMPLATFORM_GPUS:-1}"

# run a command inside a named environment: a virtual environment, else an Apptainer image
run_in_env() {
  local env_name="$1"
  shift
  if [ -x "$ENV_ROOT/$env_name/bin/python" ]; then
    (
      # shellcheck disable=SC1090
      . "$ENV_ROOT/$env_name/bin/activate"
      export LLMPLATFORM_ENV_MODE=venv
      "$@"
    )
  elif [ -f "$ENV_ROOT/$env_name.sif" ]; then
    export APPTAINERENV_LLMPLATFORM_ENV_MODE=apptainer
    apptainer exec --nv --bind "$REPO_DIR" --bind "$PLATFORM_ROOT" \
      --pwd "$REPO_DIR" "$ENV_ROOT/$env_name.sif" "$@"
  else
    die "environment '$env_name' not found under $ENV_ROOT; run: scripts/run_job.sh setup_envs"
  fi
}

# print the first environment name that exists, for jobs with an optional faster environment
pick_env() {
  local candidate
  for candidate in "$@"; do
    if [ -x "$ENV_ROOT/$candidate/bin/python" ] || [ -f "$ENV_ROOT/$candidate.sif" ]; then
      echo "$candidate"
      return 0
    fi
  done
  die "none of these environments exist: $*"
}

# fail early when a secret that a stage needs is missing, without printing it
require_secret() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] || die "$name is not set; add it to $PLATFORM_ENV_FILE"
  done
}

# record node, GPU, driver, and CUDA details at the top of the job log
log_hardware() {
  log "host=$(hostname) gpus_requested=$NUM_GPUS backend=$COMPUTE_BACKEND"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  fi
}
