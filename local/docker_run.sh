#!/usr/bin/env bash
# Run a job body inside the platform container on a single GPU machine.
# Usage: local/docker_run.sh JOB [--gpus N] [--dry-run] [-- job arguments]
# The repository and the storage root are bind mounted; secrets are forwarded by name only.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=jobs/lib.sh
. "$REPO_DIR/jobs/lib.sh"
cd "$REPO_DIR"

JOB="${1:?job name required}"
shift
GPUS=0
DRY_RUN="${DRY_RUN:-0}"
JOB_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; JOB_ARGS=("$@"); break ;;
    *) die "unknown option $1" ;;
  esac
done

PLATFORM_ENV_FILE="${PLATFORM_ENV_FILE:-$HOME/.llmplatform.env}"
if [ -f "$PLATFORM_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$PLATFORM_ENV_FILE"
  set +a
fi
eval "$(compute_query shell-env)"

DOCKER_CMD=(docker run --rm --ipc=host
  -v "$REPO_DIR:/workspace" -v "$PLATFORM_ROOT:/work"
  -e PLATFORM_ROOT=/work -e HF_TOKEN -e GROQ_API_KEY -e HF_NAMESPACE
  -w /workspace)
[ "$GPUS" -gt 0 ] && DOCKER_CMD+=(--gpus all -e "LLMPLATFORM_GPUS=$GPUS")
DOCKER_CMD+=("${EXTRA_DOCKER_ARGS[@]+"${EXTRA_DOCKER_ARGS[@]}"}")
DOCKER_CMD+=("$LOCAL_DOCKER_IMAGE" bash local/run.sh "$JOB" --no-docker --gpus "$GPUS")
[ "${#JOB_ARGS[@]}" -gt 0 ] && DOCKER_CMD+=(-- "${JOB_ARGS[@]}")

if [ "$DRY_RUN" = "1" ]; then
  echo "${DOCKER_CMD[*]}"
  exit 0
fi
exec "${DOCKER_CMD[@]}"
