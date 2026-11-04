#!/usr/bin/env bash
# Run a job body on this machine with the same arguments as slurm/submit.sh.
# Usage: local/run.sh JOB [--gpus N] [--smoke] [--chain K] [--time HH:MM:SS] [--dry-run] [-- args]
# The job restarts after a time-limit stop or a crash, up to local.max_restarts, and resumes
# from its latest checkpoint. Run it under tmux or nohup for runs longer than a session.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=jobs/lib.sh
. "$REPO_DIR/jobs/lib.sh"
cd "$REPO_DIR"

[ "$#" -ge 1 ] || die "usage: local/run.sh JOB [options] [-- job arguments]"
case "$1" in -h | --help) sed -n '2,6p' "${BASH_SOURCE[0]}"; exit 0 ;; esac
JOB="$1"
shift
GPUS=""
DRY_RUN="${DRY_RUN:-0}"
NO_DOCKER=0
JOB_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --smoke) shift ;;
    --chain | --time | --dependency) shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-docker) NO_DOCKER=1; shift ;;
    --) shift; JOB_ARGS=("$@"); break ;;
    *) die "unknown option $1" ;;
  esac
done

[ -f "jobs/$JOB.sh" ] || die "no job body jobs/$JOB.sh"
GPU_COUNT_QUERY=(gpus --job "$JOB")
[ -n "$GPUS" ] && GPU_COUNT_QUERY+=(--gpus "$GPUS")
GPU_COUNT="$(compute_query "${GPU_COUNT_QUERY[@]}")"

eval "$(compute_query shell-env)"
LIMIT_SECONDS="$(awk -v h="$LOCAL_TIME_LIMIT_HOURS" 'BEGIN {printf "%d", h * 3600}')"

if [ "$GPU_COUNT" -gt 0 ] && [ "$DRY_RUN" != "1" ]; then
  AVAILABLE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  [ "${AVAILABLE:-0}" -ge "$GPU_COUNT" ] || die "job needs $GPU_COUNT GPUs, found ${AVAILABLE:-0}"
  DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEVICES}"
fi
export LLMPLATFORM_GPUS="$GPU_COUNT"

if [ "$LOCAL_USE_DOCKER" = "1" ] && [ "$NO_DOCKER" -eq 0 ]; then
  DOCKER_FLAGS=(--gpus "$GPU_COUNT")
  [ "$DRY_RUN" = "1" ] && DOCKER_FLAGS+=(--dry-run)
  exec "$REPO_DIR/local/docker_run.sh" "$JOB" "${DOCKER_FLAGS[@]}" -- "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}"
fi

CMD=(bash "jobs/$JOB.sh" "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}")
if [ "$DRY_RUN" = "1" ]; then
  echo "LLMPLATFORM_GPUS=$GPU_COUNT limit=${LIMIT_SECONDS}s restarts=$LOCAL_MAX_RESTARTS ${CMD[*]}"
  exit 0
fi

mkdir -p logs
LOG_FILE="logs/${JOB}-$(date -u +%Y%m%dT%H%M%SZ).log"
ATTEMPT=0
while :; do
  ATTEMPT=$((ATTEMPT + 1))
  log "attempt $ATTEMPT of $((LOCAL_MAX_RESTARTS + 1)): ${CMD[*]}" | tee -a "$LOG_FILE"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM "$LIMIT_SECONDS" "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  else
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  fi
  STATUS="${PIPESTATUS[0]}"
  set -e
  [ "$STATUS" -eq 0 ] && exit 0
  [ "$ATTEMPT" -gt "$LOCAL_MAX_RESTARTS" ] && die "job $JOB failed with status $STATUS after $ATTEMPT attempts"
  log "job stopped with status $STATUS, resuming from the latest checkpoint" | tee -a "$LOG_FILE"
done
