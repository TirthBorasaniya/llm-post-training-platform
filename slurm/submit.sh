#!/usr/bin/env bash
# Submit a job body through sbatch with resources taken from configs/compute.yaml.
# Usage: slurm/submit.sh JOB [--gpus N] [--smoke] [--chain K] [--time HH:MM:SS]
#                        [--dependency JOBID] [--dry-run] [-- job arguments]
# --chain K submits K dependent jobs (afterany) so a run longer than the time limit resumes
# from its latest checkpoint; finished runs exit immediately.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=jobs/lib.sh
. "$REPO_DIR/jobs/lib.sh"
cd "$REPO_DIR"

[ "$#" -ge 1 ] || die "usage: slurm/submit.sh JOB [options] [-- job arguments]"
case "$1" in -h | --help) sed -n '2,7p' "${BASH_SOURCE[0]}"; exit 0 ;; esac
JOB="$1"
shift
GPUS=""
SMOKE=0
CHAIN=1
TIME=""
DEPENDENCY=""
DRY_RUN="${DRY_RUN:-0}"
JOB_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --chain) CHAIN="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; JOB_ARGS=("$@"); break ;;
    *) die "unknown option $1" ;;
  esac
done

QUERY=(sbatch-args --job "$JOB")
[ -n "$GPUS" ] && QUERY+=(--gpus "$GPUS")
[ "$SMOKE" -eq 1 ] && QUERY+=(--smoke)
[ -n "$TIME" ] && QUERY+=(--time "$TIME")

SBATCH_OPTS=()
while IFS= read -r line; do
  SBATCH_OPTS+=("$line")
done < <(compute_query "${QUERY[@]}")

GPU_COUNT_QUERY=(gpus --job "$JOB")
[ -n "$GPUS" ] && GPU_COUNT_QUERY+=(--gpus "$GPUS")
GPU_COUNT="$(compute_query "${GPU_COUNT_QUERY[@]}")"

eval "$(compute_query shell-env)"
if [ "$CHAIN" -gt "$SLURM_MAX_CHAIN" ]; then
  die "--chain $CHAIN exceeds slurm.max_chain=$SLURM_MAX_CHAIN"
fi

mkdir -p logs
PREVIOUS="$DEPENDENCY"
for index in $(seq 1 "$CHAIN"); do
  CMD=(sbatch --parsable "${SBATCH_OPTS[@]}" "--export=ALL,LLMPLATFORM_GPUS=$GPU_COUNT")
  [ -n "$PREVIOUS" ] && CMD+=("--dependency=afterany:$PREVIOUS")
  CMD+=(slurm/run_job.sbatch "$JOB")
  [ "${#JOB_ARGS[@]}" -gt 0 ] && CMD+=("${JOB_ARGS[@]}")
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$index/$CHAIN] ${CMD[*]}"
    PREVIOUS="DRYRUN$index"
  else
    PREVIOUS="$(command "${CMD[@]}")"
    echo "submitted $JOB ($index/$CHAIN) as job $PREVIOUS"
  fi
done
