#!/usr/bin/env bash
# Runs the k6 chat ramp as a Kubernetes Job while recording replica and queue time series.
set -euo pipefail

# ============= Configuration =============
RELEASE="${RELEASE:-llm-serve}"
NAMESPACE="${NAMESPACE:-default}"
BACKEND="${BACKEND:-llamacpp}"
MAX_VUS="${MAX_VUS:-8}"
DURATION_MINUTES="${DURATION_MINUTES:-10}"
MAX_TOKENS="${MAX_TOKENS:-64}"
MODEL_NAME="${MODEL_NAME:-default}"
K6_IMAGE="${K6_IMAGE:-grafana/k6:1.8.2}"
PROM_NAMESPACE="${PROM_NAMESPACE:-monitoring}"
PROM_POD="${PROM_POD:-prometheus-kube-prometheus-stack-prometheus-0}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-5}"
OUT_DIR="${OUT_DIR:-.tmp}"
JOB_NAME="${JOB_NAME:-llm-serve-k6}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export KUBECONFIG="${KUBECONFIG:-${REPO_ROOT}/.kube/config}"

usage() {
  cat <<USAGE
Usage: scripts/k8s_load.sh

Runs k6/chat_ramp.js as a Job against the in-cluster Service of the release, waits for it
to finish, and extracts the k6 summary. During the run it samples, every ${SAMPLE_SECONDS} seconds,
the replica count, the ready replica count, and the backend queue metric from Prometheus.

Outputs (under ${OUT_DIR}/):
  k8s_timeseries.jsonl   one JSON object per sample: ts, replicas, ready_replicas, queue_depth
  k6_summary.json        the k6 end-of-test summary

Environment:
  RELEASE, NAMESPACE   release name and namespace of the chart (defaults: ${RELEASE}, ${NAMESPACE})
  BACKEND              llamacpp or sglang, selects the queue metric (default: ${BACKEND})
  MAX_VUS              peak virtual users (default: ${MAX_VUS})
  DURATION_MINUTES     ramp duration in minutes (default: ${DURATION_MINUTES})
  MAX_TOKENS           completion token limit per request (default: ${MAX_TOKENS})
  MODEL_NAME           value of the model field in requests (default: ${MODEL_NAME})
  K6_IMAGE             pinned k6 image (default: ${K6_IMAGE})
  PROM_NAMESPACE, PROM_POD   Prometheus pod used for queries
  SAMPLE_SECONDS       sampling interval (default: ${SAMPLE_SECONDS})
  KUBECONFIG           defaults to ${REPO_ROOT}/.kube/config
  DRY_RUN=1            print commands without running them

Options:
  -h, --help           show this message
USAGE
}

# ============= Helpers =============
run() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

queue_query() {
  case "${BACKEND}" in
    llamacpp) printf '%s' 'sum%28llamacpp%3Arequests_deferred%29' ;;
    sglang) printf '%s' 'sum%28sglang%3Anum_queue_reqs%29' ;;
    *) echo "BACKEND must be llamacpp or sglang" >&2; exit 2 ;;
  esac
}

job_manifest() {
  cat <<MANIFEST
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: k6
          image: ${K6_IMAGE}
          args: ["run", "/scripts/chat_ramp.js"]
          env:
            - name: TARGET_URL
              value: http://${RELEASE}.${NAMESPACE}.svc/v1/chat/completions
            - name: MAX_VUS
              value: "${MAX_VUS}"
            - name: DURATION_MINUTES
              value: "${DURATION_MINUTES}"
            - name: MAX_TOKENS
              value: "${MAX_TOKENS}"
            - name: MODEL_NAME
              value: "${MODEL_NAME}"
          volumeMounts:
            - name: script
              mountPath: /scripts
      volumes:
        - name: script
          configMap:
            name: ${JOB_NAME}-script
MANIFEST
}

sample_loop() {
  local out="$1" query value replicas ready
  query="$(queue_query)"
  while true; do
    replicas="$(kubectl -n "${NAMESPACE}" get deployment "${RELEASE}" -o jsonpath='{.status.replicas}' 2>/dev/null || true)"
    ready="$(kubectl -n "${NAMESPACE}" get deployment "${RELEASE}" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || true)"
    value="$(kubectl -n "${PROM_NAMESPACE}" exec "${PROM_POD}" -c prometheus -- \
      wget -qO- "http://localhost:9090/api/v1/query?query=${query}" 2>/dev/null \
      | sed -n 's/.*"value":\[[^,]*,"\([^"]*\)".*/\1/p' || true)"
    printf '{"ts": "%s", "replicas": %s, "ready_replicas": %s, "queue_depth": %s}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${replicas:-0}" "${ready:-0}" "${value:-null}" >> "${out}"
    sleep "${SAMPLE_SECONDS}"
  done
}

# ============= Main =============
main() {
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  queue_query >/dev/null
  local series="${OUT_DIR}/k8s_timeseries.jsonl" summary="${OUT_DIR}/k6_summary.json"
  local timeout_seconds=$(( DURATION_MINUTES * 60 + 900 ))

  run mkdir -p "${OUT_DIR}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ kubectl -n ${NAMESPACE} create configmap ${JOB_NAME}-script --from-file=chat_ramp.js=k6/chat_ramp.js --dry-run=client -o yaml | kubectl apply -f -"
    echo "+ kubectl -n ${NAMESPACE} delete job ${JOB_NAME} --ignore-not-found"
    echo "+ kubectl apply -f - (Job ${JOB_NAME}, image ${K6_IMAGE}, MAX_VUS=${MAX_VUS}, DURATION_MINUTES=${DURATION_MINUTES})"
    echo "+ sample every ${SAMPLE_SECONDS}s into ${series}: replicas, ready_replicas, queue_depth (${BACKEND})"
    run kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" --timeout="${timeout_seconds}s"
    echo "+ kubectl -n ${NAMESPACE} logs job/${JOB_NAME} | extract summary into ${summary}"
    return 0
  fi

  : > "${series}"
  kubectl -n "${NAMESPACE}" create configmap "${JOB_NAME}-script" \
    --from-file=chat_ramp.js=k6/chat_ramp.js --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n "${NAMESPACE}" delete job "${JOB_NAME}" --ignore-not-found
  job_manifest | kubectl apply -f -

  sample_loop "${series}" &
  local sampler_pid=$!
  trap 'kill "${sampler_pid}" 2>/dev/null || true' EXIT

  local status=0
  kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB_NAME}" \
    --timeout="${timeout_seconds}s" || status=$?
  sleep "${SAMPLE_SECONDS}"
  kill "${sampler_pid}" 2>/dev/null || true

  kubectl -n "${NAMESPACE}" logs "job/${JOB_NAME}" > "${OUT_DIR}/k6_job.log" || true
  sed -n '/^K6_SUMMARY_BEGIN$/,/^K6_SUMMARY_END$/p' "${OUT_DIR}/k6_job.log" \
    | sed -e '1d' -e '$d' > "${summary}"
  echo "time series: ${series}"
  echo "k6 summary: ${summary}"
  return "${status}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
