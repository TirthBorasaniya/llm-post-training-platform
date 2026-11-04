#!/usr/bin/env bash
# Creates the local kind cluster, installs monitoring and autoscaling, and deploys the chart.
set -euo pipefail

# ============= Configuration =============
CLUSTER_NAME="${CLUSTER_NAME:-llm-serve}"
RELEASE="${RELEASE:-llm-serve}"
NAMESPACE="${NAMESPACE:-default}"
VALUES_FILE="${VALUES_FILE:-charts/llm-serve/values-cpu-kind.yaml}"
KPS_VERSION="${KPS_VERSION:-92.18.0}"
KEDA_VERSION="${KEDA_VERSION:-2.21.0}"
HELM_WAIT_TIMEOUT="${HELM_WAIT_TIMEOUT:-20m}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export KUBECONFIG="${REPO_ROOT}/.kube/config"
export HELM_CACHE_HOME="${REPO_ROOT}/.cache/helm"
export HELM_CONFIG_HOME="${REPO_ROOT}/.cache/helm-config"
export HELM_DATA_HOME="${REPO_ROOT}/.cache/helm-data"

usage() {
  cat <<USAGE
Usage: MODEL_REPO=<owner>/<name> MODEL_FILE=<file>.gguf MODEL_SHA256=<hex> scripts/k8s_up.sh

Creates the kind cluster "${CLUSTER_NAME}" (1 control plane, 2 workers), installs
kube-prometheus-stack ${KPS_VERSION} and KEDA ${KEDA_VERSION}, loads the Grafana dashboard,
and installs the llm-serve chart with the CPU values file.

Environment:
  MODEL_REPO, MODEL_FILE, MODEL_SHA256   required model settings passed to the chart
  CLUSTER_NAME      kind cluster name (default: ${CLUSTER_NAME})
  RELEASE           Helm release name (default: ${RELEASE})
  NAMESPACE         namespace of the chart release (default: ${NAMESPACE})
  VALUES_FILE       chart values file (default: ${VALUES_FILE})
  KPS_VERSION       kube-prometheus-stack chart version (default: ${KPS_VERSION})
  KEDA_VERSION      KEDA chart version (default: ${KEDA_VERSION})
  HELM_WAIT_TIMEOUT wait timeout for the chart release (default: ${HELM_WAIT_TIMEOUT})
  DRY_RUN=1         print commands without running them or contacting Docker

The kubeconfig is written to ${REPO_ROOT}/.kube/config.
The script stops if another kind cluster or any non-kind container is running.

Options:
  -h, --help        show this message
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

preflight() {
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ docker ps (skipped in dry run)"
    echo "+ kind get clusters (skipped in dry run)"
    return 0
  fi
  local tool
  for tool in docker kind helm kubectl; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "missing required tool: ${tool}" >&2; exit 1; }
  done
  local clusters other_clusters other_containers
  clusters="$(kind get clusters 2>/dev/null || true)"
  other_clusters="$(printf '%s\n' "${clusters}" | grep -v -x -e "${CLUSTER_NAME}" -e '' -e 'No kind clusters found.' || true)"
  if [ -n "${other_clusters}" ]; then
    echo "another kind cluster is running:" >&2
    echo "${other_clusters}" >&2
    echo "stop it or confirm it may be removed before continuing" >&2
    exit 1
  fi
  other_containers="$(docker ps --format '{{.Names}}' | grep -v -e "^${CLUSTER_NAME}-" || true)"
  if [ -n "${other_containers}" ]; then
    echo "other containers are running (a compose stack or another workload):" >&2
    echo "${other_containers}" >&2
    echo "stop them or confirm they may keep running before continuing" >&2
    exit 1
  fi
}

create_cluster() {
  if [ "${DRY_RUN}" != "1" ] && kind get clusters 2>/dev/null | grep -q -x "${CLUSTER_NAME}"; then
    echo "kind cluster ${CLUSTER_NAME} already exists, reusing it"
    kind export kubeconfig --name "${CLUSTER_NAME}" --kubeconfig "${KUBECONFIG}"
    return 0
  fi
  run mkdir -p "${REPO_ROOT}/.kube"
  run kind create cluster --name "${CLUSTER_NAME}" --config k8s/kind-config.yaml \
    --kubeconfig "${KUBECONFIG}" --wait 180s
}

install_monitoring_and_keda() {
  run helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  run helm repo add kedacore https://kedacore.github.io/charts
  run helm repo update
  run helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
    --version "${KPS_VERSION}" --namespace monitoring --create-namespace \
    --values k8s/monitoring-values.yaml --wait --timeout 10m
  run helm upgrade --install keda kedacore/keda \
    --version "${KEDA_VERSION}" --namespace keda --create-namespace \
    --values k8s/keda-values.yaml --wait --timeout 10m
}

load_dashboard() {
  if [ "${DRY_RUN}" = "1" ]; then
    echo "+ kubectl create configmap llm-serve-dashboard --from-file=llm-serve-dashboard.json=k8s/dashboards/llm-serve-dashboard.json --namespace monitoring --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -"
    return 0
  fi
  kubectl create configmap llm-serve-dashboard \
    --from-file=llm-serve-dashboard.json=k8s/dashboards/llm-serve-dashboard.json \
    --namespace monitoring --dry-run=client -o yaml \
    | kubectl label --local -f - grafana_dashboard=1 -o yaml \
    | kubectl apply -f -
}

install_chart() {
  local repo="${MODEL_REPO:-}" file="${MODEL_FILE:-}" sha="${MODEL_SHA256:-}"
  if [ "${DRY_RUN}" = "1" ]; then
    repo="${repo:-<MODEL_REPO>}"
    file="${file:-<MODEL_FILE>}"
    sha="${sha:-<MODEL_SHA256>}"
  elif [ -z "${repo}" ] || [ -z "${file}" ] || [ -z "${sha}" ]; then
    echo "MODEL_REPO, MODEL_FILE, and MODEL_SHA256 must all be set" >&2
    exit 1
  fi
  run helm upgrade --install "${RELEASE}" charts/llm-serve \
    --namespace "${NAMESPACE}" --create-namespace \
    --values "${VALUES_FILE}" \
    --set "model.repo=${repo}" --set "model.file=${file}" --set "model.sha256=${sha}" \
    --wait --timeout "${HELM_WAIT_TIMEOUT}"
}

# ============= Main =============
main() {
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  preflight
  create_cluster
  install_monitoring_and_keda
  load_dashboard
  install_chart
  echo "cluster ${CLUSTER_NAME} is ready; use KUBECONFIG=${KUBECONFIG}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
