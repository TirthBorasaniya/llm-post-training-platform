#!/usr/bin/env bash
# Deletes the local kind cluster created by k8s_up.sh.
set -euo pipefail

# ============= Configuration =============
CLUSTER_NAME="${CLUSTER_NAME:-llm-serve}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export KUBECONFIG="${REPO_ROOT}/.kube/config"

usage() {
  cat <<USAGE
Usage: scripts/k8s_down.sh

Deletes the kind cluster "${CLUSTER_NAME}" and its kubeconfig entry.

Environment:
  CLUSTER_NAME   kind cluster name (default: ${CLUSTER_NAME})
  DRY_RUN=1      print commands without running them or contacting Docker

Options:
  -h, --help     show this message
USAGE
}

run() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

# ============= Main =============
main() {
  case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  if [ "${DRY_RUN}" != "1" ] && ! kind get clusters 2>/dev/null | grep -q -x "${CLUSTER_NAME}"; then
    echo "kind cluster ${CLUSTER_NAME} does not exist, nothing to delete"
    exit 0
  fi
  run kind delete cluster --name "${CLUSTER_NAME}" --kubeconfig "${KUBECONFIG}"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
