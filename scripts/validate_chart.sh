#!/usr/bin/env bash
# Lint and render the chart for both backends and validate the manifests with kubeconform.
# Usage: scripts/validate_chart.sh
# Model values are synthetic and exist only on the command line of this check.
# kubeconform is taken from PATH or ./.cache/bin; without it the schema step is reported as skipped.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export HELM_CACHE_HOME="${HELM_CACHE_HOME:-$REPO_DIR/.cache/helm}"
export HELM_CONFIG_HOME="${HELM_CONFIG_HOME:-$REPO_DIR/.cache/helm-config}"
export HELM_DATA_HOME="${HELM_DATA_HOME:-$REPO_DIR/.cache/helm-data}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/.tmp/rendered}"
mkdir -p "$OUT_DIR"

CPU_SET=(--set model.repo=example/model-GGUF --set model.file=model-Q4_K_M.gguf
  --set model.sha256=0000000000000000000000000000000000000000000000000000000000000000)
GPU_SET=(--set model.repo=example/model)
CHART=charts/llm-serve

KUBECONFORM="$(command -v kubeconform || true)"
[ -n "$KUBECONFORM" ] || { [ -x .cache/bin/kubeconform ] && KUBECONFORM="$REPO_DIR/.cache/bin/kubeconform"; } || true
SCHEMA_DIR="${KUBECONFORM_SCHEMAS:-$REPO_DIR/.cache/kubeconform-schemas}"
K8S_VERSION="${K8S_VERSION:-1.30.0}"
CRD_CATALOG="https://raw.githubusercontent.com/datreeio/CRDs-catalog/main"
# schemas cached under $SCHEMA_DIR (k8s/ and crds/) are used offline; otherwise they are downloaded
if [ -d "$SCHEMA_DIR/k8s" ]; then
  SCHEMA_ARGS=(-schema-location "$SCHEMA_DIR/k8s/{{.NormalizedKubernetesVersion}}-standalone-strict/{{.ResourceKind}}{{.KindSuffix}}.json"
    -schema-location "$SCHEMA_DIR/crds/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json")
else
  SCHEMA_ARGS=(-schema-location default
    -schema-location "$CRD_CATALOG/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json")
fi

helm lint "$CHART" --values "$CHART/values-cpu-kind.yaml" "${CPU_SET[@]}"
helm lint "$CHART" --values "$CHART/values-gpu-sglang.yaml" "${GPU_SET[@]}"

for flag in true false; do
  for target in cpu gpu; do
    if [ "$target" = cpu ]; then
      values="$CHART/values-cpu-kind.yaml"; extra=("${CPU_SET[@]}")
    else
      values="$CHART/values-gpu-sglang.yaml"; extra=("${GPU_SET[@]}")
    fi
    out="$OUT_DIR/$target-$flag.yaml"
    helm template llm-serve "$CHART" --namespace llm --values "$values" "${extra[@]}" \
      --set autoscaling.enabled="$flag" --set serviceMonitor.enabled="$flag" >"$out"
    if [ -n "$KUBECONFORM" ]; then
      "$KUBECONFORM" -strict -ignore-missing-schemas -summary \
        -kubernetes-version "$K8S_VERSION" "${SCHEMA_ARGS[@]}" "$out"
    else
      echo "kubeconform not found; skipped schema validation of $out"
    fi
  done
done
