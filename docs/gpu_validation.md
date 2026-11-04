# GPU Kubernetes validation

This document describes a procedure to be executed by the repository owner on a rented single-GPU Linux virtual machine. It has not been run as part of writing it. No GPU Kubernetes claim is made anywhere in this repository until the procedure has been executed and its outputs have been committed as a results file labeled with the GPU model and the provider.

The procedure deploys the `llm-serve` chart with the SGLang backend on a single-node k3s cluster, installs the same monitoring and autoscaling stack as the local kind path, and drives it with the same k6 load profile scaled to GPU throughput.

## Pinned versions

| Component | Version |
|---|---|
| k3s | v1.37.2+k3s1 |
| NVIDIA container toolkit | 1.20.1-1 |
| NVIDIA k8s-device-plugin chart | 0.20.1 |
| kube-prometheus-stack chart | 92.18.0 |
| KEDA chart | 2.21.0 |
| SGLang image | lmsysorg/sglang:v0.5.23 |
| k6 image | grafana/k6:1.8.2 |

The kube-prometheus-stack and KEDA versions are the defaults of `scripts/k8s_up.sh`; keep them identical to the local run so that the two results are comparable.

## Machine requirements

- Ubuntu 22.04 or 24.04 on x86_64 with one NVIDIA GPU of at least 24 GB, 8 CPU cores, 32 GB RAM, and 100 GB of disk.
- A public IP is not required. All access is through the shell.
- The NVIDIA driver is installed. Verify it before anything else.

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > hardware.txt
```

## 1. NVIDIA container toolkit

Install the toolkit before k3s so that k3s detects the NVIDIA runtime when it starts.

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.1-1
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
nvidia-ctk --version
```

## 2. k3s

```bash
curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION='v1.37.2+k3s1' sh -s - --write-kubeconfig-mode 644
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get nodes
sudo grep -n nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

The last command must show an `nvidia` runtime entry. Register it as a runtime class:

```bash
kubectl apply -f - <<'YAML'
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
YAML
```

Install Helm 3 or later and kubectl using the official instructions of each project, then confirm `helm version` and `kubectl version --client`.

## 3. NVIDIA device plugin

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --version 0.20.1 --namespace nvidia-device-plugin --create-namespace \
  --set runtimeClassName=nvidia
kubectl -n nvidia-device-plugin rollout status daemonset/nvdp-nvidia-device-plugin --timeout=5m
kubectl describe node | grep -A2 'nvidia.com/gpu'
```

The node must report `nvidia.com/gpu: 1` as allocatable.

### Optional: advertise several GPU slots for scale-out

One physical GPU cannot host a second replica unless the device plugin advertises more than one `nvidia.com/gpu`. To observe scale-out mechanics on a single GPU, enable time slicing. Replicas then share the same GPU, so this validates deployment and autoscaling behavior and does not measure GPU scaling. State this in any result derived from it.

```bash
mkdir -p .tmp
cat > .tmp/device-plugin-config.yaml <<'YAML'
version: v1
sharing:
  timeSlicing:
    resources:
      - name: nvidia.com/gpu
        replicas: 3
YAML
helm upgrade --install nvdp nvdp/nvidia-device-plugin \
  --version 0.20.1 --namespace nvidia-device-plugin --create-namespace \
  --set runtimeClassName=nvidia \
  --set-file config.map.config=.tmp/device-plugin-config.yaml
```

## 4. Repository checkout

Obtain a copy of this repository on the VM and run the remaining commands from its root. The cache and temporary directories used below are inside the checkout.

```bash
export HELM_CACHE_HOME=$PWD/.cache/helm HELM_CONFIG_HOME=$PWD/.cache/helm-config HELM_DATA_HOME=$PWD/.cache/helm-data
mkdir -p .tmp
```

## 5. Monitoring and autoscaling

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --version 92.18.0 --namespace monitoring --create-namespace \
  --values k8s/monitoring-values.yaml --wait --timeout 10m
helm upgrade --install keda kedacore/keda \
  --version 2.21.0 --namespace keda --create-namespace \
  --values k8s/keda-values.yaml --wait --timeout 10m
kubectl create configmap llm-serve-dashboard \
  --from-file=llm-serve-dashboard.json=k8s/dashboards/llm-serve-dashboard.json \
  --namespace monitoring --dry-run=client -o yaml \
  | kubectl label --local -f - grafana_dashboard=1 -o yaml \
  | kubectl apply -f -
```

## 6. Deploy the chart

Set `MODEL_REPO` to the Hub repository of the model to serve. For a private repository, create a Secret holding a Hub token and add `--set model.tokenSecret=<secret-name>`. The SGLang backend downloads the snapshot itself, so the first start can take many minutes; the startup probe allows up to 30 minutes.

```bash
export MODEL_REPO=<owner>/<name>
helm lint charts/llm-serve --values charts/llm-serve/values-gpu-sglang.yaml --set model.repo="${MODEL_REPO}"
helm upgrade --install llm-serve charts/llm-serve \
  --values charts/llm-serve/values-gpu-sglang.yaml \
  --set model.repo="${MODEL_REPO}" \
  --set runtimeClassName=nvidia
kubectl rollout status deployment/llm-serve --timeout=30m
kubectl get pods -o wide
```

When time slicing is enabled and more than one replica should fit in GPU memory, limit the memory share of each SGLang process:

```bash
helm upgrade --install llm-serve charts/llm-serve \
  --values charts/llm-serve/values-gpu-sglang.yaml \
  --set model.repo="${MODEL_REPO}" \
  --set runtimeClassName=nvidia \
  --set 'sglang.extraArgs={--mem-fraction-static,0.25}'
```

Confirm the endpoint and the autoscaling metric before loading it:

```bash
kubectl port-forward svc/llm-serve 8080:80 &
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/metrics | grep sglang:num_queue_reqs
kill %1
kubectl get scaledobject llm-serve
```

## 7. Load run

The load script runs the k6 Job in the cluster and samples replica counts and queue depth every five seconds. `MAX_VUS=64` scales the ramp to GPU throughput. Set `MODEL_NAME` to the served model name if the server rejects the default value.

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
BACKEND=sglang MAX_VUS=64 DURATION_MINUTES=10 MAX_TOKENS=128 scripts/k8s_load.sh
```

## 8. Outputs

| File | Content |
|---|---|
| `.tmp/k8s_timeseries.jsonl` | One JSON object per sample with `ts`, `replicas`, `ready_replicas`, `queue_depth` |
| `.tmp/k6_summary.json` | End-of-test k6 summary, including `chat_completion_time` and the check pass rate |
| `.tmp/k6_job.log` | Full k6 job log |
| `hardware.txt` | GPU model, driver version, and memory recorded in the first step |

Also record the provider name, instance type, `kubectl version`, `helm version`, and `kubectl get pods -o yaml` for the serving pod (image digests). Copy the files to the laptop with a file transfer tool of your choice and commit the derived results file together with the code that produced it. The results file must state the GPU model and provider, and whether time slicing was used.

## 9. Cleanup

```bash
helm uninstall llm-serve
helm uninstall keda --namespace keda
helm uninstall kube-prometheus-stack --namespace monitoring
helm uninstall nvdp --namespace nvidia-device-plugin
/usr/local/bin/k3s-uninstall.sh
```

Terminate the virtual machine through the provider console when finished to stop billing.
