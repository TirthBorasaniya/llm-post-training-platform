# llm-serve

Helm chart that serves one model behind an OpenAI-compatible API with either a llama.cpp (CPU) or an SGLang (GPU) backend. It exposes Prometheus metrics, a PodDisruptionBudget, and optional KEDA autoscaling on queue depth.

## Backends

| Backend | Image | Server port | Health | Metrics | Queue metric |
|---|---|---|---|---|---|
| `llamacpp` | `ghcr.io/ggml-org/llama.cpp:server-b12110` | 8080 | `/health` | `/metrics` | `llamacpp:requests_deferred` |
| `sglang` | `lmsysorg/sglang:v0.5.23` | 30000 | `/health` | `/metrics` | `sglang:num_queue_reqs` |

Model delivery differs by backend:

- `llamacpp`: an init container downloads `model.file` from `model.repo` into `/models/model.gguf` and compares its SHA-256 with `model.sha256`. A mismatch fails the pod. Rendering fails when `model.sha256` is empty unless `model.allowUnverified=true`.
- `sglang`: the server downloads the repository snapshot through its own Hub client with `HF_HOME` on the model volume, or loads `model.path`. No checksum can be verified, so `model.allowUnverified=true` is required.

## Install

CPU backend (kind or any cluster without GPUs):

```bash
helm upgrade --install llm-serve charts/llm-serve \
  --values charts/llm-serve/values-cpu-kind.yaml \
  --set model.repo=<owner>/<name> \
  --set model.file=<file>.gguf \
  --set model.sha256=<64 hexadecimal characters>
```

GPU backend:

```bash
helm upgrade --install llm-serve charts/llm-serve \
  --values charts/llm-serve/values-gpu-sglang.yaml \
  --set model.repo=<owner>/<name>
```

Autoscaling requires KEDA and a Prometheus server reachable at `autoscaling.prometheus.serverAddress`. The ServiceMonitor requires the Prometheus Operator CRDs.

## Values

| Key | Default | Description |
|---|---|---|
| `backend` | `llamacpp` | `llamacpp` or `sglang`; any other value fails rendering |
| `images.llamacpp.{repository,tag}` | see above | llama.cpp server image |
| `images.sglang.{repository,tag}` | see above | SGLang image |
| `images.init.{repository,tag}` | `curlimages/curl:8.14.1` | Download image for the init container |
| `replicaCount` | `1` | Replicas when autoscaling is disabled |
| `model.repo` | `""` | Hub repository id |
| `model.revision` | `main` | Repository revision |
| `model.file` | `""` | GGUF file name (llamacpp) |
| `model.sha256` | `""` | Expected SHA-256 of the GGUF file (llamacpp) |
| `model.allowUnverified` | `false` | Permit rendering without checksum verification |
| `model.path` | `""` | Local model path (sglang) |
| `model.endpoint` | `https://huggingface.co` | Hub endpoint used by the init container |
| `model.tokenSecret`, `model.tokenSecretKey` | `""`, `token` | Secret holding an optional Hub token |
| `model.persistence.enabled` | `false` | Use a PVC instead of an `emptyDir` |
| `model.persistence.{existingClaim,size,storageClassName,accessModes}` | see values | PVC settings |
| `llamacpp.parallel`, `llamacpp.contextSize` | `2`, `4096` | `--parallel` and `-c` (the context is shared across slots) |
| `llamacpp.extraArgs` | `[]` | Additional server arguments |
| `sglang.extraArgs` | `[]` | Additional `launch_server` arguments |
| `sglang.shmSize` | `8Gi` | Size of the `/dev/shm` volume |
| `service.port`, `service.metricsPort` | `80`, `9100` | Service ports named `http` and `metrics` |
| `resources` | `{}` | Container resources; sglang always gets an `nvidia.com/gpu` limit of 1 unless set |
| `runtimeClassName` | `""` | Runtime class, for example `nvidia` |
| `probes.startup.failureThreshold` | `180` | Startup attempts at 10 s each |
| `pdb.enabled`, `pdb.minAvailable` | `true`, `1` | PodDisruptionBudget |
| `autoscaling.enabled` | `false` | Render the KEDA ScaledObject |
| `autoscaling.{minReplicas,maxReplicas,cooldownPeriod,pollingInterval,threshold}` | `1`, `3`, `120`, `15`, `2` | Scaling parameters |
| `autoscaling.query` | `""` | Query override; empty selects `sum(<queue metric>)` |
| `autoscaling.prometheus.serverAddress` | `http://kube-prometheus-stack-prometheus.monitoring.svc:9090` | Prometheus address used by KEDA |
| `serviceMonitor.enabled` | `false` | Render the ServiceMonitor |
| `serviceMonitor.release` | `kube-prometheus-stack` | Value of the `release` label |
| `nodeSelector`, `tolerations`, `affinity` | empty | Scheduling |

## Validation

```bash
helm lint charts/llm-serve --values charts/llm-serve/values-cpu-kind.yaml \
  --set model.repo=example/model-GGUF --set model.file=model.gguf \
  --set model.sha256=0000000000000000000000000000000000000000000000000000000000000000
```

The values above are synthetic and exist only to satisfy the required fields during linting.
