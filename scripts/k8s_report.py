"""Write the Kubernetes autoscaling results file from a load run's outputs."""

import argparse

from llmplatform.bench.k8s_report import build_report_from_files
from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import HardwareRecord, package_versions
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results

# ============= Constants =============

CPU_RESULT = "e5_k8s_cpu.json"
GPU_RESULT = "e5_k8s_gpu.json"
PACKAGES = ["httpx"]


def main() -> int:
    """Build the report and write ``results/e5_k8s_cpu.json`` or ``results/e5_k8s_gpu.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--series", default=".tmp/k8s_timeseries.jsonl")
    parser.add_argument("--requests", default=".tmp/k6_requests.jsonl")
    parser.add_argument("--summary", default=".tmp/k6_summary.json")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--provider-label", default=None, help="GPU runs: provider description")
    parser.add_argument("--gpu-model", default=None, help="GPU runs: GPU model name")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-hourly-usd", type=float, default=None)
    args = parser.parse_args()

    root = repo_root()
    compute = load_compute()
    if args.kind == "gpu":
        if not (args.provider_label and args.gpu_model and args.gpu_hourly_usd is not None):
            parser.error("--provider-label, --gpu-model, and --gpu-hourly-usd are required for gpu")
        hardware = HardwareRecord(
            gpu_model=args.gpu_model,
            gpu_count=args.gpu_count,
            provider_label=args.provider_label,
            gpu_hourly_usd=args.gpu_hourly_usd,
        )
        label, target = "GPU on a single-node cluster", GPU_RESULT
    else:
        hardware = hardware_record(compute).model_copy(update={"gpu_model": "none", "gpu_count": 0})
        label, target = "CPU on kind", CPU_RESULT
    metrics = build_report_from_files(
        args.series, args.requests, args.summary, args.threshold, args.window_seconds
    )
    metrics["label"] = label
    result = build_results(
        f"e5_k8s_{args.kind}",
        metrics,
        config_ref(root / "charts/llm-serve/values.yaml", root),
        hardware,
        package_versions(PACKAGES),
        notes=label,
    )
    write_results(results_dir(root) / target, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
