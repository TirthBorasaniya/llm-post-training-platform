"""Command-line entry point for the disaggregation runs and summary."""

import argparse
import logging
import os
from contextlib import ExitStack
from pathlib import Path

from llmplatform.bench.config import load_cost
from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.io import read_jsonl, write_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.disagg.config import load_disagg
from llmplatform.disagg.e7 import E7_RESULT, build_e7
from llmplatform.disagg.launch import SETUPS, start_stack
from llmplatform.disagg.workload import build_mixed_workload, run_rate

logger = logging.getLogger(__name__)

# ============= Constants =============

PACKAGES = ["sglang", "sglang-router", "nixl", "torch", "httpx"]
ROWS_FILE = "disagg_cells.jsonl"


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


# ============= Commands =============


def cmd_run(args: argparse.Namespace) -> int:
    """Start one setup on GPUs 0 and 1, drive every rate, and append the rows."""
    ckpt_root, data_root = _roots()
    config = load_disagg()
    model_path = resolve_model(load_models(), config.model, ckpt_root)
    extra = {"hardware": hardware_record(load_compute()).model_dump(mode="json")}
    path = data_root / "bench" / ROWS_FILE
    rows = read_jsonl(path) if path.is_file() else []
    with ExitStack() as stack:
        router_url = start_stack(stack, args.setup, config, model_path, repo_root() / "logs")
        for index, rate in enumerate(config.workload.rates):
            workload = build_mixed_workload(config.workload, seed_offset=index)
            summary = run_rate(
                f"{router_url}/v1", config.model, workload, rate, config.workload.seed + index
            )
            rows.append({"setup": args.setup, **summary, **extra})
            write_jsonl(path, rows)
    return 0


def cmd_e7(args: argparse.Namespace) -> int:
    """Write ``results/e7_disaggregation.json`` from the raw rows."""
    from llmplatform.bench.e4 import hardware_from_rows

    _, data_root = _roots()
    rows = read_jsonl(data_root / "bench" / ROWS_FILE)
    slo = load_cost().slo
    metrics = build_e7(rows, slo.ttft_p95_ms, slo.itl_p95_ms)
    compute = load_compute()
    hardware = hardware_from_rows(rows).model_copy(
        update={
            "gpu_hourly_usd": compute.gpu_hourly_usd,
            "price_source": compute.price_source,
            "price_date": compute.price_date,
            "provider_label": compute.provider_label,
        }
    )
    root = repo_root()
    result = build_results(
        "e7_disaggregation",
        metrics,
        config_ref(root / "configs/serve/disagg.yaml", root),
        hardware,
        package_versions(PACKAGES),
        notes=f"transfer backend {load_disagg().transfer_backend}",
    )
    write_results(results_dir(root) / E7_RESULT, result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/disagg.py``."""
    configure_logging("disagg")
    parser = argparse.ArgumentParser(description="Disaggregation commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--setup", choices=list(SETUPS), required=True)
    run.set_defaults(func=cmd_run)
    sub.add_parser("e7").set_defaults(func=cmd_e7)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
