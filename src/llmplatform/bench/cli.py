"""Command-line entry point for serving benchmarks, agent traces, and their summaries."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, cast

from llmplatform.bench.agent_traces import build_traces, load_sources
from llmplatform.bench.config import load_agent, load_bench, load_cost
from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.io import read_jsonl, write_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results

logger = logging.getLogger(__name__)

# ============= Constants =============

PACKAGES = ["sglang", "vllm", "torch", "transformers", "httpx"]
CELLS_FILE = "cells.jsonl"
AGENT_CELLS_FILE = "agent_cells.jsonl"


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


def _hardware_extra() -> dict[str, Any]:
    return {"hardware": hardware_record(load_compute()).model_dump(mode="json")}


# ============= Commands =============


def cmd_grid(args: argparse.Namespace) -> int:
    """Run the SGLang or vLLM benchmark grid against a running server."""
    from llmplatform.bench.grid import run_grid

    _, data_root = _roots()
    gpu_ids = [int(item) for item in args.gpu_ids.split(",")] if args.gpu_ids else None
    ran = run_grid(
        load_bench(),
        args.engine,
        args.model_name,
        args.base_url,
        args.served_model_name or args.model_name,
        data_root / "bench" / CELLS_FILE,
        gpu_ids,
        row_extra=_hardware_extra(),
    )
    logger.info("grid finished", extra={"cells_run": ran})
    return 0


def cmd_hf(args: argparse.Namespace) -> int:
    """Run the Hugging Face static batching baseline for one model."""
    from llmplatform.bench.hf_generate import run_hf_bench

    ckpt_root, data_root = _roots()
    config, models = load_bench(), load_models()
    entry = next(item for item in config.models if item.name == args.model_name)
    if entry.quantized:
        raise SystemExit("the Hugging Face baseline runs bf16 models only")
    rows = run_hf_bench(
        resolve_model(models, entry.model, ckpt_root),
        entry.name,
        config,
        revision_of(models, entry.model),
        args.cpu,
    )
    extra = _hardware_extra()
    path = data_root / "bench" / CELLS_FILE
    existing = read_jsonl(path) if path.is_file() else []
    write_jsonl(path, [*existing, *({**row, **extra} for row in rows)])
    return 0


def cmd_agent_traces(args: argparse.Namespace) -> int:
    """Generate the committed agent traces from the source prompts and tool schemas."""
    config = load_agent()
    root = repo_root()
    rows = build_traces(load_sources(root / config.source_dir), config)
    write_jsonl(root / config.output, rows)
    logger.info("wrote traces", extra={"rows": len(rows), "path": config.output})
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    """Replay the traces against a running server at every configured concurrency."""
    from llmplatform.bench.agent_bench import run_agent_level
    from llmplatform.serve.engines import Engine, ServerSpec

    _, data_root = _roots()
    config = load_agent()
    entry = next(item for item in config.engines if item.name == args.config_name)
    rows = read_jsonl(repo_root() / config.output)
    spec = ServerSpec(cast(Engine, entry.engine), "", port=args.port)
    out = data_root / "bench" / AGENT_CELLS_FILE
    extra = _hardware_extra()
    results = read_jsonl(out) if out.is_file() else []
    for concurrency in config.concurrency:
        summary = run_agent_level(
            rows,
            entry.engine,
            spec.openai_url(),
            spec.metrics_url(),
            args.served_model_name,
            concurrency,
            spec.base_url(),
            entry.prefix_cache,
        )
        results.append(
            {
                "model": args.model,
                "config_name": entry.name,
                "engine": entry.engine,
                "prefix_cache": entry.prefix_cache,
                **summary,
                **extra,
            }
        )
        write_jsonl(out, results)
    return 0


def _write(
    name: str, metrics: dict[str, Any], config_path: str, hardware: Any, notes: str = ""
) -> None:
    root = repo_root()
    result = build_results(
        name,
        metrics,
        config_ref(root / config_path, root),
        hardware,
        package_versions(PACKAGES),
        notes,
    )
    write_results(results_dir(root) / f"{name}.json", result)


def cmd_e4(args: argparse.Namespace) -> int:
    """Write ``results/e4_serving.json`` from the raw cell rows."""
    from llmplatform.bench.e4 import build_e4, hardware_from_rows, read_rows

    _, data_root = _roots()
    compute = load_compute()
    if compute.gpu_hourly_usd is None:
        raise SystemExit("set gpu_hourly_usd, price_source, and price_date in configs/compute.yaml")
    rows = read_rows(data_root / "bench" / CELLS_FILE)
    metrics = build_e4(rows, load_bench(), load_cost(), compute.gpu_hourly_usd)
    hardware = hardware_from_rows(rows).model_copy(
        update={
            "gpu_hourly_usd": compute.gpu_hourly_usd,
            "price_source": compute.price_source,
            "price_date": compute.price_date,
            "provider_label": compute.provider_label,
        }
    )
    _write("e4_serving", metrics, "configs/serve/bench.yaml", hardware)
    return 0


def cmd_e8(args: argparse.Namespace) -> int:
    """Write ``results/e8_agent_workloads.json`` from the raw agent rows."""
    from llmplatform.bench.e4 import hardware_from_rows
    from llmplatform.bench.e8 import build_e8

    _, data_root = _roots()
    rows = read_jsonl(data_root / "bench" / AGENT_CELLS_FILE)
    compute = load_compute()
    hardware = hardware_from_rows(rows).model_copy(
        update={
            "gpu_hourly_usd": compute.gpu_hourly_usd,
            "price_source": compute.price_source,
            "price_date": compute.price_date,
            "provider_label": compute.provider_label,
        }
    )
    _write("e8_agent_workloads", build_e8(rows), "configs/serve/agent.yaml", hardware)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/bench.py``."""
    configure_logging("bench")
    parser = argparse.ArgumentParser(description="Serving benchmark commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("grid")
    grid.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    grid.add_argument("--model-name", required=True)
    grid.add_argument("--served-model-name", default=None)
    grid.add_argument("--base-url", required=True)
    grid.add_argument("--gpu-ids", default=None)
    grid.set_defaults(func=cmd_grid)
    hf = sub.add_parser("hf")
    hf.add_argument("--model-name", required=True)
    hf.add_argument("--cpu", action="store_true")
    hf.set_defaults(func=cmd_hf)
    sub.add_parser("agent-traces").set_defaults(func=cmd_agent_traces)
    agent = sub.add_parser("agent")
    agent.add_argument("--config-name", required=True)
    agent.add_argument("--model", required=True)
    agent.add_argument("--served-model-name", required=True)
    agent.add_argument("--port", type=int, default=None)
    agent.set_defaults(func=cmd_agent)
    sub.add_parser("e4").set_defaults(func=cmd_e4)
    sub.add_parser("e8").set_defaults(func=cmd_e8)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
