"""Resolve benchmark model variants to server launch options."""

import argparse
import os
import shlex
from pathlib import Path

from llmplatform.bench.config import BenchConfig, BenchModel, load_bench
from llmplatform.common.models import ModelsConfig, load_models, resolve_model
from llmplatform.quantize.plan import Method

# ============= Resolution =============


def resolve_variant(
    entry: BenchModel, method: Method | None, models: ModelsConfig, ckpt_root: str | Path
) -> tuple[str, str | None]:
    """
    Return the model location and the server ``--quantization`` value for a variant.

    Parameters
    ----------
    entry : BenchModel
        Variant of the benchmark config.
    method : {"fp8", "awq"} or None
        GPU quantization method chosen for this machine; required for quantized variants.
    models, ckpt_root
        Registry and checkpoint root.

    Returns
    -------
    path, quantization : tuple[str, str or None]
        FP8 serves the original weights with online quantization; AWQ serves the checkpoint
        written by the quantization job.
    """
    if not entry.quantized:
        return resolve_model(models, entry.model, ckpt_root), None
    if method == "fp8":
        return resolve_model(models, entry.model, ckpt_root), "fp8"
    if method == "awq":
        return resolve_model(models, f"{entry.model}_awq", ckpt_root), None
    raise ValueError("a quantization method is required for quantized variants")


def launch_args(
    entry: BenchModel,
    engine: str,
    method: Method | None,
    models: ModelsConfig,
    ckpt_root: str | Path,
    config: BenchConfig,
) -> list[str]:
    """Return ``llmplatform.serve.launch`` options for a variant and engine."""
    path, quantization = resolve_variant(entry, method, models, ckpt_root)
    args = [
        "--engine",
        engine,
        "--model",
        path,
        "--served-model-name",
        entry.name,
        "--max-model-len",
        str(config.server.max_model_len),
        "--mem-fraction",
        str(
            config.server.sglang_mem_fraction
            if engine == "sglang"
            else config.server.vllm_gpu_memory_utilization
        ),
    ]
    if quantization:
        args += ["--quantization", quantization]
    return args


def main(argv: list[str] | None = None) -> int:
    """Shell helpers: ``list`` model names or print ``launch-args NAME ENGINE METHOD``."""
    parser = argparse.ArgumentParser(description="Benchmark model helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    launch = sub.add_parser("launch-args")
    launch.add_argument("name")
    launch.add_argument("engine")
    launch.add_argument("--method", choices=["fp8", "awq"], default=None)
    args = parser.parse_args(argv)
    config = load_bench()
    if args.command == "list":
        print("\n".join(entry.name for entry in config.models))
        return 0
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    ckpt_root = os.environ.get("CKPT_ROOT", str(work / "checkpoints"))
    entry = next(item for item in config.models if item.name == args.name)
    print(
        shlex.join(launch_args(entry, args.engine, args.method, load_models(), ckpt_root, config))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
