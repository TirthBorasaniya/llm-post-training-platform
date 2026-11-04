"""Command-line entry point for quantization planning, AWQ, and GGUF conversion."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.hashing import sha256_file
from llmplatform.common.io import read_json
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.quantize.config import load_quantize
from llmplatform.quantize.plan import choose_method, detect_capability, variant_names

logger = logging.getLogger(__name__)

# ============= Constants =============

RESULT_NAME = "quantization.json"
PACKAGES = ["llmcompressor", "torch", "transformers", "gguf"]


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


def record(entry_name: str, entry: dict[str, Any]) -> Path:
    """Merge one entry into ``results/quantization.json``."""
    root = repo_root()
    path = results_dir(root) / RESULT_NAME
    metrics: dict[str, Any] = dict(read_json(path)["metrics"]) if path.is_file() else {}
    metrics[entry_name] = entry
    result = build_results(
        "quantization",
        metrics,
        config_ref(root / "configs/quantize.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
    )
    return write_results(path, result)


# ============= Commands =============


def cmd_plan(args: argparse.Namespace) -> int:
    """Print the GPU quantization method chosen for this machine."""
    config = load_quantize()
    capability = detect_capability()
    if capability is None:
        print("none" if args.method_only else "no CUDA device; chosen on the target machine")
        return 0
    method = choose_method(capability, config.gpu.fp8_min_capability)
    if args.method_only:
        print(method)
    else:
        names = variant_names(config.gpu.models, method)
        print(f"capability {capability}: {method}; variants {names}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """Print one configured value per line for shell scripts."""
    config = load_quantize()
    values = {
        "gguf-models": config.gguf.models,
        "gpu-models": config.gpu.models,
        "llama-cpp-tag": [config.gguf.llama_cpp_tag],
    }[args.what]
    print("\n".join(values))
    return 0


def cmd_awq(args: argparse.Namespace) -> int:
    """Produce an AWQ checkpoint for one model."""
    from llmplatform.quantize.awq import quantize_awq

    ckpt_root, data_root = _roots()
    config, models = load_quantize(), load_models()
    source = resolve_model(models, args.model, ckpt_root)
    out = ckpt_root / config.output_subdir / f"{args.model}_awq"
    quantize_awq(
        source,
        revision_of(models, args.model),
        out,
        config.gpu.awq,
        data_root / config.gpu.awq.calibration_file,
    )
    record(
        f"{args.model}_awq", {"method": "awq", "scheme": config.gpu.awq.scheme, "path": out.name}
    )
    return 0


def cmd_gguf(args: argparse.Namespace) -> int:
    """Convert one model to Q4_K_M GGUF and verify it with five prompts."""
    from llmplatform.quantize.gguf import convert_and_quantize, verify_gguf

    ckpt_root, _ = _roots()
    config, models = load_quantize(), load_models()
    llama_dir = args.llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if not llama_dir:
        raise SystemExit("set LLAMA_CPP_DIR or pass --llama-cpp-dir")
    gguf_cfg = config.gguf
    out = ckpt_root / config.output_subdir
    quantized = convert_and_quantize(
        llama_dir,
        resolve_model(models, args.model, ckpt_root),
        out,
        args.model,
        gguf_cfg.quant_type,
        gguf_cfg.outtype,
    )
    checks = verify_gguf(
        llama_dir,
        quantized,
        gguf_cfg.verify.prompts,
        gguf_cfg.verify.n_predict,
        gguf_cfg.verify.timeout_seconds,
    )
    entry = {
        "method": "gguf",
        "quant_type": gguf_cfg.quant_type,
        "file": quantized.name,
        "bytes": quantized.stat().st_size,
        "sha256": sha256_file(quantized),
        "llama_cpp_tag": gguf_cfg.llama_cpp_tag,
        "verification": [
            {"prompt": item.prompt, "output": item.output, "ok": item.ok} for item in checks
        ],
        "verification_passed": all(item.ok for item in checks),
    }
    record(f"{args.model}_gguf", entry)
    return 0 if entry["verification_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/quantize.py``."""
    configure_logging("quantize")
    parser = argparse.ArgumentParser(description="Quantization commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--method-only", action="store_true")
    plan.set_defaults(func=cmd_plan)
    listing = sub.add_parser("list")
    listing.add_argument("what", choices=["gguf-models", "gpu-models", "llama-cpp-tag"])
    listing.set_defaults(func=cmd_list)
    awq = sub.add_parser("awq")
    awq.add_argument("--model", required=True)
    awq.set_defaults(func=cmd_awq)
    gguf = sub.add_parser("gguf")
    gguf.add_argument("--model", required=True)
    gguf.add_argument("--llama-cpp-dir", default=None)
    gguf.set_defaults(func=cmd_gguf)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
