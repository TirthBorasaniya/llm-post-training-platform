"""Command-line entry point for training, merging, and scaling reports."""

import argparse
import logging
import os
from pathlib import Path

from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.train.config import load_train_config, resolve_stage_paths
from llmplatform.train.scaling import (
    SCALING_GPU_COUNTS,
    compute_point,
    read_step_metrics,
    scaling_summary,
)

logger = logging.getLogger(__name__)

# ============= Constants =============

RECORDED_PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "bitsandbytes",
    "deepspeed",
]
SCALING_RESULT = "e2_scaling.json"
STEP_METRICS = "step_metrics.jsonl"


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the environment set by job scripts."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


# ============= Commands =============


def cmd_fit(args: argparse.Namespace) -> int:
    """Train one stage; every rank runs this command."""
    from llmplatform.train.stages import run_stage

    ckpt_root, data_root = _roots()
    config = load_train_config(args.config, args.set)
    paths = resolve_stage_paths(config, load_models(), ckpt_root, data_root, args.output_dir)
    run_stage(config, paths, o_cpu=args.cpu, o_resume=not args.no_resume)
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    """Merge the trained adapter into its base model; run on a single process."""
    from llmplatform.train.merge import MERGED_MARKER, merge_adapter
    from llmplatform.train.stages import ADAPTER_DIRNAME

    ckpt_root, data_root = _roots()
    config = load_train_config(args.config, args.set)
    paths = resolve_stage_paths(config, load_models(), ckpt_root, data_root, args.output_dir)
    destination = paths.output_dir / "merged"
    if (destination / MERGED_MARKER).is_file():
        logger.info("merged checkpoint already exists", extra={"path": str(destination)})
        return 0
    merge_adapter(
        paths.model_path,
        paths.output_dir / ADAPTER_DIRNAME,
        destination,
        paths.model_revision,
        "float32" if args.cpu else "bfloat16",
    )
    return 0


def cmd_scaling_report(args: argparse.Namespace) -> int:
    """Aggregate the step logs of every finished GPU count into the scaling results file."""
    ckpt_root, _ = _roots()
    point_list = []
    for gpus in SCALING_GPU_COUNTS:
        path = ckpt_root / "scaling" / f"gpus_{gpus}" / STEP_METRICS
        if path.is_file():
            point_list.append(compute_point(read_step_metrics(path), gpus))
    root = repo_root()
    compute = load_compute()
    summary = scaling_summary(point_list)
    result = build_results(
        "e2_scaling",
        summary,
        config_ref(root / "configs" / "train" / "dpo.yaml", root),
        hardware_record(compute),
        package_versions(RECORDED_PACKAGES),
        notes="60 DPO optimizer steps per GPU count, first 10 discarded, same per-device batch",
    )
    write_results(results_dir(root) / SCALING_RESULT, result)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/train.py``."""
    configure_logging("train")
    parser = argparse.ArgumentParser(description="Alignment training commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="train one stage and merge the adapter")
    fit.add_argument("--config", required=True)
    fit.add_argument("--set", action="append", default=[], help="override section.key=value")
    fit.add_argument("--output-dir", default=None)
    fit.add_argument("--cpu", action="store_true")
    fit.add_argument("--no-resume", action="store_true")
    fit.set_defaults(func=cmd_fit)
    merge = sub.add_parser("merge", help="merge the adapter into the base model")
    merge.add_argument("--config", required=True)
    merge.add_argument("--set", action="append", default=[])
    merge.add_argument("--output-dir", default=None)
    merge.add_argument("--cpu", action="store_true")
    merge.set_defaults(func=cmd_merge)
    report = sub.add_parser("scaling-report", help="write results/e2_scaling.json")
    report.set_defaults(func=cmd_scaling_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
