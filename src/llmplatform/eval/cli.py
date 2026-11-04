"""Command-line entry point for benchmarks, response generation, judging, and the E1 summary."""

import argparse
import logging
import os
from pathlib import Path

from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.io import read_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.settings import get_settings
from llmplatform.eval.config import load_benchmarks, load_e1, load_judge_config
from llmplatform.eval.e1 import update_checkpoint_file, write_summary
from llmplatform.eval.generate import generate_hf, generate_openai, save_responses
from llmplatform.eval.lm_eval_runner import choose_backend, run_task
from llmplatform.judge.groq_client import JudgeClient
from llmplatform.judge.run import run_judgment

logger = logging.getLogger(__name__)

# ============= Constants =============

PACKAGES = ["lm_eval", "vllm", "torch", "transformers", "httpx"]


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


def _responses_path(data_root: Path, name: str) -> Path:
    return data_root / "responses" / f"{name}.jsonl"


# ============= Commands =============


def cmd_benchmarks(args: argparse.Namespace) -> int:
    """Run every benchmark task on a checkpoint and write its benchmarks section."""
    ckpt_root, data_root = _roots()
    config = load_benchmarks()
    models = load_models()
    key = next(entry.model for entry in load_e1().checkpoints if entry.name == args.checkpoint)
    path = resolve_model(models, key, ckpt_root)
    backend = args.backend or choose_backend(config)
    work_dir = data_root / "lm_eval" / args.checkpoint
    score_dict = {}
    for task in config.tasks:
        score_dict[task.name] = {
            "metric": task.metric,
            "num_fewshot": task.num_fewshot,
            "value": run_task(
                task, config, path, backend, work_dir, revision_of(models, key), args.limit
            ),
        }
    root = repo_root()
    values = {
        "backend": backend,
        "chat_template": config.apply_chat_template,
        "limit": args.limit,
        "tasks": score_dict,
    }
    update_checkpoint_file(
        results_dir(root),
        args.checkpoint,
        "benchmarks",
        values,
        config_ref(root / "configs/eval/benchmarks.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
        notes=f"lm-eval backend {backend}",
    )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate responses to the judge prompts for a checkpoint."""
    ckpt_root, data_root = _roots()
    judge = load_judge_config()
    models = load_models()
    key = next(entry.model for entry in load_e1().checkpoints if entry.name == args.checkpoint)
    prompts = read_jsonl(repo_root() / judge.generation.prompts_file)
    if args.limit:
        prompts = prompts[: args.limit]
    if args.base_url:
        rows = generate_openai(
            args.base_url, args.served_model_name or args.checkpoint, prompts, judge.generation
        )
    else:
        rows = generate_hf(
            resolve_model(models, key, ckpt_root),
            prompts,
            judge.generation,
            revision_of(models, key),
            args.cpu,
        )
    save_responses(_responses_path(data_root, args.checkpoint), rows)
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Judge a candidate against the reference and write the judge section."""
    _, data_root = _roots()
    config = load_judge_config()
    settings = get_settings()
    reference = args.reference or config.reference
    client = JudgeClient(config, settings.require_groq_key(), data_root / config.cache_dir)
    root = repo_root()
    metrics = run_judgment(
        client,
        config,
        root / config.generation.prompts_file,
        _responses_path(data_root, args.candidate),
        _responses_path(data_root, reference),
        args.limit,
    )
    metrics["reference"] = reference
    update_checkpoint_file(
        results_dir(root),
        args.candidate,
        "judge",
        metrics,
        config_ref(root / "configs/eval/judge.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
        notes=f"judge {config.judge_model}, reference {reference}",
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    """Write results/e1_summary.json from the checkpoint files."""
    root = repo_root()
    write_summary(
        results_dir(root),
        load_e1(),
        config_ref(root / "configs/eval/e1.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/eval.py``."""
    configure_logging("eval")
    parser = argparse.ArgumentParser(description="Evaluation commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmarks")
    bench.add_argument("--checkpoint", required=True)
    bench.add_argument("--backend", choices=["vllm", "hf"], default=None)
    bench.add_argument("--limit", type=int, default=None)
    bench.set_defaults(func=cmd_benchmarks)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", required=True)
    gen.add_argument("--base-url", default=None)
    gen.add_argument("--served-model-name", default=None)
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--cpu", action="store_true")
    gen.set_defaults(func=cmd_generate)
    judge = sub.add_parser("judge")
    judge.add_argument("--candidate", required=True)
    judge.add_argument("--reference", default=None)
    judge.add_argument("--limit", type=int, default=None)
    judge.set_defaults(func=cmd_judge)
    summary = sub.add_parser("summary")
    summary.set_defaults(func=cmd_summary)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
