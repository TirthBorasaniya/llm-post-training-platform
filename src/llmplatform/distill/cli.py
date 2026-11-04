"""Command-line entry point for the vocabulary check, distillation, and baseline commands."""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.io import read_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.distill.config import load_opd, load_sft_baseline

logger = logging.getLogger(__name__)

# ============= Constants =============

PACKAGES = ["torch", "transformers", "trl", "peft", "vllm", "accelerate"]
VOCAB_RESULT = "vocab_check.json"
TEACHER_COMPLETIONS = "teacher_completions.jsonl"


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


def _prompts_file(data_root: Path, name: str) -> Path:
    primary = data_root / name
    if primary.is_file():
        return primary
    fallback = repo_root() / "eval" / "judge_prompts.jsonl"
    if fallback.is_file():
        logger.warning("using judge prompts for the check", extra={"missing": str(primary)})
        return fallback
    raise FileNotFoundError(f"prompt file not found: {primary}")


# ============= Commands =============


def cmd_check_vocab(args: argparse.Namespace) -> int:
    """Run the mandatory vocabulary check and write ``results/vocab_check.json``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmplatform.distill.opd import load_teacher, vocab_prompts
    from llmplatform.distill.vocab_check import check_and_align

    ckpt_root, data_root = _roots()
    config, models = load_opd(), load_models()
    teacher_path = resolve_model(models, config.models.teacher, ckpt_root)
    student_path = resolve_model(models, config.models.student, ckpt_root)
    teacher_rev = revision_of(models, config.models.teacher)
    student_rev = revision_of(models, config.models.student)
    teacher_tok = AutoTokenizer.from_pretrained(teacher_path, revision=teacher_rev)
    student_tok = AutoTokenizer.from_pretrained(student_path, revision=student_rev)
    teacher_tok.padding_side = "left"
    teacher = load_teacher(teacher_path, teacher_rev, args.cpu)
    student: Any = AutoModelForCausalLM.from_pretrained(
        student_path, revision=student_rev, dtype=torch.float32 if args.cpu else torch.bfloat16
    )
    prompts = vocab_prompts(
        _prompts_file(data_root, config.data.file), config.vocab_check.prompts, teacher_tok
    )
    report = check_and_align(
        teacher,
        student,
        teacher_tok,
        student_tok,
        prompts,
        config.vocab_check.max_length,
        config.vocab_check.atol_float32,
        config.vocab_check.atol_half,
    )
    root = repo_root()
    result = build_results(
        "vocab_check",
        report.as_dict(),
        config_ref(root / "configs/distill/opd.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
    )
    write_results(results_dir(root) / VOCAB_RESULT, result)
    return 0 if report.passed else 1


def cmd_opd(args: argparse.Namespace) -> int:
    """Run on-policy distillation."""
    from llmplatform.distill.opd import run_opd

    ckpt_root, data_root = _roots()
    config, models = load_opd(override_list=args.set), load_models()
    subdir = config.output.subdir + ("_alt_lr" if args.alternative_lr else "")
    learning_rate = config.optim.alternative_learning_rate if args.alternative_lr else None
    run_opd(
        config,
        resolve_model(models, config.models.teacher, ckpt_root),
        revision_of(models, config.models.teacher),
        resolve_model(models, config.models.student, ckpt_root),
        revision_of(models, config.models.student),
        data_root / config.data.file,
        ckpt_root / subdir,
        learning_rate,
        o_cpu=args.cpu,
        o_resume=not args.no_resume,
    )
    return 0


def cmd_teacher_gen(args: argparse.Namespace) -> int:
    """Generate teacher completions through a running server."""
    from llmplatform.distill.teacher_gen import generate_completions

    _, data_root = _roots()
    config = load_sft_baseline(override_list=args.set)
    prompts = read_jsonl(data_root / config.generation.prompts_file)
    if args.limit:
        prompts = prompts[: args.limit]
    summary = generate_completions(
        prompts,
        args.base_url,
        args.served_model_name,
        config.generation,
        data_root / config.generation.output_file,
        args.gpus,
    )
    logger.info("teacher generation finished", extra=summary)
    return 0


def cmd_sft_baseline(args: argparse.Namespace) -> int:
    """Run the compute-matched SFT baseline."""
    from llmplatform.distill.sft_baseline import budget_from_ledger, run_sft_baseline

    ckpt_root, data_root = _roots()
    config, models = load_sft_baseline(override_list=args.set), load_models()
    if args.budget_gpu_hours is not None:
        budget = args.budget_gpu_hours
    else:
        budget = budget_from_ledger(ckpt_root / load_opd().output.subdir / "ledger.json")
    if budget <= 0:
        raise SystemExit(
            "the distillation ledger is empty; run distillation first or pass --budget-gpu-hours"
        )
    run_sft_baseline(
        config,
        resolve_model(models, config.models.student, ckpt_root),
        revision_of(models, config.models.student),
        data_root / config.generation.output_file,
        ckpt_root / config.output.subdir,
        budget,
        o_cpu=args.cpu,
        o_resume=not args.no_resume,
    )
    return 0


def cmd_e3(args: argparse.Namespace) -> int:
    """Write ``results/e3_distillation.json``."""
    from llmplatform.distill.e3 import write_e3

    ckpt_root, data_root = _roots()
    root = repo_root()
    write_e3(
        results_dir(root),
        ckpt_root,
        data_root,
        config_ref(root / "configs/distill/opd.yaml", root),
        hardware_record(load_compute()),
        package_versions(PACKAGES),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/distill.py``."""
    configure_logging("distill")
    parser = argparse.ArgumentParser(description="Distillation commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-vocab")
    check.add_argument("--cpu", action="store_true")
    check.set_defaults(func=cmd_check_vocab)
    opd = sub.add_parser("opd")
    opd.add_argument("--set", action="append", default=[])
    opd.add_argument("--alternative-lr", action="store_true")
    opd.add_argument("--cpu", action="store_true")
    opd.add_argument("--no-resume", action="store_true")
    opd.set_defaults(func=cmd_opd)
    gen = sub.add_parser("teacher-gen")
    gen.add_argument("--base-url", required=True)
    gen.add_argument("--served-model-name", required=True)
    gen.add_argument("--gpus", type=int, default=1)
    gen.add_argument("--limit", type=int, default=None)
    gen.add_argument("--set", action="append", default=[])
    gen.set_defaults(func=cmd_teacher_gen)
    sft = sub.add_parser("sft-baseline")
    sft.add_argument("--budget-gpu-hours", type=float, default=None)
    sft.add_argument("--set", action="append", default=[])
    sft.add_argument("--cpu", action="store_true")
    sft.add_argument("--no-resume", action="store_true")
    sft.set_defaults(func=cmd_sft_baseline)
    e3 = sub.add_parser("e3")
    e3.set_defaults(func=cmd_e3)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
