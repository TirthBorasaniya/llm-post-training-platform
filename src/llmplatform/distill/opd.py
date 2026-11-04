"""On-policy distillation of the aligned teacher into the student with TRL's DistillationTrainer."""

import importlib.util
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llmplatform.common.checkpoints import is_complete, latest_checkpoint, mark_complete
from llmplatform.common.io import read_jsonl, write_json
from llmplatform.distill.accounting import (
    LEDGER_FILENAME,
    ClippedRatioGuard,
    GpuLedger,
    LedgerCallback,
    TimedSaveCallback,
)
from llmplatform.distill.config import OpdConfig
from llmplatform.distill.vocab_check import VocabReport, check_and_align

logger = logging.getLogger(__name__)

# ============= Constants =============

NEVER_SAVE_STEPS = 10**9
FINAL_DIRNAME = "final"
RUN_SUMMARY_FILENAME = "distill_run.json"


# ============= Arguments =============


def vllm_enabled(config: OpdConfig, o_cpu: bool) -> bool:
    """Decide whether rollouts use colocated vLLM: forced by config, else when importable."""
    if o_cpu or config.vllm.enabled is False:
        return False
    if config.vllm.enabled is True:
        return True
    return importlib.util.find_spec("vllm") is not None


def build_distillation_args(
    config: OpdConfig,
    output_dir: Path,
    learning_rate: float | None = None,
    o_cpu: bool = False,
    o_report: bool = True,
) -> dict[str, Any]:
    """
    Return the ``DistillationConfig`` arguments for a run.

    Parameters
    ----------
    config : OpdConfig
        Distillation settings.
    output_dir : Path
        Trainer output directory.
    learning_rate : float, optional
        Overrides the configured learning rate, for the alternative run.
    o_cpu : bool
        Float32 on the CPU without vLLM, for smoke tests.
    o_report : bool
        Report to MLflow when enabled in the config.
    """
    optim, generation = config.optim, config.generation
    dtype = "float32" if o_cpu else config.student_training.dtype
    return {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate if learning_rate is not None else optim.learning_rate,
        "lr_scheduler_type": optim.lr_scheduler_type,
        "warmup_steps": optim.warmup_ratio,
        "weight_decay": optim.weight_decay,
        "max_grad_norm": optim.max_grad_norm,
        "num_train_epochs": optim.num_train_epochs,
        "per_device_train_batch_size": optim.per_device_train_batch_size,
        "gradient_accumulation_steps": optim.gradient_accumulation_steps,
        "gradient_checkpointing": optim.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": not o_cpu,
        "use_cpu": o_cpu,
        "max_completion_length": generation.max_completion_length,
        "temperature": generation.temperature,
        "beta": generation.beta,
        "model_init_kwargs": {"dtype": dtype},
        "use_vllm": vllm_enabled(config, o_cpu),
        "vllm_mode": config.vllm.mode,
        "vllm_gpu_memory_utilization": config.vllm.gpu_memory_utilization,
        "save_strategy": "steps",
        "save_steps": NEVER_SAVE_STEPS,
        "save_total_limit": config.checkpointing.save_total_limit,
        "logging_steps": config.checkpointing.logging_steps,
        "seed": config.seed,
        "report_to": ["mlflow"] if o_report and config.tracking.enabled else [],
    }


def build_student_peft(config: OpdConfig) -> Any | None:
    """Return the LoRA config for the fallback mode, or ``None`` for full fine-tuning."""
    section = config.student_training
    if section.mode == "full":
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=section.lora_r,
        lora_alpha=section.lora_alpha,
        lora_dropout=section.lora_dropout,
        target_modules=list(section.lora_target_modules),
        modules_to_save=["lm_head"] if section.train_lm_head else None,
        task_type="CAUSAL_LM",
        bias="none",
    )


# ============= Data =============


def load_prompt_dataset(path: Path, max_prompts: int) -> Any:
    """Load the prompt file as a dataset with a conversational ``prompt`` column."""
    from datasets import Dataset

    rows = read_jsonl(path)[:max_prompts]
    return Dataset.from_list([{"prompt": row["prompt"]} for row in rows])


def vocab_prompts(path: Path, count: int, tokenizer: Any) -> list[str]:
    """Render the first ``count`` prompts with the chat template for the vocabulary check."""
    rows = read_jsonl(path)[:count]
    return [
        tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
        for row in rows
    ]


# ============= Run =============


def load_teacher(path: str, revision: str | None, o_cpu: bool) -> Any:
    """Load the merged teacher in bf16, or float32 on the CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch.float32 if o_cpu else torch.bfloat16}
    if revision and not os.path.isdir(path):
        kwargs["revision"] = revision
    model: Any = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if not o_cpu and torch.cuda.is_available():
        model.to("cuda")
    model.config.use_cache = True
    return model


def run_opd(
    config: OpdConfig,
    teacher_path: str,
    teacher_revision: str | None,
    student_path: str,
    student_revision: str | None,
    prompts_path: Path,
    output_dir: Path,
    learning_rate: float | None = None,
    o_cpu: bool = False,
    o_resume: bool = True,
) -> Path:
    """
    Distill the teacher into the student, resuming from the latest checkpoint.

    The teacher's embedding size is aligned to the student's first and the run refuses to
    start when the vocabulary check fails.

    Returns
    -------
    final_dir : Path
        Directory with the trained student.
    """
    from transformers import AutoConfig, AutoTokenizer
    from trl import DistillationConfig, DistillationTrainer

    final_dir = output_dir / FINAL_DIRNAME
    if is_complete(output_dir):
        logger.info("distillation already complete", extra={"output_dir": str(output_dir)})
        return final_dir
    if config.tracking.enabled and not o_cpu:
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config.tracking.experiment)

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, revision=teacher_revision)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path, revision=student_revision)
    teacher = load_teacher(teacher_path, teacher_revision, o_cpu)
    student_stub = SimpleNamespace(
        config=AutoConfig.from_pretrained(student_path, revision=student_revision)
    )
    teacher_tokenizer.padding_side = "left"
    report: VocabReport = check_and_align(
        teacher,
        student_stub,
        teacher_tokenizer,
        student_tokenizer,
        vocab_prompts(prompts_path, config.vocab_check.prompts, teacher_tokenizer),
        config.vocab_check.max_length,
        config.vocab_check.atol_float32,
        config.vocab_check.atol_half,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "vocab_check_run.json", report.as_dict())
    if not report.passed:
        raise RuntimeError(f"vocabulary check failed: {json.dumps(report.as_dict())}")

    args_dict = build_distillation_args(config, output_dir, learning_rate, o_cpu)
    if student_revision and not os.path.isdir(student_path):
        args_dict["model_init_kwargs"] = {
            **args_dict["model_init_kwargs"],
            "revision": student_revision,
        }
    ledger = GpuLedger(output_dir / LEDGER_FILENAME)
    trainer = DistillationTrainer(
        model=student_path,
        teacher_model=teacher,
        args=DistillationConfig(**args_dict),
        train_dataset=load_prompt_dataset(prompts_path, config.data.max_prompts),
        processing_class=student_tokenizer,
        peft_config=build_student_peft(config),
        callbacks=[
            LedgerCallback(ledger),
            TimedSaveCallback(config.checkpointing.save_every_minutes),
            ClippedRatioGuard(config.clipped_ratio.max, config.clipped_ratio.window),
        ],
    )
    resume = latest_checkpoint(output_dir) if o_resume else None
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(final_dir))
    student_tokenizer.save_pretrained(str(final_dir))
    write_json(
        output_dir / RUN_SUMMARY_FILENAME,
        {
            "learning_rate": args_dict["learning_rate"],
            "use_vllm": args_dict["use_vllm"],
            "student_mode": config.student_training.mode,
            "student_dtype": args_dict["model_init_kwargs"]["dtype"],
            "ledger": ledger.summary(),
            "vocab_check": report.as_dict(),
        },
    )
    mark_complete(output_dir)
    return final_dir
