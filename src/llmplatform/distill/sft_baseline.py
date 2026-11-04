"""Student SFT on teacher completions, stopped at the GPU-hours the distillation run used."""

import logging
import os
from pathlib import Path
from typing import Any

from llmplatform.common.checkpoints import is_complete, latest_checkpoint, mark_complete
from llmplatform.common.io import read_jsonl, write_json
from llmplatform.distill.accounting import (
    LEDGER_FILENAME,
    BudgetCallback,
    GpuLedger,
    LedgerCallback,
    TimedSaveCallback,
)
from llmplatform.distill.config import SftBaselineConfig

logger = logging.getLogger(__name__)

# ============= Constants =============

NEVER_SAVE_STEPS = 10**9
FINAL_DIRNAME = "final"
RUN_SUMMARY_FILENAME = "sft_baseline_run.json"


# ============= Arguments =============


def build_sft_args(
    config: SftBaselineConfig, output_dir: Path, o_cpu: bool = False, o_report: bool = True
) -> dict[str, Any]:
    """Return the ``SFTConfig`` arguments of the baseline run."""
    optim = config.optim
    dtype = "float32" if o_cpu else "bfloat16"
    return {
        "output_dir": str(output_dir),
        "learning_rate": optim.learning_rate,
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
        "max_length": config.lengths.max_length,
        "packing": False,
        "model_init_kwargs": {"dtype": dtype},
        "save_strategy": "steps",
        "save_steps": NEVER_SAVE_STEPS,
        "save_total_limit": config.checkpointing.save_total_limit,
        "logging_steps": config.checkpointing.logging_steps,
        "seed": config.seed,
        "report_to": ["mlflow"] if o_report and config.tracking.enabled else [],
    }


def budget_from_ledger(path: str | Path) -> float:
    """Read the distillation GPU-hours from a ledger file."""
    return GpuLedger(path).total_gpu_hours()


# ============= Run =============


def run_sft_baseline(
    config: SftBaselineConfig,
    student_path: str,
    student_revision: str | None,
    data_path: Path,
    output_dir: Path,
    budget_gpu_hours: float,
    o_cpu: bool = False,
    o_resume: bool = True,
) -> Path:
    """
    Fine-tune the student on teacher completions until the GPU-hour budget is spent.

    Returns
    -------
    final_dir : Path
        Directory with the trained student.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    final_dir = output_dir / FINAL_DIRNAME
    if is_complete(output_dir):
        return final_dir
    if config.tracking.enabled and not o_cpu:
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config.tracking.experiment)
    tokenizer = AutoTokenizer.from_pretrained(student_path, revision=student_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args_dict = build_sft_args(config, output_dir, o_cpu)
    if student_revision and not os.path.isdir(student_path):
        args_dict["model_init_kwargs"] = {
            **args_dict["model_init_kwargs"],
            "revision": student_revision,
        }
    dataset = Dataset.from_list([{"messages": row["messages"]} for row in read_jsonl(data_path)])
    ledger = GpuLedger(output_dir / LEDGER_FILENAME)
    budget = BudgetCallback(ledger, budget_gpu_hours)
    trainer = SFTTrainer(
        model=student_path,
        args=SFTConfig(**args_dict),
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            LedgerCallback(ledger),
            TimedSaveCallback(config.checkpointing.save_every_minutes),
            budget,
        ],
    )
    resume = latest_checkpoint(output_dir) if o_resume else None
    # a resumed run past its budget stops after one step, restoring the saved weights first
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_json(
        output_dir / RUN_SUMMARY_FILENAME,
        {
            "budget_gpu_hours": budget_gpu_hours,
            "ledger": ledger.summary(),
            "budget_reached": budget.reached or ledger.total_gpu_hours() >= budget_gpu_hours,
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "examples": len(dataset),
        },
    )
    mark_complete(output_dir)
    return final_dir
