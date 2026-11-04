"""Run one alignment stage (SFT, DPO, ORPO, or SimPO) with checkpoint resume."""

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

from llmplatform.common.checkpoints import is_complete, latest_checkpoint, mark_complete
from llmplatform.common.hardware import detect_cuda_version, query_nvidia_smi
from llmplatform.common.io import read_jsonl
from llmplatform.train.callbacks import StepMetricsCallback
from llmplatform.train.config import PREFERENCE_STAGES, StagePaths, TrainConfig
from llmplatform.train.qlora import build_lora_config, load_base_model, load_tokenizer

logger = logging.getLogger(__name__)

# ============= Constants =============

ADAPTER_DIRNAME = "adapter"
COLUMNS_BY_STAGE = {
    "sft": ["messages"],
    "dpo": ["prompt", "chosen", "rejected"],
    "orpo": ["prompt", "chosen", "rejected"],
    "simpo": ["prompt", "chosen", "rejected"],
}


# ============= Arguments =============


def build_common_args(
    config: TrainConfig, output_dir: Path, o_cpu: bool = False, o_report: bool = True
) -> dict[str, Any]:
    """
    Return the Trainer arguments shared by every stage.

    Parameters
    ----------
    config : TrainConfig
        Stage configuration.
    output_dir : Path
        Trainer output directory.
    o_cpu : bool
        Disable bf16 and GPU use, for smoke tests on a CPU.
    o_report : bool
        Report to MLflow when the config enables tracking.
    """
    optim = config.optim
    checkpointing = config.checkpointing
    use_mlflow = o_report and config.tracking.enabled
    return {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": optim.per_device_train_batch_size,
        "gradient_accumulation_steps": optim.gradient_accumulation_steps,
        "learning_rate": optim.learning_rate,
        "lr_scheduler_type": optim.lr_scheduler_type,
        "warmup_steps": optim.warmup_ratio,
        "num_train_epochs": optim.num_train_epochs,
        "max_steps": optim.max_steps,
        "weight_decay": optim.weight_decay,
        "max_grad_norm": optim.max_grad_norm,
        "bf16": not o_cpu,
        "use_cpu": o_cpu,
        "gradient_checkpointing": optim.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "save_strategy": "steps",
        "save_steps": checkpointing.save_steps,
        "save_total_limit": checkpointing.save_total_limit,
        "logging_steps": checkpointing.logging_steps,
        "seed": config.seed,
        "report_to": ["mlflow"] if use_mlflow else [],
        "include_num_input_tokens_seen": "all",
    }


def build_stage_args(
    config: TrainConfig, output_dir: Path, o_cpu: bool = False, o_report: bool = True
) -> dict[str, Any]:
    """Return the full argument dict for the stage's TRL config class."""
    args = build_common_args(config, output_dir, o_cpu, o_report)
    args["max_length"] = config.lengths.max_length
    stage = config.stage
    if stage == "sft":
        args["packing"] = False
    elif stage == "dpo":
        args["beta"] = config.objective.beta
        args["loss_type"] = ["sigmoid"]
    elif stage == "orpo":
        args["beta"] = config.objective.beta
    elif stage == "simpo":
        args["beta"] = config.objective.beta
        args["loss_type"] = "simpo"
        args["cpo_alpha"] = 0.0
        args["simpo_gamma"] = config.objective.simpo_gamma
    return args


def tracking_tags(stage: str) -> dict[str, str]:
    """Return node, GPU, driver, and CUDA tags stored on the MLflow run."""
    gpu_list = query_nvidia_smi()
    return {
        "stage": stage,
        "node": socket.gethostname(),
        "gpu_model": gpu_list[0]["name"] if gpu_list else "none",
        "gpu_count": str(len(gpu_list)),
        "driver": gpu_list[0]["driver"] if gpu_list else "none",
        "cuda": detect_cuda_version() or "none",
    }


# ============= Data =============


def load_stage_dataset(config: TrainConfig, data_path: Path) -> Any:
    """Load the prepared JSON Lines file as a ``datasets.Dataset`` with the stage columns only."""
    from datasets import Dataset

    rows = read_jsonl(data_path)
    if config.data.max_rows is not None:
        rows = rows[: config.data.max_rows]
    columns = COLUMNS_BY_STAGE[config.stage]
    return Dataset.from_list([{name: row[name] for name in columns} for row in rows])


# ============= Run =============


def trainer_classes(stage: str) -> tuple[Any, Any]:
    """Return the ``(config_class, trainer_class)`` pair for a stage."""
    if stage == "sft":
        from trl import SFTConfig, SFTTrainer

        return SFTConfig, SFTTrainer
    if stage == "dpo":
        from trl import DPOConfig, DPOTrainer

        return DPOConfig, DPOTrainer
    if stage == "orpo":
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

        return ORPOConfig, ORPOTrainer
    from trl.experimental.cpo import CPOConfig, CPOTrainer

    return CPOConfig, CPOTrainer


def run_stage(
    config: TrainConfig, paths: StagePaths, o_cpu: bool = False, o_resume: bool = True
) -> Path:
    """
    Train one stage, resuming from the latest checkpoint, and save the LoRA adapter.

    Parameters
    ----------
    config : TrainConfig
        Stage configuration.
    paths : StagePaths
        Resolved model, dataset, and output locations.
    o_cpu : bool
        Run on the CPU in float32 without quantization.
    o_resume : bool
        Resume from the newest checkpoint in the output directory when one exists.

    Returns
    -------
    adapter_dir : Path
        Directory holding the trained adapter.
    """
    adapter_dir = paths.output_dir / ADAPTER_DIRNAME
    if is_complete(paths.output_dir):
        logger.info("stage already complete", extra={"output_dir": str(paths.output_dir)})
        return adapter_dir
    if config.tracking.enabled and not o_cpu:
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", config.tracking.experiment)
        os.environ.setdefault("MLFLOW_TAGS", json.dumps(tracking_tags(config.stage)))

    tokenizer = load_tokenizer(paths.model_path, paths.model_revision)
    model = load_base_model(config, paths.model_path, paths.model_revision, o_cpu)
    dataset = load_stage_dataset(config, paths.data_path)
    config_class, trainer_class = trainer_classes(config.stage)
    args = config_class(**build_stage_args(config, paths.output_dir, o_cpu))
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(config),
        callbacks=[StepMetricsCallback(paths.output_dir)],
    )
    resume = latest_checkpoint(paths.output_dir) if o_resume else None
    if resume is not None:
        logger.info("resuming", extra={"checkpoint": str(resume)})
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    mark_complete(paths.output_dir)
    return adapter_dir


def is_preference_stage(stage: str) -> bool:
    """Return whether the stage trains on preference pairs."""
    return stage in PREFERENCE_STAGES
