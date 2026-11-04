"""Typed training configuration shared by SFT, DPO, ORPO, and SimPO."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.models import ModelsConfig, resolve_model, revision_of

# ============= Constants =============

STAGES = ("sft", "dpo", "orpo", "simpo")
PREFERENCE_STAGES = ("dpo", "orpo", "simpo")


# ============= Models =============


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSection(_Section):
    """Registry name of the model the stage starts from."""

    init: str


class DataSection(_Section):
    """Prepared dataset file name under the data root."""

    file: str
    max_rows: int | None = None


class LoraSection(_Section):
    """LoRA adapter settings."""

    r: int
    alpha: int
    dropout: float
    target_modules: str | list[str]


class QuantizationSection(_Section):
    """4-bit base model settings."""

    enabled: bool
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


class OptimSection(_Section):
    """Optimizer, schedule, and batch settings."""

    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    num_train_epochs: float
    max_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    max_grad_norm: float
    gradient_checkpointing: bool


class LengthSection(_Section):
    """Sequence length limits in tokens."""

    max_length: int


class ObjectiveSection(_Section):
    """Objective-specific values; unused ones are ignored by a stage."""

    beta: float
    simpo_gamma: float


class CheckpointSection(_Section):
    """Checkpoint and logging cadence."""

    save_steps: int
    save_total_limit: int
    logging_steps: int


class TrackingSection(_Section):
    """Experiment tracking."""

    enabled: bool
    experiment: str


class OutputSection(_Section):
    """Output directory relative to the checkpoint root."""

    subdir: str


class TrainConfig(ConfigModel):
    """Contents of ``configs/train/<stage>.yaml``."""

    stage: Literal["sft", "dpo", "orpo", "simpo"]
    seed: int
    model: ModelSection
    data: DataSection
    lora: LoraSection
    quantization: QuantizationSection
    optim: OptimSection
    lengths: LengthSection
    objective: ObjectiveSection
    checkpointing: CheckpointSection
    tracking: TrackingSection
    output: OutputSection

    @model_validator(mode="after")
    def _check(self) -> "TrainConfig":
        if self.optim.max_steps == 0 or self.optim.max_steps < -1:
            raise ValueError("max_steps must be -1 or positive")
        if not 0.0 <= self.optim.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")
        if self.checkpointing.save_steps < 1:
            raise ValueError("save_steps must be positive")
        return self


@dataclass(frozen=True)
class StagePaths:
    """Resolved locations for one stage run."""

    model_path: str
    model_revision: str | None
    data_path: Path
    output_dir: Path


# ============= Loading and resolution =============


def load_train_config(path: str | Path, override_list: list[str] | None = None) -> TrainConfig:
    """Load a training config and apply dotted overrides."""
    return load_config(path, TrainConfig, override_list)


def resolve_stage_paths(
    config: TrainConfig,
    models: ModelsConfig,
    ckpt_root: str | Path,
    data_root: str | Path,
    output_dir: str | Path | None = None,
) -> StagePaths:
    """Resolve the starting model, dataset file, and output directory of a stage."""
    return StagePaths(
        model_path=resolve_model(models, config.model.init, ckpt_root),
        model_revision=revision_of(models, config.model.init),
        data_path=Path(data_root) / config.data.file,
        output_dir=Path(output_dir) if output_dir else Path(ckpt_root) / config.output.subdir,
    )
