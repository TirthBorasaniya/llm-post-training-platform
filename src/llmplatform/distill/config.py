"""Typed models of the distillation configs."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Models =============


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelNames(_Section):
    """Registry names of the teacher and student."""

    teacher: str
    student: str


class StudentOnly(_Section):
    """Registry name of the student."""

    student: str


class DataSection(_Section):
    """Prompt file and size."""

    file: str
    max_prompts: int


class OptimSection(_Section):
    """Optimizer and batch settings for on-policy distillation."""

    learning_rate: float
    alternative_learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool


class GenerationSection(_Section):
    """Rollout settings."""

    max_completion_length: int
    temperature: float
    beta: float


class VllmSection(_Section):
    """Colocated vLLM generation."""

    enabled: bool | Literal["auto"]
    mode: str
    gpu_memory_utilization: float


class StudentSection(_Section):
    """Full fine-tuning or the LoRA fallback."""

    mode: Literal["full", "lora"]
    dtype: Literal["bfloat16", "float32"] = "float32"
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    train_lm_head: bool

    @model_validator(mode="after")
    def _no_head_adapter(self) -> "StudentSection":
        if any("lm_head" in name for name in self.lora_target_modules):
            raise ValueError("lm_head adapters are not allowed; use train_lm_head instead")
        return self


class CheckpointingSection(_Section):
    """Time-based checkpoint cadence."""

    save_every_minutes: float
    save_total_limit: int
    logging_steps: int


class ClippedRatioSection(_Section):
    """Guard on truncated rollouts."""

    max: float
    window: int


class VocabCheckSection(_Section):
    """Vocabulary check settings."""

    prompts: int
    max_length: int
    atol_float32: float
    atol_half: float


class OutputSection(_Section):
    """Output directory relative to the checkpoint root."""

    subdir: str


class TrackingSection(_Section):
    """Experiment tracking."""

    enabled: bool
    experiment: str


class OpdConfig(ConfigModel):
    """Contents of ``configs/distill/opd.yaml``."""

    seed: int
    models: ModelNames
    data: DataSection
    optim: OptimSection
    generation: GenerationSection
    vllm: VllmSection
    student_training: StudentSection
    checkpointing: CheckpointingSection
    clipped_ratio: ClippedRatioSection
    vocab_check: VocabCheckSection
    output: OutputSection
    tracking: TrackingSection


class TeacherGenSection(_Section):
    """Teacher completion generation."""

    prompts_file: str
    output_file: str
    temperature: float
    max_tokens: int
    concurrency: int
    keep_truncated: bool


class SftOptimSection(_Section):
    """Optimizer settings of the SFT baseline."""

    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool


class LengthSection(_Section):
    """Sequence limits."""

    max_length: int


class SftBaselineConfig(ConfigModel):
    """Contents of ``configs/distill/sft_baseline.yaml``."""

    seed: int
    models: StudentOnly
    student_dtype: Literal["bfloat16", "float32"] = "float32"
    generation: TeacherGenSection
    optim: SftOptimSection
    lengths: LengthSection
    checkpointing: CheckpointingSection
    output: OutputSection
    tracking: TrackingSection


# ============= Loading =============


def load_opd(path: str | Path | None = None, override_list: list[str] | None = None) -> OpdConfig:
    """Load the distillation config."""
    return load_config(path or repo_root() / "configs/distill/opd.yaml", OpdConfig, override_list)


def load_sft_baseline(
    path: str | Path | None = None, override_list: list[str] | None = None
) -> SftBaselineConfig:
    """Load the SFT baseline config."""
    return load_config(
        path or repo_root() / "configs/distill/sft_baseline.yaml", SftBaselineConfig, override_list
    )
