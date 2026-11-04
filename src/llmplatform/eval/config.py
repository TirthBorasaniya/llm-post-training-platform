"""Typed models of the evaluation configs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Models =============


class TaskSpec(BaseModel):
    """One benchmark task."""

    model_config = ConfigDict(extra="forbid")

    name: str
    num_fewshot: int | None
    metric: str


class BenchmarksConfig(ConfigModel):
    """Contents of ``configs/eval/benchmarks.yaml``."""

    apply_chat_template: bool
    batch_size: str
    dtype: str
    vllm_gpu_memory_utilization: float
    backend_preference: list[str]
    tasks: list[TaskSpec]


class BootstrapSpec(BaseModel):
    """Bootstrap interval settings."""

    model_config = ConfigDict(extra="forbid")

    resamples: int
    seed: int
    confidence: float


class GenerationSpec(BaseModel):
    """Candidate response generation settings."""

    model_config = ConfigDict(extra="forbid")

    temperature: float
    max_new_tokens: int
    batch_size: int
    prompts_file: str


class PricingSpec(BaseModel):
    """Judge pricing from the provider's models page."""

    model_config = ConfigDict(extra="forbid")

    input_per_million_usd: float | None
    output_per_million_usd: float | None
    source: str | None
    retrieved: str | None


class JudgeConfig(ConfigModel):
    """Contents of ``configs/eval/judge.yaml``."""

    judge_model: str
    base_url: str
    temperature: float
    max_completion_tokens: int
    extra_body: dict[str, object]
    concurrency: int
    max_retries: int
    request_timeout_seconds: float
    template_version: int
    cache_dir: str
    reference: str
    bootstrap: BootstrapSpec
    generation: GenerationSpec
    pricing: PricingSpec


class CheckpointEntry(BaseModel):
    """One checkpoint in the quality experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str


class E1Config(ConfigModel):
    """Contents of ``configs/eval/e1.yaml``."""

    checkpoints: list[CheckpointEntry]


# ============= Loading =============


def load_benchmarks(path: str | Path | None = None) -> BenchmarksConfig:
    """Load the benchmark config."""
    return load_config(path or repo_root() / "configs/eval/benchmarks.yaml", BenchmarksConfig)


def load_judge_config(path: str | Path | None = None) -> JudgeConfig:
    """Load the judge config."""
    return load_config(path or repo_root() / "configs/eval/judge.yaml", JudgeConfig)


def load_e1(path: str | Path | None = None) -> E1Config:
    """Load the quality experiment checkpoint list."""
    return load_config(path or repo_root() / "configs/eval/e1.yaml", E1Config)
