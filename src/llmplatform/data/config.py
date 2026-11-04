"""Typed model of ``configs/data.yaml``."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Constants =============

DATA_CONFIG_RELPATH = "configs/data.yaml"


# ============= Models =============


class DatasetSpec(BaseModel):
    """A pinned Hub dataset split used for training or prompts."""

    model_config = ConfigDict(extra="forbid")

    hub_id: str
    revision: str
    split: str
    license: str
    max_rows: int
    max_tokens: int
    use: str


class JudgeSpec(BaseModel):
    """Source and size of the judge prompt set."""

    model_config = ConfigDict(extra="forbid")

    hub_id: str
    revision: str
    split: str
    license: str
    n_prompts: int
    max_prompt_tokens: int
    output: str
    use: str


class NearDuplicateSettings(BaseModel):
    """MinHash parameters."""

    model_config = ConfigDict(extra="forbid")

    num_perm: int
    rows_per_band: int
    shingle_words: int
    threshold: float


class BenchmarkSource(BaseModel):
    """A benchmark question column whose text must not appear in training data."""

    model_config = ConfigDict(extra="forbid")

    name: str
    hub_id: str
    config: str | None
    split: str
    field: str


class Decontamination(BaseModel):
    """Decontamination settings."""

    model_config = ConfigDict(extra="forbid")

    near_duplicate: NearDuplicateSettings
    benchmark_sources: list[BenchmarkSource]


class TrainingSets(BaseModel):
    """Datasets for each training use."""

    model_config = ConfigDict(extra="forbid")

    sft: DatasetSpec
    preferences: DatasetSpec
    distillation_prompts: DatasetSpec


class DataConfig(ConfigModel):
    """Contents of ``configs/data.yaml``."""

    seed: int
    training: TrainingSets
    judge: JudgeSpec
    decontamination: Decontamination


# ============= Loading =============


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """Load the data configuration."""
    target = Path(path) if path else repo_root() / DATA_CONFIG_RELPATH
    return load_config(target, DataConfig)
