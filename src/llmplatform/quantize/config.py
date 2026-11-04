"""Typed model of ``configs/quantize.yaml``."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Models =============


class AwqSection(BaseModel):
    """AWQ calibration settings."""

    model_config = ConfigDict(extra="forbid")

    scheme: str
    calibration_samples: int
    max_seq_length: int
    calibration_file: str
    duo_scaling: str
    ignore: list[str]


class GpuSection(BaseModel):
    """GPU quantization policy."""

    model_config = ConfigDict(extra="forbid")

    fp8_min_capability: float
    awq: AwqSection
    models: list[str]


class VerifySection(BaseModel):
    """Prompts and limits for checking a GGUF with ``llama-cli``."""

    model_config = ConfigDict(extra="forbid")

    prompts: list[str]
    n_predict: int
    timeout_seconds: int


class GgufSection(BaseModel):
    """GGUF conversion settings."""

    model_config = ConfigDict(extra="forbid")

    llama_cpp_tag: str
    quant_type: str
    outtype: str
    verify: VerifySection
    models: list[str]


class QuantizeConfig(ConfigModel):
    """Contents of ``configs/quantize.yaml``."""

    gpu: GpuSection
    gguf: GgufSection
    output_subdir: str


def load_quantize(path: str | Path | None = None) -> QuantizeConfig:
    """Load the quantization config."""
    return load_config(path or repo_root() / "configs/quantize.yaml", QuantizeConfig)
