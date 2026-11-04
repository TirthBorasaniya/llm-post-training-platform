"""Typed model of ``configs/serve/disagg.yaml``."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.bench.config import Workload
from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Models =============


class Ports(BaseModel):
    """Ports of the two servers, the router, and the prefill bootstrap channel."""

    model_config = ConfigDict(extra="forbid")

    prefill: int
    decode: int
    router: int
    bootstrap: int


class RouterSection(BaseModel):
    """Router settings."""

    model_config = ConfigDict(extra="forbid")

    policy: str


class MixedWorkload(BaseModel):
    """Mixed long-prompt and chat traffic."""

    model_config = ConfigDict(extra="forbid")

    long: Workload
    chat: Workload
    long_fraction: float
    requests_per_rate: int
    rates: list[float]
    seed: int
    words_per_token: float


class ServerSection(BaseModel):
    """Server settings."""

    model_config = ConfigDict(extra="forbid")

    mem_fraction: float
    max_model_len: int
    ready_timeout_seconds: float


class DisaggConfig(ConfigModel):
    """Contents of ``configs/serve/disagg.yaml``."""

    model: str
    transfer_backend: str
    ports: Ports
    router: RouterSection
    workload: MixedWorkload
    server: ServerSection


def load_disagg(path: str | Path | None = None) -> DisaggConfig:
    """Load the disaggregation config."""
    return load_config(path or repo_root() / "configs/serve/disagg.yaml", DisaggConfig)
