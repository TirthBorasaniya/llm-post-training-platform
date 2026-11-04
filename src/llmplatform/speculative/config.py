"""Typed model of ``configs/serve/speculative.yaml``."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Models =============


class TrafficSection(BaseModel):
    """Chat traffic used for throughput and acceptance measurements."""

    model_config = ConfigDict(extra="forbid")

    prompts_file: str
    requests_per_cell: int
    max_tokens: int
    temperature: float
    repetitions: int


class EqualitySection(BaseModel):
    """Greedy output comparison."""

    model_config = ConfigDict(extra="forbid")

    prompts: int
    max_tokens: int


class ServerSection(BaseModel):
    """Server settings."""

    model_config = ConfigDict(extra="forbid")

    mem_fraction: float
    max_model_len: int


class SpeculativeConfig(ConfigModel):
    """Contents of ``configs/serve/speculative.yaml``."""

    target: str
    drafts: list[str]
    draft_tokens: list[int]
    topk: int
    concurrency: list[int]
    precisions: list[str]
    traffic: TrafficSection
    equality: EqualitySection
    server: ServerSection
    output_subdir: str

    def steps_for(self, draft_tokens: int) -> int:
        """Return the draft chain length for a verified token count (top-k 1)."""
        return draft_tokens - 1


def load_speculative(path: str | Path | None = None) -> SpeculativeConfig:
    """Load the speculative decoding config."""
    return load_config(path or repo_root() / "configs/serve/speculative.yaml", SpeculativeConfig)
