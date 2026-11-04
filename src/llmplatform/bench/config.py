"""Typed models of the serving benchmark, agent workload, and cost configs."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Bench grid =============


class Workload(BaseModel):
    """Prompt and completion lengths in tokens."""

    model_config = ConfigDict(extra="forbid")

    input_len: int
    output_len: int

    def label(self) -> str:
        """Return ``<input>/<output>``."""
        return f"{self.input_len}/{self.output_len}"


class GridSection(BaseModel):
    """Workloads, concurrency levels, and repetitions."""

    model_config = ConfigDict(extra="forbid")

    workloads: list[Workload]
    concurrency: list[int]
    num_prompts: int
    repetitions: int
    seed: int


class HfSection(BaseModel):
    """Hugging Face static batching baseline."""

    model_config = ConfigDict(extra="forbid")

    batch_sizes: list[int]
    num_prompts: int
    repetitions: int


class ServerSection(BaseModel):
    """Server memory and length settings shared by the engines."""

    model_config = ConfigDict(extra="forbid")

    max_model_len: int
    sglang_mem_fraction: float
    vllm_gpu_memory_utilization: float


class BenchModel(BaseModel):
    """One measured model variant."""

    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    quantized: bool


class MetricsSection(BaseModel):
    """Reported percentiles."""

    model_config = ConfigDict(extra="forbid")

    percentiles: list[int]


class BenchConfig(ConfigModel):
    """Contents of ``configs/serve/bench.yaml``."""

    engines: list[str]
    grid: GridSection
    hf: HfSection
    server: ServerSection
    models: list[BenchModel]
    metrics: MetricsSection


# ============= Cost =============


class Slo(BaseModel):
    """Latency service level objective."""

    model_config = ConfigDict(extra="forbid")

    ttft_p95_ms: float
    itl_p95_ms: float


class Primary(BaseModel):
    """The configuration behind the headline cost ratio."""

    model_config = ConfigDict(extra="forbid")

    engine: str
    workload: Workload
    precision: str


class CostConfig(ConfigModel):
    """Contents of ``configs/cost.yaml``."""

    slo: Slo
    gpus_per_replica: int
    primary: Primary


# ============= Agent workload =============


class AgentEngine(BaseModel):
    """One engine configuration of the agent workload."""

    model_config = ConfigDict(extra="forbid")

    name: str
    engine: str
    prefix_cache: bool


class AgentConfig(ConfigModel):
    """Contents of ``configs/serve/agent.yaml``."""

    source_dir: str
    output: str
    seed: int
    turns_per_conversation: int
    conversations: int
    user_words: list[int]
    tool_result_words: list[int]
    assistant_words: list[int]
    max_tokens: int
    concurrency: list[int]
    engines: list[AgentEngine]
    models: list[str]


# ============= Loading =============


def load_bench(path: str | Path | None = None) -> BenchConfig:
    """Load the benchmark grid config."""
    return load_config(path or repo_root() / "configs/serve/bench.yaml", BenchConfig)


def load_cost(path: str | Path | None = None) -> CostConfig:
    """Load the cost config."""
    return load_config(path or repo_root() / "configs/cost.yaml", CostConfig)


def load_agent(path: str | Path | None = None) -> AgentConfig:
    """Load the agent workload config."""
    return load_config(path or repo_root() / "configs/serve/agent.yaml", AgentConfig)
