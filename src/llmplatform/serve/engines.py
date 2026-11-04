"""Command lines for SGLang, vLLM, and llama.cpp servers."""

import sys
from dataclasses import dataclass, field
from typing import Literal

# ============= Constants =============

DEFAULT_PORTS = {"sglang": 30000, "vllm": 8000, "llamacpp": 8080}
LLAMACPP_BINARY = "llama-server"
DEFAULT_HOST = "127.0.0.1"
Engine = Literal["sglang", "vllm", "llamacpp"]


# ============= Specs =============


@dataclass
class SpeculativeSpec:
    """SGLang draft-model speculative decoding settings."""

    draft_model_path: str
    algorithm: str = "STANDALONE"
    num_steps: int = 2
    eagle_topk: int = 1
    num_draft_tokens: int = 3


@dataclass
class DisaggregationSpec:
    """SGLang prefill or decode server settings."""

    mode: Literal["prefill", "decode"]
    transfer_backend: str = "nixl"
    bootstrap_port: int | None = None


@dataclass
class ServerSpec:
    """Everything needed to start one inference server."""

    engine: Engine
    model_path: str
    port: int | None = None
    host: str = DEFAULT_HOST
    served_model_name: str | None = None
    quantization: str | None = None
    prefix_cache: bool = True
    max_model_len: int | None = None
    mem_fraction: float | None = None
    gpu_ids: list[int] | None = None
    extra_args: list[str] = field(default_factory=list)
    speculative: SpeculativeSpec | None = None
    disaggregation: DisaggregationSpec | None = None
    parallel: int = 2
    context_size: int = 4096
    binary: list[str] | None = None
    env_extra: dict[str, str] = field(default_factory=dict)

    def resolved_port(self) -> int:
        """Return the explicit port or the engine default."""
        return self.port if self.port is not None else DEFAULT_PORTS[self.engine]

    def base_url(self) -> str:
        """Return the server root URL."""
        return f"http://{self.host}:{self.resolved_port()}"

    def openai_url(self) -> str:
        """Return the OpenAI-compatible API root, ending in ``/v1``."""
        return f"{self.base_url()}/v1"

    def health_url(self) -> str:
        """Return the readiness endpoint; all three engines serve ``/health``."""
        return f"{self.base_url()}/health"

    def metrics_url(self) -> str:
        """Return the Prometheus metrics endpoint."""
        return f"{self.base_url()}/metrics"


# ============= Builders =============


def sglang_command(spec: ServerSpec, python: str | None = None) -> list[str]:
    """Return ``python -m sglang.launch_server`` arguments for a spec."""
    command = [
        python or sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model_path,
        "--host",
        spec.host,
        "--port",
        str(spec.resolved_port()),
        "--enable-metrics",
    ]
    if spec.served_model_name:
        command += ["--served-model-name", spec.served_model_name]
    if spec.quantization:
        command += ["--quantization", spec.quantization]
    if not spec.prefix_cache:
        command.append("--disable-radix-cache")
    if spec.max_model_len:
        command += ["--context-length", str(spec.max_model_len)]
    if spec.mem_fraction:
        command += ["--mem-fraction-static", str(spec.mem_fraction)]
    if spec.speculative:
        spec_args = spec.speculative
        command += [
            "--speculative-algorithm",
            spec_args.algorithm,
            "--speculative-draft-model-path",
            spec_args.draft_model_path,
            "--speculative-num-steps",
            str(spec_args.num_steps),
            "--speculative-eagle-topk",
            str(spec_args.eagle_topk),
            "--speculative-num-draft-tokens",
            str(spec_args.num_draft_tokens),
        ]
    if spec.disaggregation:
        command += ["--disaggregation-mode", spec.disaggregation.mode]
        command += ["--disaggregation-transfer-backend", spec.disaggregation.transfer_backend]
        if spec.disaggregation.bootstrap_port is not None:
            command += ["--disaggregation-bootstrap-port", str(spec.disaggregation.bootstrap_port)]
    return command + list(spec.extra_args)


def vllm_command(spec: ServerSpec) -> list[str]:
    """Return ``vllm serve`` arguments for a spec."""
    command = [
        "vllm",
        "serve",
        spec.model_path,
        "--host",
        spec.host,
        "--port",
        str(spec.resolved_port()),
        "--enable-prefix-caching" if spec.prefix_cache else "--disable-prefix-caching",
    ]
    if spec.served_model_name:
        command += ["--served-model-name", spec.served_model_name]
    if spec.quantization:
        command += ["--quantization", spec.quantization]
    if spec.max_model_len:
        command += ["--max-model-len", str(spec.max_model_len)]
    if spec.mem_fraction:
        command += ["--gpu-memory-utilization", str(spec.mem_fraction)]
    return command + list(spec.extra_args)


def llamacpp_command(spec: ServerSpec) -> list[str]:
    """Return ``llama-server`` arguments for a GGUF model."""
    command = [
        *(spec.binary or [LLAMACPP_BINARY]),
        "-m",
        spec.model_path,
        "--host",
        spec.host,
        "--port",
        str(spec.resolved_port()),
        "--metrics",
        "--parallel",
        str(spec.parallel),
        "-c",
        str(spec.context_size),
    ]
    return command + list(spec.extra_args)


def build_command(spec: ServerSpec, python: str | None = None) -> list[str]:
    """Return the server command for any engine."""
    if spec.engine == "sglang":
        return sglang_command(spec, python)
    if spec.engine == "vllm":
        return vllm_command(spec)
    if spec.engine == "llamacpp":
        return llamacpp_command(spec)
    raise ValueError(f"unknown engine {spec.engine!r}")


def build_env(spec: ServerSpec) -> dict[str, str]:
    """Return environment overrides: GPU pinning for the server process."""
    env = dict(spec.env_extra)
    if spec.gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in spec.gpu_ids)
    return env
