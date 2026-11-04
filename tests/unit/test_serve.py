"""Tests for server command builders, readiness polling, and metrics parsing."""

import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from llmplatform.serve.engines import (
    DisaggregationSpec,
    ServerSpec,
    SpeculativeSpec,
    build_command,
    build_env,
)
from llmplatform.serve.launch import (
    ManagedServer,
    ServerStartError,
    build_parser,
    spec_from_args,
    substitute,
    wait_for_health,
)
from llmplatform.serve.metrics import metric_names, metric_total, parse_prometheus_text, snapshot

pytestmark = pytest.mark.unit


# ============= Commands =============


def test_sglang_default_command() -> None:
    spec = ServerSpec("sglang", "/m", served_model_name="m")
    command = build_command(spec, "py")
    assert command[:3] == ["py", "-m", "sglang.launch_server"]
    assert command[command.index("--port") + 1] == "30000"
    assert "--enable-metrics" in command
    assert "--disable-radix-cache" not in command
    assert command[command.index("--served-model-name") + 1] == "m"


def test_sglang_options() -> None:
    spec = ServerSpec(
        "sglang",
        "/m",
        quantization="fp8",
        prefix_cache=False,
        max_model_len=8192,
        mem_fraction=0.8,
        speculative=SpeculativeSpec("/draft", num_steps=4, num_draft_tokens=5),
        extra_args=["--dtype", "bfloat16"],
    )
    command = build_command(spec)
    assert command[command.index("--quantization") + 1] == "fp8"
    assert "--disable-radix-cache" in command
    assert command[command.index("--speculative-algorithm") + 1] == "STANDALONE"
    assert command[command.index("--speculative-draft-model-path") + 1] == "/draft"
    assert command[command.index("--speculative-num-steps") + 1] == "4"
    assert command[command.index("--speculative-eagle-topk") + 1] == "1"
    assert command[command.index("--speculative-num-draft-tokens") + 1] == "5"
    assert command[-2:] == ["--dtype", "bfloat16"]


def test_sglang_disaggregation_flags() -> None:
    spec = ServerSpec(
        "sglang", "/m", port=30001, disaggregation=DisaggregationSpec("prefill", "nixl", 8998)
    )
    command = build_command(spec)
    assert command[command.index("--disaggregation-mode") + 1] == "prefill"
    assert command[command.index("--disaggregation-transfer-backend") + 1] == "nixl"
    assert command[command.index("--disaggregation-bootstrap-port") + 1] == "8998"


def test_vllm_prefix_cache_flags() -> None:
    on = build_command(ServerSpec("vllm", "/m"))
    off = build_command(ServerSpec("vllm", "/m", prefix_cache=False, quantization="fp8"))
    assert on[:3] == ["vllm", "serve", "/m"] and "--enable-prefix-caching" in on
    assert "--disable-prefix-caching" in off and "--quantization" in off
    assert on[on.index("--port") + 1] == "8000"


def test_llamacpp_command_matches_chart_defaults() -> None:
    command = build_command(ServerSpec("llamacpp", "/m.gguf"))
    assert command[:3] == ["llama-server", "-m", "/m.gguf"]
    assert "--metrics" in command
    assert command[command.index("--parallel") + 1] == "2"
    assert command[command.index("-c") + 1] == "4096"


def test_env_and_urls() -> None:
    spec = ServerSpec("sglang", "/m", port=31000, gpu_ids=[0, 1])
    assert build_env(spec) == {"CUDA_VISIBLE_DEVICES": "0,1"}
    assert build_env(ServerSpec("vllm", "/m")) == {}
    assert spec.openai_url() == "http://127.0.0.1:31000/v1"
    assert spec.health_url().endswith("/health") and spec.metrics_url().endswith("/metrics")
    with pytest.raises(ValueError):
        build_command(ServerSpec("nope", "/m"))  # type: ignore[arg-type]


# ============= Readiness =============


def test_wait_for_health_polls_until_ok() -> None:
    answers = iter([503, 503, 200])
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(next(answers))))
    ticks = {"now": 0.0}

    def clock() -> float:
        return ticks["now"]

    def sleep(seconds: float) -> None:
        ticks["now"] += seconds

    wait_for_health("http://x/health", 100, None, client, sleep, clock, interval=1.0)
    assert ticks["now"] == 2.0


def test_wait_for_health_times_out() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    ticks = {"now": 0.0}

    def sleep(seconds: float) -> None:
        ticks["now"] += seconds

    with pytest.raises(ServerStartError, match="not healthy"):
        wait_for_health("http://x/health", 5, None, client, sleep, lambda: ticks["now"], 1.0)


def test_wait_for_health_detects_dead_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    process.wait()
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(ServerStartError, match="status 3"):
        wait_for_health("http://x/health", 5, process, client, lambda s: None)


def test_managed_server_start_failure_cleans_up(tmp_path: Path) -> None:
    spec = ServerSpec("sglang", "/m", port=1)
    server = ManagedServer(
        spec, tmp_path / "log.txt", ready_timeout=2, python="/definitely/missing"
    )
    with pytest.raises((ServerStartError, FileNotFoundError, OSError)):
        server.start()
    server.stop()


# ============= CLI helpers =============


def test_cli_spec_and_placeholders() -> None:
    args = build_parser().parse_args(
        [
            "--engine",
            "sglang",
            "--model",
            "/m",
            "--gpu-ids",
            "0,1",
            "--draft-model",
            "/d",
            "--no-prefix-cache",
            "--",
            "echo",
            "{openai_url}",
        ]
    )
    spec = spec_from_args(args)
    assert spec.gpu_ids == [0, 1] and spec.speculative is not None and not spec.prefix_cache
    command = args.command[1:]
    assert substitute(command, spec) == ["echo", "http://127.0.0.1:30000/v1"]


# ============= Metrics =============

SAMPLE = """# HELP sglang:num_queue_reqs queue
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="m",tp_rank="0"} 3.0
sglang:num_queue_reqs{model_name="m",tp_rank="1"} 1.0
sglang:cache_hit_rate{model_name="m"} 0.75
sglang:e2e_request_latency_seconds_bucket{le="+Inf"} 12
llamacpp:requests_deferred 2
"""


def test_parse_prometheus_text() -> None:
    parsed = parse_prometheus_text(SAMPLE)
    assert metric_total(parsed, "sglang:num_queue_reqs") == 4.0
    assert metric_total(parsed, "sglang:cache_hit_rate") == 0.75
    assert metric_total(parsed, "llamacpp:requests_deferred") == 2.0
    assert metric_total(parsed, "missing") is None
    assert parsed["sglang:num_queue_reqs"][0][0]["tp_rank"] == "0"
    assert parsed["sglang:e2e_request_latency_seconds_bucket"][0][1] == 12
    assert "sglang:cache_hit_rate" in metric_names(parsed, "sglang:")


def test_snapshot_reports_absent_metrics_as_none() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=SAMPLE)))
    values = snapshot(
        "http://x/metrics", ("sglang:num_queue_reqs", "sglang:spec_accept_length"), client
    )
    assert values == {"sglang:num_queue_reqs": 4.0, "sglang:spec_accept_length": None}
