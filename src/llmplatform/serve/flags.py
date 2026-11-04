"""Flags the platform passes to each engine, checked against the engine's own help text."""

import subprocess
import sys

# ============= Constants =============

REQUIRED_FLAGS: dict[str, list[str]] = {
    "sglang.launch_server": [
        "--model-path",
        "--host",
        "--port",
        "--enable-metrics",
        "--served-model-name",
        "--quantization",
        "--disable-radix-cache",
        "--context-length",
        "--mem-fraction-static",
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-steps",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
        "--disaggregation-mode",
        "--disaggregation-transfer-backend",
        "--disaggregation-bootstrap-port",
    ],
    "sglang.benchmark.serving": [
        "--backend",
        "--host",
        "--port",
        "--model",
        "--dataset-name",
        "--random-input-len",
        "--random-output-len",
        "--random-range-ratio",
        "--num-prompts",
        "--max-concurrency",
        "--seed",
        "--flush-cache",
        "--output-file",
    ],
    "vllm.serve": [
        "--host",
        "--port",
        "--served-model-name",
        "--quantization",
        "--max-model-len",
        "--gpu-memory-utilization",
        "--enable-prefix-caching",
        "--disable-prefix-caching",
    ],
    "vllm.bench.serve": [
        "--backend",
        "--endpoint",
        "--host",
        "--port",
        "--model",
        "--dataset-name",
        "--random-input-len",
        "--random-output-len",
        "--random-range-ratio",
        "--num-prompts",
        "--max-concurrency",
        "--seed",
        "--ignore-eos",
        "--percentile-metrics",
        "--metric-percentiles",
        "--save-result",
        "--result-dir",
        "--result-filename",
    ],
    "sglang_router.launch_router": [
        "--pd-disaggregation",
        "--prefill",
        "--decode",
        "--host",
        "--port",
    ],
}
HELP_COMMANDS: dict[str, list[str]] = {
    "sglang.launch_server": [sys.executable, "-m", "sglang.launch_server", "--help"],
    "sglang.benchmark.serving": [sys.executable, "-m", "sglang.benchmark.serving", "--help"],
    "vllm.serve": ["vllm", "serve", "--help=all"],
    "vllm.bench.serve": ["vllm", "bench", "serve", "--help=all"],
    "sglang_router.launch_router": [sys.executable, "-m", "sglang_router.launch_router", "--help"],
}


# ============= Checks =============


def missing_flags(help_text: str, flags: list[str]) -> list[str]:
    """Return the flags that do not appear in an engine's help output."""
    return [flag for flag in flags if flag not in help_text]


def check_tool(tool: str) -> list[str]:
    """Run a tool's help command and return the required flags it does not list."""
    completed = subprocess.run(
        HELP_COMMANDS[tool], capture_output=True, text=True, check=False, timeout=300
    )
    return missing_flags(completed.stdout + completed.stderr, REQUIRED_FLAGS[tool])
