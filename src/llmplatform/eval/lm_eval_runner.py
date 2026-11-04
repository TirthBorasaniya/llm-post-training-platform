"""Run lm-evaluation-harness tasks and read back their metrics."""

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmplatform.eval.config import BenchmarksConfig, TaskSpec

logger = logging.getLogger(__name__)

# ============= Constants =============

RESULT_GLOB = "**/results_*.json"
VLLM_MAX_MODEL_LEN = 8192


# ============= Commands =============


def choose_backend(config: BenchmarksConfig) -> str:
    """Return the first preferred backend whose package is importable."""
    for name in config.backend_preference:
        if name == "hf" or importlib.util.find_spec(name) is not None:
            return name
    raise RuntimeError("no evaluation backend is available")


def build_command(
    task: TaskSpec,
    config: BenchmarksConfig,
    model_path: str,
    backend: str,
    output_dir: str | Path,
    revision: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """
    Build the ``lm_eval`` command for one task.

    Parameters
    ----------
    task : TaskSpec
        Task name and few-shot count.
    config : BenchmarksConfig
        Benchmark settings.
    model_path : str
        Hub id or local checkpoint directory.
    backend : str
        ``vllm`` or ``hf``.
    output_dir : str or Path
        Directory receiving the harness result files.
    revision : str, optional
        Hub revision pin.
    limit : int, optional
        Examples per task, for smoke runs only.
    """
    model_args = [f"pretrained={model_path}", f"dtype={config.dtype}"]
    if revision:
        model_args.append(f"revision={revision}")
    if backend == "vllm":
        model_args += [
            f"gpu_memory_utilization={config.vllm_gpu_memory_utilization}",
            f"max_model_len={VLLM_MAX_MODEL_LEN}",
        ]
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        task.name,
        "--batch_size",
        config.batch_size,
        "--output_path",
        str(output_dir),
    ]
    if task.num_fewshot is not None:
        command += ["--num_fewshot", str(task.num_fewshot)]
    if config.apply_chat_template:
        command.append("--apply_chat_template")
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def build_api_command(
    task: TaskSpec,
    config: BenchmarksConfig,
    model_name: str,
    base_url: str,
    output_dir: str | Path,
    limit: int | None = None,
    concurrency: int = 8,
) -> list[str]:
    """
    Build the ``lm_eval`` command that queries a running OpenAI-compatible chat server.

    Used for quantized variants served by SGLang or llama.cpp. ``base_url`` is the API root
    ending in ``/v1``.
    """
    model_args = [
        f"model={model_name}",
        f"base_url={base_url.rstrip('/')}/chat/completions",
        f"num_concurrent={concurrency}",
        "max_retries=3",
        "tokenized_requests=False",
    ]
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-chat-completions",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        task.name,
        "--output_path",
        str(output_dir),
        "--apply_chat_template",
    ]
    if task.num_fewshot is not None:
        command += ["--num_fewshot", str(task.num_fewshot)]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


# ============= Results =============


def find_results_file(output_dir: str | Path) -> Path:
    """Return the newest harness result file under ``output_dir``."""
    candidates = sorted(Path(output_dir).glob(RESULT_GLOB), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no lm_eval results under {output_dir}")
    return candidates[-1]


def read_metric(results_path: str | Path, task: str, metric: str) -> float:
    """Read one metric of a task from a harness result file."""
    with open(results_path, encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    task_results = data["results"].get(task)
    if task_results is None or metric not in task_results:
        raise KeyError(f"metric {metric!r} for task {task!r} not found in {results_path}")
    return float(task_results[metric])


def run_task(
    task: TaskSpec,
    config: BenchmarksConfig,
    model_path: str,
    backend: str,
    output_dir: str | Path,
    revision: str | None = None,
    limit: int | None = None,
    base_url: str | None = None,
    served_model_name: str | None = None,
) -> float:
    """Run one task and return its primary metric; a ``base_url`` selects the chat API path."""
    task_dir = Path(output_dir) / task.name
    if base_url:
        command = build_api_command(
            task, config, served_model_name or model_path, base_url, task_dir, limit
        )
    else:
        command = build_command(task, config, model_path, backend, task_dir, revision, limit)
    logger.info("running benchmark", extra={"task": task.name, "backend": backend})
    subprocess.run(command, check=True)
    return read_metric(find_results_file(task_dir), task.name, task.metric)
