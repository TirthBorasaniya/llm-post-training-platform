"""Run the SGLang and vLLM benchmark clients over the workload and concurrency grid."""

import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmplatform.bench.config import BenchConfig, Workload
from llmplatform.bench.gpu import GpuMemorySampler
from llmplatform.bench.parse import (
    parse_sglang_result,
    parse_vllm_result,
    read_json_file,
    read_last_json_line,
)
from llmplatform.common.io import iter_jsonl

logger = logging.getLogger(__name__)

# ============= Constants =============

RANGE_RATIO = "1.0"
PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
METRIC_PERCENTILES = "50,95,99"


# ============= Cells =============


@dataclass(frozen=True)
class Cell:
    """One benchmark measurement."""

    engine: str
    model_name: str
    workload: Workload
    concurrency: int
    repetition: int

    def key(self) -> tuple[str, str, str, int, int]:
        """Return a hashable identity used to resume interrupted grids."""
        return (
            self.engine,
            self.model_name,
            self.workload.label(),
            self.concurrency,
            self.repetition,
        )


def enumerate_cells(config: BenchConfig, engine: str, model_name: str) -> list[Cell]:
    """List every cell of the grid for one engine and model."""
    return [
        Cell(engine, model_name, workload, concurrency, repetition)
        for workload in config.grid.workloads
        for concurrency in config.grid.concurrency
        for repetition in range(config.grid.repetitions)
    ]


# ============= Commands =============


def sglang_client_command(
    cell: Cell,
    base_url: str,
    served_name: str,
    output_file: str,
    config: BenchConfig,
    python: str | None = None,
) -> list[str]:
    """Return the ``sglang.benchmark.serving`` command for a cell."""
    parsed = urlparse(base_url)
    return [
        python or sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang-oai",
        "--host",
        parsed.hostname or "127.0.0.1",
        "--port",
        str(parsed.port),
        "--model",
        served_name,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(cell.workload.input_len),
        "--random-output-len",
        str(cell.workload.output_len),
        "--random-range-ratio",
        RANGE_RATIO,
        "--num-prompts",
        str(config.grid.num_prompts),
        "--max-concurrency",
        str(cell.concurrency),
        "--seed",
        str(config.grid.seed + cell.repetition),
        "--flush-cache",
        "--output-file",
        output_file,
    ]


def vllm_client_command(
    cell: Cell,
    base_url: str,
    served_name: str,
    result_dir: str,
    result_name: str,
    config: BenchConfig,
) -> list[str]:
    """Return the ``vllm bench serve`` command for a cell."""
    parsed = urlparse(base_url)
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--host",
        parsed.hostname or "127.0.0.1",
        "--port",
        str(parsed.port),
        "--model",
        served_name,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(cell.workload.input_len),
        "--random-output-len",
        str(cell.workload.output_len),
        "--random-range-ratio",
        RANGE_RATIO,
        "--num-prompts",
        str(config.grid.num_prompts),
        "--max-concurrency",
        str(cell.concurrency),
        "--seed",
        str(config.grid.seed + cell.repetition),
        "--ignore-eos",
        "--percentile-metrics",
        PERCENTILE_METRICS,
        "--metric-percentiles",
        METRIC_PERCENTILES,
        "--save-result",
        "--result-dir",
        result_dir,
        "--result-filename",
        result_name,
    ]


# ============= Running =============


def run_cell(
    cell: Cell,
    base_url: str,
    served_name: str,
    config: BenchConfig,
    gpu_ids: list[int] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    sampler_factory: Callable[[], GpuMemorySampler] | None = None,
) -> dict[str, Any]:
    """
    Run one cell and return normalized metrics with the peak GPU memory.

    Parameters
    ----------
    runner : callable
        Replaceable ``subprocess.run`` for tests.
    """
    sampler = sampler_factory() if sampler_factory else GpuMemorySampler(gpu_ids)
    with tempfile.TemporaryDirectory() as scratch, sampler:
        if cell.engine == "sglang":
            output_file = str(Path(scratch) / "result.jsonl")
            runner(
                sglang_client_command(cell, base_url, served_name, output_file, config), check=True
            )
            parsed = parse_sglang_result(read_last_json_line(output_file))
        elif cell.engine == "vllm":
            runner(
                vllm_client_command(cell, base_url, served_name, scratch, "result.json", config),
                check=True,
            )
            parsed = parse_vllm_result(read_json_file(Path(scratch) / "result.json"))
        else:
            raise ValueError(f"unsupported engine {cell.engine!r}")
    row = {
        **asdict(cell),
        "workload": cell.workload.model_dump(),
        **parsed,
        "peak_gpu_memory_mib": sampler.peak_mib,
        "num_prompts": config.grid.num_prompts,
    }
    if parsed["completed"] != config.grid.num_prompts:
        row["incomplete"] = True
    return row


def completed_keys(path: str | Path) -> set[tuple[str, str, str, int, int]]:
    """Return the identities of cells already written to a result file."""
    if not Path(path).is_file():
        return set()
    keys = set()
    for row in iter_jsonl(path):
        workload = Workload(**row["workload"])
        keys.add(
            (
                row["engine"],
                row["model_name"],
                workload.label(),
                row["concurrency"],
                row["repetition"],
            )
        )
    return keys


def run_grid(
    config: BenchConfig,
    engine: str,
    model_name: str,
    base_url: str,
    served_name: str,
    output_path: str | Path,
    gpu_ids: list[int] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    row_extra: dict[str, Any] | None = None,
    sampler_factory: Callable[[], GpuMemorySampler] | None = None,
) -> int:
    """
    Run every missing cell of the grid, appending one JSON line per finished cell.

    Returns
    -------
    ran : int
        Number of cells executed in this call.
    """
    done = completed_keys(output_path)
    ran = 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for cell in enumerate_cells(config, engine, model_name):
        if cell.key() in done:
            continue
        logger.info("benchmark cell", extra={"cell": cell.key()})
        row = {
            **run_cell(cell, base_url, served_name, config, gpu_ids, runner, sampler_factory),
            **(row_extra or {}),
        }
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        ran += 1
    return ran
