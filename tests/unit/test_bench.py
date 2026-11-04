"""Tests for benchmark parsing, the grid runner, cost model, E4 and E8 summaries, and flags."""

import json
from pathlib import Path
from typing import Any

import pytest

from llmplatform.bench.config import Workload, load_agent, load_bench, load_cost
from llmplatform.bench.e4 import build_e4, hardware_from_rows, medians_by_cell
from llmplatform.bench.e8 import build_e8, ratio
from llmplatform.bench.gpu import GpuMemorySampler
from llmplatform.bench.grid import (
    Cell,
    completed_keys,
    enumerate_cells,
    run_cell,
    run_grid,
    sglang_client_command,
    vllm_client_command,
)
from llmplatform.bench.hf_generate import percentile, summarize_batches
from llmplatform.bench.models import launch_args, resolve_variant
from llmplatform.bench.parse import (
    METRIC_KEYS,
    median_metrics,
    parse_sglang_result,
    parse_vllm_result,
    read_last_json_line,
)
from llmplatform.common.models import load_models
from llmplatform.cost.model import (
    cost_per_million_tokens,
    cost_ratio,
    cost_summary,
    highest_concurrency_meeting_slo,
)
from llmplatform.serve.flags import REQUIRED_FLAGS, missing_flags

pytestmark = pytest.mark.unit

SGLANG_RAW = {
    "completed": 500,
    "request_throughput": 10.0,
    "output_throughput": 2560.0,
    "median_ttft_ms": 80.0,
    "p95_ttft_ms": 150.0,
    "median_itl_ms": 12.0,
    "p95_itl_ms": 20.0,
    "p95_e2e_latency_ms": 3200.0,
}
VLLM_RAW = {**SGLANG_RAW, "p95_e2el_ms": 3300.0}
VLLM_RAW.pop("p95_e2e_latency_ms")


# ============= Configs and parsing =============


def test_grid_config_matches_protocol() -> None:
    config = load_bench()
    assert [w.label() for w in config.grid.workloads] == ["512/256", "2048/512"]
    assert config.grid.concurrency == [1, 4, 16, 64]
    assert config.grid.num_prompts == 500 and config.grid.repetitions == 3
    assert config.hf.batch_sizes == [1, 8, 32]
    assert {m.name for m in config.models} == {
        "teacher_bf16",
        "teacher_quant",
        "student_bf16",
        "student_quant",
    }
    cost = load_cost()
    assert (cost.slo.ttft_p95_ms, cost.slo.itl_p95_ms) == (500, 50)


def test_result_parsers_share_one_schema() -> None:
    sglang, vllm = parse_sglang_result(SGLANG_RAW), parse_vllm_result(VLLM_RAW)
    assert set(METRIC_KEYS) <= set(sglang) and set(METRIC_KEYS) <= set(vllm)
    assert sglang["e2e_p95_ms"] == 3200.0 and vllm["e2e_p95_ms"] == 3300.0
    with pytest.raises(KeyError, match="lacks fields"):
        parse_sglang_result({"completed": 1})


def test_median_metrics_and_last_line(tmp_path: Path) -> None:
    rows = [
        {
            **parse_sglang_result({**SGLANG_RAW, "output_throughput": value}),
            "peak_gpu_memory_mib": mib,
        }
        for value, mib in ((100.0, 10.0), (300.0, 30.0), (200.0, None))
    ]
    median = median_metrics(rows)
    assert median["output_throughput"] == 200.0 and median["repetitions"] == 3
    assert median["peak_gpu_memory_mib"] == 30.0
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n")
    assert read_last_json_line(path) == {"a": 2}
    with pytest.raises(ValueError):
        median_metrics([])


# ============= Commands and grid =============


def test_client_commands() -> None:
    config = load_bench()
    cell = Cell("sglang", "teacher_bf16", Workload(input_len=512, output_len=256), 16, 2)
    sglang = sglang_client_command(cell, "http://127.0.0.1:30000", "m", "/o.jsonl", config, "py")
    assert sglang[:3] == ["py", "-m", "sglang.benchmark.serving"]
    assert sglang[sglang.index("--max-concurrency") + 1] == "16"
    assert sglang[sglang.index("--seed") + 1] == "2"
    assert sglang[sglang.index("--random-range-ratio") + 1] == "1.0"
    assert "--flush-cache" in sglang and sglang[sglang.index("--port") + 1] == "30000"
    vllm = vllm_client_command(cell, "http://127.0.0.1:8000", "m", "/d", "r.json", config)
    assert vllm[:3] == ["vllm", "bench", "serve"] and "--ignore-eos" in vllm
    assert vllm[vllm.index("--endpoint") + 1] == "/v1/completions"
    assert vllm[vllm.index("--result-filename") + 1] == "r.json"


def test_enumerate_cells_covers_the_grid() -> None:
    cells = enumerate_cells(load_bench(), "vllm", "m")
    assert len(cells) == 2 * 4 * 3
    assert len({cell.key() for cell in cells}) == len(cells)


class FakeRunner:
    """Writes the result file a real client would produce."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], check: bool) -> None:
        self.commands.append(command)
        if command[0] == "vllm":
            directory = command[command.index("--result-dir") + 1]
            name = command[command.index("--result-filename") + 1]
            Path(directory, name).write_text(json.dumps(VLLM_RAW))
        else:
            Path(command[command.index("--output-file") + 1]).write_text(
                json.dumps(SGLANG_RAW) + "\n"
            )


class FixedSampler(GpuMemorySampler):
    def __init__(self) -> None:
        super().__init__(query=lambda: [1000.0, 2048.0], interval=0.01)


def test_run_cell_parses_both_engines_and_flags_incomplete() -> None:
    config = load_bench()
    runner = FakeRunner()
    cell = Cell("sglang", "m", Workload(input_len=512, output_len=256), 4, 0)
    row = run_cell(
        cell, "http://127.0.0.1:30000", "m", config, runner=runner, sampler_factory=FixedSampler
    )
    assert row["output_throughput"] == 2560.0 and row["peak_gpu_memory_mib"] == 2048.0
    assert "incomplete" not in row
    vllm_cell = Cell("vllm", "m", Workload(input_len=512, output_len=256), 4, 0)
    vrow = run_cell(
        vllm_cell, "http://127.0.0.1:8000", "m", config, runner=runner, sampler_factory=FixedSampler
    )
    assert vrow["e2e_p95_ms"] == 3300.0
    short = config.model_copy(update={"grid": config.grid.model_copy(update={"num_prompts": 600})})
    assert run_cell(cell, "http://h:1", "m", short, runner=runner, sampler_factory=FixedSampler)[
        "incomplete"
    ]
    with pytest.raises(ValueError):
        run_cell(Cell("hf", "m", cell.workload, 1, 0), "http://h:1", "m", config, runner=runner)


def test_grid_resumes_from_written_cells(tmp_path: Path) -> None:
    config = load_bench()
    small = config.model_copy(
        update={
            "grid": config.grid.model_copy(
                update={
                    "concurrency": [1, 4],
                    "repetitions": 1,
                    "workloads": [config.grid.workloads[0]],
                }
            )
        }
    )
    out = tmp_path / "cells.jsonl"
    runner = FakeRunner()
    args = (small, "sglang", "m", "http://127.0.0.1:30000", "m", out)
    first = run_grid(*args, runner=runner, sampler_factory=FixedSampler)
    second = run_grid(*args, runner=runner, sampler_factory=FixedSampler)
    assert first == 2 and second == 0 and len(runner.commands) == 2
    assert len(completed_keys(out)) == 2


# ============= HF baseline =============


def test_hf_summary_math() -> None:
    summary = summarize_batches([0.1, 0.1], [1.1, 1.1], batch_size=4, output_len=11)
    assert summary["output_throughput"] == pytest.approx(4 * 11 * 2 / 2.2)
    assert summary["ttft_p50_ms"] == pytest.approx(100.0)
    assert summary["itl_p50_ms"] == pytest.approx(100.0)  # (1.1 - 0.1) / 10 tokens
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([5], 95) == 5
    with pytest.raises(ValueError):
        percentile([], 50)


# ============= Cost =============


def test_cost_formula_and_slo_rule() -> None:
    assert cost_per_million_tokens(3.6, 1000.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cost_per_million_tokens(1.0, 0.0)
    cells = [
        {"concurrency": 1, "output_throughput": 50.0, "ttft_p95_ms": 30.0, "itl_p95_ms": 10.0},
        {"concurrency": 4, "output_throughput": 180.0, "ttft_p95_ms": 100.0, "itl_p95_ms": 30.0},
        {"concurrency": 16, "output_throughput": 500.0, "ttft_p95_ms": 400.0, "itl_p95_ms": 49.0},
        {"concurrency": 64, "output_throughput": 900.0, "ttft_p95_ms": 450.0, "itl_p95_ms": 80.0},
    ]
    point = highest_concurrency_meeting_slo(cells, 500, 50)
    assert point is not None and point.concurrency == 16
    assert highest_concurrency_meeting_slo(cells, 10, 5) is None
    summary = cost_summary(cells, 2.0, 1, 500, 50)
    assert summary["concurrency"] == 16
    assert summary["usd_per_million_output_tokens"] == pytest.approx(2.0 / 3600 / 500 * 1e6)
    assert cost_summary(cells, 2.0, 1, 10, 5) == {
        "slo_met": False,
        "usd_per_million_output_tokens": None,
    }
    assert (
        cost_ratio({"usd_per_million_output_tokens": 1.0}, {"usd_per_million_output_tokens": 4.0})
        == 0.25
    )
    assert (
        cost_ratio({"usd_per_million_output_tokens": None}, {"usd_per_million_output_tokens": 4.0})
        is None
    )


# ============= E4 =============

HARDWARE = {"gpu_model": "Test GPU", "gpu_count": 1, "provider_label": "t", "gpu_hourly_usd": 2.0}


def cell_row(model: str, engine: str, concurrency: int, tps: float, rep: int = 0) -> dict[str, Any]:
    return {
        "engine": engine,
        "model_name": model,
        "workload": {"input_len": 512, "output_len": 256},
        "concurrency": concurrency,
        "repetition": rep,
        "output_throughput": tps,
        "request_throughput": tps / 256,
        "ttft_p50_ms": 50.0,
        "ttft_p95_ms": 100.0,
        "itl_p50_ms": 10.0,
        "itl_p95_ms": 20.0,
        "e2e_p95_ms": 3000.0,
        "peak_gpu_memory_mib": 1000.0,
        "completed": 500,
        "hardware": HARDWARE,
    }


def test_e4_medians_cost_and_ratio() -> None:
    rows = []
    for model, tps in (("teacher_bf16", 100.0), ("student_bf16", 400.0)):
        for concurrency in (1, 4):
            for rep, scale in enumerate((0.9, 1.0, 1.1)):
                rows.append(cell_row(model, "sglang", concurrency, tps * concurrency * scale, rep))
    grouped = medians_by_cell(rows)
    assert grouped[("teacher_bf16", "sglang", "512/256")][1]["output_throughput"] == pytest.approx(
        400.0
    )
    metrics = build_e4(rows, load_bench(), load_cost(), gpu_hour_usd=3.6)
    teacher_cost = metrics["cost"]["teacher_bf16"]["sglang"]["512/256"]
    student_cost = metrics["cost"]["student_bf16"]["sglang"]["512/256"]
    assert teacher_cost["concurrency"] == 4 and teacher_cost["slo_met"]
    assert (
        student_cost["usd_per_million_output_tokens"]
        < teacher_cost["usd_per_million_output_tokens"]
    )
    assert metrics["cost_ratio"]["student_to_teacher"] == pytest.approx(0.25)
    assert metrics["models"]["teacher_bf16"]["sglang"]["512/256"]["4"]["repetitions"] == 3
    assert metrics["incomplete_cells"] == []


def test_hardware_rows_must_agree() -> None:
    first = cell_row("m", "sglang", 1, 1.0)
    other = {**cell_row("m", "sglang", 4, 1.0), "hardware": {**HARDWARE, "gpu_count": 2}}
    assert hardware_from_rows([first]).gpu_model == "Test GPU"
    with pytest.raises(ValueError, match="different GPU"):
        hardware_from_rows([first, other])
    with pytest.raises(ValueError, match="no hardware"):
        hardware_from_rows([{"engine": "x"}])


# ============= E8 =============


def agent_row(
    config: str, cached: bool, concurrency: int, ttft: float, tps: float
) -> dict[str, Any]:
    return {
        "model": "teacher",
        "config_name": config,
        "engine": "sglang",
        "prefix_cache": cached,
        "concurrency": concurrency,
        "prefix_cache_hit_rate": 0.8 if cached else 0.0,
        "ttft_p50_ms": ttft,
        "ttft_p95_ms": ttft * 2,
        "output_throughput": tps,
    }


def test_e8_cache_effect() -> None:
    rows = [
        agent_row("sglang_cache_on", True, 4, 50.0, 400.0),
        agent_row("sglang_cache_off", False, 4, 200.0, 250.0),
    ]
    metrics = build_e8(rows)
    effect = metrics["cache_effect"]["teacher"]["sglang"]["4"]
    assert effect["ttft_p50_off_over_on"] == 4.0
    assert effect["throughput_on_over_off"] == pytest.approx(1.6)
    assert effect["hit_rate_on"] == 0.8
    assert ratio(None, 1.0) is None and ratio(1.0, 0) is None
    assert load_agent().concurrency == [4, 16, 64]


# ============= Models and flags =============


def test_variant_resolution() -> None:
    config, models = load_bench(), load_models()
    by_name = {entry.name: entry for entry in config.models}
    assert resolve_variant(by_name["teacher_bf16"], None, models, "/c") == (
        "/c/teacher/merged",
        None,
    )
    assert resolve_variant(by_name["teacher_quant"], "fp8", models, "/c") == (
        "/c/teacher/merged",
        "fp8",
    )
    path, quantization = resolve_variant(by_name["student_quant"], "awq", models, "/c")
    assert path.endswith("distilled_1_5b_awq") and quantization is None
    with pytest.raises(ValueError, match="method"):
        resolve_variant(by_name["teacher_quant"], None, models, "/c")
    args = launch_args(by_name["teacher_quant"], "vllm", "fp8", models, "/c", config)
    assert args[args.index("--mem-fraction") + 1] == "0.85" and "--quantization" in args


def test_flag_checker_and_required_lists() -> None:
    text = "--model-path --host --port"
    assert missing_flags(text, ["--host", "--port", "--quantization"]) == ["--quantization"]
    assert "--enable-metrics" in REQUIRED_FLAGS["sglang.launch_server"]
    assert "--disable-prefix-caching" in REQUIRED_FLAGS["vllm.serve"]


def test_gpu_sampler_tracks_the_peak() -> None:
    readings = iter([[10.0, 20.0], [30.0, 5.0], [15.0, 15.0]] + [[1.0]] * 100)
    sampler = GpuMemorySampler([0], lambda: next(readings), interval=0.001)
    with sampler:
        import time

        time.sleep(0.05)
    assert sampler.peak_mib == 30.0
