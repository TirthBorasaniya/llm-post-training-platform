"""Tests for the benchmark runner, response generation, and E1 result files."""

import json
from pathlib import Path

import httpx
import pytest

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.results import load_results
from llmplatform.eval.config import load_benchmarks, load_e1, load_judge_config
from llmplatform.eval.e1 import (
    build_summary,
    checkpoint_path,
    update_checkpoint_file,
    write_summary,
)
from llmplatform.eval.generate import generate_openai
from llmplatform.eval.lm_eval_runner import (
    build_command,
    choose_backend,
    find_results_file,
    read_metric,
)

pytestmark = pytest.mark.unit

CONFIG_REF = ConfigRef(path="configs/eval/benchmarks.yaml", sha256="0" * 64)


# ============= Benchmarks =============


def test_benchmark_config_matches_protocol() -> None:
    config = load_benchmarks()
    shots = {task.name: task.num_fewshot for task in config.tasks}
    assert shots == {
        "ifeval": None,
        "gsm8k": 5,
        "mmlu": 5,
        "arc_challenge": 25,
        "hellaswag": 10,
        "truthfulqa_mc2": 0,
    }
    assert config.apply_chat_template is True


def test_build_command_per_backend() -> None:
    config = load_benchmarks()
    gsm8k = next(task for task in config.tasks if task.name == "gsm8k")
    ifeval = next(task for task in config.tasks if task.name == "ifeval")
    vllm = build_command(gsm8k, config, "/m", "vllm", "/out", revision="abc", limit=5)
    assert vllm[vllm.index("--model") + 1] == "vllm"
    args = vllm[vllm.index("--model_args") + 1]
    assert "pretrained=/m" in args and "revision=abc" in args and "gpu_memory_utilization" in args
    assert vllm[vllm.index("--num_fewshot") + 1] == "5"
    assert "--apply_chat_template" in vllm and vllm[vllm.index("--limit") + 1] == "5"
    hf = build_command(ifeval, config, "/m", "hf", "/out")
    assert "--num_fewshot" not in hf and "--limit" not in hf
    assert "gpu_memory_utilization" not in hf[hf.index("--model_args") + 1]


def test_choose_backend_falls_back_to_hf() -> None:
    config = load_benchmarks()
    assert choose_backend(config) in ("vllm", "hf")
    only_hf = config.model_copy(update={"backend_preference": ["hf"]})
    assert choose_backend(only_hf) == "hf"
    none = config.model_copy(update={"backend_preference": ["definitely_missing_backend"]})
    with pytest.raises(RuntimeError):
        choose_backend(none)


def test_read_metric_from_harness_output(tmp_path: Path) -> None:
    nested = tmp_path / "model" / "results_2026-01-01.json"
    nested.parent.mkdir()
    nested.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.75}}}))
    assert find_results_file(tmp_path) == nested
    assert read_metric(nested, "gsm8k", "exact_match,strict-match") == 0.75
    with pytest.raises(KeyError):
        read_metric(nested, "gsm8k", "acc,none")
    with pytest.raises(FileNotFoundError):
        find_results_file(tmp_path / "empty")


# ============= Generation =============


def test_generate_openai_uses_greedy_settings() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "reply to " + body["messages"][0]["content"]}}],
                "usage": {"completion_tokens": 4},
            },
        )

    spec = load_judge_config().generation
    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = generate_openai("http://x/v1", "m", [{"id": "a", "prompt": "hi"}], spec, 1, client)
    assert rows == [{"id": "a", "prompt": "hi", "response": "reply to hi", "completion_tokens": 4}]
    assert seen[0]["temperature"] == 0.0 and seen[0]["max_tokens"] == 1024


# ============= E1 files =============


def hardware() -> HardwareRecord:
    return HardwareRecord(gpu_model="none", gpu_count=0, provider_label="t", gpu_hourly_usd=None)


def test_sections_are_merged_and_summarized(tmp_path: Path) -> None:
    update_checkpoint_file(
        tmp_path,
        "sft",
        "benchmarks",
        {"tasks": {"gsm8k": {"value": 0.5}}},
        CONFIG_REF,
        hardware(),
        {},
    )
    update_checkpoint_file(tmp_path, "sft", "judge", {"win_rate": 0.6}, CONFIG_REF, hardware(), {})
    path = checkpoint_path(tmp_path, "sft")
    loaded = load_results(path)
    assert set(loaded.metrics) >= {"benchmarks", "judge", "provenance", "checkpoint"}
    assert set(loaded.metrics["provenance"]) == {"benchmarks", "judge"}
    # rewriting one section keeps the other
    update_checkpoint_file(tmp_path, "sft", "judge", {"win_rate": 0.7}, CONFIG_REF, hardware(), {})
    again = load_results(path)
    assert again.metrics["judge"]["win_rate"] == 0.7
    assert again.metrics["benchmarks"]["tasks"]["gsm8k"]["value"] == 0.5
    with pytest.raises(ValueError, match="section"):
        update_checkpoint_file(tmp_path, "sft", "other", {}, CONFIG_REF, hardware(), {})

    config = load_e1()
    summary = build_summary(tmp_path, config)
    assert list(summary["checkpoints"]) == ["sft"]
    assert "dpo" in summary["missing"]
    result = write_summary(tmp_path, config, CONFIG_REF, hardware(), {})
    assert (tmp_path / "e1_summary.json").is_file()
    assert result.metrics["checkpoints"]["sft"]["judge"]["win_rate"] == 0.7


def test_summary_requires_some_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_summary(tmp_path, load_e1())


def test_e1_lists_every_required_checkpoint() -> None:
    names = [entry.name for entry in load_e1().checkpoints]
    assert names == [
        "base_7b",
        "sft",
        "dpo",
        "orpo",
        "simpo",
        "teacher",
        "base_1_5b",
        "sft_baseline_1_5b",
        "distilled_1_5b",
    ]
