"""Tests for the results renderer: blocks fill only when results exist and --check is stable."""

from pathlib import Path
from typing import Any

import pytest

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.render import BLOCKS, process, render_block, render_text, table
from llmplatform.common.results import build_results, write_results

pytestmark = pytest.mark.unit

REF = ConfigRef(path="configs/x.yaml", sha256="0" * 64)
HW = HardwareRecord(gpu_model="Test GPU", gpu_count=1, provider_label="lab", gpu_hourly_usd=2.0)


def write(results_root: Path, filename: str, metrics: dict[str, Any]) -> None:
    result = build_results(filename.removesuffix(".json"), metrics, REF, HW, {})
    write_results(results_root / filename, result)


def readme(tmp_path: Path, names: list[str]) -> Path:
    text = "# Title\n\n" + "\n\n".join(
        f"## {name}\n<!-- results:{name}:start -->\n<!-- results:{name}:end -->" for name in names
    )
    path = tmp_path / "README.md"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def all_names() -> list[str]:
    return ["headline", *BLOCKS]


# ============= Behavior =============


def test_blocks_stay_empty_without_results(tmp_path: Path) -> None:
    path = readme(tmp_path, all_names())
    results = tmp_path / "results"
    results.mkdir()
    assert process(tmp_path, results, o_check=True) == []
    assert process(tmp_path, results, o_check=False) == []
    assert path.read_text() == readme(tmp_path, all_names()).read_text()
    assert all(render_block(name, results) == "" for name in all_names())


def test_block_fills_when_results_exist_and_check_detects_staleness(tmp_path: Path) -> None:
    path = readme(tmp_path, all_names())
    results = tmp_path / "results"
    write(
        results,
        "e2_scaling.json",
        {
            "points": {
                "1": {
                    "gpus": 1,
                    "tokens_per_second_total": 1000.0,
                    "tokens_per_second_per_gpu": 1000.0,
                    "scaling_efficiency": 1.0,
                    "peak_memory_gib_per_gpu": 30.5,
                },
                "2": {
                    "gpus": 2,
                    "tokens_per_second_total": 1900.0,
                    "tokens_per_second_per_gpu": 950.0,
                    "scaling_efficiency": 0.95,
                    "peak_memory_gib_per_gpu": 30.9,
                },
            },
            "complete": False,
            "missing": [4, 8],
        },
    )
    problems = process(tmp_path, results, o_check=True)
    assert problems == ["README.md: results blocks are out of date"]
    assert process(tmp_path, results, o_check=False) == []
    filled = path.read_text()
    assert "| 2 | 1900 | 950 | 0.950 | 30.9 |" in filled
    assert "Hardware: 1 x Test GPU, provider `lab`" in filled
    assert process(tmp_path, results, o_check=True) == []
    assert "- Scaling efficiency at 2 GPUs: 0.950." in filled
    # other blocks stay empty
    assert "<!-- results:e1:start -->\n<!-- results:e1:end -->" in filled


def test_results_without_a_block_and_unknown_blocks_are_reported(tmp_path: Path) -> None:
    readme(tmp_path, ["e1"])
    results = tmp_path / "results"
    write(results, "e8_agent_workloads.json", {"runs": {}, "cache_effect": {}})
    problems = process(tmp_path, results, o_check=True)
    assert any("has no 'e8' block" in problem for problem in problems)
    (tmp_path / "README.md").write_text("<!-- results:nope:start -->\n<!-- results:nope:end -->\n")
    assert any(
        "unknown results block" in problem for problem in process(tmp_path, results, o_check=True)
    )


def test_render_text_is_idempotent(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write(
        results,
        "e8_agent_workloads.json",
        {
            "runs": {
                "teacher": {
                    "sglang_cache_on": {
                        "4": {
                            "prefix_cache_hit_rate": 0.8,
                            "ttft_p50_ms": 40.0,
                            "ttft_p95_ms": 90.0,
                            "output_throughput": 500.0,
                        }
                    }
                }
            },
            "cache_effect": {},
        },
    )
    text = "<!-- results:e8:start -->\n<!-- results:e8:end -->"
    once = render_text(text, results)
    assert render_text(once, results) == once and "0.800" in once


# ============= Individual renderers =============


def test_each_renderer_handles_its_schema(tmp_path: Path) -> None:
    results = tmp_path / "results"
    judge = {
        "win_rate": 0.6,
        "ci95_low": 0.5,
        "ci95_high": 0.7,
        "candidate_length": {"mean_words": 120.0},
    }
    write(
        results,
        "e1_summary.json",
        {
            "checkpoints": {
                "sft": {"benchmarks": {"tasks": {"gsm8k": {"value": 0.7}}}, "judge": judge}
            },
            "missing": [],
        },
    )
    write(
        results,
        "e3_distillation.json",
        {
            "roles": {"teacher": {"judge": judge}, "distilled_student": {"judge": judge}},
            "retention_vs_teacher": {"distilled_student": 0.75},
            "pairwise_distilled_vs_base_student": {"win_rate": 0.7},
            "gpu_hours": {"distillation": 4.0, "sft_baseline": 4.0, "teacher_generation": 0.5},
            "student_to_teacher_cost_ratio": {"student_to_teacher": 0.2},
        },
    )
    write(
        results,
        "e4_serving.json",
        {
            "cost": {
                "teacher_bf16": {
                    "sglang": {
                        "512/256": {
                            "slo_met": True,
                            "concurrency": 16,
                            "output_tokens_per_second": 900.0,
                            "usd_per_million_output_tokens": 1.5,
                        }
                    }
                }
            },
            "slo": {"ttft_p95_ms": 500, "itl_p95_ms": 50},
            "cost_ratio": {"student_to_teacher": 0.2},
        },
    )
    write(
        results,
        "e5_k8s_cpu.json",
        {
            "label": "CPU on kind",
            "scale_out_observed": True,
            "seconds_from_breach_to_pod_ready": 42.0,
            "peak_ready_replicas": 2,
            "latency_before_scale_out": {"p50_ms": 4000.0},
            "latency_after_scale_out": {"p50_ms": 2500.0},
            "tokens_per_second": 12.5,
        },
    )
    write(
        results,
        "e6_speculative.json",
        {
            "results": {
                "bf16": {
                    "distilled_1_5b_k5": {
                        "meta": {},
                        "by_concurrency": {
                            "1": {
                                "accept_length_mean": 3.2,
                                "output_throughput": 90.0,
                                "speedup_vs_baseline": 2.1,
                            }
                        },
                        "equality": {"passed": True},
                    }
                }
            }
        },
    )
    write(
        results,
        "e7_disaggregation.json",
        {
            "setups": {
                "colocated": {
                    "breaking_rate": 2.0,
                    "by_rate": {
                        "1.0": {
                            "chat_ttft_p95_ms": 300.0,
                            "long_ttft_p95_ms": 3000.0,
                            "itl_p95_ms": 30.0,
                            "output_throughput": 800.0,
                            "slo_met": True,
                        }
                    },
                }
            }
        },
    )
    write(
        results,
        "data_manifest.json",
        {
            "sft": {"hub_id": "a/b", "rows_before": 100, "rows_after": 50},
            "reference_sizes": {"x": 1},
        },
    )
    outputs = {
        name: render_block(name, results)
        for name in ("e1", "e3", "e4", "e5_cpu", "e6", "e7", "data_manifest")
    }
    assert "| sft | 0.700 | 0.600 | 0.500 to 0.700 | 120 |" in outputs["e1"]
    assert (
        "Retention vs teacher" in outputs["e3"]
        and "0.750" in outputs["e3"]
        and "0.200" in outputs["e3"]
    )
    assert "| teacher_bf16 | sglang | 512/256 | 16 | 900 | 1.500 |" in outputs["e4"]
    assert "| Seconds from threshold breach to new pod Ready | 42.0 |" in outputs["e5_cpu"]
    assert "no GPU" not in outputs["e5_cpu"] and "Test GPU" in outputs["e5_cpu"]
    assert "| bf16 | distilled_1_5b_k5 | 1 | 3.20 | 90 | 2.10 | true |" in outputs["e6"]
    assert "colocated: 2.000" in outputs["e7"]
    assert "| sft | `a/b` | 100 | 50 |" in outputs["data_manifest"]
    assert render_block("headline", results).count("\n") >= 0
    assert table(["a"], [["1"]]) == "| a |\n|---|\n| 1 |"
