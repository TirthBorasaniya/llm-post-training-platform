"""Normalize benchmark client outputs from SGLang and vLLM into one metric schema."""

import json
import statistics
from pathlib import Path
from typing import Any

# ============= Constants =============

METRIC_KEYS = (
    "output_throughput",
    "request_throughput",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "e2e_p95_ms",
)
SGLANG_KEYS = {
    "output_throughput": "output_throughput",
    "request_throughput": "request_throughput",
    "ttft_p50_ms": "median_ttft_ms",
    "ttft_p95_ms": "p95_ttft_ms",
    "itl_p50_ms": "median_itl_ms",
    "itl_p95_ms": "p95_itl_ms",
    "e2e_p95_ms": "p95_e2e_latency_ms",
}
VLLM_KEYS = {
    "output_throughput": "output_throughput",
    "request_throughput": "request_throughput",
    "ttft_p50_ms": "median_ttft_ms",
    "ttft_p95_ms": "p95_ttft_ms",
    "itl_p50_ms": "median_itl_ms",
    "itl_p95_ms": "p95_itl_ms",
    "e2e_p95_ms": "p95_e2el_ms",
}


# ============= Parsing =============


def _normalize(raw: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    missing = [source for source in mapping.values() if source not in raw]
    if missing:
        raise KeyError(f"benchmark output lacks fields: {missing}")
    result = {name: float(raw[source]) for name, source in mapping.items()}
    result["completed"] = int(raw.get("completed", 0))
    return result


def parse_sglang_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Map one ``sglang.benchmark.serving`` result object to the common metric names."""
    return _normalize(raw, SGLANG_KEYS)


def parse_vllm_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Map one ``vllm bench serve`` result object to the common metric names."""
    return _normalize(raw, VLLM_KEYS)


def read_last_json_line(path: str | Path) -> dict[str, Any]:
    """Return the last JSON object of a JSON Lines file."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    return dict(json.loads(lines[-1]))


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a single JSON object, as written by ``vllm bench serve --save-result``."""
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ============= Aggregation =============


def median_metrics(row_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Return the per-metric median across repetitions.

    Parameters
    ----------
    row_list : list[dict]
        Normalized rows of one cell, one per repetition.

    Returns
    -------
    result : dict
        Medians of every metric key plus ``repetitions``.
    """
    if not row_list:
        raise ValueError("no repetitions to aggregate")
    result: dict[str, Any] = {
        key: statistics.median(row[key] for row in row_list) for key in METRIC_KEYS
    }
    result["repetitions"] = len(row_list)
    peaks = [row["peak_gpu_memory_mib"] for row in row_list if row.get("peak_gpu_memory_mib")]
    result["peak_gpu_memory_mib"] = max(peaks) if peaks else None
    return result
