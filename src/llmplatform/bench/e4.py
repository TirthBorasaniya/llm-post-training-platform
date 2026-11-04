"""Serving summary: medians per cell, SLO-based cost, and the student to teacher ratio."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from llmplatform.bench.config import BenchConfig, CostConfig
from llmplatform.bench.parse import median_metrics
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import iter_jsonl
from llmplatform.cost.model import cost_ratio, cost_summary

# ============= Constants =============

E4_RESULT = "e4_serving.json"
STUDENT_PREFIX, TEACHER_PREFIX = "student", "teacher"


# ============= Helpers =============


def hardware_from_rows(row_list: list[dict[str, Any]]) -> HardwareRecord:
    """Return the hardware recorded on benchmark rows, requiring one machine configuration."""
    records = [row["hardware"] for row in row_list if "hardware" in row]
    if not records:
        raise ValueError("benchmark rows carry no hardware record")
    first = records[0]
    for record in records[1:]:
        if (record["gpu_model"], record["gpu_count"]) != (first["gpu_model"], first["gpu_count"]):
            raise ValueError("benchmark rows come from different GPU configurations")
    return HardwareRecord(**first)


def medians_by_cell(
    row_list: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """
    Group rows by ``(model, engine, workload)`` and return one median row per concurrency.

    Returns
    -------
    grouped : dict
        For each key, a list of dicts with ``concurrency`` and the median metrics, sorted by
        concurrency.
    """
    cells: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in row_list:
        workload = f"{row['workload']['input_len']}/{row['workload']['output_len']}"
        cells[(row["model_name"], row["engine"], workload, int(row["concurrency"]))].append(row)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model, engine, workload, concurrency), rows in sorted(cells.items()):
        grouped[(model, engine, workload)].append(
            {"concurrency": concurrency, **median_metrics(rows)}
        )
    return grouped


def precision_of(model_name: str) -> str:
    """Return ``quantized`` for names ending in ``_quant`` and ``bf16`` otherwise."""
    return "quantized" if model_name.endswith("_quant") else "bf16"


# ============= Summary =============


def build_e4(
    row_list: list[dict[str, Any]], bench: BenchConfig, cost: CostConfig, gpu_hour_usd: float
) -> dict[str, Any]:
    """
    Assemble the serving metrics.

    Parameters
    ----------
    row_list : list[dict]
        Raw cell rows from SGLang, vLLM, and Hugging Face runs.
    bench, cost : config models
        Grid and SLO settings.
    gpu_hour_usd : float
        Price of one GPU hour from the compute config.

    Returns
    -------
    metrics : dict
        ``models`` (medians per cell), ``cost`` (SLO operating point and USD per million output
        tokens), ``cost_ratio`` (student over teacher), and ``incomplete_cells``.
    """
    grouped = medians_by_cell(row_list)
    models: dict[str, Any] = {}
    cost_out: dict[str, Any] = {}
    for (model, engine, workload), cells in grouped.items():
        models.setdefault(model, {}).setdefault(engine, {})[workload] = {
            str(cell["concurrency"]): {k: v for k, v in cell.items() if k != "concurrency"}
            for cell in cells
        }
        cost_out.setdefault(model, {}).setdefault(engine, {})[workload] = cost_summary(
            cells, gpu_hour_usd, cost.gpus_per_replica, cost.slo.ttft_p95_ms, cost.slo.itl_p95_ms
        )
    ratios: dict[str, Any] = {}
    for engine in {key[1] for key in grouped}:
        for workload in {key[2] for key in grouped if key[1] == engine}:
            for precision, student, teacher in (
                ("bf16", "student_bf16", "teacher_bf16"),
                ("quantized", "student_quant", "teacher_quant"),
            ):
                try:
                    ratio = cost_ratio(
                        cost_out[student][engine][workload], cost_out[teacher][engine][workload]
                    )
                except KeyError:
                    continue
                ratios.setdefault(engine, {}).setdefault(workload, {})[precision] = ratio
    primary = cost.primary
    label = primary.workload.label()
    headline = ratios.get(primary.engine, {}).get(label, {}).get(primary.precision)
    incomplete = [
        {k: row[k] for k in ("engine", "model_name", "concurrency", "repetition")}
        for row in row_list
        if row.get("incomplete")
    ]
    return {
        "models": models,
        "slo": {"ttft_p95_ms": cost.slo.ttft_p95_ms, "itl_p95_ms": cost.slo.itl_p95_ms},
        "gpu_hour_usd": gpu_hour_usd,
        "cost": cost_out,
        "cost_ratio": {
            "definition": "student cost per million output tokens divided by teacher cost",
            "by_engine": ratios,
            "student_to_teacher": headline,
            "primary": {
                "engine": primary.engine,
                "workload": label,
                "precision": primary.precision,
            },
        },
        "incomplete_cells": incomplete,
        "hf_notes": (
            "Hugging Face rows use static batching: concurrency is the batch size, and every "
            "sequence of a batch shares its time to first token."
        ),
    }


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read raw cell rows."""
    return list(iter_jsonl(path))
