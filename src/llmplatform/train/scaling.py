"""Data-parallel scaling metrics computed from per-step training logs."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ============= Constants =============

DISCARD_STEPS = 10
SCALING_GPU_COUNTS = (1, 2, 4, 8)


# ============= Models =============


@dataclass(frozen=True)
class ScalingPoint:
    """Throughput measured for one GPU count."""

    gpus: int
    tokens_per_second_total: float
    tokens_per_second_per_gpu: float
    peak_memory_gib_per_gpu: float
    steps_measured: int


# ============= Computation =============


def read_step_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read ``step_metrics.jsonl`` rows in step order, keeping the last row of repeated steps."""
    row_dict: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row_dict[int(row["step"])] = row
    return [row_dict[step] for step in sorted(row_dict)]


def compute_point(
    row_list: list[dict[str, Any]], gpus: int, discard: int = DISCARD_STEPS
) -> ScalingPoint:
    """
    Compute throughput after discarding the first ``discard`` optimizer steps.

    Parameters
    ----------
    row_list : list[dict]
        Step rows with ``step``, ``time_s`` (seconds, monotonic), ``tokens_seen`` (global,
        cumulative), and ``peak_mem_gib``.
    gpus : int
        Number of GPUs of the run.
    discard : int
        Warm-up steps excluded from the measurement.

    Returns
    -------
    point : ScalingPoint
        Total and per-GPU tokens per second and the peak memory per GPU.
    """
    by_step = {int(row["step"]): row for row in row_list}
    last_step = max(by_step)
    if discard not in by_step or last_step <= discard:
        raise ValueError(f"need steps beyond {discard} to measure throughput, got {last_step}")
    start, end = by_step[discard], by_step[last_step]
    elapsed = end["time_s"] - start["time_s"]
    if elapsed <= 0:
        raise ValueError("non-positive elapsed time between measured steps")
    total = (end["tokens_seen"] - start["tokens_seen"]) / elapsed
    measured = [row for step, row in by_step.items() if step > discard]
    return ScalingPoint(
        gpus=gpus,
        tokens_per_second_total=total,
        tokens_per_second_per_gpu=total / gpus,
        peak_memory_gib_per_gpu=max(row["peak_mem_gib"] for row in measured),
        steps_measured=last_step - discard,
    )


def scaling_summary(point_list: list[ScalingPoint]) -> dict[str, Any]:
    """
    Compute scaling efficiency ``throughput_N / (N * throughput_1)`` for each point.

    Returns
    -------
    summary : dict
        ``points`` keyed by GPU count, ``complete`` (all of 1, 2, 4, 8 present), and
        ``missing`` GPU counts.
    """
    by_gpus = {point.gpus: point for point in point_list}
    if 1 not in by_gpus:
        raise ValueError("a 1-GPU point is required as the scaling baseline")
    base = by_gpus[1].tokens_per_second_total
    points = {
        str(gpus): {
            "gpus": gpus,
            "tokens_per_second_total": point.tokens_per_second_total,
            "tokens_per_second_per_gpu": point.tokens_per_second_per_gpu,
            "scaling_efficiency": point.tokens_per_second_total / (gpus * base),
            "peak_memory_gib_per_gpu": point.peak_memory_gib_per_gpu,
            "steps_measured": point.steps_measured,
        }
        for gpus, point in sorted(by_gpus.items())
    }
    missing = [gpus for gpus in SCALING_GPU_COUNTS if gpus not in by_gpus]
    return {"points": points, "complete": not missing, "missing": missing}
