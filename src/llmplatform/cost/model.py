"""Cost per million output tokens at the highest concurrency that meets the latency SLO."""

from dataclasses import dataclass
from typing import Any

# ============= Constants =============

SECONDS_PER_HOUR = 3600.0
TOKENS_PER_MILLION = 1e6


# ============= Cost =============


def cost_per_million_tokens(gpu_hour_usd: float, output_tokens_per_second: float) -> float:
    """
    Return ``gpu_hour_usd / 3600 / output_tokens_per_second * 1e6``.

    Parameters
    ----------
    gpu_hour_usd : float
        Price of the GPUs serving one replica for one hour.
    output_tokens_per_second : float
        Sustained output throughput of that replica.
    """
    if output_tokens_per_second <= 0:
        raise ValueError("output throughput must be positive")
    return gpu_hour_usd / SECONDS_PER_HOUR / output_tokens_per_second * TOKENS_PER_MILLION


@dataclass(frozen=True)
class SloPoint:
    """The operating point chosen by the SLO rule."""

    concurrency: int
    output_tokens_per_second: float
    ttft_p95_ms: float
    itl_p95_ms: float


def highest_concurrency_meeting_slo(
    cell_list: list[dict[str, Any]], ttft_p95_ms: float, itl_p95_ms: float
) -> SloPoint | None:
    """
    Pick the highest concurrency whose p95 time to first token and p95 inter-token latency
    are both within the SLO.

    Parameters
    ----------
    cell_list : list[dict]
        Medians per concurrency, each with ``concurrency``, ``output_throughput``,
        ``ttft_p95_ms``, and ``itl_p95_ms``.

    Returns
    -------
    point : SloPoint or None
        ``None`` when no concurrency meets the SLO.
    """
    passing = [
        cell
        for cell in cell_list
        if cell["ttft_p95_ms"] <= ttft_p95_ms and cell["itl_p95_ms"] <= itl_p95_ms
    ]
    if not passing:
        return None
    best = max(passing, key=lambda cell: cell["concurrency"])
    return SloPoint(
        best["concurrency"], best["output_throughput"], best["ttft_p95_ms"], best["itl_p95_ms"]
    )


def cost_summary(
    cell_list: list[dict[str, Any]],
    gpu_hour_usd: float,
    gpus: int,
    ttft_p95_ms: float,
    itl_p95_ms: float,
) -> dict[str, Any]:
    """
    Summarize cost for one model, engine, and workload.

    Returns
    -------
    summary : dict
        ``slo_met`` and, when met, the operating point and ``usd_per_million_output_tokens``.
    """
    point = highest_concurrency_meeting_slo(cell_list, ttft_p95_ms, itl_p95_ms)
    if point is None:
        return {"slo_met": False, "usd_per_million_output_tokens": None}
    return {
        "slo_met": True,
        "concurrency": point.concurrency,
        "output_tokens_per_second": point.output_tokens_per_second,
        "usd_per_million_output_tokens": cost_per_million_tokens(
            gpu_hour_usd * gpus, point.output_tokens_per_second
        ),
    }


def cost_ratio(student: dict[str, Any], teacher: dict[str, Any]) -> float | None:
    """Return student cost divided by teacher cost, or ``None`` if either misses the SLO."""
    first, second = (
        student["usd_per_million_output_tokens"],
        teacher["usd_per_million_output_tokens"],
    )
    if first is None or second is None:
        return None
    return float(first / second)
