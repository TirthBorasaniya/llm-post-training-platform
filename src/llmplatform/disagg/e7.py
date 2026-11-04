"""Disaggregation summary: latency per traffic class, SLO status, and the breaking rate."""

from collections import defaultdict
from typing import Any

# ============= Constants =============

E7_RESULT = "e7_disaggregation.json"


# ============= SLO =============


def slo_met(row: dict[str, Any], ttft_p95_ms: float, itl_p95_ms: float) -> bool:
    """
    Return whether a run met the latency SLO.

    The chat class must keep p95 time to first token within the limit, which is what long
    prefills interfere with, and overall p95 inter-token latency must stay within its limit.
    """
    chat_ttft = row["chat"]["ttft_p95_ms"]
    itl = row["overall"]["itl_p95_ms"]
    if chat_ttft is None or itl is None or row["overall"]["errors"] > 0:
        return False
    return bool(chat_ttft <= ttft_p95_ms and itl <= itl_p95_ms)


def breaking_rate(rates: list[tuple[float, bool]]) -> float | None:
    """Return the lowest tested rate that breaks the SLO, or ``None`` if none did."""
    broken = [rate for rate, met in sorted(rates) if not met]
    return broken[0] if broken else None


# ============= Summary =============


def build_e7(
    row_list: list[dict[str, Any]], ttft_p95_ms: float, itl_p95_ms: float
) -> dict[str, Any]:
    """
    Assemble the disaggregation metrics.

    Parameters
    ----------
    row_list : list[dict]
        One row per setup and rate, with ``setup``, ``rate``, ``overall``, ``long``, ``chat``.
    ttft_p95_ms, itl_p95_ms : float
        SLO limits.

    Returns
    -------
    metrics : dict
        ``setups`` with per-rate latency and throughput, ``slo_met`` per rate, and the
        ``breaking_rate`` of each setup (``None`` means it held the SLO at every tested rate).
        A null or negative result is reported as measured.
    """
    setups: dict[str, Any] = {}
    rates_by_setup: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in row_list:
        met = slo_met(row, ttft_p95_ms, itl_p95_ms)
        rates_by_setup[row["setup"]].append((row["rate"], met))
        setups.setdefault(row["setup"], {"by_rate": {}})["by_rate"][str(row["rate"])] = {
            "slo_met": met,
            "chat_ttft_p50_ms": row["chat"]["ttft_p50_ms"],
            "chat_ttft_p95_ms": row["chat"]["ttft_p95_ms"],
            "long_ttft_p50_ms": row["long"]["ttft_p50_ms"],
            "long_ttft_p95_ms": row["long"]["ttft_p95_ms"],
            "itl_p95_ms": row["overall"]["itl_p95_ms"],
            "output_throughput": row["overall"]["output_throughput"],
            "request_throughput": row["overall"]["request_throughput"],
            "errors": row["overall"]["errors"],
            "mean_prompt_tokens_long": row["long"].get("mean_prompt_tokens"),
            "mean_prompt_tokens_chat": row["chat"].get("mean_prompt_tokens"),
        }
    for name, rates in rates_by_setup.items():
        setups[name]["breaking_rate"] = breaking_rate(rates)
    return {
        "setups": setups,
        "slo": {"chat_ttft_p95_ms": ttft_p95_ms, "itl_p95_ms": itl_p95_ms},
        "slo_definition": "chat p95 time to first token and overall p95 inter-token latency",
    }
