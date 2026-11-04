"""Agent workload summary: prefix caching on versus off per engine, model, and concurrency."""

from collections import defaultdict
from typing import Any

# ============= Constants =============

E8_RESULT = "e8_agent_workloads.json"
SUMMARY_KEYS = (
    "prefix_cache_hit_rate",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_first_turn",
    "ttft_later_turns",
    "output_throughput",
    "request_throughput",
    "errors",
)


# ============= Summary =============


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return ``numerator / denominator`` or ``None`` when either is missing or zero."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def build_e8(row_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Assemble the agent workload metrics.

    Parameters
    ----------
    row_list : list[dict]
        Rows with ``model``, ``config_name`` (for example ``sglang_cache_on``), ``engine``,
        ``prefix_cache``, ``concurrency``, and the replay summary.

    Returns
    -------
    metrics : dict
        ``runs`` keyed by model, configuration, and concurrency, and ``cache_effect`` with the
        off-over-on TTFT ratio and the on-over-off throughput ratio per engine.
    """
    runs: dict[str, Any] = {}
    index: dict[tuple[str, str, bool, int], dict[str, Any]] = {}
    for row in row_list:
        entry = {key: row.get(key) for key in SUMMARY_KEYS}
        runs.setdefault(row["model"], {}).setdefault(row["config_name"], {})[
            str(row["concurrency"])
        ] = entry
        index[(row["model"], row["engine"], bool(row["prefix_cache"]), int(row["concurrency"]))] = (
            entry
        )
    effect: dict[str, Any] = defaultdict(dict)
    for (model, engine, cached, concurrency), on in index.items():
        if not cached:
            continue
        off = index.get((model, engine, False, concurrency))
        if off is None:
            continue
        effect[model].setdefault(engine, {})[str(concurrency)] = {
            "ttft_p50_off_over_on": ratio(off["ttft_p50_ms"], on["ttft_p50_ms"]),
            "ttft_p95_off_over_on": ratio(off["ttft_p95_ms"], on["ttft_p95_ms"]),
            "throughput_on_over_off": ratio(on["output_throughput"], off["output_throughput"]),
            "hit_rate_on": on["prefix_cache_hit_rate"],
        }
    return {"runs": runs, "cache_effect": dict(effect)}
