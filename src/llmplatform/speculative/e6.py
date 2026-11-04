"""Speculative decoding summary: acceptance, speedup over the baseline, and output equality."""

import statistics
from collections import defaultdict
from typing import Any

# ============= Constants =============

E6_RESULT = "e6_speculative.json"
BASELINE = "baseline"


# ============= Helpers =============


def config_name(draft: str, draft_tokens: int) -> str:
    """Return the label of a speculative configuration, for example ``distilled_1_5b_k5``."""
    return f"{draft}_k{draft_tokens}"


def median_of(rows: list[dict[str, Any]], key: str) -> float | None:
    """Median of a field over rows, skipping missing values."""
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


# ============= Summary =============


def build_e6(row_list: list[dict[str, Any]], equality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Assemble the speculative decoding metrics.

    Parameters
    ----------
    row_list : list[dict]
        Traffic rows with ``precision``, ``config`` (``baseline`` or a ``config_name``),
        ``draft`` and ``draft_tokens`` (``None`` for the baseline), ``concurrency``,
        ``repetition``, and the run summary including ``accept_length_mean``.
    equality : dict
        Equality report per ``"<precision>/<config>"`` from the 50-prompt greedy comparison.

    Returns
    -------
    metrics : dict
        ``results`` per precision, configuration, and concurrency with medians, acceptance
        length, speedup over the baseline of the same precision and concurrency, and the
        equality verdict; ``equality_passed`` for the whole experiment.
    """
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in row_list:
        grouped[(row["precision"], row["config"], int(row["concurrency"]))].append(row)
        meta[(row["precision"], row["config"])] = {
            "draft": row.get("draft"),
            "draft_tokens": row.get("draft_tokens"),
        }
    results: dict[str, Any] = {}
    for (precision, config, concurrency), rows in sorted(grouped.items()):
        baseline_rows = grouped.get((precision, BASELINE, concurrency), [])
        baseline_tps = median_of(baseline_rows, "output_throughput")
        tps = median_of(rows, "output_throughput")
        entry = {
            "output_throughput": tps,
            "itl_p50_ms": median_of(rows, "itl_p50_ms"),
            "itl_p95_ms": median_of(rows, "itl_p95_ms"),
            "ttft_p95_ms": median_of(rows, "ttft_p95_ms"),
            "accept_length_mean": median_of(rows, "accept_length_mean"),
            "speedup_vs_baseline": tps / baseline_tps if tps and baseline_tps else None,
            "repetitions": len(rows),
        }
        node = results.setdefault(precision, {}).setdefault(
            config, {"meta": meta[(precision, config)]}
        )
        node.setdefault("by_concurrency", {})[str(concurrency)] = entry
    for key, report in equality.items():
        precision, config = key.split("/", 1)
        results.setdefault(precision, {}).setdefault(config, {"meta": {}})["equality"] = report
    failed = [key for key, report in equality.items() if not report["passed"]]
    return {
        "results": results,
        "equality_passed": not failed,
        "equality_failures": failed,
        "note": "speedup is output tokens per second divided by the baseline at equal concurrency",
    }
