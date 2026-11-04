"""Kubernetes autoscaling report from the replica time series and the per-request load log."""

import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llmplatform.common.io import iter_jsonl, read_json

# ============= Constants =============

DEFAULT_THRESHOLD = 2.0
DEFAULT_WINDOW_SECONDS = 60.0


# ============= Parsing =============


def parse_timestamp(value: str) -> float:
    """Convert an ISO 8601 UTC timestamp such as ``2026-01-01T00:00:05Z`` to epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp()


def read_series(path: str | Path) -> list[dict[str, Any]]:
    """Read the sampled time series with epoch-second timestamps added as ``t``."""
    rows = []
    for row in iter_jsonl(path):
        rows.append({**row, "t": parse_timestamp(row["ts"])})
    return sorted(rows, key=lambda row: row["t"])


def read_requests(path: str | Path) -> list[dict[str, Any]]:
    """Read per-request records with ``end_s`` in epoch seconds, ordered by completion."""
    rows = [{**row, "end_s": row["end_ms"] / 1000.0} for row in iter_jsonl(path)]
    return sorted(rows, key=lambda row: row["end_s"])


# ============= Analysis =============


def latency_stats(duration_ms: list[float]) -> dict[str, float | None]:
    """Median, 95th percentile, and count of request durations."""
    if not duration_ms:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    ordered = sorted(duration_ms)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {"count": len(ordered), "p50_ms": statistics.median(ordered), "p95_ms": ordered[index]}


def find_breach(series: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    """Return the first sample whose queue depth reaches the autoscaling threshold."""
    for row in series:
        depth = row.get("queue_depth")
        if depth is not None and depth >= threshold:
            return row
    return None


def find_ready(series: list[dict[str, Any]], after: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first sample after the breach with more ready replicas than at the breach."""
    base = after["ready_replicas"]
    for row in series:
        if row["t"] > after["t"] and row["ready_replicas"] > base:
            return row
    return None


def window(requests: list[dict[str, Any]], start: float, end: float) -> list[float]:
    """Durations of successful requests that finished in ``[start, end)``."""
    return [
        row["duration_ms"]
        for row in requests
        if start <= row["end_s"] < end and row["status"] == 200
    ]


def build_report(
    series: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """
    Summarize one autoscaling run.

    Parameters
    ----------
    series : list[dict]
        Samples with ``t``, ``replicas``, ``ready_replicas``, ``queue_depth``.
    requests : list[dict]
        Requests with ``end_s``, ``duration_ms``, ``status``, ``tokens``.
    summary : dict, optional
        k6 end-of-test summary, kept for its aggregate request counts.
    threshold : float
        Queue depth that triggers scale-out, the value in the ScaledObject.
    window_seconds : float
        Width of the latency windows before the breach and after the new pod is Ready.

    Returns
    -------
    report : dict
        Queue and replica series, breach and readiness times, seconds from breach to Ready,
        latency before and after scale-out, and tokens per second.
    """
    if not series:
        raise ValueError("the time series is empty")
    origin = series[0]["t"]
    breach = find_breach(series, threshold)
    ready = find_ready(series, breach) if breach else None
    ok = [row for row in requests if row["status"] == 200]
    span = (
        (requests[-1]["end_s"] - requests[0]["end_s"] + requests[0]["duration_ms"] / 1000.0)
        if requests
        else 0.0
    )
    report: dict[str, Any] = {
        "samples": len(series),
        "queue_depth_over_time": [
            {"seconds": round(row["t"] - origin, 3), "queue_depth": row.get("queue_depth")}
            for row in series
        ],
        "replicas_over_time": [
            {
                "seconds": round(row["t"] - origin, 3),
                "replicas": row["replicas"],
                "ready_replicas": row["ready_replicas"],
            }
            for row in series
        ],
        "peak_replicas": max(row["replicas"] for row in series),
        "peak_ready_replicas": max(row["ready_replicas"] for row in series),
        "threshold": threshold,
        "scale_out_observed": ready is not None,
        "seconds_to_threshold_breach": breach["t"] - origin if breach else None,
        "seconds_from_breach_to_pod_ready": ready["t"] - breach["t"] if breach and ready else None,
        "requests": {"total": len(requests), "ok": len(ok), "failed": len(requests) - len(ok)},
        "tokens_per_second": (sum(row["tokens"] for row in ok) / span) if span > 0 else None,
        "latency_window_seconds": window_seconds,
    }
    if breach and ready:
        report["latency_before_scale_out"] = latency_stats(
            window(requests, breach["t"] - window_seconds, breach["t"])
        )
        report["latency_after_scale_out"] = latency_stats(
            window(requests, ready["t"], ready["t"] + window_seconds)
        )
    else:
        report["latency_before_scale_out"] = report["latency_after_scale_out"] = None
    if summary is not None:
        values = summary.get("metrics", {}).get("http_reqs", {}).get("values", {})
        report["k6_http_reqs"] = values.get("count")
    return report


def build_report_from_files(
    series_path: str | Path,
    requests_path: str | Path,
    summary_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Read the load run outputs and build the report."""
    summary = read_json(summary_path) if summary_path and Path(summary_path).is_file() else None
    return build_report(
        read_series(series_path), read_requests(requests_path), summary, threshold, window_seconds
    )
