"""Replay agent traces against a server and report latency, throughput, and prefix cache hits."""

import asyncio
import logging
from collections import defaultdict
from typing import Any

import httpx

from llmplatform.bench.client import (
    RequestFn,
    RequestStats,
    make_client,
    run_closed_loop,
    stream_chat,
    summarize,
)
from llmplatform.serve.metrics import metric_total, scrape

logger = logging.getLogger(__name__)

# ============= Cache accounting =============

CACHE_COUNTERS = {
    "sglang": ("sglang:cached_tokens_total", "sglang:prompt_tokens_total"),
    "vllm": ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
}


def counter_totals(engine: str, metric_dict: dict[str, list[Any]]) -> tuple[float, float] | None:
    """Return ``(cache hits, queries)`` totals, or ``None`` when the counters are absent."""
    hit_name, query_name = CACHE_COUNTERS[engine]
    hits = metric_total(metric_dict, hit_name)
    queries = metric_total(metric_dict, query_name)
    if hits is None:
        hits = metric_total(metric_dict, f"{hit_name}_total")
    if queries is None:
        queries = metric_total(metric_dict, f"{query_name}_total")
    if hits is None or queries is None:
        return None
    return hits, queries


def hit_rate(before: tuple[float, float] | None, after: tuple[float, float] | None) -> float | None:
    """Return the prefix cache hit rate over a run from counter deltas."""
    if before is None or after is None:
        return None
    queries = after[1] - before[1]
    return (after[0] - before[0]) / queries if queries > 0 else None


def flush_cache(engine: str, base_url: str, client: httpx.Client | None = None) -> None:
    """Empty the prefix cache between runs so earlier runs do not warm later ones."""
    http = client or httpx.Client(timeout=60)
    path = "/flush_cache" if engine == "sglang" else "/reset_prefix_cache"
    response = http.post(f"{base_url.rstrip('/')}{path}")
    if response.status_code != 200:
        raise RuntimeError(f"cache flush failed with status {response.status_code} at {path}")


# ============= Replay =============


def group_conversations(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group trace rows by conversation, ordered by turn."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["conversation_id"]].append(row)
    return [sorted(groups[key], key=lambda row: row["turn"]) for key in sorted(groups)]


async def replay(
    rows: list[dict[str, Any]],
    openai_url: str,
    model: str,
    concurrency: int,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[RequestStats], float]:
    """Replay conversations with ``concurrency`` conversations in flight."""
    http = client or make_client()
    url = f"{openai_url.rstrip('/')}/chat/completions"

    def sender(row: dict[str, Any]) -> RequestFn:
        async def send() -> RequestStats:
            return await stream_chat(
                http, url, model, row["messages"], row["max_tokens"], tag=str(row["turn"])
            )

        return send

    sequences = [
        [sender(row) for row in conversation] for conversation in group_conversations(rows)
    ]
    try:
        return await run_closed_loop(sequences, concurrency)
    finally:
        if client is None:
            await http.aclose()


def run_agent_level(
    rows: list[dict[str, Any]],
    engine: str,
    openai_url: str,
    metrics_url: str,
    model: str,
    concurrency: int,
    base_url: str,
    o_prefix_cache: bool,
) -> dict[str, Any]:
    """
    Run one concurrency level and return its summary.

    The cache is flushed first (when enabled) and hit rate comes from counter deltas around
    the run. TTFT is also reported separately for the first turn and for later turns.
    """
    if o_prefix_cache:
        flush_cache(engine, base_url)
    before = counter_totals(engine, scrape(metrics_url))
    stats_list, wall = asyncio.run(replay(rows, openai_url, model, concurrency))
    after = counter_totals(engine, scrape(metrics_url))
    summary = summarize(stats_list, wall)
    first = [item for item in stats_list if item.tag == "0"]
    later = [item for item in stats_list if item.tag != "0"]
    summary["ttft_first_turn"] = summarize(first, wall)["ttft_p50_ms"]
    summary["ttft_later_turns"] = summarize(later, wall)["ttft_p50_ms"]
    summary["prefix_cache_hit_rate"] = hit_rate(before, after)
    summary["concurrency"] = concurrency
    summary["conversations"] = len(group_conversations(rows))
    return summary
