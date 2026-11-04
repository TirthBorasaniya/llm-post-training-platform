"""Asynchronous streaming chat client that measures time to first token and inter-token latency."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

# ============= Constants =============

REQUEST_TIMEOUT_SECONDS = 1800.0
DATA_PREFIX = "data:"
DONE_MARKER = "[DONE]"


# ============= Models =============


@dataclass
class RequestStats:
    """Timing and token counts of one streamed request."""

    tag: str
    ttft: float | None = None
    itls: list[float] = field(default_factory=list)
    e2e: float = 0.0
    prompt_tokens: int | None = None
    output_tokens: int = 0
    cached_tokens: int | None = None
    start: float = 0.0
    error: str | None = None


# ============= Streaming =============


async def stream_chat(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    tag: str = "",
    temperature: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
) -> RequestStats:
    """
    Send one streaming chat request and time its chunks.

    Parameters
    ----------
    client : httpx.AsyncClient
        Shared client.
    url : str
        Full chat completions URL.
    model : str
        Served model name.
    messages : list[dict]
        Chat messages.
    max_tokens : int
        Completion limit.
    tag : str
        Caller label stored on the result, for example the turn index.

    Returns
    -------
    stats : RequestStats
        Time to first content chunk, gaps between later chunks, end-to-end time, and counts.
    """
    stats = RequestStats(tag=tag)
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = clock()
    stats.start = started
    previous: float | None = None
    chunks = 0
    try:
        async with client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                await response.aread()
                stats.error = f"status {response.status_code}"
                stats.e2e = clock() - started
                return stats
            async for line in response.aiter_lines():
                if not line.startswith(DATA_PREFIX):
                    continue
                payload = line[len(DATA_PREFIX) :].strip()
                if payload == DONE_MARKER:
                    break
                event = json.loads(payload)
                usage = event.get("usage")
                if usage:
                    stats.prompt_tokens = usage.get("prompt_tokens")
                    stats.output_tokens = usage.get("completion_tokens", stats.output_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    stats.cached_tokens = details.get("cached_tokens")
                choices = event.get("choices") or []
                if not choices or not choices[0].get("delta", {}).get("content"):
                    continue
                now = clock()
                chunks += 1
                if stats.ttft is None:
                    stats.ttft = now - started
                elif previous is not None:
                    stats.itls.append(now - previous)
                previous = now
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        stats.error = type(exc).__name__
    stats.e2e = clock() - started
    if not stats.output_tokens:
        stats.output_tokens = chunks
    return stats


# ============= Summary =============


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Return a percentile or ``None`` for an empty sequence."""
    return float(np.percentile(values, pct)) if len(values) else None


def summarize(stats_list: list[RequestStats], wall_seconds: float) -> dict[str, Any]:
    """
    Aggregate request stats into latency percentiles and throughput.

    Parameters
    ----------
    stats_list : list[RequestStats]
        Requests of one run.
    wall_seconds : float
        Wall-clock duration of the run, from the first send to the last completion.
    """
    ok = [item for item in stats_list if item.error is None and item.ttft is not None]
    ttfts = [item.ttft * 1000 for item in ok if item.ttft is not None]
    itls = [gap * 1000 for item in ok for gap in item.itls]
    e2es = [item.e2e * 1000 for item in ok]
    tokens = sum(item.output_tokens for item in ok)
    return {
        "requests": len(stats_list),
        "errors": len(stats_list) - len(ok),
        "output_tokens": tokens,
        "output_throughput": tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "request_throughput": len(ok) / wall_seconds if wall_seconds > 0 else 0.0,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "itl_p50_ms": percentile(itls, 50),
        "itl_p95_ms": percentile(itls, 95),
        "e2e_p95_ms": percentile(e2es, 95),
        "wall_seconds": wall_seconds,
    }


# ============= Runners =============


RequestFn = Callable[[], Awaitable[RequestStats]]


async def run_closed_loop(
    sequences: list[list[RequestFn]],
    concurrency: int,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[RequestStats], float]:
    """
    Run request sequences with at most ``concurrency`` sequences in flight.

    Requests within a sequence run one after another, as turns of a conversation do.

    Returns
    -------
    stats_list : list[RequestStats]
        Every request's stats.
    wall_seconds : float
        Duration from the first start to the last completion.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestStats] = []

    async def run_sequence(sequence: list[RequestFn]) -> None:
        async with semaphore:
            for send in sequence:
                results.append(await send())

    started = clock()
    await asyncio.gather(*(run_sequence(sequence) for sequence in sequences))
    return results, clock() - started


async def run_open_loop(
    schedule: list[tuple[float, RequestFn]], clock: Callable[[], float] = time.perf_counter
) -> tuple[list[RequestStats], float]:
    """
    Fire requests at fixed offsets regardless of completions, as independent clients do.

    Parameters
    ----------
    schedule : list[tuple[float, callable]]
        Pairs of seconds after the start and the request function to launch then.
    """
    started = clock()

    async def fire(offset: float, send: RequestFn) -> RequestStats:
        delay = offset - (clock() - started)
        if delay > 0:
            await asyncio.sleep(delay)
        return await send()

    results = await asyncio.gather(*(fire(offset, send) for offset, send in schedule))
    return list(results), clock() - started


def poisson_offsets(count: int, rate: float, seed: int) -> list[float]:
    """Return ``count`` arrival offsets of a Poisson process with ``rate`` requests per second."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    gaps = np.random.default_rng(seed).exponential(1.0 / rate, size=count)
    return [float(value) for value in np.cumsum(gaps)]


def make_client() -> httpx.AsyncClient:
    """Create the shared async client with a long timeout and no connection limit."""
    return httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
    )
