"""Traffic runs against a speculative or baseline server: greedy outputs, throughput, acceptance."""

import asyncio
import threading
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
from llmplatform.serve.metrics import snapshot

# ============= Constants =============

ACCEPT_METRIC = "sglang:spec_accept_length"
SAMPLE_INTERVAL_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 900.0


# ============= Greedy outputs =============


def collect_greedy_texts(
    openai_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    client: httpx.Client | None = None,
) -> list[str]:
    """Request one greedy completion per prompt, sequentially, for exact comparison."""
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    url = f"{openai_url.rstrip('/')}/chat/completions"
    texts = []
    for prompt in prompts:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        response = http.post(url, json=body)
        response.raise_for_status()
        texts.append(response.json()["choices"][0]["message"]["content"] or "")
    return texts


# ============= Acceptance sampling =============


class AcceptLengthSampler:
    """Polls the acceptance length gauge while traffic runs and averages the nonzero samples."""

    def __init__(self, metrics_url: str, interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.metrics_url = metrics_url
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                value = snapshot(self.metrics_url, (ACCEPT_METRIC,))[ACCEPT_METRIC]
            except httpx.HTTPError:
                value = None
            if value:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def __enter__(self) -> "AcceptLengthSampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def mean(self) -> float | None:
        """Return the mean of the nonzero samples, or ``None`` when none were seen."""
        return sum(self.samples) / len(self.samples) if self.samples else None


# ============= Traffic =============


def run_traffic(
    openai_url: str,
    metrics_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Send each prompt once with ``concurrency`` requests in flight and summarize the run.

    Returns
    -------
    summary : dict
        Throughput and latency percentiles, plus ``accept_length_mean`` (``None`` for servers
        that do not speculate) and the number of acceptance samples.
    """
    url = f"{openai_url.rstrip('/')}/chat/completions"

    async def run() -> Any:
        client = make_client()
        try:

            def sender(text: str) -> RequestFn:
                async def send() -> RequestStats:
                    return await stream_chat(
                        client,
                        url,
                        model,
                        [{"role": "user", "content": text}],
                        max_tokens,
                        temperature=temperature,
                    )

                return send

            sequences = [[sender(prompt)] for prompt in prompts]
            return await run_closed_loop(sequences, concurrency)
        finally:
            await client.aclose()

    with AcceptLengthSampler(metrics_url) as sampler:
        stats_list, wall = asyncio.run(run())
    summary = summarize(stats_list, wall)
    summary["accept_length_mean"] = sampler.mean()
    summary["accept_length_samples"] = len(sampler.samples)
    summary["concurrency"] = concurrency
    return summary
