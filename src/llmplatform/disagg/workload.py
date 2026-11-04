"""Mixed long-prompt and chat workload driven at fixed request rates."""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from llmplatform.bench.agent_traces import VOCABULARY
from llmplatform.bench.client import (
    RequestFn,
    RequestStats,
    make_client,
    poisson_offsets,
    run_open_loop,
    stream_chat,
    summarize,
)
from llmplatform.disagg.config import MixedWorkload

# ============= Models =============


@dataclass(frozen=True)
class MixedRequest:
    """One request of the mixed workload."""

    cls: str
    prompt: str
    max_tokens: int


# ============= Building =============


def make_prompt(rng: random.Random, tokens: int, words_per_token: float) -> str:
    """Return a unique prompt of about ``tokens`` tokens made of common words."""
    return " ".join(rng.choice(VOCABULARY) for _ in range(max(1, round(tokens * words_per_token))))


def build_mixed_workload(config: MixedWorkload, seed_offset: int = 0) -> list[MixedRequest]:
    """
    Build ``requests_per_rate`` requests with a seeded mix of long and chat classes.

    Every prompt is unique so prefix caching cannot favor either setup.
    """
    rng = random.Random(config.seed + seed_offset)
    request_list = []
    for _ in range(config.requests_per_rate):
        is_long = rng.random() < config.long_fraction
        spec = config.long if is_long else config.chat
        request_list.append(
            MixedRequest(
                "long" if is_long else "chat",
                make_prompt(rng, spec.input_len, config.words_per_token),
                spec.output_len,
            )
        )
    return request_list


# ============= Running =============


def run_rate(
    openai_url: str, model: str, request_list: list[MixedRequest], rate: float, seed: int
) -> dict[str, Any]:
    """
    Send the requests as a Poisson arrival process at ``rate`` per second and summarize.

    Returns
    -------
    summary : dict
        ``overall`` and per-class summaries (``long`` and ``chat``) with latency percentiles
        and throughput, plus mean prompt tokens per class.
    """
    url = f"{openai_url.rstrip('/')}/chat/completions"

    async def run() -> tuple[list[RequestStats], float]:
        client = make_client()
        try:

            def sender(request: MixedRequest) -> RequestFn:
                async def send() -> RequestStats:
                    return await stream_chat(
                        client,
                        url,
                        model,
                        [{"role": "user", "content": request.prompt}],
                        request.max_tokens,
                        tag=request.cls,
                    )

                return send

            offsets = poisson_offsets(len(request_list), rate, seed)
            return await run_open_loop(
                [(offset, sender(req)) for offset, req in zip(offsets, request_list, strict=True)]
            )
        finally:
            await client.aclose()

    stats_list, wall = asyncio.run(run())
    summary: dict[str, Any] = {"overall": summarize(stats_list, wall), "rate": rate}
    for cls in ("long", "chat"):
        subset = [item for item in stats_list if item.tag == cls]
        entry = summarize(subset, wall)
        prompts = [item.prompt_tokens for item in subset if item.prompt_tokens]
        entry["mean_prompt_tokens"] = sum(prompts) / len(prompts) if prompts else None
        summary[cls] = entry
    return summary
