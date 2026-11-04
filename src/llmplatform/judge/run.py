"""Judge one candidate response file against a reference and summarize the result."""

import logging
from pathlib import Path
from typing import Any

from llmplatform.common.io import read_jsonl
from llmplatform.eval.config import JudgeConfig
from llmplatform.judge.groq_client import JudgeClient
from llmplatform.judge.pairwise import first_position_rate, judge_all, token_totals
from llmplatform.judge.stats import win_rate

logger = logging.getLogger(__name__)

# ============= Helpers =============


def load_responses(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a response file into a dict keyed by prompt id."""
    return {row["id"]: row for row in read_jsonl(path)}


def length_stats(row_list: list[dict[str, Any]]) -> dict[str, float | None]:
    """Mean response length in words and characters, and tokens when recorded."""
    responses = [row["response"] for row in row_list]
    token_counts = [row["completion_tokens"] for row in row_list if "completion_tokens" in row]
    return {
        "mean_words": sum(len(text.split()) for text in responses) / len(responses),
        "mean_chars": sum(len(text) for text in responses) / len(responses),
        "mean_tokens": sum(token_counts) / len(token_counts) if token_counts else None,
    }


def estimated_cost(config: JudgeConfig, totals: dict[str, int]) -> float | None:
    """Return the judge cost in USD from configured prices, or ``None`` while prices are unset."""
    pricing = config.pricing
    if pricing.input_per_million_usd is None or pricing.output_per_million_usd is None:
        return None
    return (
        totals["prompt_tokens"] * pricing.input_per_million_usd
        + totals["completion_tokens"] * pricing.output_per_million_usd
    ) / 1e6


# ============= Run =============


def run_judgment(
    client: JudgeClient,
    config: JudgeConfig,
    prompts_path: str | Path,
    candidate_path: str | Path,
    reference_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """
    Judge a candidate against a reference on the judge prompts.

    Returns
    -------
    metrics : dict
        Win, loss, and tie counts, win rate with bootstrap interval, mean lengths of both
        response sets, position bias, judge token usage, and cost when prices are configured.
    """
    prompt_rows = read_jsonl(prompts_path)
    if limit is not None:
        prompt_rows = prompt_rows[:limit]
    candidate = load_responses(candidate_path)
    reference = load_responses(reference_path)
    result_list = judge_all(
        client,
        prompt_rows,
        {key: row["response"] for key, row in candidate.items()},
        {key: row["response"] for key, row in reference.items()},
        config.concurrency,
    )
    stats = win_rate(
        [item.outcome for item in result_list],
        config.bootstrap.resamples,
        config.bootstrap.seed,
        config.bootstrap.confidence,
    )
    totals = token_totals(result_list)
    ids = [row["id"] for row in prompt_rows]
    return {
        "judge_model": config.judge_model,
        "n": stats.n,
        "wins": stats.wins,
        "losses": stats.losses,
        "ties": stats.ties,
        "win_rate": stats.win_rate,
        "ci95_low": stats.ci_low,
        "ci95_high": stats.ci_high,
        "bootstrap_resamples": config.bootstrap.resamples,
        "candidate_length": length_stats([candidate[key] for key in ids]),
        "reference_length": length_stats([reference[key] for key in ids]),
        "first_position_pick_rate": first_position_rate(result_list),
        "judge_usage": totals,
        "judge_cost_usd": estimated_cost(config, totals),
        "pricing_source": config.pricing.source,
        "pricing_retrieved": config.pricing.retrieved,
    }
