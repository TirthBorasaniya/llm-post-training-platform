"""Position-swapped pairwise judging of candidate responses against reference responses."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from llmplatform.judge.groq_client import JudgeClient, Verdict
from llmplatform.judge.stats import combine_orders

logger = logging.getLogger(__name__)

# ============= Models =============


@dataclass(frozen=True)
class PairResult:
    """Judgment of one prompt in both orders."""

    prompt_id: str
    outcome: str
    candidate_first: Verdict
    reference_first: Verdict


# ============= Judging =============


def _label(verdict: Verdict, o_candidate_is_a: bool) -> str:
    """Map an A/B/tie verdict to ``candidate``, ``reference``, or ``tie``."""
    if verdict.winner == "tie":
        return "tie"
    picked_a = verdict.winner == "A"
    return "candidate" if picked_a == o_candidate_is_a else "reference"


def judge_pair(
    client: JudgeClient, prompt_id: str, prompt: str, candidate: str, reference: str
) -> PairResult:
    """Judge one prompt twice with swapped positions and combine the verdicts."""
    candidate_first = client.judge(prompt, candidate, reference)
    reference_first = client.judge(prompt, reference, candidate)
    outcome = combine_orders(
        _label(candidate_first, o_candidate_is_a=True),
        _label(reference_first, o_candidate_is_a=False),
    )
    return PairResult(prompt_id, outcome, candidate_first, reference_first)


def judge_all(
    client: JudgeClient,
    prompt_rows: list[dict[str, Any]],
    candidate_by_id: dict[str, str],
    reference_by_id: dict[str, str],
    concurrency: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[PairResult]:
    """
    Judge every prompt that has both a candidate and a reference response.

    Parameters
    ----------
    client : JudgeClient
        Cached judge client.
    prompt_rows : list[dict]
        Rows with ``id`` and ``prompt``.
    candidate_by_id, reference_by_id : dict[str, str]
        Responses keyed by prompt id.
    concurrency : int
        Worker threads issuing requests.
    on_progress : callable, optional
        Called with ``(done, total)`` after each prompt.

    Returns
    -------
    result_list : list[PairResult]
        One result per prompt, in prompt order.
    """
    missing = [
        row["id"]
        for row in prompt_rows
        if row["id"] not in candidate_by_id or row["id"] not in reference_by_id
    ]
    if missing:
        raise KeyError(f"missing responses for {len(missing)} prompts, first: {missing[0]}")

    def work(row: dict[str, Any]) -> PairResult:
        return judge_pair(
            client, row["id"], row["prompt"], candidate_by_id[row["id"]], reference_by_id[row["id"]]
        )

    result_list: list[PairResult] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for done, result in enumerate(pool.map(work, prompt_rows), start=1):
            result_list.append(result)
            if on_progress:
                on_progress(done, len(prompt_rows))
    return result_list


def token_totals(result_list: list[PairResult]) -> dict[str, int]:
    """Sum judge prompt and completion tokens, and count cached and unparsable replies."""
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_calls": 0,
        "parse_failures": 0,
        "calls": 0,
    }
    for result in result_list:
        for verdict in (result.candidate_first, result.reference_first):
            totals["prompt_tokens"] += verdict.prompt_tokens
            totals["completion_tokens"] += verdict.completion_tokens
            totals["cached_calls"] += int(verdict.o_cached)
            totals["parse_failures"] += int(verdict.o_parse_failure)
            totals["calls"] += 1
    return totals


def first_position_rate(result_list: list[PairResult]) -> float:
    """Return how often the judge picked the response shown first, among decisive verdicts."""
    decisive = first = 0
    for result in result_list:
        for verdict in (result.candidate_first, result.reference_first):
            if verdict.winner != "tie":
                decisive += 1
                first += int(verdict.winner == "A")
    return first / decisive if decisive else 0.0
