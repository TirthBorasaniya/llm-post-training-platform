"""Tests for the judge client, pairwise logic, statistics, and run summary."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmplatform.common.io import write_jsonl
from llmplatform.eval.config import load_judge_config
from llmplatform.judge.groq_client import JudgeClient, Verdict, parse_verdict
from llmplatform.judge.pairwise import first_position_rate, judge_all, judge_pair, token_totals
from llmplatform.judge.run import estimated_cost, length_stats, run_judgment
from llmplatform.judge.stats import combine_orders, win_rate

pytestmark = pytest.mark.unit


def reply(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    )


def make_client(tmp_path: Path, handler: Any, sleeps: list[float] | None = None) -> JudgeClient:
    config = load_judge_config()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    recorded = sleeps if sleeps is not None else []
    return JudgeClient(config, "key", tmp_path / "cache", http, sleep=recorded.append)


# ============= Parsing =============


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"winner": "A"}', "A"),
        ('{"winner":"b","reason":"x"}', "B"),
        ('Sure. {"winner": "tie"}', "tie"),
        ("Winner: B", "B"),
        ("no verdict", None),
        ("", None),
    ],
)
def test_parse_verdict(text: str, expected: str | None) -> None:
    assert parse_verdict(text) == expected


# ============= Client =============


def test_cache_hit_avoids_second_request(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return reply('{"winner": "A"}')

    client = make_client(tmp_path, handler)
    first = client.judge("p", "r1", "r2")
    second = client.judge("p", "r1", "r2")
    assert first.winner == "A" and not first.o_cached
    assert second.winner == "A" and second.o_cached
    assert len(calls) == 1
    assert calls[0]["model"] == "openai/gpt-oss-120b"
    assert calls[0]["temperature"] == 0.0
    # swapping positions is a different judgment
    client.judge("p", "r2", "r1")
    assert len(calls) == 2


def test_cache_key_covers_model_prompt_and_responses(tmp_path: Path) -> None:
    client = make_client(tmp_path, lambda r: reply("{}"))
    base = client.cache_key("p", "a", "b")
    assert base == client.cache_key("p", "a", "b")
    assert len({base, client.cache_key("q", "a", "b"), client.cache_key("p", "b", "a")}) == 3
    assert len(base) == 64


def test_retries_rate_limit_then_succeeds(tmp_path: Path) -> None:
    statuses = iter([429, 503, 200])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 429:
            return httpx.Response(429, headers={"retry-after": "7"})
        if status == 503:
            return httpx.Response(503)
        return reply('{"winner": "B"}')

    client = make_client(tmp_path, handler, sleeps)
    assert client.judge("p", "a", "b").winner == "B"
    assert sleeps == [7.0, 2.0]


def test_non_retryable_error_raises(tmp_path: Path) -> None:
    client = make_client(tmp_path, lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(RuntimeError, match="401"):
        client.judge("p", "a", "b")


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    client = make_client(tmp_path, lambda r: httpx.Response(500))
    with pytest.raises(RuntimeError, match="attempts"):
        client.judge("p", "a", "b")


def test_unparsable_reply_becomes_uncached_tie(tmp_path: Path) -> None:
    count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        return reply("I cannot decide", 3, 2)

    client = make_client(tmp_path, handler)
    verdict = client.judge("p", "a", "b")
    assert verdict.winner == "tie" and verdict.o_parse_failure
    assert count["n"] == 3
    assert verdict.prompt_tokens == 9 and verdict.completion_tokens == 6
    client.judge("p", "a", "b")
    assert count["n"] == 6


# ============= Pairwise =============


class FakeClient:
    """Stands in for ``JudgeClient``; ``rule`` maps the two responses to a winner."""

    def __init__(self, rule: Any) -> None:
        self.rule = rule
        self.calls = 0

    def judge(self, prompt: str, response_a: str, response_b: str) -> Verdict:
        self.calls += 1
        return Verdict(self.rule(response_a, response_b), 1, 1)


def test_combine_orders_requires_agreement() -> None:
    assert combine_orders("candidate", "candidate") == "win"
    assert combine_orders("reference", "reference") == "loss"
    assert combine_orders("candidate", "reference") == "tie"
    assert combine_orders("tie", "candidate") == "tie"


def test_content_preference_wins_and_position_bias_ties() -> None:
    prefers_candidate = FakeClient(lambda a, b: "A" if a == "cand" else "B")
    result = judge_pair(prefers_candidate, "1", "p", "cand", "ref")  # type: ignore[arg-type]
    assert result.outcome == "win"
    prefers_reference = FakeClient(lambda a, b: "A" if a == "ref" else "B")
    assert judge_pair(prefers_reference, "1", "p", "cand", "ref").outcome == "loss"  # type: ignore[arg-type]
    always_first = FakeClient(lambda a, b: "A")
    biased = judge_pair(always_first, "1", "p", "cand", "ref")  # type: ignore[arg-type]
    assert biased.outcome == "tie"
    assert first_position_rate([biased]) == 1.0


def test_judge_all_checks_coverage_and_order() -> None:
    rows = [{"id": "a", "prompt": "p1"}, {"id": "b", "prompt": "p2"}]
    client = FakeClient(lambda a, b: "A" if a.startswith("c") else "B")
    seen: list[tuple[int, int]] = []
    results = judge_all(
        client,  # type: ignore[arg-type]
        rows,
        {"a": "c1", "b": "c2"},
        {"a": "r1", "b": "r2"},
        2,
        lambda done, total: seen.append((done, total)),
    )
    assert [r.prompt_id for r in results] == ["a", "b"]
    assert seen[-1] == (2, 2)
    assert token_totals(results)["calls"] == 4
    with pytest.raises(KeyError, match="missing responses"):
        judge_all(client, rows, {"a": "c1"}, {"a": "r1", "b": "r2"}, 1)  # type: ignore[arg-type]


# ============= Statistics =============


def test_win_rate_counts_ties_as_half() -> None:
    outcomes = ["win"] * 6 + ["tie"] * 2 + ["loss"] * 2
    stats = win_rate(outcomes, 2000, 1, 0.95)
    assert stats.win_rate == pytest.approx(0.7)
    assert (stats.wins, stats.ties, stats.losses, stats.n) == (6, 2, 2, 10)
    assert stats.ci_low < stats.win_rate < stats.ci_high
    assert stats == win_rate(outcomes, 2000, 1, 0.95)
    assert stats != win_rate(outcomes, 2000, 2, 0.95)


def test_win_rate_interval_degenerate_and_empty() -> None:
    stats = win_rate(["win"] * 20, 500, 0, 0.95)
    assert stats.ci_low == stats.ci_high == 1.0
    with pytest.raises(ValueError, match="no outcomes"):
        win_rate([], 10, 0, 0.95)


# ============= Run =============


def test_run_judgment_end_to_end(tmp_path: Path) -> None:
    prompts = [{"id": f"j{i}", "prompt": f"question {i}"} for i in range(4)]
    write_jsonl(tmp_path / "prompts.jsonl", prompts)
    write_jsonl(
        tmp_path / "cand.jsonl",
        [{"id": p["id"], "response": "good answer here", "completion_tokens": 3} for p in prompts],
    )
    write_jsonl(
        tmp_path / "ref.jsonl",
        [{"id": p["id"], "response": "bad", "completion_tokens": 1} for p in prompts],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        a_text = content.split("[Response A]\n")[1].split("\n\n[Response B]")[0]
        return reply('{"winner": "%s"}' % ("A" if a_text.startswith("good") else "B"))

    client = make_client(tmp_path, handler)
    metrics = run_judgment(
        client,
        load_judge_config(),
        tmp_path / "prompts.jsonl",
        tmp_path / "cand.jsonl",
        tmp_path / "ref.jsonl",
    )
    assert metrics["win_rate"] == 1.0 and metrics["wins"] == 4
    assert metrics["candidate_length"]["mean_words"] == 3
    assert metrics["reference_length"]["mean_tokens"] == 1
    assert metrics["judge_usage"]["calls"] == 8
    assert metrics["judge_cost_usd"] is None
    limited = run_judgment(
        client,
        load_judge_config(),
        tmp_path / "prompts.jsonl",
        tmp_path / "cand.jsonl",
        tmp_path / "ref.jsonl",
        2,
    )
    assert limited["n"] == 2
    assert limited["judge_usage"]["cached_calls"] == 4


def test_cost_requires_configured_prices() -> None:
    config = load_judge_config()
    totals = {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}
    assert estimated_cost(config, totals) is None
    priced = config.model_copy(
        update={
            "pricing": config.pricing.model_copy(
                update={"input_per_million_usd": 0.1, "output_per_million_usd": 0.4}
            )
        }
    )
    assert estimated_cost(priced, totals) == pytest.approx(0.3)
    assert length_stats([{"response": "a b"}])["mean_tokens"] is None
