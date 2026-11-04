"""Tests for the streaming client, agent trace generation, and agent replay."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmplatform.bench.agent_bench import (
    counter_totals,
    flush_cache,
    group_conversations,
    hit_rate,
    replay,
)
from llmplatform.bench.agent_traces import (
    SYSTEM_SUFFIX,
    TOOLS_SUFFIX,
    build_traces,
    load_sources,
    system_message,
)
from llmplatform.bench.client import (
    RequestStats,
    poisson_offsets,
    run_closed_loop,
    run_open_loop,
    stream_chat,
    summarize,
)
from llmplatform.bench.config import load_agent
from llmplatform.serve.metrics import parse_prometheus_text

pytestmark = pytest.mark.unit


def sse(chunks: list[str], usage: dict[str, Any] | None = None) -> bytes:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n" for text in chunks
    ]
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def async_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ============= Client =============


def test_stream_chat_measures_chunks_and_usage() -> None:
    usage = {
        "prompt_tokens": 40,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 32},
    }
    body = sse(["a", "b", "c"], usage)
    ticks = iter([0.0, 0.5, 0.75, 1.0, 1.25])

    async def run() -> RequestStats:
        async with async_client(lambda r: httpx.Response(200, content=body)) as client:
            return await stream_chat(
                client, "http://x/v1/chat/completions", "m", [], 8, "t", clock=lambda: next(ticks)
            )

    stats = asyncio.run(run())
    assert stats.ttft == 0.5 and stats.itls == [0.25, 0.25]
    assert stats.output_tokens == 3 and stats.prompt_tokens == 40 and stats.cached_tokens == 32
    assert stats.error is None and stats.e2e == 1.25


def test_stream_chat_reports_errors() -> None:
    async def run(handler: Any) -> RequestStats:
        async with async_client(handler) as client:
            return await stream_chat(client, "http://x", "m", [], 8)

    assert asyncio.run(run(lambda r: httpx.Response(500))).error == "status 500"

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert asyncio.run(run(broken)).error == "ConnectError"
    assert (
        asyncio.run(run(lambda r: httpx.Response(200, content=b"data: {oops\n\n"))).error
        == "JSONDecodeError"
    )


def test_summary_excludes_failed_requests() -> None:
    good = RequestStats("0", ttft=0.1, itls=[0.01, 0.03], e2e=0.5, output_tokens=10)
    bad = RequestStats("0", error="status 500")
    summary = summarize([good, good, bad], wall_seconds=2.0)
    assert summary["requests"] == 3 and summary["errors"] == 1
    assert summary["output_throughput"] == 10.0 and summary["request_throughput"] == 1.0
    assert summary["ttft_p50_ms"] == pytest.approx(100.0)
    assert summary["itl_p95_ms"] == pytest.approx(30.0)
    empty = summarize([], 1.0)
    assert empty["ttft_p50_ms"] is None and empty["output_throughput"] == 0.0


def test_closed_loop_limits_concurrency_and_orders_turns() -> None:
    state = {"active": 0, "peak": 0}
    log: list[str] = []

    def sender(name: str) -> Any:
        async def send() -> RequestStats:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            log.append(name)
            return RequestStats(name, ttft=0.0)

        return send

    sequences = [[sender(f"c{i}t{t}") for t in range(2)] for i in range(5)]
    results, wall = asyncio.run(run_closed_loop(sequences, concurrency=2))
    assert state["peak"] == 2 and len(results) == 10 and wall > 0
    for i in range(5):
        assert log.index(f"c{i}t0") < log.index(f"c{i}t1")


def test_open_loop_fires_on_schedule_and_poisson_is_seeded() -> None:
    fired: list[float] = []

    def sender() -> Any:
        async def send() -> RequestStats:
            fired.append(1.0)
            return RequestStats("x", ttft=0.0)

        return send

    results, _ = asyncio.run(run_open_loop([(0.0, sender()), (0.02, sender())]))
    assert len(results) == 2
    offsets = poisson_offsets(50, 10.0, seed=3)
    assert offsets == poisson_offsets(50, 10.0, seed=3) and offsets == sorted(offsets)
    assert offsets != poisson_offsets(50, 10.0, seed=4)
    with pytest.raises(ValueError):
        poisson_offsets(1, 0.0, 0)


# ============= Traces =============


def write_sources(root: Path, names: tuple[str, ...] = ("alpha", "beta")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / f"{name}{SYSTEM_SUFFIX}").write_text(f"System prompt of {name}. " * 20)
        (root / f"{name}{TOOLS_SUFFIX}").write_text(
            json.dumps([{"name": f"{name}_tool", "parameters": {}}])
        )
    return root


def small_config(**updates: Any) -> Any:
    return load_agent().model_copy(
        update={"conversations": 4, "turns_per_conversation": 3, **updates}
    )


def test_traces_share_prefix_and_grow(tmp_path: Path) -> None:
    sources = load_sources(write_sources(tmp_path / "src"))
    rows = build_traces(sources, small_config())
    assert len(rows) == 12
    by_conversation = group_conversations(rows)
    assert len(by_conversation) == 4
    for conversation in by_conversation:
        for previous, current in zip(conversation, conversation[1:], strict=False):
            assert current["messages"][: len(previous["messages"])] == previous["messages"]
            assert len(current["messages"]) == len(previous["messages"]) + 2
        assert conversation[0]["messages"][0]["role"] == "system"
    systems = {row["agent"]: row["messages"][0]["content"] for row in rows}
    assert set(systems) == {"alpha", "beta"}
    assert all(row["messages"][0]["content"] == systems[row["agent"]] for row in rows)
    assert "alpha_tool" in systems["alpha"]
    assert [conversation[0]["agent"] for conversation in by_conversation] == [
        "alpha",
        "beta",
        "alpha",
        "beta",
    ]


def test_traces_are_deterministic_and_seeded(tmp_path: Path) -> None:
    sources = load_sources(write_sources(tmp_path / "src"))
    assert build_traces(sources, small_config()) == build_traces(sources, small_config())
    assert build_traces(sources, small_config()) != build_traces(sources, small_config(), seed=999)


def test_sources_are_required(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no agent sources"):
        load_sources(tmp_path)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / f"x{SYSTEM_SUFFIX}").write_text("p")
    with pytest.raises(FileNotFoundError, match="tools"):
        load_sources(broken)
    (broken / f"x{TOOLS_SUFFIX}").write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="JSON list"):
        load_sources(broken)


def test_system_message_embeds_tools(tmp_path: Path) -> None:
    source = load_sources(write_sources(tmp_path / "s", ("one",)))[0]
    text = system_message(source)
    assert text.startswith("System prompt of one.") and "# Tools" in text and "one_tool" in text


def test_committed_trace_inputs_exist(root: Path) -> None:
    config = load_agent()
    assert (root / config.source_dir / "README.md").is_file()
    traces = root / config.output
    if traces.exists():
        first = json.loads(traces.read_text().splitlines()[0])
        assert {"conversation_id", "turn", "messages", "max_tokens"} <= set(first)


# ============= Replay and cache accounting =============

SGLANG_TEXT = "sglang:cached_tokens_total 30\nsglang:prompt_tokens_total 100\n"
VLLM_TEXT = "vllm:prefix_cache_hits_total 5\nvllm:prefix_cache_queries_total 20\n"


def test_cache_counters_and_hit_rate() -> None:
    before = counter_totals("sglang", parse_prometheus_text(SGLANG_TEXT))
    after = counter_totals(
        "sglang",
        parse_prometheus_text("sglang:cached_tokens_total 130\nsglang:prompt_tokens_total 300\n"),
    )
    assert before == (30.0, 100.0) and hit_rate(before, after) == pytest.approx(0.5)
    assert counter_totals("vllm", parse_prometheus_text(VLLM_TEXT)) == (5.0, 20.0)
    assert counter_totals("sglang", parse_prometheus_text("other 1")) is None
    assert hit_rate(None, after) is None and hit_rate(before, before) is None


def test_flush_cache_paths_and_failure() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200 if request.url.path != "/bad" else 500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    flush_cache("sglang", "http://x", client)
    flush_cache("vllm", "http://x", client)
    assert seen == ["/flush_cache", "/reset_prefix_cache"]
    failing = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(RuntimeError, match="flush failed"):
        flush_cache("vllm", "http://x", failing)


def test_replay_sends_turns_in_order(tmp_path: Path) -> None:
    sources = load_sources(write_sources(tmp_path / "src", ("alpha",)))
    rows = build_traces(sources, small_config(conversations=2))
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(len(body["messages"]))
        return httpx.Response(
            200, content=sse(["ok"], {"prompt_tokens": 1, "completion_tokens": 1})
        )

    async def run() -> tuple[list[RequestStats], float]:
        async with async_client(handler) as client:
            return await replay(rows, "http://x/v1", "m", 1, client)

    stats, wall = asyncio.run(run())
    assert len(stats) == 6 and all(item.error is None for item in stats)
    assert seen == [2, 4, 6, 2, 4, 6]
    assert [item.tag for item in stats] == ["0", "1", "2", "0", "1", "2"]
    assert wall >= 0
