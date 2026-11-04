"""Tests for speculative decoding utilities and the disaggregation stack, workload, and summary."""

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmplatform.disagg.config import load_disagg
from llmplatform.disagg.e7 import breaking_rate, build_e7, slo_met
from llmplatform.disagg.launch import router_command, server_specs
from llmplatform.disagg.workload import build_mixed_workload, make_prompt
from llmplatform.serve.engines import build_command, build_env
from llmplatform.smoke.tiny import build_tiny_model
from llmplatform.speculative import run as spec_run
from llmplatform.speculative.config import load_speculative
from llmplatform.speculative.e6 import BASELINE, build_e6, config_name
from llmplatform.speculative.equality import compare_outputs, first_difference
from llmplatform.speculative.export_draft import pad_draft, zero_padded_rows
from llmplatform.speculative.run import AcceptLengthSampler, collect_greedy_texts

pytestmark = pytest.mark.unit


# ============= Equality =============


def test_equality_requires_exact_match() -> None:
    assert compare_outputs(["a b", "c"], ["a b", "c"]).passed
    report = compare_outputs(["hello world", "same"], ["hello there", "same"])
    assert not report.passed and report.mismatches[0]["index"] == 0
    assert report.mismatches[0]["first_difference_char"] == 6
    assert report.as_dict()["mismatch_count"] == 1
    assert first_difference("abc", "abcd") == 3
    with pytest.raises(ValueError, match="counts differ"):
        compare_outputs(["a"], [])


# ============= Draft export =============


def load_model(path: Path) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path), dtype="float32").eval()


PROMPTS = ["Explain binary search.", "The water cycle.", "Compound interest grows."]


@pytest.mark.parametrize("tied", [True, False])
def test_pad_draft_preserves_logits_and_never_picks_padding(
    tmp_path: Path, mini_tokenizer: Any, tied: bool
) -> None:
    n_real = len(mini_tokenizer)
    draft = load_model(
        build_tiny_model(
            tmp_path / "d", n_real + 20, 32, o_tie_embeddings=tied, tokenizer=mini_tokenizer
        )
    )
    report = pad_draft(draft, mini_tokenizer, mini_tokenizer, n_real + 52, PROMPTS, 32)
    assert report.passed and report.draft_size_before == n_real + 20
    assert report.draft_size_after == n_real + 52 and report.padded_argmax_count == 0
    assert report.real_logits_max_abs_diff <= report.logits_atol
    weight = draft.get_input_embeddings().weight
    assert float(weight[n_real:].detach().abs().max()) == 0.0
    assert float(weight[:n_real].detach().abs().max()) > 0.0


def test_pad_draft_refuses_mismatched_tokenizers_or_oversized_draft(
    tmp_path: Path, mini_tokenizer: Any
) -> None:
    n_real = len(mini_tokenizer)
    other = copy.deepcopy(mini_tokenizer)
    other.add_tokens(["<extra>"])
    draft = load_model(build_tiny_model(tmp_path / "d", n_real + 20, 32, tokenizer=mini_tokenizer))
    report = pad_draft(draft, mini_tokenizer, other, n_real + 52, PROMPTS, 32)
    assert not report.passed and not report.tokenizers_identical
    assert report.draft_size_after == n_real + 20
    larger = pad_draft(draft, mini_tokenizer, mini_tokenizer, n_real + 4, PROMPTS, 32)
    assert not larger.passed and larger.draft_size_after == n_real + 20
    _ = zero_padded_rows


# ============= Speculative runs and summary =============


def test_speculative_config_matches_protocol() -> None:
    config = load_speculative()
    assert config.draft_tokens == [3, 5, 8] and config.topk == 1
    assert config.concurrency == [1, 4, 16] and config.equality.prompts == 50
    assert [config.steps_for(tokens) for tokens in config.draft_tokens] == [2, 4, 7]
    assert set(config.drafts) == {"base_1_5b", "distilled_1_5b"}
    assert config.traffic.temperature == 0.0


def test_collect_greedy_texts_uses_temperature_zero() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": body["messages"][0]["content"][::-1]}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    texts = collect_greedy_texts("http://x/v1", "m", ["ab", "cd"], 16, client)
    assert texts == ["ba", "dc"] and all(body["temperature"] == 0.0 for body in seen)


def test_accept_length_sampler_averages_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter([0.0, 2.0, 4.0] + [None] * 1000)
    monkeypatch.setattr(spec_run, "snapshot", lambda url, names: {names[0]: next(values)})
    with AcceptLengthSampler("http://x/metrics", interval=0.001) as sampler:
        import time

        time.sleep(0.05)
    assert sampler.mean() == 3.0 and len(sampler.samples) == 2
    assert AcceptLengthSampler("http://x").mean() is None


def spec_row(
    precision: str, config: str, concurrency: int, tps: float, accept: float | None, rep: int = 0
) -> dict:
    draft, separator, tokens = config.rpartition("_k")
    return {
        "precision": precision,
        "config": config,
        "draft": draft if separator else None,
        "draft_tokens": int(tokens) if separator else None,
        "concurrency": concurrency,
        "repetition": rep,
        "output_throughput": tps,
        "itl_p50_ms": 10.0,
        "itl_p95_ms": 20.0,
        "ttft_p95_ms": 100.0,
        "accept_length_mean": accept,
    }


def test_e6_speedup_and_equality() -> None:
    name = config_name("distilled_1_5b", 5)
    rows = []
    for concurrency, base, fast in ((1, 40.0, 100.0), (16, 400.0, 480.0)):
        for rep in range(3):
            rows.append(spec_row("bf16", BASELINE, concurrency, base + rep, None, rep))
            rows.append(spec_row("bf16", name, concurrency, fast + rep, 3.5, rep))
    equality = {f"bf16/{name}": {"passed": True, "n": 50}}
    metrics = build_e6(rows, equality)
    node = metrics["results"]["bf16"][name]
    assert node["by_concurrency"]["1"]["speedup_vs_baseline"] == pytest.approx(101 / 41)
    assert (
        node["by_concurrency"]["16"]["speedup_vs_baseline"]
        < node["by_concurrency"]["1"]["speedup_vs_baseline"]
    )
    assert node["by_concurrency"]["1"]["accept_length_mean"] == 3.5
    assert node["meta"] == {"draft": "distilled_1_5b", "draft_tokens": 5}
    assert metrics["results"]["bf16"][BASELINE]["by_concurrency"]["1"]["speedup_vs_baseline"] == 1.0
    assert metrics["equality_passed"] is True
    failing = build_e6(rows, {f"bf16/{name}": {"passed": False, "n": 50}})
    assert failing["equality_passed"] is False and failing["equality_failures"] == [f"bf16/{name}"]


# ============= Disaggregation =============


def test_router_commands_for_both_setups() -> None:
    config = load_disagg()
    pd = router_command("disaggregated", config, "py")
    assert pd[:3] == ["py", "-m", "sglang_router.launch_router"]
    assert "--pd-disaggregation" in pd
    prefill = pd.index("--prefill")
    assert pd[prefill + 1 : prefill + 3] == ["http://127.0.0.1:30000", "8998"]
    assert pd[pd.index("--decode") + 1] == "http://127.0.0.1:30001"
    colocated = router_command("colocated", config)
    assert "--pd-disaggregation" not in colocated and "--worker-urls" in colocated
    with pytest.raises(ValueError):
        router_command("other", config)


def test_server_specs_pin_one_gpu_each() -> None:
    config = load_disagg()
    prefill, decode = server_specs("disaggregated", config, "/m")
    assert build_env(prefill) == {"CUDA_VISIBLE_DEVICES": "0"} and build_env(decode) == {
        "CUDA_VISIBLE_DEVICES": "1"
    }
    assert "--disaggregation-mode" in build_command(prefill) and "prefill" in build_command(prefill)
    decode_command = build_command(decode)
    assert decode_command[decode_command.index("--disaggregation-mode") + 1] == "decode"
    assert decode_command[decode_command.index("--disaggregation-transfer-backend") + 1] == "nixl"
    replicas = server_specs("colocated", config, "/m")
    assert all(spec.disaggregation is None for spec in replicas)
    assert [spec.resolved_port() for spec in replicas] == [30000, 30001]


def test_mixed_workload_is_seeded_unique_and_mixed() -> None:
    config = load_disagg().workload
    first = build_mixed_workload(config)
    assert first == build_mixed_workload(config)
    assert first != build_mixed_workload(config, seed_offset=1)
    counts = Counter(request.cls for request in first)
    assert counts["long"] + counts["chat"] == config.requests_per_rate
    assert 0.1 < counts["long"] / config.requests_per_rate < 0.35
    assert len({request.prompt for request in first}) == len(first)
    long_request = next(request for request in first if request.cls == "long")
    chat_request = next(request for request in first if request.cls == "chat")
    assert len(long_request.prompt.split()) == config.long.input_len
    assert len(chat_request.prompt.split()) == config.chat.input_len
    assert chat_request.max_tokens == 256
    import random

    assert len(make_prompt(random.Random(0), 10, 2.0).split()) == 20


def disagg_row(setup: str, rate: float, chat_ttft: float, itl: float, errors: int = 0) -> dict:
    cls = {"ttft_p50_ms": chat_ttft / 2, "ttft_p95_ms": chat_ttft, "mean_prompt_tokens": 500.0}
    return {
        "setup": setup,
        "rate": rate,
        "chat": cls,
        "long": {**cls, "ttft_p95_ms": 4000.0},
        "overall": {
            "itl_p95_ms": itl,
            "output_throughput": 900.0,
            "request_throughput": 3.0,
            "errors": errors,
        },
    }


def test_slo_and_breaking_rate() -> None:
    assert slo_met(disagg_row("a", 1, 400, 40), 500, 50)
    assert not slo_met(disagg_row("a", 1, 600, 40), 500, 50)
    assert not slo_met(disagg_row("a", 1, 400, 60), 500, 50)
    assert not slo_met(disagg_row("a", 1, 400, 40, errors=2), 500, 50)
    assert breaking_rate([(1.0, True), (4.0, False), (2.0, False)]) == 2.0
    assert breaking_rate([(1.0, True)]) is None


def test_e7_reports_both_setups_as_measured() -> None:
    rows = [
        disagg_row("colocated", 1.0, 300, 30),
        disagg_row("colocated", 2.0, 900, 70),
        disagg_row("disaggregated", 1.0, 350, 25),
        disagg_row("disaggregated", 2.0, 450, 35),
    ]
    metrics = build_e7(rows, 500, 50)
    assert metrics["setups"]["colocated"]["breaking_rate"] == 2.0
    assert metrics["setups"]["disaggregated"]["breaking_rate"] is None
    assert metrics["setups"]["colocated"]["by_rate"]["2.0"]["slo_met"] is False
    assert metrics["setups"]["disaggregated"]["by_rate"]["1.0"]["long_ttft_p95_ms"] == 4000.0
    worse = build_e7(
        [disagg_row("disaggregated", 1.0, 800, 70), disagg_row("colocated", 1.0, 300, 30)], 500, 50
    )
    assert worse["setups"]["disaggregated"]["breaking_rate"] == 1.0
    assert worse["setups"]["colocated"]["breaking_rate"] is None


def test_disagg_config_matches_protocol() -> None:
    config = load_disagg()
    assert config.workload.long.input_len == 4096 and config.workload.chat.input_len == 512
    assert config.workload.long.output_len == config.workload.chat.output_len == 256
    assert len(config.workload.rates) == 3 and config.transfer_backend == "nixl"
