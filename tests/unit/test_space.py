"""Tests for the demo's rate limiter, input limits, and generation helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))

from limits import (  # noqa: E402
    MAX_NEW_TOKENS_CAP,
    MAX_PROMPT_CHARS,
    SessionRateLimiter,
    clamp_max_tokens,
    clamp_temperature,
    validate_prompt,
)
from logic import build_messages, prepare_request, stream_text  # noqa: E402

pytestmark = pytest.mark.unit


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ============= Rate limiting =============


def test_limiter_blocks_after_the_quota_and_recovers() -> None:
    clock = Clock()
    limiter = SessionRateLimiter(3, 60.0, clock)
    assert [limiter.acquire("a") for _ in range(4)] == [True, True, True, False]
    assert limiter.retry_after("a") == pytest.approx(60.0)
    assert limiter.acquire("b")
    clock.now = 30.0
    assert limiter.retry_after("a") == pytest.approx(30.0)
    clock.now = 61.0
    assert limiter.retry_after("a") == 0.0 and limiter.acquire("a")


def test_limiter_uses_a_sliding_window() -> None:
    clock = Clock()
    limiter = SessionRateLimiter(2, 10.0, clock)
    limiter.acquire("s")
    clock.now = 6.0
    limiter.acquire("s")
    clock.now = 11.0
    assert limiter.acquire("s") is True
    assert limiter.acquire("s") is False


# ============= Limits =============


def test_input_limits() -> None:
    assert clamp_max_tokens(10_000) == MAX_NEW_TOKENS_CAP and clamp_max_tokens(0) == 1
    assert clamp_max_tokens(None) == MAX_NEW_TOKENS_CAP and clamp_max_tokens(64.0) == 64
    assert (
        clamp_temperature(3) == 1.0
        and clamp_temperature(-1) == 0.0
        and clamp_temperature(None) == 0.0
    )
    assert validate_prompt("  hi  ") == "hi"
    assert len(validate_prompt("x" * (MAX_PROMPT_CHARS + 50))) == MAX_PROMPT_CHARS
    with pytest.raises(ValueError, match="prompt"):
        validate_prompt("   ")
    with pytest.raises(ValueError):
        validate_prompt(None)


def test_prepare_request_charges_the_session() -> None:
    limiter = SessionRateLimiter(1, 60.0, Clock())
    text, tokens, temperature = prepare_request(limiter, "s", " hello ", 9999, 5)
    assert (text, tokens, temperature) == ("hello", MAX_NEW_TOKENS_CAP, 1.0)
    with pytest.raises(PermissionError, match="rate limit"):
        prepare_request(limiter, "s", "again", 16, 0.0)
    with pytest.raises(ValueError):
        prepare_request(limiter, "other", "", 16, 0.0)
    # an empty prompt does not consume the quota
    assert prepare_request(limiter, "other", "ok", 16, 0.0)[0] == "ok"


# ============= Generation =============


class FakeLlm:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        for piece in ("Hel", None, "lo", ""):
            yield {"choices": [{"delta": {"content": piece} if piece is not None else {}}]}


def test_stream_text_accumulates_and_passes_settings() -> None:
    llm = FakeLlm()
    outputs = list(stream_text(llm, "hi", 32, 0.2))
    assert outputs == ["Hel", "Hello"]
    call = llm.calls[0]
    assert call["max_tokens"] == 32 and call["temperature"] == 0.2 and call["stream"] is True
    assert call["messages"] == build_messages("hi")
    assert build_messages("hi")[1] == {"role": "user", "content": "hi"}


def test_space_files_are_complete(root: Path) -> None:
    requirements = (root / "space/requirements.txt").read_text()
    assert "gradio==" in requirements and "llama-cpp-python==" in requirements
    assert "whl/cpu" in requirements
    card = (root / "space/README.md").read_text()
    assert card.startswith("---\n") and "sdk: gradio" in card and "app_file: app.py" in card
    assert "sleeps when idle" in card
