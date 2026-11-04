"""Generation helpers for the demo: chat messages and streamed completions."""

from collections.abc import Iterator
from typing import Any

from limits import SessionRateLimiter, clamp_max_tokens, clamp_temperature, validate_prompt

# ============= Constants =============

SYSTEM_PROMPT = "You are a helpful assistant."


# ============= Requests =============


def prepare_request(
    limiter: SessionRateLimiter,
    session: str,
    prompt: str | None,
    max_tokens: float | None,
    temperature: float | None,
) -> tuple[str, int, float]:
    """
    Validate a request and charge it to the session.

    Returns
    -------
    prompt, max_tokens, temperature : tuple
        The cleaned prompt and the clamped generation settings.

    Raises
    ------
    ValueError
        For an empty prompt.
    PermissionError
        When the session exceeded its rate limit; the message states the wait.
    """
    text = validate_prompt(prompt)
    wait = limiter.retry_after(session)
    if wait > 0 or not limiter.acquire(session):
        raise PermissionError(f"rate limit reached; try again in {max(1, int(wait))} seconds")
    return text, clamp_max_tokens(max_tokens), clamp_temperature(temperature)


def build_messages(prompt: str) -> list[dict[str, str]]:
    """Return the chat messages for a single-turn prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def stream_text(llm: Any, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """
    Yield the growing completion text from a ``llama_cpp.Llama`` chat stream.

    Each yielded string holds everything generated so far, which is what a text box shows.
    """
    text = ""
    for chunk in llm.create_chat_completion(
        messages=build_messages(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ):
        piece = chunk["choices"][0].get("delta", {}).get("content")
        if piece:
            text += piece
            yield text
