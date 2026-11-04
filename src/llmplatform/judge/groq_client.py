"""Judge client for an OpenAI-compatible chat endpoint with retry and an on-disk verdict cache."""

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from llmplatform.common.hashing import sha256_json
from llmplatform.common.io import atomic_write_text
from llmplatform.eval.config import JudgeConfig
from llmplatform.judge.prompts import RETRY_SUFFIX, build_messages

logger = logging.getLogger(__name__)

# ============= Constants =============

VERDICT_PATTERN = re.compile(r'"winner"\s*:\s*"(A|B|tie)"', re.IGNORECASE)
LOOSE_PATTERN = re.compile(r"winner\W+(A|B|tie)\b", re.IGNORECASE)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF_SECONDS = 60.0
PARSE_ATTEMPTS = 3


# ============= Parsing =============


def parse_verdict(text: str) -> str | None:
    """Extract ``A``, ``B``, or ``tie`` from a judge reply, or ``None`` when absent."""
    if not text:
        return None
    for pattern in (VERDICT_PATTERN, LOOSE_PATTERN):
        match = pattern.search(text)
        if match:
            value = match.group(1)
            return "tie" if value.lower() == "tie" else value.upper()
    return None


@dataclass(frozen=True)
class Verdict:
    """One judged ordering of a response pair."""

    winner: str
    prompt_tokens: int
    completion_tokens: int
    o_parse_failure: bool = False
    o_cached: bool = False


# ============= Client =============


class JudgeClient:
    """Pairwise judge calls with disk caching keyed by model, prompt, and both responses."""

    def __init__(
        self,
        config: JudgeConfig,
        api_key: str,
        cache_dir: str | Path,
        client: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=config.request_timeout_seconds)
        self._sleep = sleep

    def cache_key(self, prompt: str, response_a: str, response_b: str) -> str:
        """Return the SHA-256 key of one judgment."""
        return sha256_json(
            {
                "judge_model": self.config.judge_model,
                "template_version": self.config.template_version,
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
            }
        )

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> Verdict | None:
        path = self._cache_path(key)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Verdict(
            winner=data["winner"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            o_cached=True,
        )

    def _write_cache(self, key: str, verdict: Verdict) -> None:
        payload = {
            "winner": verdict.winner,
            "prompt_tokens": verdict.prompt_tokens,
            "completion_tokens": verdict.completion_tokens,
        }
        atomic_write_text(self._cache_path(key), json.dumps(payload))

    def _post(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send one request, retrying transient failures with exponential backoff."""
        body = {
            "model": self.config.judge_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            **self.config.extra_body,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(url, json=body, headers=headers)
            except httpx.TransportError as exc:
                logger.warning("judge transport error", extra={"error": type(exc).__name__})
                delay = min(2.0**attempt, MAX_BACKOFF_SECONDS)
            else:
                if response.status_code == 200:
                    return dict(response.json())
                if response.status_code not in RETRYABLE_STATUS:
                    raise RuntimeError(
                        f"judge request failed with status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                retry_after = response.headers.get("retry-after")
                delay = (
                    float(retry_after) if retry_after else min(2.0**attempt, MAX_BACKOFF_SECONDS)
                )
                logger.warning("judge retry", extra={"status": response.status_code})
            if attempt == self.config.max_retries:
                break
            self._sleep(delay)
        raise RuntimeError(f"judge request failed after {self.config.max_retries + 1} attempts")

    def judge(self, prompt: str, response_a: str, response_b: str) -> Verdict:
        """
        Judge one ordering of a response pair.

        Cached verdicts are returned without a request. An unparsable reply is retried with a
        reminder; after three attempts the verdict is recorded as a tie with ``o_parse_failure``
        set, and it is not cached.
        """
        key = self.cache_key(prompt, response_a, response_b)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        messages = build_messages(prompt, response_a, response_b)
        prompt_tokens = completion_tokens = 0
        for _ in range(PARSE_ATTEMPTS):
            data = self._post(messages)
            usage = data.get("usage", {})
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            text = data["choices"][0]["message"].get("content") or ""
            winner = parse_verdict(text)
            if winner is not None:
                verdict = Verdict(winner, prompt_tokens, completion_tokens)
                self._write_cache(key, verdict)
                return verdict
            messages = [
                *messages[:-1],
                {**messages[-1], "content": messages[-1]["content"] + RETRY_SUFFIX},
            ]
        logger.error("judge reply could not be parsed", extra={"cache_key": key})
        return Verdict("tie", prompt_tokens, completion_tokens, o_parse_failure=True)
