"""Per-session rate limiting and input limits for the public demo."""

import time
from collections import defaultdict, deque
from collections.abc import Callable

# ============= Constants =============

MAX_NEW_TOKENS_CAP = 256
MAX_PROMPT_CHARS = 2000
RATE_LIMIT_REQUESTS = 6
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_TRACKED_SESSIONS = 5000


# ============= Rate limiting =============


class SessionRateLimiter:
    """Allow at most ``max_requests`` per session within a sliding time window."""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: float = RATE_LIMIT_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, session: str, now: float) -> deque[float]:
        events = self._events[session]
        while events and now - events[0] >= self.window_seconds:
            events.popleft()
        if not events:
            self._events.pop(session, None)
            events = self._events[session]
        return events

    def retry_after(self, session: str) -> float:
        """Return seconds until the session may send again, or 0 when it may send now."""
        now = self._clock()
        events = self._prune(session, now)
        if len(events) < self.max_requests:
            return 0.0
        return self.window_seconds - (now - events[0])

    def acquire(self, session: str) -> bool:
        """Record a request and return True, or return False when the session is over its limit."""
        now = self._clock()
        events = self._prune(session, now)
        if len(events) >= self.max_requests:
            return False
        if len(self._events) > MAX_TRACKED_SESSIONS:
            stale = [key for key, value in self._events.items() if not value]
            for key in stale:
                self._events.pop(key, None)
        events.append(now)
        return True


# ============= Input limits =============


def clamp_max_tokens(requested: int | float | None, cap: int = MAX_NEW_TOKENS_CAP) -> int:
    """Return the requested completion length limited to ``[1, cap]``."""
    if requested is None:
        return cap
    return max(1, min(int(requested), cap))


def clamp_temperature(requested: float | None) -> float:
    """Return the sampling temperature limited to ``[0, 1]``."""
    if requested is None:
        return 0.0
    return max(0.0, min(float(requested), 1.0))


def validate_prompt(prompt: str | None, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """
    Return the stripped prompt, rejecting empty input and truncating very long input.

    Raises
    ------
    ValueError
        If the prompt is empty after stripping.
    """
    text = (prompt or "").strip()
    if not text:
        raise ValueError("enter a prompt")
    return text[:max_chars]
