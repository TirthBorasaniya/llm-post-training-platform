"""Structured JSON logging with a fixed field set."""

import json
import logging
import sys
from datetime import UTC, datetime

# ============= Constants =============

RESERVED_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


# ============= Formatter =============


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object with ``ts``, ``level``, ``service``, and ``event``."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "service": self.service,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(service: str = "llmplatform", level: int = logging.INFO) -> None:
    """Install a JSON stream handler on the root logger, replacing earlier handlers."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
