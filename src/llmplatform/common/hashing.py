"""SHA-256 helpers for files, text, and JSON-serializable objects."""

import hashlib
import json
from pathlib import Path
from typing import Any

# ============= Constants =============

CHUNK_BYTES = 1 << 20


# ============= Hashing =============


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of the file at ``path``, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    """Return the hex SHA-256 digest of ``obj`` serialized with sorted keys."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
