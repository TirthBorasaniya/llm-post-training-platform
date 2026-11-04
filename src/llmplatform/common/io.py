"""JSON and JSON Lines readers and atomic writers."""

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# ============= JSON =============


def read_json(path: str | Path) -> Any:
    """Load a JSON document from ``path``."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """
    Write ``text`` to ``path`` through a temporary file and an atomic rename.

    Parameters
    ----------
    path : str or Path
        Destination file. Parent directories are created.
    text : str
        Full file content.

    Returns
    -------
    path : Path
        The destination path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    """Write ``obj`` as pretty JSON with sorted keys and a trailing newline."""
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"
    return atomic_write_text(path, text)


# ============= JSON Lines =============


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one object per non-empty line of a JSON Lines file."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load every record of a JSON Lines file into a list."""
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write ``rows`` as JSON Lines through an atomic rename."""
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return atomic_write_text(path, text)
