"""Generate teacher completions for the distillation prompts through an OpenAI-compatible server."""

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from llmplatform.common.io import read_jsonl, write_json, write_jsonl
from llmplatform.distill.config import TeacherGenSection

logger = logging.getLogger(__name__)

# ============= Constants =============

REQUEST_TIMEOUT_SECONDS = 900.0
SECONDS_PER_HOUR = 3600.0
PARTIAL_SUFFIX = ".partial"


# ============= Generation =============


def request_completion(
    client: httpx.Client,
    url: str,
    model_name: str,
    prompt: list[dict[str, str]],
    spec: TeacherGenSection,
) -> dict[str, Any]:
    """Request one completion and return its text and finish reason."""
    body = {
        "model": model_name,
        "messages": prompt,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    response = client.post(url, json=body)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),
    }


def generate_completions(
    prompt_rows: list[dict[str, Any]],
    base_url: str,
    model_name: str,
    spec: TeacherGenSection,
    output_path: str | Path,
    gpus: int,
    client: httpx.Client | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """
    Generate and store completions, resuming from a partial file.

    Finished prompts are appended to ``<output>.partial`` as they complete, so an interrupted
    job continues where it stopped. Truncated completions are dropped unless the spec keeps
    them. The final file holds ``{"id", "messages"}`` records in prompt order.

    Returns
    -------
    summary : dict
        Counts, wall-clock seconds of this invocation, and GPU-hours of this invocation.
    """
    out = Path(output_path)
    partial = out.with_name(out.name + PARTIAL_SUFFIX)
    done: dict[str, dict[str, Any]] = {}
    if partial.is_file():
        done = {row["id"]: row for row in read_jsonl(partial)}
    pending = [row for row in prompt_rows if row["id"] not in done]
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    url = f"{base_url.rstrip('/')}/chat/completions"
    lock = threading.Lock()
    started = clock()
    partial.parent.mkdir(parents=True, exist_ok=True)

    def work(row: dict[str, Any]) -> None:
        result = request_completion(http, url, model_name, row["prompt"], spec)
        record = {"id": row["id"], "prompt": row["prompt"], **result}
        with lock:
            done[row["id"]] = record
            with open(partial, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=max(1, spec.concurrency)) as pool:
        list(pool.map(work, pending))

    kept, truncated, empty = [], 0, 0
    for row in prompt_rows:
        record = done[row["id"]]
        if record["finish_reason"] == "length" and not spec.keep_truncated:
            truncated += 1
        elif not record["text"].strip():
            empty += 1
        else:
            kept.append(
                {
                    "id": row["id"],
                    "messages": [*row["prompt"], {"role": "assistant", "content": record["text"]}],
                }
            )
    write_jsonl(out, kept)
    elapsed = clock() - started
    summary = {
        "requested": len(prompt_rows),
        "resumed_from_partial": len(prompt_rows) - len(pending),
        "kept": len(kept),
        "dropped_truncated": truncated,
        "dropped_empty": empty,
        "wall_seconds_this_run": elapsed,
        "gpus": gpus,
        "gpu_hours_this_run": elapsed * gpus / SECONDS_PER_HOUR,
    }
    write_json(out.with_name(out.stem + ".summary.json"), summary)
    return summary
