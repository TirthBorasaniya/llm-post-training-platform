"""Generate responses to the judge prompts with Hugging Face or an OpenAI-compatible server."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from llmplatform.common.io import write_jsonl
from llmplatform.eval.config import GenerationSpec

logger = logging.getLogger(__name__)

# ============= Constants =============

REQUEST_TIMEOUT_SECONDS = 600.0


# ============= Hugging Face backend =============


def generate_hf(
    model_path: str,
    prompt_rows: list[dict[str, Any]],
    spec: GenerationSpec,
    revision: str | None = None,
    o_cpu: bool = False,
) -> list[dict[str, Any]]:
    """
    Greedy-decode a response per prompt with ``transformers``, using the model's chat template.

    Prompts are batched by length to limit padding; output order matches ``prompt_rows``.

    Returns
    -------
    row_list : list[dict]
        Rows with ``id``, ``prompt``, ``response``, and ``completion_tokens``.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"dtype": torch.float32 if o_cpu else torch.bfloat16}
    if revision and not os.path.isdir(model_path):
        kwargs["revision"] = revision
    tokenizer: Any = AutoTokenizer.from_pretrained(model_path, revision=kwargs.get("revision"))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    device = "cpu" if o_cpu or not torch.cuda.is_available() else "cuda"
    model.to(device).eval()

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=False
        )
        for row in prompt_rows
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    output: dict[int, tuple[str, int]] = {}
    for start in range(0, len(order), spec.batch_size):
        batch_ids = order[start : start + spec.batch_size]
        batch = tokenizer([texts[i] for i in batch_ids], return_tensors="pt", padding=True).to(
            device
        )
        with torch.no_grad():
            generated = model.generate(
                **batch,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=spec.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        for position, index in enumerate(batch_ids):
            ids = new_tokens[position]
            real = ids[ids != tokenizer.pad_token_id]
            output[index] = (
                str(tokenizer.decode(real, skip_special_tokens=True)),
                int(real.numel()),
            )
    return [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "response": output[i][0],
            "completion_tokens": output[i][1],
        }
        for i, row in enumerate(prompt_rows)
    ]


# ============= OpenAI-compatible backend =============


def generate_openai(
    base_url: str,
    model_name: str,
    prompt_rows: list[dict[str, Any]],
    spec: GenerationSpec,
    concurrency: int = 8,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """
    Request a greedy response per prompt from an OpenAI-compatible chat endpoint.

    The server applies the model's chat template, so SGLang, vLLM, and llama.cpp servers all
    receive the same plain user message.
    """
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    url = f"{base_url.rstrip('/')}/chat/completions"

    def work(row: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": spec.temperature,
            "max_tokens": spec.max_new_tokens,
        }
        response = http.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return {
            "id": row["id"],
            "prompt": row["prompt"],
            "response": data["choices"][0]["message"]["content"] or "",
            "completion_tokens": int(data.get("usage", {}).get("completion_tokens", 0)),
        }

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, prompt_rows))


def save_responses(path: str | Path, row_list: list[dict[str, Any]]) -> Path:
    """Write generated responses as JSON Lines."""
    return write_jsonl(path, row_list)
