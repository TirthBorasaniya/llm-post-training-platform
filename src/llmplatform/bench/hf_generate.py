"""Hugging Face generate baseline with static batching."""

import statistics
import time
from typing import Any

from llmplatform.bench.config import BenchConfig, Workload

# ============= Aggregation =============


def percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0 to 100) with linear interpolation."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no values")
    rank = (len(ordered) - 1) * pct / 100.0
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarize_batches(
    ttft_s: list[float], e2e_s: list[float], batch_size: int, output_len: int
) -> dict[str, float]:
    """
    Convert per-batch timings into the common metric schema.

    Parameters
    ----------
    ttft_s : list[float]
        Seconds to produce the first token of each batch (prefill plus one decode step).
    e2e_s : list[float]
        Seconds to produce all ``output_len`` tokens of each batch.
    batch_size : int
        Sequences per batch; every sequence of a batch shares its timings.
    output_len : int
        Generated tokens per sequence.
    """
    itl_s = [(e2e - ttft) / max(1, output_len - 1) for ttft, e2e in zip(ttft_s, e2e_s, strict=True)]
    wall = sum(e2e_s)
    return {
        "output_throughput": batch_size * output_len * len(e2e_s) / wall,
        "request_throughput": batch_size * len(e2e_s) / wall,
        "ttft_p50_ms": statistics.median(ttft_s) * 1000,
        "ttft_p95_ms": percentile(ttft_s, 95) * 1000,
        "itl_p50_ms": statistics.median(itl_s) * 1000,
        "itl_p95_ms": percentile(itl_s, 95) * 1000,
        "e2e_p95_ms": percentile(e2e_s, 95) * 1000,
    }


# ============= Benchmark =============


def run_hf_bench(
    model_path: str,
    model_name: str,
    config: BenchConfig,
    revision: str | None = None,
    o_cpu: bool = False,
) -> list[dict[str, Any]]:
    """
    Measure static-batch generation at each configured batch size and workload.

    Prompts are random token ids of the workload's length. Time to first token is measured
    with a one-token generation, total time with a full generation of exactly ``output_len``
    tokens, and inter-token latency is the difference spread over the remaining tokens.

    Returns
    -------
    row_list : list[dict]
        One row per workload, batch size, and repetition, in the common metric schema with
        ``engine`` set to ``hf`` and ``concurrency`` equal to the batch size.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch.float32 if o_cpu else torch.bfloat16}
    if revision and not os.path.isdir(model_path):
        kwargs["revision"] = revision
    model: Any = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    device = "cpu" if o_cpu or not torch.cuda.is_available() else "cuda"
    model.to(device).eval()
    vocab = int(model.config.get_text_config().vocab_size)
    on_gpu = device == "cuda"

    def timed_generate(input_ids: Any, new_tokens: int) -> float:
        if on_gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=new_tokens,
                min_new_tokens=new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=0,
            )
        if on_gpu:
            torch.cuda.synchronize()
        return time.perf_counter() - started

    row_list: list[dict[str, Any]] = []
    for workload in config.grid.workloads:
        for batch_size in config.hf.batch_sizes:
            for repetition in range(config.hf.repetitions):
                generator = torch.Generator().manual_seed(config.grid.seed + repetition)
                ttft_s, e2e_s = [], []
                if on_gpu:
                    torch.cuda.reset_peak_memory_stats()
                n_batches = config.hf.num_prompts // batch_size
                for index in range(n_batches + 1):  # the first batch is a warm-up
                    ids = torch.randint(
                        0, vocab, (batch_size, workload.input_len), generator=generator
                    ).to(device)
                    ttft = timed_generate(ids, 1)
                    total = timed_generate(ids, workload.output_len)
                    if index > 0:
                        ttft_s.append(ttft)
                        e2e_s.append(total)
                peak = torch.cuda.max_memory_allocated() / (1 << 20) if on_gpu else None
                row_list.append(
                    {
                        "engine": "hf",
                        "model_name": model_name,
                        "workload": workload.model_dump(),
                        "concurrency": batch_size,
                        "repetition": repetition,
                        **summarize_batches(ttft_s, e2e_s, batch_size, workload.output_len),
                        "peak_gpu_memory_mib": peak,
                        "num_prompts": n_batches * batch_size,
                        "completed": n_batches * batch_size,
                    }
                )
    return row_list


def workload_label(workload: Workload) -> str:
    """Return ``<input>/<output>``."""
    return workload.label()
