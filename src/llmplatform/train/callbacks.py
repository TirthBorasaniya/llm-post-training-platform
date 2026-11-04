"""Training callback that records per-step wall time, token counts, and peak GPU memory."""

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

# ============= Constants =============

BYTES_PER_GIB = 1 << 30
STEP_METRICS_FILENAME = "step_metrics.jsonl"


# ============= Helpers =============


def peak_memory_gib() -> float:
    """Return peak allocated GPU memory in GiB, reduced to the maximum across ranks."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    value = torch.tensor([torch.cuda.max_memory_allocated() / BYTES_PER_GIB], device="cuda")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return float(value.item())


# ============= Callback =============


class StepMetricsCallback(TrainerCallback):
    """Append one JSON line per optimizer step on the main process."""

    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / STEP_METRICS_FILENAME

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if args.process_index == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        memory = peak_memory_gib()
        if args.process_index != 0:
            return
        row = {
            "step": state.global_step,
            "time_s": time.perf_counter(),
            "tokens_seen": int(getattr(state, "num_input_tokens_seen", 0)),
            "peak_mem_gib": memory,
            "world_size": args.world_size,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
