"""GPU-hour ledger and training callbacks for time-based saves, budgets, and rollout guards."""

import json
import time
import uuid
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from llmplatform.common.io import atomic_write_text

# ============= Constants =============

LEDGER_FILENAME = "ledger.json"
SECONDS_PER_HOUR = 3600.0
CLIPPED_METRIC = "completions/clipped_ratio"


# ============= Ledger =============


class GpuLedger:
    """Accumulates GPU-hours over every chained segment of a run, surviving kills."""

    def __init__(self, path: str | Path, clock: Any = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self._segment_id: str | None = None

    def _read(self) -> dict[str, Any]:
        if self.path.is_file():
            return dict(json.loads(self.path.read_text(encoding="utf-8")))
        return {"segments": []}

    def start_segment(self, gpus: int) -> None:
        """Open a new segment for the current job."""
        data = self._read()
        self._segment_id = uuid.uuid4().hex
        now = self._clock()
        data["segments"].append({"id": self._segment_id, "start": now, "last": now, "gpus": gpus})
        atomic_write_text(self.path, json.dumps(data))

    def heartbeat(self) -> None:
        """Record the current time as the end of the open segment."""
        if self._segment_id is None:
            return
        data = self._read()
        for segment in data["segments"]:
            if segment["id"] == self._segment_id:
                segment["last"] = self._clock()
        atomic_write_text(self.path, json.dumps(data))

    def total_gpu_hours(self) -> float:
        """Return the summed GPU-hours of all segments."""
        return sum(
            (segment["last"] - segment["start"]) * segment["gpus"] / SECONDS_PER_HOUR
            for segment in self._read()["segments"]
        )

    def summary(self) -> dict[str, Any]:
        """Return segment count, wall-clock seconds, and total GPU-hours."""
        segments = self._read()["segments"]
        return {
            "segments": len(segments),
            "wall_seconds": sum(item["last"] - item["start"] for item in segments),
            "gpu_hours": self.total_gpu_hours(),
        }


# ============= Callbacks =============


class LedgerCallback(TrainerCallback):
    """Open a ledger segment at train start and heartbeat at every step."""

    def __init__(self, ledger: GpuLedger) -> None:
        self.ledger = ledger

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if args.process_index == 0:
            self.ledger.start_segment(args.world_size)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if args.process_index == 0:
            self.ledger.heartbeat()

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if args.process_index == 0:
            self.ledger.heartbeat()


class TimedSaveCallback(TrainerCallback):
    """Request a checkpoint whenever ``minutes`` passed since the last one."""

    def __init__(self, minutes: float, clock: Any = time.monotonic) -> None:
        self.interval = minutes * 60.0
        self._clock = clock
        self._last = clock()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._clock() - self._last >= self.interval:
            control.should_save = True

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._last = self._clock()


class BudgetCallback(TrainerCallback):
    """Stop training and save once the ledger reaches a GPU-hour budget."""

    def __init__(self, ledger: GpuLedger, budget_gpu_hours: float) -> None:
        self.ledger = ledger
        self.budget = budget_gpu_hours
        self.reached = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.ledger.total_gpu_hours() >= self.budget:
            self.reached = True
            control.should_save = True
            control.should_training_stop = True


class ClippedRatioError(RuntimeError):
    """Raised when too many rollouts hit the completion length limit."""


class ClippedRatioGuard(TrainerCallback):
    """Fail fast when the rolling mean of ``completions/clipped_ratio`` exceeds a limit."""

    def __init__(self, maximum: float, window: int) -> None:
        self.maximum = maximum
        self.window = window
        self.values: list[float] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if not logs or CLIPPED_METRIC not in logs:
            return
        self.values.append(float(logs[CLIPPED_METRIC]))
        recent = self.values[-self.window :]
        if len(recent) >= self.window and sum(recent) / len(recent) > self.maximum:
            raise ClippedRatioError(
                f"mean {CLIPPED_METRIC} over the last {self.window} logs is "
                f"{sum(recent) / len(recent):.3f}, above {self.maximum}; raise "
                "generation.max_completion_length and restart"
            )
