"""Peak GPU memory sampling during a benchmark cell."""

import subprocess
import threading
from collections.abc import Callable

# ============= Constants =============

QUERY = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
SAMPLE_INTERVAL_SECONDS = 1.0


# ============= Sampling =============


def query_memory_mib() -> list[float]:
    """Return used memory in MiB for each visible GPU, or an empty list without ``nvidia-smi``."""
    try:
        completed = subprocess.run(QUERY, capture_output=True, text=True, check=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return []
    return [float(line) for line in completed.stdout.split() if line.replace(".", "", 1).isdigit()]


class GpuMemorySampler:
    """Context manager that records the maximum per-GPU memory seen while it is active."""

    def __init__(
        self,
        gpu_ids: list[int] | None = None,
        query: Callable[[], list[float]] = query_memory_mib,
        interval: float = SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self.gpu_ids = gpu_ids
        self._query = query
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib: float | None = None

    def _sample(self) -> None:
        values = self._query()
        if self.gpu_ids is not None:
            values = [values[index] for index in self.gpu_ids if index < len(values)]
        if values:
            peak = max(values)
            self.peak_mib = peak if self.peak_mib is None else max(self.peak_mib, peak)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self._interval)

    def __enter__(self) -> "GpuMemorySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()
