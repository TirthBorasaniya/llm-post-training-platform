"""Checkpoint discovery and completion markers so long jobs resume on any backend."""

import re
from pathlib import Path

# ============= Constants =============

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
STATE_FILENAME = "trainer_state.json"
DONE_FILENAME = "DONE"


# ============= Discovery =============


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """
    Return the newest complete ``checkpoint-<step>`` directory, or ``None``.

    A directory counts as complete when it contains ``trainer_state.json``, which the Trainer
    writes last, so a checkpoint interrupted mid-write is skipped.

    Parameters
    ----------
    output_dir : str or Path
        Trainer output directory.

    Returns
    -------
    checkpoint : Path or None
        Directory with the highest step number, or ``None`` when no complete checkpoint exists.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return None
    step_dict: dict[int, Path] = {}
    for child in root.iterdir():
        match = CHECKPOINT_PATTERN.match(child.name)
        if match and child.is_dir() and (child / STATE_FILENAME).is_file():
            step_dict[int(match.group(1))] = child
    if not step_dict:
        return None
    return step_dict[max(step_dict)]


def mark_complete(output_dir: str | Path) -> Path:
    """Create the completion marker that tells chained jobs to stop."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / DONE_FILENAME
    marker.write_text("done\n", encoding="utf-8")
    return marker


def is_complete(output_dir: str | Path) -> bool:
    """Return whether ``output_dir`` carries the completion marker."""
    return (Path(output_dir) / DONE_FILENAME).is_file()
