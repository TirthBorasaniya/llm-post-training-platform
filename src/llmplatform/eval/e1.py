"""Per-checkpoint result files and the quality experiment summary."""

from pathlib import Path
from typing import Any

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import read_json
from llmplatform.common.results import ResultsFile, build_results, load_results, write_results
from llmplatform.eval.config import E1Config

# ============= Constants =============

E1_DIRNAME = "e1"
SUMMARY_NAME = "e1_summary.json"
SECTIONS = ("benchmarks", "judge")


# ============= Checkpoint files =============


def checkpoint_path(results_root: str | Path, name: str) -> Path:
    """Return ``results/e1/<name>.json``."""
    return Path(results_root) / E1_DIRNAME / f"{name}.json"


def update_checkpoint_file(
    results_root: str | Path,
    name: str,
    section: str,
    values: dict[str, Any],
    config: ConfigRef,
    hardware: HardwareRecord,
    packages: dict[str, str],
    notes: str = "",
) -> Path:
    """
    Write one section (``benchmarks`` or ``judge``) of a checkpoint's result file.

    The other section of an existing file is preserved. Each section keeps its own code
    version, config hash, and hardware, because benchmarks and judging run on different
    machines.
    """
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {SECTIONS}")
    path = checkpoint_path(results_root, name)
    metrics: dict[str, Any] = {}
    if path.is_file():
        metrics = dict(load_results(path).metrics)
    result = build_results(f"e1_{name}_{section}", {}, config, hardware, packages, notes)
    metrics[section] = values
    provenance = dict(metrics.get("provenance", {}))
    provenance[section] = {
        "timestamp_utc": result.timestamp_utc,
        "git_sha": result.git_sha,
        "sync_sha": result.sync_sha,
        "config_path": config.path,
        "config_sha256": config.sha256,
        "hardware": hardware.model_dump(mode="json"),
        "notes": notes,
    }
    metrics["provenance"] = provenance
    metrics["checkpoint"] = name
    final = result.model_copy(update={"metrics": metrics})
    return write_results(path, final)


# ============= Summary =============


def build_summary(results_root: str | Path, config: E1Config) -> dict[str, Any]:
    """
    Collect every checkpoint file into one comparison.

    Returns
    -------
    summary : dict
        ``checkpoints`` keyed by name with ``benchmarks`` and ``judge`` sections when present,
        and ``missing`` lists of checkpoints without any file.
    """
    entries: dict[str, Any] = {}
    missing: list[str] = []
    for entry in config.checkpoints:
        path = checkpoint_path(results_root, entry.name)
        if not path.is_file():
            missing.append(entry.name)
            continue
        metrics = read_json(path)["metrics"]
        entries[entry.name] = {
            section: metrics[section] for section in SECTIONS if section in metrics
        }
    if not entries:
        raise FileNotFoundError("no checkpoint result files found")
    return {"checkpoints": entries, "missing": missing}


def write_summary(
    results_root: str | Path,
    config: E1Config,
    config_ref_value: ConfigRef,
    hardware: HardwareRecord,
    packages: dict[str, str],
) -> ResultsFile:
    """Build and write ``results/e1_summary.json``."""
    summary = build_summary(results_root, config)
    result = build_results(
        "e1_summary",
        summary,
        config_ref_value,
        hardware,
        packages,
        notes="aggregated from results/e1",
    )
    write_results(Path(results_root) / SUMMARY_NAME, result)
    return result
