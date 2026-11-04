"""Distillation experiment summary: win rates, retention, benchmarks, and GPU-hours."""

from pathlib import Path
from typing import Any

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import read_json
from llmplatform.common.results import ResultsFile, build_results, write_results
from llmplatform.eval.e1 import checkpoint_path, pairwise_path

# ============= Constants =============

E3_RESULT = "e3_distillation.json"
E4_RESULT = "e4_serving.json"
ROLE_CHECKPOINTS = {
    "teacher": "teacher",
    "base_student": "base_1_5b",
    "sft_baseline_student": "sft_baseline_1_5b",
    "distilled_student": "distilled_1_5b",
}


# ============= Helpers =============


def retention(student_win_rate: float, teacher_win_rate: float) -> float:
    """Return the student's win rate divided by the teacher's."""
    if teacher_win_rate <= 0:
        raise ValueError("teacher win rate must be positive to compute retention")
    return student_win_rate / teacher_win_rate


def compact_judge(section: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields of a judge section that the summary reports."""
    keys = ("reference", "n", "wins", "losses", "ties", "win_rate", "ci95_low", "ci95_high")
    summary = {key: section[key] for key in keys if key in section}
    summary["mean_length_words"] = section["candidate_length"]["mean_words"]
    return summary


def compact_benchmarks(section: dict[str, Any]) -> dict[str, float]:
    """Return ``{task: value}`` from a benchmarks section."""
    return {name: item["value"] for name, item in section["tasks"].items()}


def gpu_hours_from(path: Path, key: str = "gpu_hours") -> float | None:
    """Read GPU-hours from a ledger summary file when it exists."""
    if not path.is_file():
        return None
    data = read_json(path)
    if "segments" in data:
        return float(sum((s["last"] - s["start"]) * s["gpus"] / 3600.0 for s in data["segments"]))
    return float(data[key]) if key in data else None


# ============= Summary =============


def build_e3(
    results_root: str | Path,
    ckpt_root: str | Path,
    data_root: str | Path,
    distill_subdir: str = "distill",
    sft_subdir: str = "sft_baseline",
) -> dict[str, Any]:
    """
    Assemble the distillation metrics from checkpoint files, pairwise results, and ledgers.

    Parameters
    ----------
    results_root : str or Path
        Directory holding ``e1/`` and ``e3/`` result files.
    ckpt_root : str or Path
        Checkpoint root holding the distillation and SFT baseline ledgers.
    data_root : str or Path
        Data root holding the teacher generation summary.

    Returns
    -------
    metrics : dict
        Per-role win rates and benchmarks, retention, the direct pairwise result, GPU-hours of
        each stage, and the student-to-teacher cost ratio when serving results exist.
    """
    roles: dict[str, Any] = {}
    for role, checkpoint in ROLE_CHECKPOINTS.items():
        path = checkpoint_path(results_root, checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"missing result file for {checkpoint}: {path}")
        metrics = read_json(path)["metrics"]
        roles[role] = {
            "checkpoint": checkpoint,
            "judge": compact_judge(metrics["judge"]) if "judge" in metrics else None,
            "benchmarks": compact_benchmarks(metrics["benchmarks"])
            if "benchmarks" in metrics
            else None,
        }
    teacher_wr = roles["teacher"]["judge"]["win_rate"]
    retention_by_role = {
        role: retention(roles[role]["judge"]["win_rate"], teacher_wr)
        for role in ("base_student", "sft_baseline_student", "distilled_student")
    }
    pairwise_file = pairwise_path(results_root, "distilled_1_5b", "base_1_5b")
    pairwise = None
    if pairwise_file.is_file():
        pairwise = compact_judge(read_json(pairwise_file)["metrics"]["judge"])
    root = Path(ckpt_root)
    distill_hours = gpu_hours_from(root / distill_subdir / "ledger.json")
    sft_hours = gpu_hours_from(root / sft_subdir / "ledger.json")
    generation_summary = Path(data_root) / "teacher_completions.summary.json"
    metrics_out: dict[str, Any] = {
        "roles": roles,
        "retention_vs_teacher": retention_by_role,
        "pairwise_distilled_vs_base_student": pairwise,
        "gpu_hours": {
            "distillation": distill_hours,
            "sft_baseline": sft_hours,
            "teacher_generation": gpu_hours_from(generation_summary, "gpu_hours_this_run"),
        },
    }
    e4 = Path(results_root) / E4_RESULT
    if e4.is_file():
        ratio = read_json(e4)["metrics"].get("cost_ratio")
        metrics_out["student_to_teacher_cost_ratio"] = ratio
    return metrics_out


def write_e3(
    results_root: str | Path,
    ckpt_root: str | Path,
    data_root: str | Path,
    config: ConfigRef,
    hardware: HardwareRecord,
    packages: dict[str, str],
) -> ResultsFile:
    """Build and write ``results/e3_distillation.json``."""
    metrics = build_e3(results_root, ckpt_root, data_root)
    result = build_results(
        "e3_distillation", metrics, config, hardware, packages, notes="aggregated from results/e1"
    )
    write_results(Path(results_root) / E3_RESULT, result)
    return result
