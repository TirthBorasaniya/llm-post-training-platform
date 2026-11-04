"""Render Hub model cards from results files; sections without results are omitted."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmplatform.common.io import read_json
from llmplatform.common.models import ModelsConfig
from llmplatform.data.config import DataConfig

# ============= Constants =============

REQUIRED_LICENSE = "apache-2.0"
BENCH_ORDER = ["ifeval", "gsm8k", "mmlu", "arc_challenge", "hellaswag", "truthfulqa_mc2"]
TEACHER_ROWS = ["base_7b", "sft", "dpo", "orpo", "simpo", "teacher"]
STUDENT_ROWS = ["base_1_5b", "sft_baseline_1_5b", "distilled_1_5b"]


# ============= Helpers =============


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a number, or ``n/a`` when absent."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_checkpoint_metrics(results_root: Path, name: str) -> dict[str, Any] | None:
    """Return the metrics of ``results/e1/<name>.json`` or ``None`` when it does not exist."""
    path = results_root / "e1" / f"{name}.json"
    return dict(read_json(path)["metrics"]) if path.is_file() else None


def quality_table(results_root: Path, names: list[str]) -> str | None:
    """
    Build the benchmark and win-rate table for a list of checkpoints.

    Returns ``None`` when no listed checkpoint has any result.
    """
    rows, found = [], False
    for name in names:
        metrics = load_checkpoint_metrics(results_root, name)
        if metrics is None:
            continue
        found = True
        tasks = metrics.get("benchmarks", {}).get("tasks", {})
        judge = metrics.get("judge")
        win = fmt(judge["win_rate"]) if judge else "n/a"
        interval = f"{fmt(judge['ci95_low'])} to {fmt(judge['ci95_high'])}" if judge else "n/a"
        length = fmt(judge["candidate_length"]["mean_words"], 0) if judge else "n/a"
        rows.append(
            [name]
            + [fmt(tasks[task]["value"]) if task in tasks else "n/a" for task in BENCH_ORDER]
            + [win, interval, length]
        )
    if not found:
        return None
    headers = ["Checkpoint", *BENCH_ORDER, "Win rate", "95% interval", "Mean words"]
    return markdown_table(headers, rows)


# ============= Cards =============


@dataclass(frozen=True)
class CardContext:
    """Inputs shared by every card."""

    kind: str
    repo_id: str
    base_model_id: str
    base_model_license: str
    models: ModelsConfig
    data: DataConfig
    results_root: Path
    gguf_files: list[str] | None = None


def front_matter(ctx: CardContext, library: str) -> str:
    """Return the YAML header of a card."""
    datasets = [
        ctx.data.training.sft.hub_id,
        ctx.data.training.preferences.hub_id,
        ctx.data.training.distillation_prompts.hub_id,
    ]
    lines = [
        "---",
        f"license: {REQUIRED_LICENSE}",
        f"base_model: {ctx.base_model_id}",
        f"library_name: {library}",
        "pipeline_tag: text-generation",
        "language:",
        "- en",
        "datasets:",
        *[f"- {name}" for name in sorted(set(datasets))],
        "tags:",
        "- llm-post-training-platform",
        "---",
    ]
    return "\n".join(lines)


def training_section(ctx: CardContext) -> str:
    """List the datasets and their pinned revisions."""
    items = [
        ("supervised fine-tuning", ctx.data.training.sft),
        ("preference optimization", ctx.data.training.preferences),
        ("distillation prompts", ctx.data.training.distillation_prompts),
    ]
    lines = ["## Data", ""]
    for use, spec in items:
        lines.append(
            f"- {use}: `{spec.hub_id}` at revision `{spec.revision}` (split `{spec.split}`)"
        )
    lines.append(
        "- All training and prompt sets are decontaminated against the judge prompts and the "
        "benchmark questions with exact and MinHash near-duplicate matching."
    )
    return "\n".join(lines)


def provenance_section(results_root: Path, names: list[str]) -> str | None:
    """Report the code version of the newest listed result file."""
    for name in names:
        metrics = load_checkpoint_metrics(results_root, name)
        if metrics and "provenance" in metrics:
            first = next(iter(metrics["provenance"].values()))
            return f"## Provenance\n\nResults were produced by code revision `{first['git_sha']}`."
    return None


LIMITATIONS = (
    "## Intended use and limitations\n\n"
    "These checkpoints are research artifacts of an open post-training pipeline. They inherit the "
    "limitations of the base model and of the public datasets listed above, can produce incorrect "
    "or unsafe text, and were evaluated on the benchmarks and the 300 judge prompts reported here "
    "only. The judge is a language model and its preferences are an estimate of quality."
)


def render_model_card(ctx: CardContext) -> str:
    """
    Render the card for a published repository.

    Parameters
    ----------
    ctx : CardContext
        Repository kind (``teacher``, ``student``, or ``gguf``), base model, and results location.

    Returns
    -------
    card : str
        Markdown with a YAML header. Every number comes from a results file.
    """
    if ctx.base_model_license != REQUIRED_LICENSE:
        raise ValueError(
            f"base model license is {ctx.base_model_license!r}; refusing to render a card "
            f"that declares {REQUIRED_LICENSE}"
        )
    results = ctx.results_root
    if ctx.kind == "teacher":
        names, library = TEACHER_ROWS, "transformers"
        intro = (
            f"`{ctx.repo_id}` is `{ctx.base_model_id}` aligned with QLoRA supervised fine-tuning "
            "followed by direct preference optimization, merged into a bf16 checkpoint."
        )
        heading = "## Evaluation of the alignment objectives"
    elif ctx.kind == "student":
        names, library = STUDENT_ROWS, "transformers"
        intro = (
            f"`{ctx.repo_id}` is `{ctx.base_model_id}` distilled from the aligned 7B teacher with "
            "on-policy reverse-KL distillation."
        )
        heading = "## Evaluation against the compute-matched SFT baseline"
    elif ctx.kind == "gguf":
        names, library = ["teacher_q4km", "distilled_1_5b_q4km", "base_1_5b_q4km"], "gguf"
        intro = f"`{ctx.repo_id}` holds Q4_K_M GGUF files for CPU inference with llama.cpp."
        heading = "## Evaluation of the quantized variants"
    else:
        raise ValueError(f"unknown card kind {ctx.kind!r}")

    parts = [front_matter(ctx, library), f"# {ctx.repo_id}", intro]
    table = quality_table(results, names)
    if table:
        note = (
            "Win rate is measured against the base 7B model's responses on 300 held-out prompts "
            "with a position-swapped LLM judge; ties count as half."
        )
        parts += [heading, note, table]
    if ctx.kind == "gguf" and ctx.gguf_files:
        parts += ["## Files", "\n".join(f"- `{name}`" for name in ctx.gguf_files)]
        parts.append(
            '## Usage\n\n```bash\nllama cli -m <file>.gguf -p "Hello" -n 64\n```\n\n'
            "Older llama.cpp builds name the same tool `llama-cli`."
        )
    if ctx.kind != "gguf":
        parts.append(training_section(ctx))
    provenance = provenance_section(results, names)
    if provenance:
        parts.append(provenance)
    parts += [LIMITATIONS, f"## License\n\nApache 2.0, inherited from `{ctx.base_model_id}`."]
    return "\n\n".join(parts) + "\n"
