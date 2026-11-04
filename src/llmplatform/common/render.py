"""Fill README and docs result blocks from results files; blocks stay empty until a file exists."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmplatform.common.io import read_json

# ============= Constants =============

BLOCK_PATTERN = re.compile(
    r"(?P<start><!-- results:(?P<name>[a-z0-9_]+):start -->)(?P<body>.*?)"
    r"(?P<end><!-- results:(?P=name):end -->)",
    re.DOTALL,
)
SCANNED_FILES = ("README.md", "docs/DATA.md")


# ============= Formatting =============


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def num(value: Any, digits: int = 3) -> str:
    """Format a number, or ``n/a`` for ``None``."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def caption(result: dict[str, Any]) -> str:
    """Return the hardware and code version line shown above every table."""
    hardware = result["environment"]["hardware"]
    gpu = (
        f"{hardware['gpu_count']} x {hardware['gpu_model']}"
        if hardware["gpu_count"]
        else "no GPU (CPU)"
    )
    return (
        f"Hardware: {gpu}, provider `{hardware['provider_label']}`. "
        f"Code revision `{result['git_sha']}`, produced {result['timestamp_utc']}."
    )


# ============= Renderers =============


def render_e1(result: dict[str, Any]) -> str:
    """Quality table: benchmarks and judge win rate for every checkpoint."""
    checkpoints = result["metrics"]["checkpoints"]
    tasks = sorted(
        {
            task
            for entry in checkpoints.values()
            for task in entry.get("benchmarks", {}).get("tasks", {})
        }
    )
    rows = []
    for name, entry in checkpoints.items():
        scores = entry.get("benchmarks", {}).get("tasks", {})
        judge = entry.get("judge")
        rows.append(
            [name]
            + [num(scores[task]["value"]) if task in scores else "n/a" for task in tasks]
            + [
                num(judge["win_rate"]) if judge else "n/a",
                f"{num(judge['ci95_low'])} to {num(judge['ci95_high'])}" if judge else "n/a",
                num(judge["candidate_length"]["mean_words"], 0) if judge else "n/a",
            ]
        )
    body = table(["Checkpoint", *tasks, "Win rate", "95% interval", "Mean words"], rows)
    return f"{caption(result)}\n\n{body}"


def render_e2(result: dict[str, Any]) -> str:
    """Scaling table."""
    points = result["metrics"]["points"]
    rows = [
        [
            str(point["gpus"]),
            num(point["tokens_per_second_total"], 0),
            num(point["tokens_per_second_per_gpu"], 0),
            num(point["scaling_efficiency"]),
            num(point["peak_memory_gib_per_gpu"], 1),
        ]
        for point in points.values()
    ]
    body = table(
        ["GPUs", "Tokens/s total", "Tokens/s per GPU", "Efficiency", "Peak GiB per GPU"], rows
    )
    return f"{caption(result)}\n\n{body}"


def render_e3(result: dict[str, Any]) -> str:
    """Distillation table with retention and GPU-hours."""
    metrics = result["metrics"]
    rows = []
    for role, entry in metrics["roles"].items():
        judge = entry["judge"]
        retention = metrics["retention_vs_teacher"].get(role)
        rows.append(
            [
                role,
                num(judge["win_rate"]) if judge else "n/a",
                f"{num(judge['ci95_low'])} to {num(judge['ci95_high'])}" if judge else "n/a",
                num(retention),
            ]
        )
    body = table(["Model", "Win rate vs reference", "95% interval", "Retention vs teacher"], rows)
    hours = metrics["gpu_hours"]
    lines = [
        caption(result),
        "",
        body,
        "",
        f"GPU-hours: distillation {num(hours['distillation'], 2)}, SFT baseline "
        f"{num(hours['sft_baseline'], 2)}, teacher generation "
        f"{num(hours['teacher_generation'], 2)}.",
    ]
    pairwise = metrics.get("pairwise_distilled_vs_base_student")
    if pairwise:
        lines.append(f"Distilled versus base student: win rate {num(pairwise['win_rate'])}.")
    ratio = metrics.get("student_to_teacher_cost_ratio")
    if ratio:
        lines.append(f"Student to teacher serving cost ratio: {num(ratio['student_to_teacher'])}.")
    return "\n".join(lines)


def render_e4(result: dict[str, Any]) -> str:
    """Serving table at the highest SLO-compliant concurrency, with cost."""
    metrics = result["metrics"]
    rows = []
    for model, engines in metrics["cost"].items():
        for engine, workloads in engines.items():
            for workload, summary in workloads.items():
                rows.append(
                    [
                        model,
                        engine,
                        workload,
                        num(summary.get("concurrency")),
                        num(summary.get("output_tokens_per_second"), 0),
                        num(summary["usd_per_million_output_tokens"], 3),
                    ]
                )
    slo = metrics["slo"]
    body = table(
        [
            "Model",
            "Engine",
            "Workload (in/out)",
            "SLO concurrency",
            "Output tokens/s",
            "USD per 1M output tokens",
        ],
        rows,
    )
    ratio = metrics["cost_ratio"]["student_to_teacher"]
    lines = [
        caption(result),
        "",
        f"SLO: p95 time to first token at most {num(slo['ttft_p95_ms'], 0)} ms and p95 "
        f"inter-token latency at most {num(slo['itl_p95_ms'], 0)} ms.",
        "",
        body,
        "",
        f"Student to teacher cost ratio: {num(ratio)}.",
    ]
    return "\n".join(lines)


def render_e5(result: dict[str, Any]) -> str:
    """Autoscaling summary."""
    metrics = result["metrics"]
    before, after = metrics.get("latency_before_scale_out"), metrics.get("latency_after_scale_out")
    rows = [
        ["Run label", str(metrics["label"])],
        ["Scale-out observed", num(metrics["scale_out_observed"])],
        [
            "Seconds from threshold breach to new pod Ready",
            num(metrics["seconds_from_breach_to_pod_ready"], 1),
        ],
        ["Peak ready replicas", num(metrics["peak_ready_replicas"])],
        ["Latency p50 before scale-out (ms)", num(before["p50_ms"], 0) if before else "n/a"],
        ["Latency p50 after scale-out (ms)", num(after["p50_ms"], 0) if after else "n/a"],
        ["Tokens per second", num(metrics["tokens_per_second"], 1)],
    ]
    return f"{caption(result)}\n\n{table(['Measure', 'Value'], rows)}"


def render_e6(result: dict[str, Any]) -> str:
    """Speculative decoding table."""
    rows = []
    for precision, configs in result["metrics"]["results"].items():
        for config, node in configs.items():
            verdict = node.get("equality", {}).get("passed")
            for concurrency, entry in node.get("by_concurrency", {}).items():
                rows.append(
                    [
                        precision,
                        config,
                        concurrency,
                        num(entry["accept_length_mean"], 2),
                        num(entry["output_throughput"], 0),
                        num(entry["speedup_vs_baseline"], 2),
                        num(verdict),
                    ]
                )
    body = table(
        [
            "Precision",
            "Configuration",
            "Concurrency",
            "Accept length",
            "Tokens/s",
            "Speedup",
            "Outputs identical",
        ],
        rows,
    )
    return f"{caption(result)}\n\n{body}"


def render_e7(result: dict[str, Any]) -> str:
    """Disaggregation table."""
    rows = []
    for setup, node in result["metrics"]["setups"].items():
        for rate, entry in node["by_rate"].items():
            rows.append(
                [
                    setup,
                    rate,
                    num(entry["chat_ttft_p95_ms"], 0),
                    num(entry["long_ttft_p95_ms"], 0),
                    num(entry["itl_p95_ms"], 1),
                    num(entry["output_throughput"], 0),
                    num(entry["slo_met"]),
                ]
            )
    body = table(
        [
            "Setup",
            "Rate (req/s)",
            "Chat TTFT p95 (ms)",
            "Long TTFT p95 (ms)",
            "ITL p95 (ms)",
            "Tokens/s",
            "SLO met",
        ],
        rows,
    )
    breaks = ", ".join(
        f"{setup}: {num(node['breaking_rate'])}"
        for setup, node in result["metrics"]["setups"].items()
    )
    note = f"Lowest tested rate that broke the SLO (n/a means none): {breaks}."
    return f"{caption(result)}\n\n{body}\n\n{note}"


def render_e8(result: dict[str, Any]) -> str:
    """Agent workload table."""
    rows = []
    for model, configs in result["metrics"]["runs"].items():
        for config, levels in configs.items():
            for concurrency, entry in levels.items():
                rows.append(
                    [
                        model,
                        config,
                        concurrency,
                        num(entry["prefix_cache_hit_rate"]),
                        num(entry["ttft_p50_ms"], 0),
                        num(entry["ttft_p95_ms"], 0),
                        num(entry["output_throughput"], 0),
                    ]
                )
    body = table(
        [
            "Model",
            "Configuration",
            "Concurrency",
            "Cache hit rate",
            "TTFT p50 (ms)",
            "TTFT p95 (ms)",
            "Tokens/s",
        ],
        rows,
    )
    return f"{caption(result)}\n\n{body}"


def render_data(result: dict[str, Any]) -> str:
    """Row counts per dataset before and after filtering."""
    metrics = result["metrics"]
    rows = []
    for name, entry in metrics.items():
        if not isinstance(entry, dict) or "hub_id" not in entry:
            continue
        before = entry.get("rows_before", entry.get("rows"))
        after = entry.get("rows_after", entry.get("selected"))
        rows.append([name, f"`{entry['hub_id']}`", num(before), num(after)])
    return f"{caption(result)}\n\n{table(['Set', 'Hub id', 'Rows before', 'Rows after'], rows)}"


def render_headline(results_root: Path) -> str:
    """Compact list of the main numbers from whichever results exist."""
    lines = []
    e3 = results_root / "e3_distillation.json"
    if e3.is_file():
        metrics = read_json(e3)["metrics"]
        retained = num(metrics["retention_vs_teacher"]["distilled_student"])
        lines.append(f"- Distilled student retains {retained} of the teacher's win rate.")
        ratio = metrics.get("student_to_teacher_cost_ratio")
        if ratio:
            lines.append(
                f"- Student serving cost is {num(ratio['student_to_teacher'])} of the teacher's."
            )
    e2 = results_root / "e2_scaling.json"
    if e2.is_file():
        points = read_json(e2)["metrics"]["points"]
        top = max(points.values(), key=lambda point: point["gpus"])
        lines.append(
            f"- Scaling efficiency at {top['gpus']} GPUs: {num(top['scaling_efficiency'])}."
        )
    e6 = results_root / "e6_speculative.json"
    if e6.is_file():
        best = max(
            (
                entry["speedup_vs_baseline"]
                for configs in read_json(e6)["metrics"]["results"].values()
                for node in configs.values()
                for entry in node.get("by_concurrency", {}).values()
                if entry["speedup_vs_baseline"]
            ),
            default=None,
        )
        if best is not None:
            lines.append(f"- Best speculative decoding speedup: {num(best, 2)}x.")
    return "\n".join(lines)


# ============= Registry =============


@dataclass(frozen=True)
class Block:
    """A results-backed README block."""

    filename: str
    render: Callable[[dict[str, Any]], str]


BLOCKS: dict[str, Block] = {
    "e1": Block("e1_summary.json", render_e1),
    "e2": Block("e2_scaling.json", render_e2),
    "e3": Block("e3_distillation.json", render_e3),
    "e4": Block("e4_serving.json", render_e4),
    "e5_cpu": Block("e5_k8s_cpu.json", render_e5),
    "e5_gpu": Block("e5_k8s_gpu.json", render_e5),
    "e6": Block("e6_speculative.json", render_e6),
    "e7": Block("e7_disaggregation.json", render_e7),
    "e8": Block("e8_agent_workloads.json", render_e8),
    "data_manifest": Block("data_manifest.json", render_data),
}
HEADLINE = "headline"


def render_block(name: str, results_root: Path) -> str:
    """Return the Markdown of a block, or an empty string when its results file is absent."""
    if name == HEADLINE:
        return render_headline(results_root)
    block = BLOCKS[name]
    path = results_root / block.filename
    return block.render(read_json(path)) if path.is_file() else ""


def known_block_names() -> list[str]:
    """All block names a README may contain."""
    return [HEADLINE, *BLOCKS]


# ============= Processing =============


def render_text(text: str, results_root: Path) -> str:
    """
    Replace every results block in ``text`` with its current rendering.

    Blocks without results become empty (start marker, newline, end marker).
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in known_block_names():
            raise KeyError(f"unknown results block {name!r}")
        body = render_block(name, results_root)
        inner = f"\n\n{body}\n\n" if body else "\n"
        return f"{match.group('start')}{inner}{match.group('end')}"

    return BLOCK_PATTERN.sub(replace, text)


def process(root: Path, results_root: Path, o_check: bool) -> list[str]:
    """
    Render every scanned file and either write it or report differences.

    Returns
    -------
    problem_list : list[str]
        Files whose blocks differ from the rendering, results files that no block displays,
        and unknown block names. Empty when everything is current.
    """
    problem_list: list[str] = []
    seen: set[str] = set()
    for relative in SCANNED_FILES:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        seen.update(match.group("name") for match in BLOCK_PATTERN.finditer(text))
        try:
            updated = render_text(text, results_root)
        except KeyError as exc:
            problem_list.append(f"{relative}: {exc}")
            continue
        if updated != text:
            if o_check:
                problem_list.append(f"{relative}: results blocks are out of date")
            else:
                path.write_text(updated, encoding="utf-8")
    for name, block in BLOCKS.items():
        if (results_root / block.filename).is_file() and name not in seen:
            problem_list.append(
                f"results/{block.filename} has no '{name}' block in the README or docs"
            )
    return problem_list
