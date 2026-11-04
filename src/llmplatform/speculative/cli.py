"""Command-line entry point for draft export, speculative traffic runs, and the E6 summary."""

import argparse
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from llmplatform.common.compute import hardware_record, load_compute
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.io import read_json, read_jsonl, write_json, write_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.speculative.config import load_speculative
from llmplatform.speculative.e6 import BASELINE, E6_RESULT, build_e6, config_name
from llmplatform.speculative.equality import compare_outputs

logger = logging.getLogger(__name__)

# ============= Constants =============

PACKAGES = ["sglang", "torch", "transformers", "httpx"]
ROWS_FILE = "spec_cells.jsonl"
OUTPUT_DIR = "spec_outputs"


def _roots() -> tuple[Path, Path]:
    """Return ``(ckpt_root, data_root)`` from the job environment."""
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    return Path(os.environ.get("CKPT_ROOT", work / "checkpoints")), Path(
        os.environ.get("DATA_ROOT", work / "data")
    )


def _prompts(count: int | None = None) -> list[str]:
    config = load_speculative()
    rows = read_jsonl(repo_root() / config.traffic.prompts_file)
    texts = [row["prompt"] for row in rows]
    return texts[:count] if count else texts


# ============= Commands =============


def cmd_export_draft(args: argparse.Namespace) -> int:
    """Pad a draft model to the target's embedding size and save it."""
    from transformers import AutoTokenizer

    from llmplatform.speculative.export_draft import export_draft

    ckpt_root, _ = _roots()
    config, models = load_speculative(), load_models()
    target_path = resolve_model(models, config.target, ckpt_root)
    tokenizer = AutoTokenizer.from_pretrained(
        target_path, revision=revision_of(models, config.target)
    )
    tokenizer.padding_side = "left"
    prompts = [
        str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
            )
        )
        for text in _prompts(config.equality.prompts)
    ]
    export_draft(
        resolve_model(models, args.draft, ckpt_root),
        revision_of(models, args.draft),
        target_path,
        revision_of(models, config.target),
        ckpt_root / config.output_subdir / f"{args.draft}_padded",
        prompts,
        args.cpu,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Measure one server configuration: greedy outputs, then traffic at each concurrency."""
    from llmplatform.serve.engines import ServerSpec
    from llmplatform.speculative.run import collect_greedy_texts, run_traffic

    _, data_root = _roots()
    config = load_speculative()
    spec = ServerSpec("sglang", "", port=args.port)
    name = BASELINE if args.draft is None else config_name(args.draft, args.draft_tokens)
    outputs = collect_greedy_texts(
        spec.openai_url(),
        args.served_model_name,
        _prompts(config.equality.prompts),
        config.equality.max_tokens,
    )
    out_dir = data_root / "bench" / OUTPUT_DIR / args.precision
    write_json(out_dir / f"{name}.json", {"texts": outputs})
    prompts = _prompts()
    traffic = config.traffic
    extra = {"hardware": hardware_record(load_compute()).model_dump(mode="json")}
    path = data_root / "bench" / ROWS_FILE
    rows = read_jsonl(path) if path.is_file() else []
    for concurrency in config.concurrency:
        for repetition in range(traffic.repetitions):
            offset = (repetition * traffic.requests_per_cell) % max(1, len(prompts))
            window = (prompts + prompts)[offset : offset + traffic.requests_per_cell]
            summary = run_traffic(
                spec.openai_url(),
                spec.metrics_url(),
                args.served_model_name,
                window,
                concurrency,
                traffic.max_tokens,
                traffic.temperature,
            )
            rows.append(
                {
                    "precision": args.precision,
                    "config": name,
                    "draft": args.draft,
                    "draft_tokens": args.draft_tokens,
                    "repetition": repetition,
                    "outputs_sha256": hashlib.sha256("\n".join(outputs).encode()).hexdigest(),
                    **summary,
                    **extra,
                }
            )
            write_jsonl(path, rows)
    return 0


def cmd_e6(args: argparse.Namespace) -> int:
    """Compare outputs, aggregate rows, and write ``results/e6_speculative.json``."""
    from llmplatform.bench.e4 import hardware_from_rows

    _, data_root = _roots()
    rows = read_jsonl(data_root / "bench" / ROWS_FILE)
    outputs_root = data_root / "bench" / OUTPUT_DIR
    equality: dict[str, dict[str, Any]] = {}
    for precision_dir in sorted(outputs_root.iterdir()):
        baseline_file = precision_dir / f"{BASELINE}.json"
        if not baseline_file.is_file():
            continue
        baseline = read_json(baseline_file)["texts"]
        for candidate_file in sorted(precision_dir.glob("*.json")):
            if candidate_file.stem == BASELINE:
                continue
            report = compare_outputs(baseline, read_json(candidate_file)["texts"])
            equality[f"{precision_dir.name}/{candidate_file.stem}"] = report.as_dict()
    metrics = build_e6(rows, equality)
    compute = load_compute()
    hardware = hardware_from_rows(rows).model_copy(
        update={
            "gpu_hourly_usd": compute.gpu_hourly_usd,
            "price_source": compute.price_source,
            "price_date": compute.price_date,
            "provider_label": compute.provider_label,
        }
    )
    root = repo_root()
    result = build_results(
        "e6_speculative",
        metrics,
        config_ref(root / "configs/serve/speculative.yaml", root),
        hardware,
        package_versions(PACKAGES),
        notes="outputs must be identical to non-speculative decoding at temperature 0",
    )
    write_results(results_dir(root) / E6_RESULT, result)
    if not metrics["equality_passed"]:
        logger.error(
            "greedy outputs differ from the baseline",
            extra={"configs": metrics["equality_failures"]},
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``scripts/speculative.py``."""
    configure_logging("speculative")
    parser = argparse.ArgumentParser(description="Speculative decoding commands.")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-draft")
    export.add_argument("--draft", required=True)
    export.add_argument("--cpu", action="store_true")
    export.set_defaults(func=cmd_export_draft)
    run = sub.add_parser("run")
    run.add_argument("--precision", choices=["bf16", "fp8"], required=True)
    run.add_argument("--draft", default=None, help="omit for the non-speculative baseline")
    run.add_argument("--draft-tokens", type=int, default=None)
    run.add_argument("--served-model-name", required=True)
    run.add_argument("--port", type=int, default=None)
    run.set_defaults(func=cmd_run)
    sub.add_parser("e6").set_defaults(func=cmd_e6)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
