"""Build the SFT, preference, distillation prompt, and judge prompt sets with decontamination."""

import argparse
import logging
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from llmplatform.common.compute import hardware_record, load_compute, resolve_work_root
from llmplatform.common.config import config_ref
from llmplatform.common.hardware import package_versions
from llmplatform.common.hashing import sha256_file
from llmplatform.common.io import read_jsonl, write_jsonl
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.results import build_results, write_results
from llmplatform.data import sources
from llmplatform.data.config import DataConfig, DatasetSpec, load_data_config
from llmplatform.data.dedup import (
    FilterStats,
    NearDuplicateConfig,
    ReferenceIndex,
    exact_key,
)
from llmplatform.data.formats import (
    last_user_text,
    preference_record,
    prompt_record,
    record_text,
    sft_record,
)

logger = logging.getLogger(__name__)

# ============= Constants =============

RECORD_BUILDERS: dict[str, Callable[[dict[str, Any], str], dict[str, Any] | None]] = {
    "sft": sft_record,
    "preferences": preference_record,
    "distillation_prompts": prompt_record,
}
OUTPUT_NAMES = {
    "sft": "sft.jsonl",
    "preferences": "preferences.jsonl",
    "distillation_prompts": "distillation_prompts.jsonl",
}
MANIFEST_RESULT_NAME = "data_manifest"
RECORD_PACKAGES = ["datasets", "numpy"]


# ============= Reference index =============


def build_reference_index(
    data_config: DataConfig, judge_prompts: Iterable[str], benchmark_text_list: Iterable[str]
) -> ReferenceIndex:
    """Index judge prompts and benchmark questions for decontamination."""
    settings = data_config.decontamination.near_duplicate
    index = ReferenceIndex(
        NearDuplicateConfig(
            num_perm=settings.num_perm,
            rows_per_band=settings.rows_per_band,
            shingle_words=settings.shingle_words,
            threshold=settings.threshold,
            seed=data_config.seed,
        )
    )
    index.add_many(judge_prompts)
    index.add_many(benchmark_text_list)
    return index


# ============= Selection =============


def select_records(
    rows: Iterable[dict[str, Any]],
    kind: str,
    max_rows: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    index: ReferenceIndex,
) -> tuple[list[dict[str, Any]], list[FilterStats]]:
    """
    Convert, filter, and truncate a stream of rows to at most ``max_rows`` records.

    Parameters
    ----------
    rows : iterable
        Raw dataset rows in the order they should be considered, already seeded.
    kind : str
        One of ``sft``, ``preferences``, ``distillation_prompts``.
    max_rows : int
        Stop once this many records survive every filter.
    max_tokens : int
        Maximum tokens of the full record text (prompt plus completions).
    count_tokens : callable
        Token counter for a string.
    index : ReferenceIndex
        Decontamination reference.

    Returns
    -------
    records : list[dict]
        Surviving records.
    stats_list : list[FilterStats]
        One entry per filter with counts.
    """
    builder = RECORD_BUILDERS[kind]
    convert = FilterStats(name="convert")
    length = FilterStats(name="token_length")
    contam = FilterStats(name="decontaminate")
    uniq = FilterStats(name="unique")
    seen_texts: set[str] = set()
    records: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if len(records) >= max_rows:
            break
        convert.seen += 1
        record = builder(row, f"{kind}-{position}")
        if record is None:
            convert.drop("invalid")
            continue
        convert.kept += 1
        length.seen += 1
        if count_tokens(_full_text(record)) > max_tokens:
            length.drop("too_long")
            continue
        length.kept += 1
        contam.seen += 1
        match = index.match(record_text(record))
        if match:
            contam.drop(match)
            continue
        contam.kept += 1
        uniq.seen += 1
        key = record_text(record)
        digest = exact_key(key)
        if digest in seen_texts:
            uniq.drop("exact")
            continue
        seen_texts.add(digest)
        uniq.kept += 1
        records.append(record)
    return records, [convert, length, contam, uniq]


def _full_text(record: dict[str, Any]) -> str:
    """Concatenate every message of a record for length counting."""
    parts: list[str] = []
    for key in ("messages", "prompt", "chosen", "rejected"):
        value = record.get(key)
        if isinstance(value, list):
            parts.extend(item["content"] for item in value)
    return "\n".join(parts)


def select_judge_prompts(
    rows: Iterable[dict[str, Any]],
    n_prompts: int,
    max_prompt_tokens: int,
    seed: int,
    count_tokens: Callable[[str], int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Choose judge prompts from a held-out preference split.

    Eligible rows are single-turn prompts within the token limit and unique by text. The
    sample is drawn with a seeded generator, so reruns select the same prompts.

    Returns
    -------
    prompts : list[dict]
        Records with ``id``, ``prompt_id``, and ``prompt`` (plain text).
    counts : dict[str, int]
        ``rows``, ``eligible``, and ``selected``.
    """
    row_list = list(rows)
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for row in row_list:
        text = last_user_text(row.get("prompt", "")).strip()
        if not text or text in seen or count_tokens(text) > max_prompt_tokens:
            continue
        seen.add(text)
        eligible.append({"prompt_id": str(row.get("prompt_id", "")), "prompt": text})
    if len(eligible) < n_prompts:
        raise ValueError(f"only {len(eligible)} eligible prompts, need {n_prompts}")
    chosen = random.Random(seed).sample(eligible, n_prompts)
    prompts = [{"id": f"judge-{i:03d}", **item} for i, item in enumerate(chosen)]
    return prompts, {"rows": len(row_list), "eligible": len(eligible), "selected": n_prompts}


# ============= Pipeline =============


def build_judge_file(
    data_config: DataConfig, count_tokens: Callable[[str], int], output: Path
) -> dict[str, int]:
    """Write the judge prompt file and return selection counts."""
    spec = data_config.judge
    dataset = sources.load_split(spec.hub_id, spec.split, revision=spec.revision)
    prompts, counts = select_judge_prompts(
        dataset, spec.n_prompts, spec.max_prompt_tokens, data_config.seed, count_tokens
    )
    write_jsonl(output, prompts)
    return counts


def shuffled_rows(dataset: Any, seed: int) -> Iterable[dict[str, Any]]:
    """Yield dataset rows in a seeded random order."""
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    return sources.iter_rows(dataset, order)


def build_training_set(
    kind: str,
    spec: DatasetSpec,
    seed: int,
    count_tokens: Callable[[str], int],
    index: ReferenceIndex,
    output: Path,
) -> dict[str, Any]:
    """Build one training set and return its manifest entry."""
    dataset = sources.load_split(spec.hub_id, spec.split, revision=spec.revision)
    records, stats_list = select_records(
        shuffled_rows(dataset, seed), kind, spec.max_rows, spec.max_tokens, count_tokens, index
    )
    write_jsonl(output, records)
    return {
        "hub_id": spec.hub_id,
        "revision": spec.revision,
        "split": spec.split,
        "license": spec.license,
        "rows_before": len(dataset),
        "rows_after": len(records),
        "filters": [stats.as_dict() for stats in stats_list],
        "output_sha256": sha256_file(output),
    }


def run(o_skip_training: bool = False) -> int:
    """Prepare every dataset and write the manifest results file."""
    configure_logging("prepare_data")
    root = repo_root()
    compute = load_compute()
    data_config = load_data_config()
    models = load_models()
    out_dir = resolve_work_root(compute) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_key = models.roles["tokenizer_source"]
    count_tokens = sources.tokenizer_counter(
        resolve_model(models, tokenizer_key, out_dir), revision_of(models, tokenizer_key)
    )
    judge_path = root / data_config.judge.output
    judge_counts = build_judge_file(data_config, count_tokens, judge_path)
    manifest: dict[str, Any] = {
        "judge_prompts": {
            "hub_id": data_config.judge.hub_id,
            "revision": data_config.judge.revision,
            "split": data_config.judge.split,
            "license": data_config.judge.license,
            **judge_counts,
            "output_sha256": sha256_file(judge_path),
        }
    }
    if not o_skip_training:
        judge_texts = [row["prompt"] for row in read_jsonl(judge_path)]
        bench_texts: list[str] = []
        benchmark_counts: dict[str, int] = {}
        for source in data_config.decontamination.benchmark_sources:
            texts = sources.benchmark_texts(source)
            benchmark_counts[source.name] = len(texts)
            bench_texts.extend(texts)
        index = build_reference_index(data_config, judge_texts, bench_texts)
        manifest["reference_sizes"] = {"judge_prompts": len(judge_texts), **benchmark_counts}
        for kind in RECORD_BUILDERS:
            spec = getattr(data_config.training, kind)
            manifest[kind] = build_training_set(
                kind, spec, data_config.seed, count_tokens, index, out_dir / OUTPUT_NAMES[kind]
            )
    if o_skip_training:
        logger.info("wrote %s without a manifest", judge_path)
        return 0
    result = build_results(
        MANIFEST_RESULT_NAME,
        manifest,
        config_ref(root / "configs" / "data.yaml", root),
        hardware_record(compute),
        package_versions(RECORD_PACKAGES),
        notes="rows_before is the split size; filters list rows seen and dropped in order",
    )
    path = write_results(results_dir(root) / f"{MANIFEST_RESULT_NAME}.json", result)
    logger.info("wrote %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Prepare datasets with decontamination.")
    parser.add_argument(
        "--judge-only", action="store_true", help="write only eval/judge_prompts.jsonl"
    )
    args = parser.parse_args(argv)
    return run(o_skip_training=args.judge_only)


if __name__ == "__main__":
    raise SystemExit(main())
