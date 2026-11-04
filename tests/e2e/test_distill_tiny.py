"""Distillation and the SFT baseline on tiny CPU models with mismatched padded vocabularies."""

from pathlib import Path

import pytest

from llmplatform.common.checkpoints import is_complete
from llmplatform.common.io import read_json, write_jsonl
from llmplatform.distill.config import load_opd, load_sft_baseline
from llmplatform.distill.opd import run_opd
from llmplatform.distill.sft_baseline import run_sft_baseline
from llmplatform.smoke.synthetic import sft_rows
from llmplatform.smoke.tiny import TEACHER_PADDED_VOCAB, build_tiny_model

pytestmark = pytest.mark.slow

OPD_OVERRIDES = [
    "vllm.enabled=false",
    "tracking.enabled=false",
    "optim.per_device_train_batch_size=2",
    "optim.gradient_accumulation_steps=1",
    "optim.gradient_checkpointing=false",
    "optim.warmup_ratio=0.0",
    "generation.max_completion_length=6",
    "clipped_ratio.max=1.0",
    "checkpointing.logging_steps=1",
    "vocab_check.prompts=4",
    "vocab_check.max_length=48",
    "data.max_prompts=4",
]


def test_opd_aligns_vocab_trains_and_records_gpu_hours(
    tiny_student: Path, tokenizer, data_root: Path, tmp_path: Path
) -> None:
    teacher = build_tiny_model(
        tmp_path / "teacher",
        TEACHER_PADDED_VOCAB,
        64,
        o_tie_embeddings=False,
        seed=1,
        tokenizer=tokenizer,
    )
    config = load_opd(override_list=OPD_OVERRIDES)
    out = tmp_path / "distill"
    final = run_opd(
        config,
        str(teacher),
        None,
        str(tiny_student),
        None,
        data_root / "distillation_prompts.jsonl",
        out,
        o_cpu=True,
    )
    assert (final / "config.json").is_file()
    assert is_complete(out)
    check = read_json(out / "vocab_check_run.json")
    assert check["passed"] and check["resized_teacher"]
    assert check["teacher_padded_size"] == TEACHER_PADDED_VOCAB
    assert check["teacher_padded_size_after"] == 151936
    summary = read_json(out / "distill_run.json")
    assert summary["ledger"]["segments"] == 1 and summary["ledger"]["gpu_hours"] > 0
    assert (
        run_opd(
            config,
            str(teacher),
            None,
            str(tiny_student),
            None,
            data_root / "distillation_prompts.jsonl",
            out,
            o_cpu=True,
        )
        == final
    )


def test_sft_baseline_stops_at_budget(tiny_student: Path, data_root: Path, tmp_path: Path) -> None:
    write_jsonl(tmp_path / "completions.jsonl", sft_rows(8))
    config = load_sft_baseline(
        override_list=[
            "tracking.enabled=false",
            "optim.per_device_train_batch_size=2",
            "optim.gradient_accumulation_steps=1",
            "optim.gradient_checkpointing=false",
            "optim.warmup_ratio=0.0",
            "optim.num_train_epochs=500",
            "lengths.max_length=64",
            "checkpointing.logging_steps=1",
        ]
    )
    budget_hours = 3.0 / 3600.0  # three seconds of one CPU "GPU"
    out = tmp_path / "sft_baseline"
    final = run_sft_baseline(
        config,
        str(tiny_student),
        None,
        tmp_path / "completions.jsonl",
        out,
        budget_hours,
        o_cpu=True,
    )
    summary = read_json(out / "sft_baseline_run.json")
    assert (final / "config.json").is_file()
    assert summary["budget_reached"] is True
    assert summary["global_step"] >= 1
    assert summary["ledger"]["gpu_hours"] >= budget_hours
