"""Tests for the vocabulary check, accounting, distillation arguments, and E3 summary."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import read_jsonl, write_json, write_jsonl
from llmplatform.common.results import build_results, load_results, write_results
from llmplatform.distill.accounting import (
    BudgetCallback,
    ClippedRatioError,
    ClippedRatioGuard,
    GpuLedger,
    LedgerCallback,
    TimedSaveCallback,
)
from llmplatform.distill.config import load_opd, load_sft_baseline
from llmplatform.distill.e3 import build_e3, retention, write_e3
from llmplatform.distill.opd import build_distillation_args, build_student_peft, vllm_enabled
from llmplatform.distill.sft_baseline import budget_from_ledger, build_sft_args
from llmplatform.distill.teacher_gen import generate_completions
from llmplatform.distill.vocab_check import check_and_align, padded_vocab_size
from llmplatform.eval.e1 import update_checkpoint_file, write_pairwise_file
from llmplatform.smoke.tiny import build_tiny_model

pytestmark = pytest.mark.unit


# ============= Vocabulary check =============


def load_model(path: Path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path), dtype="float32").eval()


PROMPTS = ["Explain binary search.", "The water cycle.", "Compound interest grows."]


def test_check_resizes_teacher_and_preserves_logits(tmp_path: Path, mini_tokenizer) -> None:
    n_real = len(mini_tokenizer)
    student = load_model(
        build_tiny_model(
            tmp_path / "s", n_real + 20, 32, o_tie_embeddings=True, tokenizer=mini_tokenizer
        )
    )
    teacher = load_model(
        build_tiny_model(
            tmp_path / "t",
            n_real + 52,
            48,
            o_tie_embeddings=False,
            seed=1,
            tokenizer=mini_tokenizer,
        )
    )
    assert padded_vocab_size(teacher) != padded_vocab_size(student)
    mini_tokenizer.padding_side = "left"
    report = check_and_align(teacher, student, mini_tokenizer, mini_tokenizer, PROMPTS, 32)
    assert report.passed and report.resized_teacher
    assert (
        report.teacher_padded_size == n_real + 52
        and report.teacher_padded_size_after == n_real + 20
    )
    assert padded_vocab_size(teacher) == n_real + 20
    assert report.logits_max_abs_diff <= report.logits_atol
    assert report.argmax_identical and report.prompts_checked == 3
    assert 0.0 <= report.dropped_mass_max <= 1.0


def test_check_skips_resize_when_sizes_match(tmp_path: Path, mini_tokenizer) -> None:
    size = len(mini_tokenizer) + 8
    student = load_model(build_tiny_model(tmp_path / "s", size, 32, tokenizer=mini_tokenizer))
    teacher = load_model(
        build_tiny_model(tmp_path / "t", size, 32, seed=2, tokenizer=mini_tokenizer)
    )
    mini_tokenizer.padding_side = "left"
    report = check_and_align(teacher, student, mini_tokenizer, mini_tokenizer, PROMPTS, 32)
    assert report.passed and not report.resized_teacher and report.logits_max_abs_diff == 0.0


def test_check_fails_on_different_tokenizers(tmp_path: Path, mini_tokenizer) -> None:
    import copy

    other = copy.deepcopy(mini_tokenizer)
    other.add_tokens(["<extra_token>"])
    size = len(other) + 8
    student = load_model(build_tiny_model(tmp_path / "s", size, 32, tokenizer=other))
    teacher = load_model(build_tiny_model(tmp_path / "t", size + 8, 32, seed=3, tokenizer=other))
    report = check_and_align(teacher, student, mini_tokenizer, other, PROMPTS, 32)
    assert not report.tokenizers_identical and not report.passed
    assert padded_vocab_size(teacher) == size + 8


def test_check_fails_when_tokenizer_fills_padded_size(mini_tokenizer) -> None:
    n_real = len(mini_tokenizer)
    tiny = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(vocab_size=n_real))
    )
    import torch

    class Teacher:
        config = tiny.config

        def parameters(self):
            return iter([torch.zeros(1)])

    report = check_and_align(Teacher(), tiny, mini_tokenizer, mini_tokenizer, [], 8)
    assert not report.tokenizer_below_both_padded_sizes and not report.passed


# ============= Accounting =============


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_ledger_accumulates_segments_and_survives_kill(tmp_path: Path) -> None:
    clock = FakeClock()
    first = GpuLedger(tmp_path / "ledger.json", clock)
    first.start_segment(gpus=2)
    clock.now += 1800
    first.heartbeat()
    assert first.total_gpu_hours() == pytest.approx(1.0)
    clock.now += 600  # killed here: the last heartbeat was 10 minutes ago, so they are not counted
    second = GpuLedger(tmp_path / "ledger.json", clock)
    second.start_segment(gpus=1)
    clock.now += 3600
    second.heartbeat()
    assert second.total_gpu_hours() == pytest.approx(2.0)
    summary = second.summary()
    assert summary["segments"] == 2 and summary["gpu_hours"] == pytest.approx(2.0)
    assert budget_from_ledger(tmp_path / "ledger.json") == pytest.approx(2.0)


def test_callbacks_drive_training_control(tmp_path: Path) -> None:
    clock = FakeClock()
    ledger = GpuLedger(tmp_path / "l.json", clock)
    args = SimpleNamespace(process_index=0, world_size=2)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    LedgerCallback(ledger).on_train_begin(args, None, control)
    budget = BudgetCallback(ledger, 1.0)
    budget.on_step_end(args, None, control)
    assert not control.should_training_stop
    clock.now += 1800
    LedgerCallback(ledger).on_step_end(args, None, control)
    budget.on_step_end(args, None, control)
    assert control.should_training_stop and control.should_save and budget.reached

    ticks = {"now": 0.0}
    timed = TimedSaveCallback(1.0, lambda: ticks["now"])
    save_control = SimpleNamespace(should_save=False)
    timed.on_step_end(args, None, save_control)
    assert not save_control.should_save
    ticks["now"] = 61.0
    timed.on_step_end(args, None, save_control)
    assert save_control.should_save
    timed.on_save(args, None, save_control)
    save_control.should_save = False
    ticks["now"] = 100.0
    timed.on_step_end(args, None, save_control)
    assert not save_control.should_save


def test_clipped_ratio_guard_stops_on_sustained_truncation() -> None:
    guard = ClippedRatioGuard(maximum=0.05, window=3)
    metric = "completions/clipped_ratio"
    guard.on_log(None, None, None, {"loss": 1.0})
    for value in (0.2, 0.0):
        guard.on_log(None, None, None, {metric: value})
    with pytest.raises(ClippedRatioError, match="max_completion_length"):
        guard.on_log(None, None, None, {metric: 0.1})
    calm = ClippedRatioGuard(0.05, 3)
    for value in (0.0, 0.01, 0.0, 0.02):
        calm.on_log(None, None, None, {metric: value})


# ============= Arguments =============


def test_opd_args_follow_the_protocol() -> None:
    config = load_opd()
    args = build_distillation_args(config, Path("/out"))
    assert args["beta"] == 1.0 and args["temperature"] == 1.0
    assert args["learning_rate"] == 1e-6
    assert args["max_completion_length"] == 1024
    assert args["per_device_train_batch_size"] == 4 and args["gradient_accumulation_steps"] == 8
    assert args["model_init_kwargs"] == {"dtype": "float32"}
    assert args["bf16"] is True
    assert args["vllm_mode"] == "colocate" and args["vllm_gpu_memory_utilization"] == 0.3
    assert args["gradient_checkpointing"] is True and args["bf16"] is True
    assert config.data.max_prompts == 20000 and config.optim.num_train_epochs == 1
    assert build_distillation_args(config, Path("/o"), 3e-6)["learning_rate"] == 3e-6
    cpu = build_distillation_args(config, Path("/o"), o_cpu=True)
    assert cpu["use_vllm"] is False and cpu["model_init_kwargs"] == {"dtype": "float32"}
    assert build_student_peft(config) is None


def test_vllm_switch() -> None:
    config = load_opd()
    forced_off = load_opd(override_list=["vllm.enabled=false"])
    forced_on = load_opd(override_list=["vllm.enabled=true"])
    assert vllm_enabled(forced_off, False) is False
    assert vllm_enabled(forced_on, False) is True
    assert vllm_enabled(forced_on, True) is False
    assert isinstance(vllm_enabled(config, False), bool)


def test_lora_fallback_and_head_rule() -> None:
    lora = load_opd(
        override_list=["student_training.mode=lora", "student_training.train_lm_head=true"]
    )
    peft = build_student_peft(lora)
    assert peft is not None
    assert peft.r == 64 and peft.modules_to_save == ["lm_head"]
    assert "lm_head" not in peft.target_modules
    with pytest.raises(ValueError, match="lm_head"):
        load_opd(override_list=["student_training.lora_target_modules=[q_proj, lm_head]"])


def test_sft_baseline_args() -> None:
    config = load_sft_baseline()
    args = build_sft_args(config, Path("/o"))
    assert args["lr_scheduler_type"] == "constant_with_warmup"
    assert args["model_init_kwargs"] == {"dtype": "float32"} and args["bf16"] is True
    assert config.generation.keep_truncated is False


# ============= Teacher generation =============


def test_teacher_generation_filters_and_resumes(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][-1]["content"]
        calls.append(text)
        finish = "length" if text == "long" else "stop"
        content = "" if text == "empty" else f"answer to {text}"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        )

    prompts = [
        {"id": name, "prompt": [{"role": "user", "content": name}]}
        for name in ("a", "long", "empty", "b")
    ]
    spec = load_sft_baseline().generation.model_copy(update={"concurrency": 1})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    ticks = iter([0.0, 3600.0])
    out = tmp_path / "teacher.jsonl"
    summary = generate_completions(
        prompts[:2], "http://x/v1", "m", spec, out, 2, client, lambda: next(ticks)
    )
    assert summary["kept"] == 1 and summary["dropped_truncated"] == 1
    assert summary["gpu_hours_this_run"] == pytest.approx(2.0)
    ticks = iter([0.0, 1.0])
    summary = generate_completions(
        prompts, "http://x/v1", "m", spec, out, 1, client, lambda: next(ticks)
    )
    assert summary["resumed_from_partial"] == 2
    assert calls == ["a", "long", "empty", "b"]
    rows = read_jsonl(out)
    assert [row["id"] for row in rows] == ["a", "b"]
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "answer to a"}
    assert (tmp_path / "teacher.summary.json").is_file()


# ============= E3 =============

REF = ConfigRef(path="configs/eval/judge.yaml", sha256="0" * 64)
HW = HardwareRecord(gpu_model="none", gpu_count=0, provider_label="t", gpu_hourly_usd=None)


def judge_values(win_rate: float, words: float = 100.0) -> dict:
    return {
        "reference": "base_7b",
        "n": 300,
        "wins": 1,
        "losses": 1,
        "ties": 1,
        "win_rate": win_rate,
        "ci95_low": win_rate - 0.05,
        "ci95_high": win_rate + 0.05,
        "candidate_length": {"mean_words": words},
    }


def populate(results: Path) -> None:
    rates = {"teacher": 0.6, "base_1_5b": 0.2, "sft_baseline_1_5b": 0.3, "distilled_1_5b": 0.45}
    for name, rate in rates.items():
        update_checkpoint_file(results, name, "judge", judge_values(rate), REF, HW, {})
        update_checkpoint_file(
            results, name, "benchmarks", {"tasks": {"gsm8k": {"value": rate / 2}}}, REF, HW, {}
        )


def test_e3_summary(tmp_path: Path) -> None:
    results, ckpt, data = tmp_path / "results", tmp_path / "ckpt", tmp_path / "data"
    populate(results)
    write_pairwise_file(results, "distilled_1_5b", "base_1_5b", judge_values(0.7), REF, HW, {})
    clock = FakeClock()
    for subdir, hours in (("distill", 2.0), ("sft_baseline", 2.0)):
        ledger = GpuLedger(ckpt / subdir / "ledger.json", clock)
        ledger.start_segment(1)
        clock.now += hours * 3600
        ledger.heartbeat()
    write_json(data / "teacher_completions.summary.json", {"gpu_hours_this_run": 0.5})
    write_json(
        results / "e4_serving.json", {"metrics": {"cost_ratio": {"student_to_teacher": 0.2}}}
    )
    metrics = build_e3(results, ckpt, data)
    assert metrics["retention_vs_teacher"]["distilled_student"] == pytest.approx(0.75)
    assert metrics["pairwise_distilled_vs_base_student"]["win_rate"] == 0.7
    assert metrics["gpu_hours"] == {
        "distillation": pytest.approx(2.0),
        "sft_baseline": pytest.approx(2.0),
        "teacher_generation": 0.5,
    }
    assert metrics["roles"]["distilled_student"]["benchmarks"] == {"gsm8k": 0.225}
    assert metrics["student_to_teacher_cost_ratio"] == {"student_to_teacher": 0.2}
    out = write_e3(results, ckpt, data, REF, HW, {})
    assert load_results(results / "e3_distillation.json").run_id == out.run_id


def test_e3_requires_every_role_and_positive_teacher(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_e3(tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="positive"):
        retention(0.3, 0.0)
    assert retention(0.3, 0.6) == pytest.approx(0.5)
    _ = (write_results, build_results, write_jsonl)
