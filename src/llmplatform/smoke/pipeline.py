"""CPU smoke pipeline on tiny models: align, distill, judge from cache, pad a draft, convert."""

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from llmplatform.common.config import config_ref
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import read_jsonl, write_json, write_jsonl
from llmplatform.common.paths import repo_root
from llmplatform.eval.config import load_judge_config
from llmplatform.eval.e1 import checkpoint_path, update_checkpoint_file
from llmplatform.eval.generate import generate_hf
from llmplatform.judge.groq_client import JudgeClient
from llmplatform.judge.run import run_judgment
from llmplatform.smoke.synthetic import (
    judge_prompt_rows,
    preference_rows,
    prompt_rows,
    sft_rows,
)
from llmplatform.smoke.tiny import (
    STUDENT_PADDED_VOCAB,
    TEACHER_PADDED_VOCAB,
    build_tiny_model,
    load_smoke_tokenizer,
)

logger = logging.getLogger(__name__)

# ============= Constants =============

SMOKE_STEPS = 5
JUDGE_PROMPTS = 5
GENERATION_TOKENS = 8
TRAIN_OVERRIDES = [
    "quantization.enabled=false",
    "tracking.enabled=false",
    f"optim.max_steps={SMOKE_STEPS}",
    "optim.per_device_train_batch_size=2",
    "optim.gradient_accumulation_steps=1",
    "optim.gradient_checkpointing=false",
    "optim.warmup_ratio=0.0",
    "lengths.max_length=96",
    "checkpointing.save_steps=2",
    "checkpointing.logging_steps=1",
    "lora.r=4",
    "lora.alpha=8",
]
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
    "data.max_prompts=10",
]
SFT_BASELINE_OVERRIDES = [
    "tracking.enabled=false",
    "optim.per_device_train_batch_size=2",
    "optim.gradient_accumulation_steps=1",
    "optim.gradient_checkpointing=false",
    "optim.warmup_ratio=0.0",
    "optim.num_train_epochs=500",
    "lengths.max_length=64",
    "checkpointing.logging_steps=1",
]
CPU_HARDWARE = HardwareRecord(
    gpu_model="none", gpu_count=0, provider_label="smoke test", gpu_hourly_usd=None
)


# ============= Models =============


@dataclass
class StepResult:
    """Outcome of one smoke step."""

    name: str
    status: str
    seconds: float
    detail: str = ""


class StepRunner:
    """Times each step and records failures instead of stopping at the first one."""

    def __init__(self) -> None:
        self.results: list[StepResult] = []

    def run(self, name: str, function: Any) -> Any:
        """Run ``function``; record ``ok``, ``skipped`` (returned string), or ``failed``."""
        started = time.perf_counter()
        try:
            outcome = function()
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and reported
            logger.exception("smoke step failed", extra={"step": name})
            self.results.append(
                StepResult(name, "failed", time.perf_counter() - started, repr(exc))
            )
            return None
        status = "skipped" if isinstance(outcome, str) else "ok"
        detail = outcome if isinstance(outcome, str) else ""
        self.results.append(StepResult(name, status, time.perf_counter() - started, detail))
        return outcome

    @property
    def failed(self) -> bool:
        """Whether any step failed."""
        return any(item.status == "failed" for item in self.results)


# ============= Simulated judge =============


def simulated_judge_transport() -> httpx.MockTransport:
    """Return an offline judge that prefers the response with more words, whatever its position."""

    def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        first = content.split("[Response A]\n")[1].split("\n\n[Response B]")[0]
        second = content.split("[Response B]\n")[1].split("\n\nWhich response")[0]
        if len(first.split()) == len(second.split()):
            verdict = "tie"
        else:
            verdict = "A" if len(first.split()) > len(second.split()) else "B"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({"winner": verdict})}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    return httpx.MockTransport(handler)


# ============= Stages =============


def train_stages(work: Path, student: Path, data: Path) -> dict[str, Path]:
    """Run SFT then the three preference objectives, merging each adapter."""
    from llmplatform.train.config import StagePaths, load_train_config
    from llmplatform.train.merge import merge_adapter
    from llmplatform.train.stages import run_stage

    merged: dict[str, Path] = {}
    for stage in ("sft", "dpo", "orpo", "simpo"):
        config = load_train_config(
            repo_root() / "configs" / "train" / f"{stage}.yaml", TRAIN_OVERRIDES
        )
        base = student if stage == "sft" else merged["sft"]
        paths = StagePaths(str(base), None, data / config.data.file, work / "ckpt" / stage)
        adapter = run_stage(config, paths, o_cpu=True)
        merged[stage] = merge_adapter(
            str(base), adapter, paths.output_dir / "merged", None, "float32"
        )
    return merged


def distill_stages(work: Path, student: Path, teacher: Path, data: Path) -> Path:
    """Distill with vocabulary alignment, then run the budget-matched SFT baseline."""
    from llmplatform.distill.config import load_opd, load_sft_baseline
    from llmplatform.distill.opd import run_opd
    from llmplatform.distill.sft_baseline import budget_from_ledger, run_sft_baseline

    opd_out = work / "ckpt" / "distill"
    final = run_opd(
        load_opd(override_list=OPD_OVERRIDES),
        str(teacher),
        None,
        str(student),
        None,
        data / "distillation_prompts.jsonl",
        opd_out,
        o_cpu=True,
    )
    spec = load_judge_config().generation.model_copy(
        update={"max_new_tokens": GENERATION_TOKENS, "batch_size": 4}
    )
    prompts = [
        {"id": row["id"], "prompt": row["prompt"][0]["content"]}
        for row in read_jsonl(data / "distillation_prompts.jsonl")
    ]
    completions = generate_hf(str(teacher), prompts, spec, o_cpu=True)
    write_jsonl(
        data / "teacher_completions.jsonl",
        [
            {
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["response"] or "ok"},
                ],
            }
            for row in completions
        ],
    )
    run_sft_baseline(
        load_sft_baseline(override_list=SFT_BASELINE_OVERRIDES),
        str(student),
        None,
        data / "teacher_completions.jsonl",
        work / "ckpt" / "sft_baseline",
        max(budget_from_ledger(opd_out / "ledger.json"), 1.0 / 3600.0),
        o_cpu=True,
    )
    return final


def judge_from_cache(work: Path, reference: Path, candidates: dict[str, Path]) -> dict[str, Any]:
    """
    Generate responses to five judge prompts and judge them twice through the real client.

    The first pass fills the on-disk cache through a simulated judge endpoint, and the second
    pass must be answered entirely from the cache.
    """
    config = load_judge_config()
    spec = config.generation.model_copy(
        update={"max_new_tokens": GENERATION_TOKENS, "batch_size": 5}
    )
    prompts = judge_prompt_rows(JUDGE_PROMPTS)
    prompts_path = work / "data" / "judge_prompts.jsonl"
    write_jsonl(prompts_path, prompts)
    responses = work / "data" / "responses"
    write_jsonl(
        responses / "reference.jsonl", generate_hf(str(reference), prompts, spec, o_cpu=True)
    )
    root = repo_root()
    ref = config_ref(root / "configs/eval/judge.yaml", root)
    summary: dict[str, Any] = {}
    for name, path in candidates.items():
        write_jsonl(responses / f"{name}.jsonl", generate_hf(str(path), prompts, spec, o_cpu=True))
        client = JudgeClient(
            config,
            "offline-key",
            work / "data" / config.cache_dir,
            httpx.Client(transport=simulated_judge_transport()),
        )
        args = (
            client,
            config,
            prompts_path,
            responses / f"{name}.jsonl",
            responses / "reference.jsonl",
        )
        first = run_judgment(*args)
        second = run_judgment(*args)
        if second["judge_usage"]["cached_calls"] != second["judge_usage"]["calls"]:
            raise RuntimeError("second judgment pass was not served from the cache")
        if first["win_rate"] != second["win_rate"]:
            raise RuntimeError("cached judgments changed the win rate")
        update_checkpoint_file(
            work / "results",
            name,
            "judge",
            first,
            ref,
            CPU_HARDWARE,
            {},
            notes="smoke, simulated judge",
        )
        summary[name] = {"win_rate": first["win_rate"], "n": first["n"]}
    return summary


def export_draft_stage(work: Path, draft: Path, teacher: Path) -> dict[str, Any]:
    """Pad the draft to the teacher's embedding size and require the verification to pass."""
    from llmplatform.speculative.export_draft import export_draft

    tokenizer = load_smoke_tokenizer()
    prompts = [
        str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
        for row in judge_prompt_rows(6)
    ]
    report = export_draft(
        str(draft), None, str(teacher), None, work / "ckpt" / "draft_padded", prompts, o_cpu=True
    )
    return asdict(report)


def gguf_stage(work: Path, model: Path) -> str | dict[str, Any]:
    """Convert a model to GGUF when a llama.cpp checkout is configured, else report a skip."""
    from llmplatform.quantize.config import load_quantize
    from llmplatform.quantize.gguf import (
        CONVERT_SCRIPT,
        convert_and_quantize,
        convert_command,
        resolve_tool,
        verify_gguf,
    )

    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    if not llama_dir or not (Path(llama_dir) / CONVERT_SCRIPT).is_file():
        return "set LLAMA_CPP_DIR to a llama.cpp checkout to run the GGUF conversion"
    out = work / "gguf"
    out.mkdir(parents=True, exist_ok=True)
    gguf_cfg = load_quantize().gguf
    try:
        resolve_tool(llama_dir, "quantize")
    except FileNotFoundError:
        import subprocess

        target = out / "tiny-f16.gguf"
        subprocess.run(convert_command(llama_dir, model, target, gguf_cfg.outtype), check=True)
        return {"converted": target.name, "bytes": target.stat().st_size, "quantized": False}
    quantized = convert_and_quantize(
        llama_dir, model, out, "tiny", gguf_cfg.quant_type, gguf_cfg.outtype
    )
    checks = verify_gguf(
        llama_dir, quantized, gguf_cfg.verify.prompts, 8, gguf_cfg.verify.timeout_seconds
    )
    if not all(item.ok for item in checks):
        raise RuntimeError("a GGUF verification prompt produced no output")
    return {"converted": quantized.name, "bytes": quantized.stat().st_size, "quantized": True}


# ============= Pipeline =============


def run_smoke(work: Path, o_clean: bool = True) -> list[StepResult]:
    """
    Run every smoke step on tiny models and return the step outcomes.

    Parameters
    ----------
    work : Path
        Scratch directory; nothing is written to the repository's ``results/``.
    o_clean : bool
        Remove ``work`` first so reruns start from nothing.
    """
    if o_clean and work.exists():
        shutil.rmtree(work)
    data = work / "data"
    data.mkdir(parents=True)
    runner = StepRunner()
    tokenizer = runner.run("load tokenizer", load_smoke_tokenizer)
    if tokenizer is None:
        return runner.results

    def build_models() -> dict[str, Path]:
        return {
            "student": build_tiny_model(
                work / "models" / "student",
                STUDENT_PADDED_VOCAB,
                32,
                o_tie_embeddings=True,
                seed=0,
                tokenizer=tokenizer,
            ),
            "teacher": build_tiny_model(
                work / "models" / "teacher",
                TEACHER_PADDED_VOCAB,
                64,
                o_tie_embeddings=False,
                seed=1,
                tokenizer=tokenizer,
            ),
        }

    models = runner.run("build tiny models", build_models)

    def write_data() -> None:
        write_jsonl(data / "sft.jsonl", sft_rows(12))
        write_jsonl(data / "preferences.jsonl", preference_rows(12))
        write_jsonl(data / "distillation_prompts.jsonl", prompt_rows(10))

    runner.run("write synthetic data", write_data)
    if models is None:
        return runner.results
    merged = runner.run(
        "sft, dpo, orpo, simpo", lambda: train_stages(work, models["student"], data)
    )
    runner.run(
        "distillation and sft baseline",
        lambda: distill_stages(work, models["student"], models["teacher"], data),
    )
    final = work / "ckpt" / "distill" / "final"
    if merged is not None and final.is_dir():
        runner.run(
            "judge from cache",
            lambda: judge_from_cache(
                work, models["student"], {"distilled_1_5b": final, "dpo": merged["dpo"]}
            ),
        )
    runner.run(
        "draft export", lambda: export_draft_stage(work, models["student"], models["teacher"])
    )
    runner.run("gguf conversion", lambda: gguf_stage(work, models["student"]))
    write_json(work / "smoke_summary.json", {"steps": [asdict(item) for item in runner.results]})
    return runner.results


def judge_file_exists(work: Path, name: str) -> bool:
    """Return whether a judge section was written for a checkpoint in the scratch results."""
    return checkpoint_path(work / "results", name).is_file()
