"""Tests for training configs, argument mapping, scaling math, and the step metrics callback."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from llmplatform.common.config import apply_overrides
from llmplatform.common.models import load_models
from llmplatform.train.config import STAGES, load_train_config, resolve_stage_paths
from llmplatform.train.scaling import (
    ScalingPoint,
    compute_point,
    read_step_metrics,
    scaling_summary,
)
from llmplatform.train.stages import build_stage_args, tracking_tags

pytestmark = pytest.mark.unit


def config_path(root: Path, stage: str) -> Path:
    return root / "configs" / "train" / f"{stage}.yaml"


# ============= Configs =============


@pytest.mark.parametrize("stage", STAGES)
def test_stage_configs_load(root: Path, stage: str) -> None:
    config = load_train_config(config_path(root, stage))
    assert config.stage == stage
    assert config.quantization.quant_type == "nf4"
    assert config.quantization.double_quant is True
    assert config.lora.target_modules == "all-linear"


def test_preference_stages_share_optimizer_settings(root: Path) -> None:
    dumps = {
        stage: load_train_config(config_path(root, stage)).model_dump(mode="json")
        for stage in ("dpo", "orpo", "simpo")
    }
    for key in (
        "optim",
        "lora",
        "quantization",
        "lengths",
        "checkpointing",
        "model",
        "data",
        "seed",
    ):
        assert dumps["dpo"][key] == dumps["orpo"][key] == dumps["simpo"][key], key


def test_overrides_apply_and_validate(root: Path) -> None:
    config = load_train_config(
        config_path(root, "dpo"), ["optim.max_steps=60", "quantization.enabled=false"]
    )
    assert config.optim.max_steps == 60
    assert config.quantization.enabled is False
    with pytest.raises(KeyError):
        apply_overrides({"a": {"b": 1}}, ["a.c=2"])
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({"a": 1}, ["bad"])
    with pytest.raises(ValueError, match="max_steps"):
        load_train_config(config_path(root, "dpo"), ["optim.max_steps=0"])


def test_unknown_key_rejected(root: Path, tmp_path: Path) -> None:
    data = yaml.safe_load(config_path(root, "sft").read_text())
    data["lora"]["unexpected"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="invalid config"):
        load_train_config(bad)


def test_stage_paths(root: Path) -> None:
    config = load_train_config(config_path(root, "dpo"))
    paths = resolve_stage_paths(config, load_models(), "/ckpt", "/data")
    assert paths.model_path == "/ckpt/sft/merged"
    assert paths.model_revision is None
    assert paths.data_path == Path("/data/preferences.jsonl")
    assert paths.output_dir == Path("/ckpt/dpo")
    sft = resolve_stage_paths(
        load_train_config(config_path(root, "sft")), load_models(), "/c", "/d"
    )
    assert sft.model_path == "Qwen/Qwen2.5-7B-Instruct"
    assert sft.model_revision is not None


# ============= Argument mapping =============


def test_stage_args_map_objectives(root: Path) -> None:
    out = Path("/out")
    sft = build_stage_args(load_train_config(config_path(root, "sft")), out)
    assert sft["packing"] is False and "beta" not in sft
    dpo = build_stage_args(load_train_config(config_path(root, "dpo")), out)
    assert dpo["beta"] == 0.1 and dpo["loss_type"] == ["sigmoid"]
    orpo = build_stage_args(load_train_config(config_path(root, "orpo")), out)
    assert orpo["beta"] == 0.1 and "loss_type" not in orpo
    simpo = build_stage_args(load_train_config(config_path(root, "simpo")), out)
    assert simpo["loss_type"] == "simpo" and simpo["cpo_alpha"] == 0.0
    assert simpo["simpo_gamma"] == 1.0
    for args in (sft, dpo, orpo, simpo):
        assert args["bf16"] is True
        assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
        assert args["include_num_input_tokens_seen"] == "all"
        assert args["report_to"] == ["mlflow"]


def test_cpu_args_disable_gpu_features(root: Path) -> None:
    args = build_stage_args(load_train_config(config_path(root, "sft")), Path("/o"), True, False)
    assert args["bf16"] is False and args["use_cpu"] is True and args["report_to"] == []


def test_tracking_tags_have_hardware_fields() -> None:
    tags = tracking_tags("dpo")
    assert {"stage", "node", "gpu_model", "gpu_count", "driver", "cuda"} <= set(tags)


# ============= Scaling =============


def rows(
    step_seconds: float, tokens_per_step: int, steps: int = 60, mem: float = 10.0
) -> list[dict]:
    return [
        {
            "step": step,
            "time_s": 100.0 + step * step_seconds,
            "tokens_seen": step * tokens_per_step,
            "peak_mem_gib": mem + step * 0.001,
        }
        for step in range(1, steps + 1)
    ]


def test_compute_point_discards_warmup() -> None:
    point = compute_point(rows(2.0, 1000), gpus=2)
    assert point.tokens_per_second_total == pytest.approx(500.0)
    assert point.tokens_per_second_per_gpu == pytest.approx(250.0)
    assert point.steps_measured == 50
    assert point.peak_memory_gib_per_gpu == pytest.approx(10.06)


def test_compute_point_needs_enough_steps() -> None:
    with pytest.raises(ValueError, match="beyond"):
        compute_point(rows(1.0, 10, steps=10), gpus=1)


def test_scaling_efficiency() -> None:
    one = compute_point(rows(2.0, 1000), 1)
    four = compute_point(rows(2.5, 4000), 4)
    summary = scaling_summary([one, four])
    assert summary["points"]["1"]["scaling_efficiency"] == pytest.approx(1.0)
    assert summary["points"]["4"]["scaling_efficiency"] == pytest.approx(0.8)
    assert summary["complete"] is False
    assert summary["missing"] == [2, 8]
    with pytest.raises(ValueError, match="1-GPU"):
        scaling_summary([ScalingPoint(2, 1.0, 0.5, 1.0, 50)])


def test_read_step_metrics_keeps_last_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r) for r in [{"step": 2, "x": 1}, {"step": 1, "x": 0}, {"step": 2, "x": 9}]
        )
    )
    result = read_step_metrics(path)
    assert [r["step"] for r in result] == [1, 2]
    assert result[1]["x"] == 9


def test_step_metrics_callback_writes_rows(tmp_path: Path) -> None:
    from llmplatform.train.callbacks import StepMetricsCallback

    callback = StepMetricsCallback(tmp_path)
    args = SimpleNamespace(process_index=0, world_size=1)
    state = SimpleNamespace(global_step=3, num_input_tokens_seen=123)
    callback.on_train_begin(args, state, None)
    callback.on_step_end(args, state, None)
    row = json.loads((tmp_path / "step_metrics.jsonl").read_text().splitlines()[0])
    assert row["step"] == 3 and row["tokens_seen"] == 123 and row["world_size"] == 1


def test_accelerate_config_is_zero2(root: Path) -> None:
    data = yaml.safe_load((root / "configs" / "accelerate" / "zero2.yaml").read_text())
    assert data["distributed_type"] == "DEEPSPEED"
    assert data["deepspeed_config"]["zero_stage"] == 2
    assert data["mixed_precision"] == "bf16"
