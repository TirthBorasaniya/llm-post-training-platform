"""Run every alignment stage for two optimizer steps on a tiny CPU model."""

from pathlib import Path

import pytest

from llmplatform.common.checkpoints import is_complete
from llmplatform.train.config import StagePaths, load_train_config
from llmplatform.train.merge import merge_adapter
from llmplatform.train.stages import run_stage

pytestmark = pytest.mark.slow

OVERRIDES = [
    "quantization.enabled=false",
    "tracking.enabled=false",
    "optim.max_steps=2",
    "optim.per_device_train_batch_size=2",
    "optim.gradient_accumulation_steps=1",
    "optim.gradient_checkpointing=false",
    "optim.warmup_ratio=0.0",
    "lengths.max_length=96",
    "checkpointing.save_steps=1",
    "checkpointing.logging_steps=1",
    "lora.r=4",
    "lora.alpha=8",
]


@pytest.mark.parametrize("stage", ["sft", "dpo", "orpo", "simpo"])
def test_stage_trains_merges_and_resumes(
    root: Path, tiny_student: Path, data_root: Path, tmp_path: Path, stage: str
) -> None:
    config = load_train_config(root / "configs" / "train" / f"{stage}.yaml", OVERRIDES)
    paths = StagePaths(
        model_path=str(tiny_student),
        model_revision=None,
        data_path=data_root / config.data.file,
        output_dir=tmp_path / stage,
    )
    adapter = run_stage(config, paths, o_cpu=True)
    assert (adapter / "adapter_config.json").is_file()
    assert (paths.output_dir / "step_metrics.jsonl").is_file()
    assert is_complete(paths.output_dir)

    merged = merge_adapter(
        str(tiny_student), adapter, tmp_path / f"{stage}_merged", None, "float32"
    )
    assert (merged / "config.json").is_file()
    assert (merged / "model.safetensors").is_file()

    # a finished stage returns immediately without retraining
    assert run_stage(config, paths, o_cpu=True) == adapter
