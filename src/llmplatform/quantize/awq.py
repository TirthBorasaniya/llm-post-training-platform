"""AWQ 4-bit weight quantization with llm-compressor, for GPUs without FP8 support."""

import logging
from pathlib import Path
from typing import Any

from llmplatform.common.io import read_jsonl
from llmplatform.quantize.config import AwqSection

logger = logging.getLogger(__name__)

# ============= Recipe =============


def build_recipe(section: AwqSection) -> list[Any]:
    """
    Return the llm-compressor recipe: AWQ smoothing followed by weight-only quantization.

    The language-model head stays in higher precision through ``ignore``.
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    return [
        AWQModifier(duo_scaling=section.duo_scaling if section.duo_scaling == "both" else True),
        QuantizationModifier(
            ignore=list(section.ignore), scheme=section.scheme, targets=["Linear"]
        ),
    ]


def calibration_dataset(path: Path, tokenizer: Any, samples: int, max_length: int) -> Any:
    """Tokenize chat records into a calibration dataset of at most ``samples`` rows."""
    from datasets import Dataset

    rows = read_jsonl(path)[:samples]
    texts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        for row in rows
    ]
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(
        lambda item: tokenizer(
            item["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        ),
        remove_columns=["text"],
    )


def quantize_awq(
    model_path: str,
    revision: str | None,
    output_dir: Path,
    section: AwqSection,
    calibration_path: Path,
) -> Path:
    """
    Quantize a model to AWQ 4-bit weights and save a compressed checkpoint.

    Returns
    -------
    output_dir : Path
        Directory with the quantized model and tokenizer, loadable by SGLang.
    """
    import os

    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"dtype": "auto"}
    if revision and not os.path.isdir(model_path):
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=kwargs.get("revision"))
    dataset = calibration_dataset(
        calibration_path, tokenizer, section.calibration_samples, section.max_seq_length
    )
    oneshot(
        model=model,
        dataset=dataset,
        recipe=build_recipe(section),
        max_seq_length=section.max_seq_length,
        num_calibration_samples=section.calibration_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))
    return output_dir
