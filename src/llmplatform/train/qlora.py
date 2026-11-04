"""Model, tokenizer, quantization, and LoRA construction for QLoRA training."""

import os
from typing import Any

from llmplatform.train.config import TrainConfig

# ============= Constants =============

ALL_LINEAR = "all-linear"


# ============= Builders =============


def torch_dtype(name: str) -> Any:
    """Return the ``torch.dtype`` named ``name``, for example ``bfloat16``."""
    import torch

    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r}")
    return dtype


def build_quantization_config(config: TrainConfig) -> Any | None:
    """Return a bitsandbytes 4-bit config, or ``None`` when quantization is disabled."""
    section = config.quantization
    if not section.enabled:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=section.quant_type,
        bnb_4bit_use_double_quant=section.double_quant,
        bnb_4bit_compute_dtype=torch_dtype(section.compute_dtype),
    )


def build_lora_config(config: TrainConfig) -> Any:
    """Return a PEFT LoRA config targeting all linear layers unless modules are listed."""
    from peft import LoraConfig

    lora = config.lora
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_tokenizer(path: str, revision: str | None) -> Any:
    """Load the tokenizer with a pad token and right padding for training."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: TrainConfig, path: str, revision: str | None, o_cpu: bool) -> Any:
    """
    Load the starting model for a stage.

    Parameters
    ----------
    config : TrainConfig
        Stage configuration; its quantization section decides the 4-bit setup.
    path : str
        Hub id or local directory.
    revision : str or None
        Hub revision pin; ignored for local paths.
    o_cpu : bool
        Load in float32 on the CPU without quantization, for smoke tests.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": torch.float32 if o_cpu else torch.bfloat16}
    quantization = None if o_cpu else build_quantization_config(config)
    if quantization is not None:
        kwargs["quantization_config"] = quantization
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if revision and not os.path.isdir(path):
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = False
    return model
