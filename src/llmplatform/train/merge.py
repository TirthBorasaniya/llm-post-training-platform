"""Merge a LoRA adapter into a full-precision base model and save a standalone checkpoint."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============= Constants =============

MERGED_MARKER = "config.json"

# ============= Merge =============


def merge_adapter(
    base_path: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    base_revision: str | None = None,
    dtype_name: str = "bfloat16",
) -> Path:
    """
    Merge adapter weights into the base model.

    The base is loaded unquantized in ``dtype_name``, so the merged weights are a standard
    checkpoint that vLLM, SGLang, and llama.cpp load directly.

    Parameters
    ----------
    base_path : str
        Hub id or directory of the model the adapter was trained on.
    adapter_dir : str or Path
        Directory written by the trainer.
    output_dir : str or Path
        Destination for the merged model and tokenizer.
    base_revision : str, optional
        Revision pin for Hub models.
    dtype_name : str
        Torch dtype name for the merged weights.

    Returns
    -------
    output_dir : Path
        The destination directory.
    """
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    kwargs = {"dtype": dtype}
    if base_revision and not os.path.isdir(base_path):
        kwargs["revision"] = base_revision
    base = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(destination), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(destination))
    logger.info("merged adapter", extra={"output_dir": str(destination)})
    return destination
