"""Build tiny randomly initialized Qwen2 models that share a real tokenizer for CPU tests."""

import os
from pathlib import Path
from typing import Any

# ============= Constants =============

DEFAULT_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_DIR_ENV = "LLMPLATFORM_TOKENIZER_DIR"
# padded embedding sizes of the real 1.5B and 7B models
STUDENT_PADDED_VOCAB = 151936
TEACHER_PADDED_VOCAB = 152064


# ============= Builders =============


def tokenizer_source() -> str:
    """Return the tokenizer directory from the environment, else the Hub id."""
    return os.environ.get(TOKENIZER_DIR_ENV) or DEFAULT_TOKENIZER_ID


def load_smoke_tokenizer(source: str | None = None) -> Any:
    """Load the shared tokenizer with a pad token set."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source or tokenizer_source())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_tiny_model(
    output_dir: str | Path,
    vocab_size: int = STUDENT_PADDED_VOCAB,
    hidden_size: int = 32,
    num_layers: int = 2,
    o_tie_embeddings: bool = True,
    seed: int = 0,
    tokenizer: Any | None = None,
) -> Path:
    """
    Save a tiny Qwen2 causal language model and its tokenizer.

    Parameters
    ----------
    output_dir : str or Path
        Destination directory.
    vocab_size : int
        Embedding rows; at least the tokenizer size. Use different values for teacher and
        student to mirror the padded sizes of the real models.
    hidden_size : int
        Model width; must be divisible by four attention heads.
    num_layers : int
        Number of decoder layers.
    o_tie_embeddings : bool
        Tie the output projection to the embedding matrix.
    seed : int
        Initialization seed.
    tokenizer : optional
        Tokenizer to save; loaded from the configured source when omitted.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = tokenizer or load_smoke_tokenizer()
    if vocab_size < len(tok):
        raise ValueError("vocab_size is smaller than the tokenizer")
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=o_tie_embeddings,
        bos_token_id=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination), safe_serialization=True)
    tok.save_pretrained(str(destination))
    return destination
