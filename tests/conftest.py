"""Shared fixtures."""

from pathlib import Path

import pytest

from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.paths import repo_root


@pytest.fixture(scope="session")
def root() -> Path:
    """Repository root."""
    return repo_root()


@pytest.fixture()
def cpu_hardware() -> HardwareRecord:
    """Hardware record for a host without a GPU."""
    return HardwareRecord(
        gpu_model="none",
        gpu_count=0,
        provider_label="test-host",
        gpu_hourly_usd=None,
        cpu="test-cpu",
    )


@pytest.fixture()
def gpu_hardware() -> HardwareRecord:
    """Hardware record for a priced single-GPU host."""
    return HardwareRecord(
        gpu_model="Test GPU 80GB",
        gpu_count=1,
        provider_label="test-provider",
        gpu_hourly_usd=2.0,
        price_source="unit test",
        price_date="2026-01-01",
        cpu="test-cpu",
    )


@pytest.fixture(scope="session")
def mini_tokenizer():
    """A small byte-level BPE tokenizer trained locally, so no download is needed."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = [
        "Explain binary search in two sentences.",
        "The water cycle moves water between the sea, the air, and the land.",
        "Compound interest grows faster than simple interest over time.",
    ] * 20
    tokenizer.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|im_end|>", pad_token="<|endoftext|>"
    )
