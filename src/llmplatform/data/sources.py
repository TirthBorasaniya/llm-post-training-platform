"""Hub dataset access. Heavy imports are deferred so the module loads without the training stack."""

import logging
from collections.abc import Callable, Iterator
from typing import Any

from llmplatform.data.config import BenchmarkSource

logger = logging.getLogger(__name__)

# ============= Loading =============


def load_split(
    hub_id: str, split: str, revision: str | None = None, config: str | None = None
) -> Any:
    """
    Load one dataset split from the Hub.

    Parameters
    ----------
    hub_id : str
        Dataset repository id.
    split : str
        Split name.
    revision : str, optional
        Commit hash to pin. ``None`` follows the default branch and is recorded by the caller.
    config : str, optional
        Dataset configuration name.
    """
    from datasets import load_dataset

    return load_dataset(hub_id, config, split=split, revision=revision)


def resolve_revision(hub_id: str, revision: str | None) -> str:
    """Return the pinned revision, or the current head commit of the default branch."""
    if revision:
        return revision
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(hub_id)
    if info.sha is None:
        raise RuntimeError(f"could not resolve a revision for {hub_id}")
    return info.sha


def benchmark_texts(source: BenchmarkSource) -> list[str]:
    """Return the question strings of one benchmark source."""
    dataset = load_split(source.hub_id, source.split, config=source.config)
    return [str(value) for value in dataset[source.field]]


def tokenizer_counter(tokenizer_id: str, revision: str | None = None) -> Callable[[str], int]:
    """Return a function counting tokens of a string with a Hub tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def iter_rows(dataset: Any, order: list[int]) -> Iterator[dict[str, Any]]:
    """Yield dataset rows in a given index order."""
    for index in order:
        yield dataset[index]
