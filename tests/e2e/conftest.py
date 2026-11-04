"""Fixtures for tiny-model tests; skipped when no tokenizer can be loaded."""

from pathlib import Path

import pytest

from llmplatform.common.io import write_jsonl
from llmplatform.smoke.synthetic import preference_rows, prompt_rows, sft_rows
from llmplatform.smoke.tiny import build_tiny_model, load_smoke_tokenizer


@pytest.fixture(scope="session")
def tokenizer():
    try:
        return load_smoke_tokenizer()
    except OSError as exc:
        pytest.skip(f"tokenizer unavailable: {exc}")


@pytest.fixture(scope="session")
def tiny_student(tmp_path_factory: pytest.TempPathFactory, tokenizer) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("student"), tokenizer=tokenizer)


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    write_jsonl(tmp_path / "sft.jsonl", sft_rows(8))
    write_jsonl(tmp_path / "preferences.jsonl", preference_rows(8))
    write_jsonl(tmp_path / "distillation_prompts.jsonl", prompt_rows(8))
    return tmp_path
