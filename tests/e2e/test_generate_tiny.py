"""Greedy generation with Hugging Face on a tiny CPU model."""

from pathlib import Path

import pytest

from llmplatform.eval.config import load_judge_config
from llmplatform.eval.generate import generate_hf, save_responses
from llmplatform.smoke.synthetic import judge_prompt_rows

pytestmark = pytest.mark.slow


def test_generate_hf_is_deterministic(tiny_student: Path, tmp_path: Path) -> None:
    spec = load_judge_config().generation.model_copy(update={"max_new_tokens": 6, "batch_size": 2})
    prompts = judge_prompt_rows(3)
    first = generate_hf(str(tiny_student), prompts, spec, o_cpu=True)
    second = generate_hf(str(tiny_student), prompts, spec, o_cpu=True)
    assert [row["id"] for row in first] == [row["id"] for row in prompts]
    assert first == second
    assert all(row["completion_tokens"] <= 6 for row in first)
    out = save_responses(tmp_path / "r.jsonl", first)
    assert out.is_file()
