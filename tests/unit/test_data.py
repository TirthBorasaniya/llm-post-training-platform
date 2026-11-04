"""Tests for dedup, formats, selection, and data and model configs."""

from pathlib import Path

import pytest

from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.data.config import load_data_config
from llmplatform.data.dedup import (
    NearDuplicateConfig,
    ReferenceIndex,
    decontaminate,
    exact_key,
    normalize_text,
    shingle_hashes,
    unique_by_text,
)
from llmplatform.data.formats import (
    is_valid_dialogue,
    preference_record,
    prompt_record,
    record_text,
    sft_record,
)
from llmplatform.data.prepare import build_reference_index, select_judge_prompts, select_records

pytestmark = pytest.mark.unit

LONG = (
    "Write a detailed essay about the history of the printing press and its influence on "
    "european science during the early modern period with several concrete examples"
)


def words(text: str) -> int:
    return len(text.split())


# ============= Dedup =============


def test_normalization_ignores_case_punctuation_and_spacing() -> None:
    assert normalize_text("Hello,   World!") == "hello world"
    assert exact_key("Hello, World!") == exact_key("hello   world")


def test_short_text_has_one_shingle() -> None:
    assert len(shingle_hashes("two words", 5)) == 1


def test_index_detects_exact_and_near_duplicates() -> None:
    index = ReferenceIndex(NearDuplicateConfig(threshold=0.7))
    index.add(LONG)
    assert index.match(LONG.upper()) == "exact"
    near = LONG + " today"
    assert index.match(near) == "near"
    assert (
        index.match("Explain how photosynthesis converts light into chemical energy in plants")
        is None
    )
    assert len(index) == 1


def test_index_rejects_incompatible_bands() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ReferenceIndex(NearDuplicateConfig(num_perm=100, rows_per_band=3))


def test_decontaminate_and_unique() -> None:
    index = ReferenceIndex()
    index.add(LONG)
    rows = [
        {"t": LONG},
        {"t": "What is the capital of France?"},
        {"t": "What is the capital of France"},
    ]
    kept, stats = decontaminate(rows, lambda r: r["t"], index)
    assert [r["t"] for r in kept] == [rows[1]["t"], rows[2]["t"]]
    assert stats.dropped == {"exact": 1}
    unique, ustats = unique_by_text(kept, lambda r: r["t"])
    assert len(unique) == 1
    assert ustats.dropped == {"exact": 1}


# ============= Formats =============


def dialogue(*pairs: str) -> list[dict[str, str]]:
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": text} for i, text in enumerate(pairs)]


def test_dialogue_validation() -> None:
    assert is_valid_dialogue(dialogue("q", "a"))
    assert not is_valid_dialogue(dialogue("q"))
    assert not is_valid_dialogue(dialogue("q", " "))
    assert not is_valid_dialogue(
        [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]
    )
    system = [{"role": "system", "content": "s"}, *dialogue("q", "a")]
    assert is_valid_dialogue(system)


def test_sft_and_preference_records() -> None:
    assert sft_record({"messages": dialogue("q", "a")}, "x") == {
        "id": "x",
        "messages": dialogue("q", "a"),
    }
    assert sft_record({"messages": dialogue("q")}, "x") is None
    row = {"chosen": dialogue("q", "good"), "rejected": dialogue("q", "bad")}
    record = preference_record(row, "p")
    assert record is not None
    assert record["prompt"] == dialogue("q")
    assert record["chosen"][0]["content"] == "good"
    same = {"chosen": dialogue("q", "a"), "rejected": dialogue("q", "a")}
    assert preference_record(same, "p") is None
    mismatch = {"chosen": dialogue("q1", "a"), "rejected": dialogue("q2", "b")}
    assert preference_record(mismatch, "p") is None


def test_prompt_record_accepts_strings_and_messages() -> None:
    assert prompt_record({"prompt": "hi"}, "a") == {
        "id": "a",
        "prompt": [{"role": "user", "content": "hi"}],
    }
    assert prompt_record({"prompt": dialogue("q")}, "a") is not None
    assert prompt_record({"prompt": "  "}, "a") is None
    assert record_text({"messages": dialogue("first", "a", "second", "b")}) == "first"


# ============= Selection =============


def test_select_records_applies_all_filters() -> None:
    index = build_reference_index(load_data_config(), [LONG], [])
    rows = [
        {"messages": dialogue("short question one", "answer")},
        {"messages": dialogue(LONG, "contaminated")},
        {"messages": dialogue("x " * 50, "too long")},
        {"messages": dialogue("short question one", "another answer")},
        {"messages": dialogue("broken")},
        {"messages": dialogue("short question two", "answer")},
        {"messages": dialogue("short question three", "answer")},
    ]
    records, stats_list = select_records(rows, "sft", 2, 40, words, index)
    assert [r["id"] for r in records] == ["sft-0", "sft-5"]
    by_name = {s.name: s for s in stats_list}
    assert by_name["convert"].dropped == {"invalid": 1}
    assert by_name["token_length"].dropped == {"too_long": 1}
    assert by_name["decontaminate"].dropped == {"exact": 1}
    assert by_name["unique"].dropped == {"exact": 1}


def test_judge_prompt_selection_is_seeded_and_filtered() -> None:
    rows = [{"prompt": f"prompt number {i}", "prompt_id": str(i)} for i in range(30)]
    rows.append({"prompt": "word " * 100, "prompt_id": "long"})
    rows.append({"prompt": "prompt number 1", "prompt_id": "dup"})
    first, counts = select_judge_prompts(rows, 10, 10, 7, words)
    second, _ = select_judge_prompts(rows, 10, 10, 7, words)
    assert first == second
    assert counts == {"rows": 32, "eligible": 30, "selected": 10}
    assert all(item["prompt_id"] not in {"long", "dup"} for item in first)
    third, _ = select_judge_prompts(rows, 10, 10, 8, words)
    assert third != first
    with pytest.raises(ValueError, match="eligible"):
        select_judge_prompts(rows, 31, 10, 7, words)


# ============= Configs =============


def test_data_config_pins_revisions(root: Path) -> None:
    config = load_data_config(root / "configs" / "data.yaml")
    for spec in (
        config.training.sft,
        config.training.preferences,
        config.training.distillation_prompts,
    ):
        assert len(spec.revision) == 40
    assert config.judge.n_prompts == 300
    assert config.judge.max_prompt_tokens == 1024
    assert {s.name for s in config.decontamination.benchmark_sources} >= {"gsm8k", "mmlu", "ifeval"}


def test_models_registry_resolution(root: Path) -> None:
    models = load_models(root / "configs" / "models.yaml")
    assert resolve_model(models, "reference", "/ckpt") == "Qwen/Qwen2.5-7B-Instruct"
    assert resolve_model(models, "dpo", "/ckpt") == "/ckpt/dpo/merged"
    assert revision_of(models, "student_base") is not None
    assert revision_of(models, "sft") is None
    with pytest.raises(KeyError):
        resolve_model(models, "missing", "/ckpt")


def test_data_doc_lists_every_dataset(root: Path) -> None:
    text = (root / "docs" / "DATA.md").read_text(encoding="utf-8")
    config = load_data_config(root / "configs" / "data.yaml")
    for spec in (
        config.training.sft,
        config.training.preferences,
        config.training.distillation_prompts,
        config.judge,
    ):
        assert spec.hub_id in text
        assert spec.revision in text
    for source in config.decontamination.benchmark_sources:
        assert source.hub_id in text


def test_judge_prompt_file_when_present(root: Path) -> None:
    path = root / "eval" / "judge_prompts.jsonl"
    if not path.exists():
        pytest.skip("judge prompts are generated by make judge-prompts")
    from llmplatform.common.io import read_jsonl

    rows = read_jsonl(path)
    config = load_data_config(root / "configs" / "data.yaml")
    assert len(rows) == config.judge.n_prompts
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["prompt"].strip() for row in rows)
