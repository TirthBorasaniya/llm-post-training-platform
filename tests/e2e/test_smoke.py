"""The CPU smoke pipeline runs end to end on tiny models."""

from pathlib import Path

import pytest

from llmplatform.smoke.pipeline import judge_file_exists, run_smoke

pytestmark = pytest.mark.slow


def test_smoke_pipeline_completes(tmp_path: Path, tokenizer: object) -> None:
    results = run_smoke(tmp_path / "smoke")
    by_name = {item.name: item for item in results}
    failed = {name: item.detail for name, item in by_name.items() if item.status == "failed"}
    assert failed == {}
    for name in (
        "sft, dpo, orpo, simpo",
        "distillation and sft baseline",
        "judge from cache",
        "draft export",
    ):
        assert by_name[name].status == "ok"
    assert by_name["gguf conversion"].status in {"ok", "skipped"}
    assert judge_file_exists(tmp_path / "smoke", "distilled_1_5b")
    assert (tmp_path / "smoke" / "smoke_summary.json").is_file()
    assert not any((Path(__file__).resolve().parents[2] / "results").glob("*.json"))
