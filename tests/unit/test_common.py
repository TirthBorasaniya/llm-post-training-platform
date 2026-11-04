"""Tests for configuration, results, IO, hashing, hardware, and checkpoints."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from llmplatform.common.checkpoints import is_complete, latest_checkpoint, mark_complete
from llmplatform.common.config import ConfigModel, config_ref, load_config
from llmplatform.common.hardware import (
    HardwareRecord,
    build_hardware_record,
    package_versions,
    parse_nvidia_smi_csv,
)
from llmplatform.common.hashing import sha256_file, sha256_json, sha256_text
from llmplatform.common.io import read_jsonl, write_jsonl
from llmplatform.common.logging_utils import JsonFormatter
from llmplatform.common.results import build_results, load_results, write_results
from llmplatform.common.settings import Settings

pytestmark = pytest.mark.unit


class _Demo(ConfigModel):
    learning_rate: float


# ============= Config =============


def test_load_config_validates_version_and_keys(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("version: 1\nlearning_rate: 0.5\n")
    assert load_config(good, _Demo).learning_rate == 0.5

    wrong_version = tmp_path / "v2.yaml"
    wrong_version.write_text("version: 2\nlearning_rate: 0.5\n")
    with pytest.raises(ValueError, match="unsupported config version"):
        load_config(wrong_version, _Demo)

    extra = tmp_path / "extra.yaml"
    extra.write_text("version: 1\nlearning_rate: 0.5\nunknown: 1\n")
    with pytest.raises(ValueError, match="invalid config"):
        load_config(extra, _Demo)


def test_config_ref_hashes_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("version: 1\n")
    ref = config_ref(path, root=tmp_path)
    assert ref.path == "c.yaml"
    assert ref.sha256 == sha256_file(path)


# ============= Hashing and IO =============


def test_hashes_are_stable() -> None:
    assert sha256_text("a") == sha256_text("a")
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
    assert sha256_text("a") != sha256_text("b")


def test_jsonl_roundtrip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    rows: list[dict[str, Any]] = [{"a": 1}, {"a": "é"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert not list(path.parent.glob("*.tmp"))


# ============= Hardware =============


def test_parse_nvidia_smi_csv() -> None:
    text = "NVIDIA A100-SXM4-80GB, 550.54, 81920\nNVIDIA A100-SXM4-80GB, 550.54, 81920\n"
    gpu_list = parse_nvidia_smi_csv(text)
    assert len(gpu_list) == 2
    assert gpu_list[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert parse_nvidia_smi_csv("") == []


def test_build_hardware_record_applies_overrides() -> None:
    record = build_hardware_record(
        "lab", 1.5, gpu_model_override="Override GPU", gpu_count_override=4
    )
    assert record.gpu_model == "Override GPU"
    assert record.gpu_count == 4
    assert record.provider_label == "lab"


def test_package_versions_skips_missing() -> None:
    versions = package_versions(["pydantic", "definitely-not-installed-pkg"])
    assert "pydantic" in versions
    assert "definitely-not-installed-pkg" not in versions


# ============= Results =============


def test_results_roundtrip(tmp_path: Path, gpu_hardware: HardwareRecord) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("version: 1\n")
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / ".sync_sha").write_text("abc1234-dirty\n")
    results = build_results(
        "e0_test",
        {"x": 1},
        config_ref(cfg, root=tmp_path),
        gpu_hardware,
        {"pydantic": "2"},
        root=tmp_path,
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    assert results.run_id == "2026-01-02T03-04-05Z_e0_test"
    assert results.git_sha == "abc1234"
    assert results.sync_sha == "abc1234-dirty"
    out = write_results(tmp_path / "out.json", results)
    loaded = load_results(out)
    assert loaded.metrics == {"x": 1}
    assert loaded.environment.hardware.gpu_hourly_usd == 2.0
    assert json.loads(out.read_text())["schema_version"] == 1


def test_results_require_price_when_gpu_used(
    tmp_path: Path, gpu_hardware: HardwareRecord, cpu_hardware: HardwareRecord
) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("version: 1\n")
    ref = config_ref(cfg, root=tmp_path)
    unpriced = gpu_hardware.model_copy(update={"gpu_hourly_usd": None})
    with pytest.raises(ValidationError, match="gpu_hourly_usd"):
        build_results("x", {}, ref, unpriced, {}, root=tmp_path)
    build_results("x", {}, ref, cpu_hardware, {}, root=tmp_path)


# ============= Checkpoints =============


def test_latest_checkpoint_skips_incomplete(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path / "missing") is None
    for step, complete in [(10, True), (20, True), (30, False)]:
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        if complete:
            (directory / "trainer_state.json").write_text("{}")
    (tmp_path / "other").mkdir()
    assert latest_checkpoint(tmp_path) == tmp_path / "checkpoint-20"
    assert not is_complete(tmp_path)
    mark_complete(tmp_path)
    assert is_complete(tmp_path)


# ============= Settings and logging =============


def test_settings_hide_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    settings = Settings()
    assert "secret-value" not in repr(settings)
    assert settings.require_groq_key() == "secret-value"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        Settings().require_hf_token()


def test_json_formatter_fields() -> None:
    import logging

    record = logging.LogRecord("n", logging.INFO, "f", 1, "hello", (), None)
    record.run_id = "r1"
    payload = json.loads(JsonFormatter("svc").format(record))
    assert payload["event"] == "hello"
    assert payload["service"] == "svc"
    assert payload["level"] == "info"
    assert payload["run_id"] == "r1"
    assert "ts" in payload
