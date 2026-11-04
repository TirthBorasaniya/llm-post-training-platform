"""Tests for quantization planning, GGUF tooling, model cards, and Hub publishing."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmplatform.cards.render import CardContext, markdown_table, render_model_card
from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.models import load_models
from llmplatform.data.config import load_data_config
from llmplatform.eval.config import load_e1
from llmplatform.eval.e1 import update_checkpoint_file
from llmplatform.eval.variants import launch_args
from llmplatform.publish.hub import (
    confirm_apache,
    list_upload_files,
    load_publish,
    publish_files,
    publish_folder,
)
from llmplatform.quantize.config import load_quantize
from llmplatform.quantize.gguf import (
    convert_and_quantize,
    convert_command,
    quantize_command,
    resolve_tool,
    verify_command,
    verify_gguf,
)
from llmplatform.quantize.plan import choose_method, parse_capability, variant_names

pytestmark = pytest.mark.unit

REF = ConfigRef(path="configs/eval/judge.yaml", sha256="0" * 64)
HW = HardwareRecord(gpu_model="none", gpu_count=0, provider_label="t", gpu_hourly_usd=None)


# ============= Planning =============


@pytest.mark.parametrize(
    ("capability", "method"), [(8.0, "awq"), (8.6, "awq"), (8.9, "fp8"), (9.0, "fp8")]
)
def test_method_by_gpu_generation(capability: float, method: str) -> None:
    assert choose_method(capability, load_quantize().gpu.fp8_min_capability) == method


def test_capability_parsing_and_names() -> None:
    assert parse_capability((8, 9)) == 8.9 and parse_capability("9.0") == 9.0
    assert variant_names(["teacher"], "fp8") == {"teacher": "teacher_fp8"}


def test_quantize_config_matches_protocol() -> None:
    config = load_quantize()
    assert config.gguf.quant_type == "Q4_K_M"
    assert len(config.gguf.verify.prompts) == 5
    assert set(config.gguf.models) == {"teacher", "distilled_1_5b", "base_1_5b"}
    assert "lm_head" in config.gpu.awq.ignore
    assert config.gguf.llama_cpp_tag == "b12110"


# ============= GGUF =============


def fake_build(tmp_path: Path, unified: bool) -> Path:
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    names = ["llama"] if unified else ["llama-quantize", "llama-cli", "llama-server"]
    for name in names:
        (bin_dir / name).write_text("")
    return tmp_path


def test_tool_resolution_prefers_unified_binary(tmp_path: Path) -> None:
    unified = fake_build(tmp_path / "u", True)
    assert resolve_tool(unified, "quantize")[-1] == "quantize"
    assert resolve_tool(unified, "cli")[0].endswith("/llama")
    legacy = fake_build(tmp_path / "l", False)
    assert resolve_tool(legacy, "quantize")[0].endswith("llama-quantize")
    assert resolve_tool(legacy, "serve")[0].endswith("llama-server")
    with pytest.raises(FileNotFoundError):
        resolve_tool(tmp_path / "none", "cli")
    with pytest.raises(ValueError):
        resolve_tool(unified, "other")


def test_gguf_commands(tmp_path: Path) -> None:
    build = fake_build(tmp_path, True)
    convert = convert_command(build, "/m", "/o.gguf", "f16")
    assert convert[1].endswith("convert_hf_to_gguf.py")
    assert convert[convert.index("--outfile") + 1] == "/o.gguf"
    assert "--outtype" in convert and convert[convert.index("--outtype") + 1] == "f16"
    assert quantize_command(build, "a.gguf", "b.gguf")[-3:] == ["a.gguf", "b.gguf", "Q4_K_M"]
    verify = verify_command(build, "m.gguf", "hi", 16)
    assert "-st" in verify and verify[verify.index("-p") + 1] == "hi"
    assert verify[verify.index("-n") + 1] == "16" and verify[verify.index("--temp") + 1] == "0"


def test_convert_and_quantize_runs_both_steps_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = fake_build(tmp_path / "build_dir", True)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        if command[0].endswith("python") or "convert" in command[1]:
            Path(command[command.index("--outfile") + 1]).write_text("float")
        else:
            Path(command[-2]).write_text("quantized")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = convert_and_quantize(build, "/m", tmp_path / "out", "teacher")
    assert out.name == "teacher-Q4_K_M.gguf" and out.read_text() == "quantized"
    assert not (tmp_path / "out" / "teacher-f16.gguf").exists()
    assert len(calls) == 2


def test_verify_requires_clean_exit_and_text(tmp_path: Path) -> None:
    build = fake_build(tmp_path, True)

    def runner(command: list[str], **kwargs: Any) -> Any:
        prompt = command[command.index("-p") + 1]
        if prompt == "bad":
            return SimpleNamespace(returncode=1, stdout="text")
        if prompt == "silent":
            return SimpleNamespace(returncode=0, stdout="  ")
        return SimpleNamespace(returncode=0, stdout=" ok ")

    results = verify_gguf(build, "m.gguf", ["good", "bad", "silent"], 8, 10, runner)
    assert [item.ok for item in results] == [True, False, False]
    assert results[0].output == "ok"


# ============= Variants =============


def test_variant_launch_args() -> None:
    entries = {entry.name: entry for entry in load_e1().checkpoints}
    fp8 = launch_args(entries["teacher_fp8"], "/ckpt/teacher/merged")
    assert fp8[:2] == ["--engine", "sglang"] and "--quantization" in fp8
    assert fp8[fp8.index("--served-model-name") + 1] == "teacher_fp8"
    gguf = launch_args(entries["teacher_q4km"], "/m.gguf", "/llama")
    assert gguf[:2] == ["--engine", "llamacpp"] and gguf[-2:] == ["--llama-cpp-dir", "/llama"]
    with pytest.raises(ValueError, match="LLAMA_CPP_DIR"):
        launch_args(entries["teacher_q4km"], "/m.gguf")
    with pytest.raises(ValueError, match="no serve"):
        launch_args(entries["sft"], "/m")


def test_quantized_variants_are_registered_and_optional_flags_set() -> None:
    entries = {entry.name: entry for entry in load_e1().checkpoints}
    models = load_models().models
    for name in ("teacher_q4km", "distilled_1_5b_q4km", "base_1_5b_q4km"):
        assert not entries[name].optional and entries[name].tasks == ["ifeval"]
        assert entries[name].model in models
    assert entries["teacher_fp8"].optional and entries["teacher_awq"].optional


# ============= Cards =============


def make_context(
    results_root: Path, kind: str = "teacher", license_id: str = "apache-2.0"
) -> CardContext:
    return CardContext(
        kind=kind,
        repo_id="ns/qwen2.5-7b-aligned",
        base_model_id="Qwen/Qwen2.5-7B-Instruct",
        base_model_license=license_id,
        models=load_models(),
        data=load_data_config(),
        results_root=results_root,
        gguf_files=["teacher-Q4_K_M.gguf"],
    )


def test_card_without_results_has_no_tables(tmp_path: Path) -> None:
    card = render_model_card(make_context(tmp_path))
    assert card.startswith("---\nlicense: apache-2.0")
    assert "| Checkpoint" not in card and "Evaluation" not in card
    assert "HuggingFaceH4/ultrachat_200k" in card and "## License" in card


def test_card_tables_come_from_results(tmp_path: Path) -> None:
    judge = {
        "win_rate": 0.612,
        "ci95_low": 0.55,
        "ci95_high": 0.67,
        "candidate_length": {"mean_words": 210.0},
    }
    update_checkpoint_file(
        tmp_path, "teacher", "benchmarks", {"tasks": {"gsm8k": {"value": 0.8123}}}, REF, HW, {}
    )
    update_checkpoint_file(tmp_path, "teacher", "judge", judge, REF, HW, {})
    card = render_model_card(make_context(tmp_path))
    assert "| teacher |" in card
    assert "0.812" in card and "0.612" in card and "0.550 to 0.670" in card
    assert "Provenance" in card
    student = render_model_card(make_context(tmp_path, "student"))
    assert "| teacher |" not in student


def test_gguf_card_and_license_gate(tmp_path: Path) -> None:
    card = render_model_card(make_context(tmp_path, "gguf"))
    assert "teacher-Q4_K_M.gguf" in card and "library_name: gguf" in card
    with pytest.raises(ValueError, match="license"):
        render_model_card(make_context(tmp_path, license_id="other"))
    with pytest.raises(ValueError, match="unknown card kind"):
        render_model_card(make_context(tmp_path, "weird"))
    assert markdown_table(["a"], [["1"]]) == "| a |\n|---|\n| 1 |"


# ============= Publishing =============


class FakeApi:
    def __init__(self, license_id: str | None = "apache-2.0") -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.license_id = license_id

    def model_info(self, hub_id: str, revision: str) -> Any:
        return SimpleNamespace(card_data=SimpleNamespace(license=self.license_id))

    def create_repo(self, repo_id: str, **kwargs: Any) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kwargs}))

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", kwargs))

    def upload_file(self, **kwargs: Any) -> None:
        self.calls.append(("upload_file", kwargs))


def test_license_confirmation() -> None:
    assert confirm_apache(FakeApi(), "Qwen/x", "rev") == "apache-2.0"
    with pytest.raises(RuntimeError, match="expected"):
        confirm_apache(FakeApi("mit"), "Qwen/x", "rev")
    with pytest.raises(RuntimeError):
        confirm_apache(FakeApi(None), "Qwen/x", "rev")


def test_upload_listing_skips_training_state(tmp_path: Path) -> None:
    for name in ("model.safetensors", "config.json", "optimizer.pt", "DONE", "checkpoint-5/x.bin"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    patterns = load_publish().ignore_patterns
    assert list_upload_files(tmp_path, patterns) == ["config.json", "model.safetensors"]


def test_publish_folder_dry_run_and_real(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    api = FakeApi()
    patterns = load_publish().ignore_patterns
    files = publish_folder(api, tmp_path, "ns/m", "card", True, patterns, o_dry_run=True)
    assert files == ["config.json", "README.md"] and api.calls == []
    publish_folder(api, tmp_path, "ns/m", "card text", True, patterns)
    kinds = [name for name, _ in api.calls]
    assert kinds == ["create_repo", "upload_folder", "upload_file"]
    assert api.calls[0][1]["private"] is True
    assert api.calls[2][1]["path_in_repo"] == "README.md"
    assert api.calls[2][1]["path_or_fileobj"] == b"card text"


def test_publish_files_uploads_each_gguf(tmp_path: Path) -> None:
    first = tmp_path / "a.gguf"
    first.write_text("x")
    api = FakeApi()
    sent = publish_files(api, [first], "ns/g", "card", False)
    assert sent == ["a.gguf", "README.md"]
    assert [name for name, _ in api.calls].count("upload_file") == 2


def test_publish_config_defaults_to_private_and_has_no_account() -> None:
    config = load_publish()
    assert config.private is True
    assert set(config.repos) == {"teacher", "student", "gguf"}
    assert all("/" not in repo.name for repo in config.repos.values())
