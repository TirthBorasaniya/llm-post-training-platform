"""Documentation, Makefile, workflow, and job consistency checks."""

import re
from pathlib import Path

import pytest
import yaml

from llmplatform.common.compute import load_compute
from llmplatform.common.render import BLOCKS, process

pytestmark = pytest.mark.unit

README_SECTIONS = [
    "## Headline results",
    "## Architecture",
    "## Quickstart",
    "## How it works",
    "## Evaluation",
    "## Design decisions",
    "## Limitations",
    "## Repository layout",
]
GENERATED_PATHS = {
    "eval/judge_prompts.jsonl",
    "eval/agent_traces/traces.jsonl",
}
DOC_FILES = ["README.md", "DECISIONS.md", "SECURITY.md"]


def doc_paths(root: Path) -> list[Path]:
    return [root / name for name in DOC_FILES] + sorted((root / "docs").glob("*.md"))


def makefile_targets(root: Path) -> set[str]:
    text = (root / "Makefile").read_text()
    return set(re.findall(r"^([a-zA-Z0-9_-]+):", text, flags=re.MULTILINE))


# ============= README =============


def test_readme_follows_the_template(root: Path) -> None:
    text = (root / "README.md").read_text()
    positions = [text.index(section) for section in README_SECTIONS]
    assert positions == sorted(positions)
    assert text.index("# LLM Post-Training") == 0
    assert (
        "```mermaid" in text and "make setup" in text and "make up" in text and "make demo" in text
    )


def test_readme_has_a_block_for_every_experiment_and_none_are_filled(root: Path) -> None:
    text = (root / "README.md").read_text()
    for name in ["headline", *[n for n in BLOCKS if n != "data_manifest"]]:
        assert f"<!-- results:{name}:start -->\n<!-- results:{name}:end -->" in text, name
    assert (
        "<!-- results:data_manifest:start -->\n<!-- results:data_manifest:end -->"
        in (root / "docs/DATA.md").read_text()
    )


def test_results_check_passes_with_no_results(root: Path, tmp_path: Path) -> None:
    assert process(root, tmp_path / "empty-results", o_check=True) == []


def test_readme_contains_no_status_text_or_roadmap(root: Path) -> None:
    text = (root / "README.md").read_text().lower()
    for phrase in ("todo", "coming soon", "roadmap", "work in progress", "tbd", "placeholder"):
        assert phrase not in text, phrase


# ============= References =============


def test_make_targets_named_in_docs_exist(root: Path) -> None:
    targets = makefile_targets(root)
    missing = []
    for path in doc_paths(root):
        for target in re.findall(r"`make ([a-z0-9-]+)", path.read_text()):
            if target not in targets:
                missing.append(f"{path.name}: make {target}")
    assert missing == []


def test_paths_named_in_docs_exist(root: Path) -> None:
    pattern = re.compile(
        r"`((?:configs|scripts|jobs|src|docs|charts|k8s|k6|space|envs|slurm|local|docker|eval|tests)"
        r"/[A-Za-z0-9_./-]+)`"
    )
    missing = []
    for path in doc_paths(root):
        for match in pattern.findall(path.read_text()):
            candidate = match.rstrip("/.")
            if "*" in candidate or candidate in GENERATED_PATHS:
                continue
            if not (root / candidate).exists():
                missing.append(f"{path.name}: {candidate}")
    assert missing == []


def test_jobs_used_by_make_are_defined(root: Path) -> None:
    compute = load_compute(root / "configs/compute.yaml")
    text = (root / "Makefile").read_text()
    for job in set(re.findall(r"\$\(RUN\) ([a-z0-9_]+)", text)):
        assert (root / "jobs" / f"{job}.sh").is_file(), job
        assert job in compute.jobs, job


def test_every_job_body_has_a_compute_entry(root: Path) -> None:
    compute = load_compute(root / "configs/compute.yaml")
    helpers = {"common", "lib"}
    for path in (root / "jobs").glob("*.sh"):
        if path.stem not in helpers:
            assert path.stem in compute.jobs, path.stem


def test_job_scripts_are_executable_and_source_common(root: Path) -> None:
    for path in (root / "jobs").glob("*.sh"):
        if path.stem in {"common", "lib"}:
            continue
        assert path.stat().st_mode & 0o111, path.name
        assert "common.sh" in path.read_text(), path.name


# ============= Workflows =============


def test_unproven_workflows_run_only_on_manual_dispatch(root: Path) -> None:
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        data = yaml.safe_load(path.read_text())
        triggers = data.get(True, data.get("on"))
        assert set(triggers) == {"workflow_dispatch"}, path.name


def test_workflows_pin_actions_by_commit(root: Path) -> None:
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        for line in path.read_text().splitlines():
            if "uses:" in line:
                assert re.search(r"@[0-9a-f]{40}", line), f"{path.name}: {line.strip()}"
