"""Tests for the compute configuration, scheduler flags, and job wrapper scripts."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from llmplatform.common.compute import (
    ComputeConfig,
    format_exports,
    job_gpus,
    load_compute,
    sbatch_args,
    shell_env,
)
from llmplatform.common.config import load_config

pytestmark = pytest.mark.unit

BASE = {
    "version": 1,
    "backend": "slurm",
    "provider_label": "test",
    "slurm": {
        "partition": "p1",
        "cpu_partition": "pc",
        "smoke_partition": "ps",
        "gpu_type": "a100",
        "max_gpus_per_job": 4,
        "time_limit": "02:00:00",
        "smoke_time_limit": "00:10:00",
        "account": "acct",
        "qos": "normal",
        "extra_args": ["--constraint=x"],
    },
    "jobs": {"train": {"gpus": 1}, "gguf": {"gpus": 0}, "disagg": {"gpus": 2}},
}


def make_config(tmp_path: Path, **overrides: object) -> ComputeConfig:
    data = {**BASE, **overrides}
    path = tmp_path / "compute.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path, ComputeConfig)


# ============= Config =============


def test_repo_compute_config_is_valid(root: Path) -> None:
    config = load_compute(root / "configs" / "compute.yaml")
    assert config.backend in ("slurm", "local")
    assert config.gpu_hourly_usd is None
    assert "disagg" in config.jobs


def test_invalid_backend_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        make_config(tmp_path, backend="kubernetes")


def test_bad_time_limit_rejected(tmp_path: Path) -> None:
    bad = {**BASE["slurm"], "time_limit": "8h"}  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="time limit"):
        make_config(tmp_path, slurm=bad)


# ============= Scheduler flags =============


def test_sbatch_args_for_gpu_job(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    args = sbatch_args(config, "train", gpus=2)
    assert "--partition=p1" in args
    assert "--gres=gpu:a100:2" in args
    assert "--time=02:00:00" in args
    assert "--cpus-per-task=16" in args
    assert "--account=acct" in args
    assert "--qos=normal" in args
    assert "--constraint=x" in args


def test_sbatch_args_smoke_and_cpu(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    smoke = sbatch_args(config, "train", o_smoke=True)
    assert "--partition=ps" in smoke
    assert "--time=00:10:00" in smoke
    cpu = sbatch_args(config, "gguf")
    assert "--partition=pc" in cpu
    assert not any(item.startswith("--gres") for item in cpu)


def test_gpu_limit_enforced(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert job_gpus(config, "disagg") == 2
    with pytest.raises(ValueError, match="at most 4"):
        job_gpus(config, "train", 8)
    with pytest.raises(KeyError):
        job_gpus(config, "nope")


def test_partition_required_for_submission(tmp_path: Path) -> None:
    config = make_config(tmp_path, slurm={"max_gpus_per_job": 1})
    with pytest.raises(ValueError, match="partition"):
        sbatch_args(config, "train")


def test_gpu_type_optional(tmp_path: Path) -> None:
    slurm = {**BASE["slurm"], "gpu_type": None}  # type: ignore[dict-item]
    args = sbatch_args(make_config(tmp_path, slurm=slurm), "train")
    assert "--gres=gpu:1" in args


def test_shell_env_quotes_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLATFORM_ROOT", str(tmp_path / "with space"))
    config = make_config(tmp_path, work_root="~/ignored")
    env_dict = shell_env(config)
    assert env_dict["ENV_ROOT"].endswith("with space/envs")
    exports = format_exports(env_dict)
    assert "'" in exports
    assert "export COMPUTE_BACKEND=slurm" in exports


# ============= Scripts =============


def run_script(
    root: Path, args: list[str], tmp_path: Path, **extra: str
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "PLATFORM_ROOT": str(tmp_path / "work"),
        "ENV_FILE": "/dev/null",
        **extra,
    }
    return subprocess.run(
        [str(root / args[0]), *args[1:]], capture_output=True, text=True, env=env, cwd=root
    )


@pytest.mark.parametrize(
    "script",
    [
        "jobs/lib.sh",
        "jobs/common.sh",
        "jobs/env_info.sh",
        "jobs/setup_envs.sh",
        "slurm/submit.sh",
        "slurm/run_job.sbatch",
        "local/run.sh",
        "local/docker_run.sh",
        "scripts/run_job.sh",
        "scripts/sync.sh",
        "scripts/pull_results.sh",
    ],
)
def test_shell_syntax(root: Path, script: str) -> None:
    result = subprocess.run(["bash", "-n", str(root / script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_submit_dry_run_chain(root: Path, tmp_path: Path) -> None:
    compute = tmp_path / "compute.yaml"
    compute.write_text(yaml.safe_dump(BASE))
    result = run_script(
        root,
        [
            "slurm/submit.sh",
            "train",
            "--gpus",
            "2",
            "--chain",
            "2",
            "--dry-run",
            "--",
            "--stage",
            "dpo",
        ],
        tmp_path,
        LLMPLATFORM_COMPUTE_CONFIG=str(compute),
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2
    assert "--gres=gpu:a100:2" in lines[0]
    assert "LLMPLATFORM_GPUS=2" in lines[0]
    assert "slurm/run_job.sbatch train --stage dpo" in lines[0]
    assert "--dependency=afterany:DRYRUN1" in lines[1]


def test_submit_rejects_excess_gpus_and_chain(root: Path, tmp_path: Path) -> None:
    compute = tmp_path / "compute.yaml"
    compute.write_text(yaml.safe_dump(BASE))
    env = {"LLMPLATFORM_COMPUTE_CONFIG": str(compute)}
    too_many = run_script(
        root, ["slurm/submit.sh", "train", "--gpus", "9", "--dry-run"], tmp_path, **env
    )
    assert too_many.returncode != 0
    too_long = run_script(
        root, ["slurm/submit.sh", "train", "--chain", "99", "--dry-run"], tmp_path, **env
    )
    assert too_long.returncode != 0
    assert "max_chain" in too_long.stderr


def test_local_dry_run(root: Path, tmp_path: Path) -> None:
    result = run_script(root, ["local/run.sh", "env_info", "--dry-run"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "jobs/env_info.sh" in result.stdout
    assert "LLMPLATFORM_GPUS=0" in result.stdout


def test_sync_dry_run_uses_syncignore(root: Path, tmp_path: Path) -> None:
    result = run_script(
        root,
        ["scripts/sync.sh"],
        tmp_path,
        DRY_RUN="1",
        REMOTE_HOST="example-host",
        REMOTE_DIR="/srv/example",
    )
    try:
        assert result.returncode == 0, result.stderr
        assert "--exclude-from=.syncignore" in result.stdout
        assert "example-host:/srv/example/" in result.stdout
        assert (root / ".sync_sha").is_file()
    finally:
        (root / ".sync_sha").unlink(missing_ok=True)


def test_sync_requires_remote(root: Path, tmp_path: Path) -> None:
    result = run_script(root, ["scripts/sync.sh"], tmp_path, DRY_RUN="1")
    assert result.returncode != 0
    assert "REMOTE_HOST" in result.stderr


def test_pull_results_dry_run(root: Path, tmp_path: Path) -> None:
    result = run_script(
        root,
        ["scripts/pull_results.sh"],
        tmp_path,
        DRY_RUN="1",
        REMOTE_HOST="example-host",
        REMOTE_DIR="/srv/example",
    )
    assert result.returncode == 0, result.stderr
    assert "example-host:/srv/example/results/" in result.stdout
    assert "example-host:/srv/example/logs/" in result.stdout


def test_syncignore_excludes_every_private_path(root: Path) -> None:
    patterns = (root / ".syncignore").read_text().splitlines()
    required = [
        ".planning/",
        "CLAUDE.md",
        ".claude/",
        ".env",
        ".venv/",
        ".cache/",
        ".tmp/",
        ".kube/",
        "node_modules/",
        "data/",
        "models/",
        "mlruns/",
        ".git/",
    ]
    for entry in required:
        assert entry in patterns, entry


def test_env_files_never_leave_the_machine(root: Path) -> None:
    patterns = (root / ".syncignore").read_text().splitlines()
    assert ".env.*" in patterns
