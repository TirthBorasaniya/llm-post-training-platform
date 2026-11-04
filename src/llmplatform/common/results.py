"""Results file schema, construction, validation, and writing."""

import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from llmplatform.common.config import ConfigRef
from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.io import read_json, write_json
from llmplatform.common.paths import SYNC_SHA_FILENAME, repo_root

# ============= Constants =============

SCHEMA_VERSION = 1
GIT_TIMEOUT_SECONDS = 20
RUN_ID_TIME_FORMAT = "%Y-%m-%dT%H-%M-%SZ"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
RUNTIME_ENV_VAR = "LLMPLATFORM_ENV_MODE"


# ============= Models =============


class Environment(BaseModel):
    """Software and hardware context of a run."""

    python: str
    packages: dict[str, str]
    hardware: HardwareRecord
    runtime: str | None = None


class ResultsFile(BaseModel):
    """One evaluation, benchmark, or load-test artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    git_sha: str
    sync_sha: str | None = None
    timestamp_utc: str
    config_path: str
    config_sha256: str
    environment: Environment
    metrics: dict[str, Any]
    notes: str = ""

    @model_validator(mode="after")
    def _require_price_for_gpu_runs(self) -> "ResultsFile":
        hardware = self.environment.hardware
        if hardware.gpu_count > 0 and hardware.gpu_hourly_usd is None:
            raise ValueError(
                "gpu_hourly_usd must be set in the compute config for runs that use a GPU"
            )
        return self


# ============= Code version =============


def read_code_version(root: str | Path | None = None) -> tuple[str, str | None]:
    """
    Return ``(git_sha, sync_sha)`` for the code that produced a result.

    The synced working tree carries a ``.sync_sha`` file written by the sync target. When it is
    absent the value comes from git directly, with a ``-dirty`` suffix for uncommitted changes.

    Parameters
    ----------
    root : str or Path, optional
        Repository root. Defaults to the discovered root.

    Returns
    -------
    git_sha : str
        Commit hash without the dirty suffix, or ``unknown`` outside a git checkout.
    sync_sha : str or None
        Full content of ``.sync_sha`` when the file exists.
    """
    base = Path(root) if root else repo_root()
    sync_file = base / SYNC_SHA_FILENAME
    if sync_file.is_file():
        sync_value = sync_file.read_text(encoding="utf-8").strip()
        return sync_value.removesuffix("-dirty"), sync_value
    try:
        sha = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown", None
    return sha, None if not dirty else f"{sha}-dirty"


# ============= Construction =============


def build_results(
    name: str,
    metrics: dict[str, Any],
    config: ConfigRef,
    hardware: HardwareRecord,
    packages: dict[str, str],
    notes: str = "",
    root: str | Path | None = None,
    now: datetime | None = None,
) -> ResultsFile:
    """
    Assemble a results file.

    Parameters
    ----------
    name : str
        Experiment name used in the run id, for example ``e1_model_quality``.
    metrics : dict
        Measured values. Never estimated or hand-edited.
    config : ConfigRef
        Config file path and SHA-256 that drove the run.
    hardware : HardwareRecord
        GPU model, GPU count, provider label, and hourly price.
    packages : dict[str, str]
        Versions of the packages that matter to the result.
    notes : str
        Free-text caveats such as the serving backend used.
    """
    moment = now or datetime.now(UTC)
    git_sha, sync_sha = read_code_version(root)
    return ResultsFile(
        run_id=f"{moment.strftime(RUN_ID_TIME_FORMAT)}_{name}",
        git_sha=git_sha,
        sync_sha=sync_sha,
        timestamp_utc=moment.strftime(TIMESTAMP_FORMAT),
        config_path=config.path,
        config_sha256=config.sha256,
        environment=Environment(
            python=platform.python_version(),
            packages=packages,
            hardware=hardware,
            runtime=os.environ.get(RUNTIME_ENV_VAR),
        ),
        metrics=metrics,
        notes=notes,
    )


# ============= IO =============


def write_results(path: str | Path, results: ResultsFile) -> Path:
    """Write a results file as sorted, indented JSON."""
    return write_json(path, results.model_dump(mode="json"))


def load_results(path: str | Path) -> ResultsFile:
    """Read and validate a results file."""
    return ResultsFile.model_validate(read_json(path))
