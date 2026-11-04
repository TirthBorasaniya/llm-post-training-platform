"""Compute backend configuration shared by shell job wrappers and Python results writers."""

import argparse
import os
import re
import shlex
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.hardware import HardwareRecord, build_hardware_record
from llmplatform.common.paths import repo_root

# ============= Constants =============

COMPUTE_CONFIG_RELPATH = "configs/compute.yaml"
CONFIG_ENV_VAR = "LLMPLATFORM_COMPUTE_CONFIG"
BACKENDS = ("slurm", "local")
TIME_PATTERN = re.compile(r"^(\d+-)?\d{1,3}:\d{2}:\d{2}$")
DEFAULT_WORK_ROOT = "~/llmplatform"
LOG_DIRNAME = "logs"


# ============= Models =============


class SlurmSettings(BaseModel):
    """Scheduler settings for the slurm backend."""

    model_config = ConfigDict(extra="forbid")

    partition: str | None = None
    cpu_partition: str | None = None
    smoke_partition: str | None = None
    gpu_type: str | None = None
    max_gpus_per_job: int = 1
    time_limit: str = "08:00:00"
    smoke_time_limit: str = "00:30:00"
    cpus_per_gpu: int = 8
    mem_per_gpu_gb: int = 64
    account: str | None = None
    qos: str | None = None
    extra_args: list[str] = []
    max_chain: int = 6

    @model_validator(mode="after")
    def _check(self) -> "SlurmSettings":
        for value in (self.time_limit, self.smoke_time_limit):
            if not TIME_PATTERN.match(value):
                raise ValueError(f"time limit {value!r} must look like HH:MM:SS or D-HH:MM:SS")
        if self.max_gpus_per_job < 1:
            raise ValueError("max_gpus_per_job must be at least 1")
        return self


class LocalSettings(BaseModel):
    """Settings for the local backend, a single machine with zero or more GPUs."""

    model_config = ConfigDict(extra="forbid")

    max_gpus: int | None = None
    time_limit_hours: float = 8
    max_restarts: int = 6
    use_docker: bool = False
    docker_image: str = "llm-platform:local"
    docker_extra_args: list[str] = []


class JobSpec(BaseModel):
    """Default resources of one job body."""

    model_config = ConfigDict(extra="forbid")

    gpus: int = 1


class ComputeConfig(ConfigModel):
    """Contents of ``configs/compute.yaml``."""

    backend: str
    provider_label: str
    gpu_hourly_usd: float | None = None
    price_source: str | None = None
    price_date: str | None = None
    gpu_model: str | None = None
    work_root: str = DEFAULT_WORK_ROOT
    slurm: SlurmSettings = SlurmSettings()
    local: LocalSettings = LocalSettings()
    jobs: dict[str, JobSpec]

    @model_validator(mode="after")
    def _check(self) -> "ComputeConfig":
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.gpu_hourly_usd is not None and self.gpu_hourly_usd < 0:
            raise ValueError("gpu_hourly_usd must not be negative")
        return self


# ============= Loading =============


def load_compute(path: str | Path | None = None) -> ComputeConfig:
    """
    Load the compute config.

    The path comes from the argument, then ``LLMPLATFORM_COMPUTE_CONFIG``, then the default
    ``configs/compute.yaml`` in the repository.
    """
    explicit = path or os.environ.get(CONFIG_ENV_VAR)
    target = Path(explicit) if explicit else repo_root() / COMPUTE_CONFIG_RELPATH
    return load_config(target, ComputeConfig)


def resolve_work_root(config: ComputeConfig) -> Path:
    """Return the storage root, letting ``PLATFORM_ROOT`` override the config value."""
    raw = os.environ.get("PLATFORM_ROOT") or config.work_root
    return Path(raw).expanduser()


def hardware_record(config: ComputeConfig) -> HardwareRecord:
    """Detect local hardware and merge the provider label and price from the config."""
    return build_hardware_record(
        provider_label=config.provider_label,
        gpu_hourly_usd=config.gpu_hourly_usd,
        price_source=config.price_source,
        price_date=config.price_date,
        gpu_model_override=config.gpu_model,
    )


# ============= Resources =============


def job_gpus(config: ComputeConfig, job: str, requested: int | None = None) -> int:
    """
    Return the GPU count for a job after checking the backend limit.

    Parameters
    ----------
    config : ComputeConfig
        Loaded compute config.
    job : str
        Job body name, a key of ``config.jobs``.
    requested : int, optional
        Explicit GPU count overriding the job default.

    Returns
    -------
    gpus : int
        GPU count to request.
    """
    if job not in config.jobs:
        raise KeyError(f"unknown job {job!r}; known jobs: {', '.join(sorted(config.jobs))}")
    gpus = config.jobs[job].gpus if requested is None else requested
    if gpus < 0:
        raise ValueError("GPU count must not be negative")
    limit = config.slurm.max_gpus_per_job if config.backend == "slurm" else config.local.max_gpus
    if limit is not None and gpus > limit:
        raise ValueError(
            f"job {job!r} needs {gpus} GPUs but the {config.backend} backend allows at most "
            f"{limit} per job; raise the limit in configs/compute.yaml if the hardware has more"
        )
    return gpus


def sbatch_args(
    config: ComputeConfig,
    job: str,
    gpus: int | None = None,
    o_smoke: bool = False,
    time_limit: str | None = None,
    log_dir: str = LOG_DIRNAME,
) -> list[str]:
    """
    Build the ``sbatch`` option list for a job from the slurm settings.

    Parameters
    ----------
    config : ComputeConfig
        Loaded compute config with ``backend: slurm`` settings.
    job : str
        Job body name.
    gpus : int, optional
        Overrides the job default GPU count.
    o_smoke : bool
        Use the smoke partition and time limit.
    time_limit : str, optional
        Overrides the configured time limit.
    log_dir : str
        Directory receiving scheduler output files.

    Returns
    -------
    arg_list : list[str]
        One option per element, ready to pass to ``sbatch``.
    """
    slurm = config.slurm
    count = job_gpus(config, job, gpus)
    if count > 0:
        partition = (slurm.smoke_partition if o_smoke else None) or slurm.partition
    else:
        partition = slurm.cpu_partition or slurm.partition
    if partition is None:
        raise ValueError("set slurm.partition in configs/compute.yaml before submitting")
    limit = time_limit or (slurm.smoke_time_limit if o_smoke else slurm.time_limit)
    if not TIME_PATTERN.match(limit):
        raise ValueError(f"time limit {limit!r} must look like HH:MM:SS")
    arg_list = [
        f"--job-name={job}",
        f"--partition={partition}",
        f"--time={limit}",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        f"--cpus-per-task={max(1, count) * slurm.cpus_per_gpu}",
        f"--mem={max(1, count) * slurm.mem_per_gpu_gb}G",
    ]
    if count > 0:
        gres = f"gpu:{slurm.gpu_type}:{count}" if slurm.gpu_type else f"gpu:{count}"
        arg_list.append(f"--gres={gres}")
    if slurm.account:
        arg_list.append(f"--account={slurm.account}")
    if slurm.qos:
        arg_list.append(f"--qos={slurm.qos}")
    arg_list.extend(slurm.extra_args)
    return arg_list


def shell_env(config: ComputeConfig) -> dict[str, str]:
    """
    Return the environment variables that job scripts source.

    Parameters
    ----------
    config : ComputeConfig
        Loaded compute config.

    Returns
    -------
    env_dict : dict[str, str]
        Variable names mapped to values; paths are absolute.
    """
    root = resolve_work_root(config)
    env_dict = {
        "PLATFORM_ROOT": str(root),
        "ENV_ROOT": str(root / "envs"),
        "HF_HOME": str(root / "hf"),
        "DATA_ROOT": str(root / "data"),
        "CKPT_ROOT": str(root / "checkpoints"),
        "MODEL_ROOT": str(root / "models"),
        "COMPUTE_BACKEND": config.backend,
        "PROVIDER_LABEL": config.provider_label,
        "SLURM_MAX_CHAIN": str(config.slurm.max_chain),
        "LOCAL_TIME_LIMIT_HOURS": str(config.local.time_limit_hours),
        "LOCAL_MAX_RESTARTS": str(config.local.max_restarts),
        "LOCAL_USE_DOCKER": "1" if config.local.use_docker else "0",
        "LOCAL_DOCKER_IMAGE": config.local.docker_image,
    }
    return env_dict


def format_exports(env_dict: dict[str, str]) -> str:
    """Render a variable dict as shell ``export`` lines with safe quoting."""
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env_dict.items())


# ============= CLI =============


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m llmplatform.common.compute``."""
    parser = argparse.ArgumentParser(description="Query the compute configuration.")
    parser.add_argument("--config", default=None, help="path of compute.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("backend", help="print the configured backend")
    sub.add_parser("shell-env", help="print export lines for job scripts")
    sub.add_parser("validate", help="validate the config file")
    sbatch = sub.add_parser("sbatch-args", help="print sbatch options, one per line")
    sbatch.add_argument("--job", required=True)
    sbatch.add_argument("--gpus", type=int, default=None)
    sbatch.add_argument("--smoke", action="store_true")
    sbatch.add_argument("--time", default=None)
    gpus = sub.add_parser("gpus", help="print the checked GPU count for a job")
    gpus.add_argument("--job", required=True)
    gpus.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_compute(args.config)
    if args.command == "backend":
        print(config.backend)
    elif args.command == "shell-env":
        print(format_exports(shell_env(config)))
    elif args.command == "validate":
        print(f"ok: backend={config.backend}")
    elif args.command == "sbatch-args":
        for item in sbatch_args(config, args.job, args.gpus, args.smoke, args.time):
            print(item)
    elif args.command == "gpus":
        print(job_gpus(config, args.job, args.gpus))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
