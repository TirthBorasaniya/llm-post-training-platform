"""Hardware detection for results files: GPU model and count, driver, CUDA, and host CPU."""

import platform
import shutil
import subprocess
from importlib import metadata

from pydantic import BaseModel

# ============= Constants =============

NVIDIA_SMI_QUERY = ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
NO_GPU_LABEL = "none"
SMI_TIMEOUT_SECONDS = 20


# ============= Models =============


class HardwareRecord(BaseModel):
    """Hardware and pricing context stored in every results file."""

    gpu_model: str
    gpu_count: int
    provider_label: str
    gpu_hourly_usd: float | None
    price_source: str | None = None
    price_date: str | None = None
    driver: str | None = None
    cuda: str | None = None
    gpu_memory_gb: float | None = None
    cpu: str = ""


# ============= Detection =============


def parse_nvidia_smi_csv(text: str) -> list[dict[str, str]]:
    """
    Parse ``nvidia-smi --query-gpu=name,driver_version,memory.total`` CSV output.

    Parameters
    ----------
    text : str
        Raw command output, one GPU per line.

    Returns
    -------
    gpu_list : list[dict[str, str]]
        One dict per GPU with keys ``name``, ``driver``, and ``memory_mib``.
    """
    gpu_list = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        gpu_list.append({"name": parts[0], "driver": parts[1], "memory_mib": parts[2]})
    return gpu_list


def query_nvidia_smi() -> list[dict[str, str]]:
    """Return GPU descriptions from ``nvidia-smi`` or an empty list when unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        completed = subprocess.run(
            [binary, *NVIDIA_SMI_QUERY],
            capture_output=True,
            text=True,
            check=True,
            timeout=SMI_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_nvidia_smi_csv(completed.stdout)


def detect_cuda_version() -> str | None:
    """Return the CUDA version PyTorch was built with, when PyTorch is importable."""
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.version, "cuda", None)


def host_cpu_description() -> str:
    """Return a short CPU description for the host."""
    return f"{platform.machine()} {platform.processor() or platform.system()}".strip()


def package_versions(name_list: list[str]) -> dict[str, str]:
    """Return installed versions for ``name_list``, skipping packages that are not installed."""
    version_dict = {}
    for name in name_list:
        try:
            version_dict[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return version_dict


def build_hardware_record(
    provider_label: str,
    gpu_hourly_usd: float | None,
    price_source: str | None = None,
    price_date: str | None = None,
    gpu_model_override: str | None = None,
    gpu_count_override: int | None = None,
) -> HardwareRecord:
    """
    Detect local hardware and combine it with the provider label and price from the compute config.

    Parameters
    ----------
    provider_label : str
        Free-text provider description from the compute config.
    gpu_hourly_usd : float or None
        Hourly price per GPU used for cost figures.
    gpu_model_override, gpu_count_override : optional
        Values that replace detection, for hosts where ``nvidia-smi`` is hidden from the job.
    """
    gpu_list = query_nvidia_smi()
    detected_model = gpu_list[0]["name"] if gpu_list else NO_GPU_LABEL
    detected_count = len(gpu_list)
    memory_gb = None
    if gpu_list and gpu_list[0]["memory_mib"].replace(".", "", 1).isdigit():
        memory_gb = round(float(gpu_list[0]["memory_mib"]) / 1024.0, 1)
    return HardwareRecord(
        gpu_model=gpu_model_override or detected_model,
        gpu_count=gpu_count_override if gpu_count_override is not None else detected_count,
        provider_label=provider_label,
        gpu_hourly_usd=gpu_hourly_usd,
        price_source=price_source,
        price_date=price_date,
        driver=gpu_list[0]["driver"] if gpu_list else None,
        cuda=detect_cuda_version(),
        gpu_memory_gb=memory_gb,
        cpu=host_cpu_description(),
    )
