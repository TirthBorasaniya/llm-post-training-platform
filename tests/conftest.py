"""Shared fixtures."""

from pathlib import Path

import pytest

from llmplatform.common.hardware import HardwareRecord
from llmplatform.common.paths import repo_root


@pytest.fixture(scope="session")
def root() -> Path:
    """Repository root."""
    return repo_root()


@pytest.fixture()
def cpu_hardware() -> HardwareRecord:
    """Hardware record for a host without a GPU."""
    return HardwareRecord(
        gpu_model="none",
        gpu_count=0,
        provider_label="test-host",
        gpu_hourly_usd=None,
        cpu="test-cpu",
    )


@pytest.fixture()
def gpu_hardware() -> HardwareRecord:
    """Hardware record for a priced single-GPU host."""
    return HardwareRecord(
        gpu_model="Test GPU 80GB",
        gpu_count=1,
        provider_label="test-provider",
        gpu_hourly_usd=2.0,
        price_source="unit test",
        price_date="2026-01-01",
        cpu="test-cpu",
    )
