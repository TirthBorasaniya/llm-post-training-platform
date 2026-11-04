"""Choose the GPU quantization method from the device's compute capability."""

from typing import Literal

# ============= Constants =============

Method = Literal["fp8", "awq"]


# ============= Planning =============


def parse_capability(value: str | tuple[int, int]) -> float:
    """Convert ``(8, 9)`` or ``"8.9"`` to the float ``8.9``."""
    if isinstance(value, tuple):
        return value[0] + value[1] / 10.0
    return float(value)


def choose_method(capability: float, fp8_min_capability: float) -> Method:
    """
    Return ``fp8`` on Ada and Hopper class GPUs and ``awq`` on older ones.

    Parameters
    ----------
    capability : float
        CUDA compute capability, for example 8.0 for Ampere A100 or 9.0 for Hopper.
    fp8_min_capability : float
        Lowest capability with FP8 tensor cores.
    """
    return "fp8" if capability >= fp8_min_capability else "awq"


def detect_capability() -> float | None:
    """Return the capability of GPU 0, or ``None`` without a CUDA device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return parse_capability(torch.cuda.get_device_capability(0))


def variant_names(models: list[str], method: Method) -> dict[str, str]:
    """Map each model to its GPU quantization variant name, for example ``teacher_fp8``."""
    return {model: f"{model}_{method}" for model in models}
