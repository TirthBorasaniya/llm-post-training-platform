"""GGUF conversion and Q4_K_M quantization with llama.cpp, plus a generation check."""

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============= Constants =============

UNIFIED_BINARY = "llama"
BUILD_BIN = Path("build") / "bin"
CONVERT_SCRIPT = "convert_hf_to_gguf.py"
SUBCOMMANDS = {"quantize": "quantize", "cli": "cli", "serve": "serve"}
LEGACY_NAMES = {"quantize": "llama-quantize", "cli": "llama-cli", "serve": "llama-server"}


# ============= Tool resolution =============


def resolve_tool(llama_cpp_dir: str | Path, tool: str) -> list[str]:
    """
    Return the command prefix for a llama.cpp tool.

    Recent llama.cpp builds ship one ``llama`` binary with subcommands (``llama quantize``);
    older builds ship ``llama-quantize``, ``llama-cli``, and ``llama-server``. The unified
    binary is preferred when it exists.

    Parameters
    ----------
    llama_cpp_dir : str or Path
        Source checkout that was built with CMake into ``build/bin``.
    tool : str
        One of ``quantize``, ``cli``, ``serve``.
    """
    if tool not in SUBCOMMANDS:
        raise ValueError(f"unknown llama.cpp tool {tool!r}")
    bin_dir = Path(llama_cpp_dir) / BUILD_BIN
    unified = bin_dir / UNIFIED_BINARY
    if unified.is_file():
        return [str(unified), SUBCOMMANDS[tool]]
    legacy = bin_dir / LEGACY_NAMES[tool]
    if legacy.is_file():
        return [str(legacy)]
    raise FileNotFoundError(
        f"no built llama.cpp {tool} tool under {bin_dir}; build llama.cpp first"
    )


# ============= Commands =============


def convert_command(
    llama_cpp_dir: str | Path, model_dir: str | Path, outfile: str | Path, outtype: str = "f16"
) -> list[str]:
    """Return the ``convert_hf_to_gguf.py`` command for a Hugging Face checkpoint."""
    return [
        sys.executable,
        str(Path(llama_cpp_dir) / CONVERT_SCRIPT),
        str(model_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]


def quantize_command(
    llama_cpp_dir: str | Path, source: str | Path, target: str | Path, quant_type: str = "Q4_K_M"
) -> list[str]:
    """Return the quantization command turning a float GGUF into ``quant_type``."""
    return [*resolve_tool(llama_cpp_dir, "quantize"), str(source), str(target), quant_type]


def verify_command(
    llama_cpp_dir: str | Path, gguf: str | Path, prompt: str, n_predict: int
) -> list[str]:
    """Return a single-turn greedy ``llama cli`` command for one prompt."""
    return [
        *resolve_tool(llama_cpp_dir, "cli"),
        "-m",
        str(gguf),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "-st",
    ]


# ============= Pipeline =============


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of one verification prompt."""

    prompt: str
    output: str
    ok: bool


def convert_and_quantize(
    llama_cpp_dir: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    name: str,
    quant_type: str = "Q4_K_M",
    outtype: str = "f16",
    o_keep_float: bool = False,
) -> Path:
    """
    Convert a checkpoint to GGUF and quantize it.

    Parameters
    ----------
    name : str
        Base name of the output files, for example ``teacher``.
    o_keep_float : bool
        Keep the intermediate float GGUF instead of deleting it.

    Returns
    -------
    quantized : Path
        Path of ``<name>-<quant_type>.gguf``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    float_path = out / f"{name}-{outtype}.gguf"
    quantized = out / f"{name}-{quant_type}.gguf"
    subprocess.run(convert_command(llama_cpp_dir, model_dir, float_path, outtype), check=True)
    subprocess.run(quantize_command(llama_cpp_dir, float_path, quantized, quant_type), check=True)
    if not o_keep_float:
        float_path.unlink(missing_ok=True)
    return quantized


def verify_gguf(
    llama_cpp_dir: str | Path,
    gguf: str | Path,
    prompts: list[str],
    n_predict: int,
    timeout: int,
    runner: Any = subprocess.run,
) -> list[VerifyResult]:
    """
    Generate from each prompt and require a clean exit with non-empty output.

    Parameters
    ----------
    runner : callable
        Replaceable ``subprocess.run`` for tests.

    Returns
    -------
    result_list : list[VerifyResult]
        One entry per prompt.
    """
    result_list = []
    for prompt in prompts:
        completed = runner(
            verify_command(llama_cpp_dir, gguf, prompt, n_predict),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout or "").strip()
        result_list.append(VerifyResult(prompt, output, completed.returncode == 0 and bool(output)))
    return result_list
