"""Server launch arguments for quantized checkpoint variants, printed for shell job scripts."""

import argparse
import os
import shlex
from pathlib import Path

from llmplatform.common.models import load_models, resolve_model
from llmplatform.eval.config import CheckpointEntry, load_e1

# ============= Arguments =============


def launch_args(
    entry: CheckpointEntry, model_path: str, llama_cpp_dir: str | None = None
) -> list[str]:
    """
    Return ``llmplatform.serve.launch`` options that serve one variant.

    Parameters
    ----------
    entry : CheckpointEntry
        Variant with a ``serve`` section.
    model_path : str
        Directory or GGUF file of the model to serve.
    llama_cpp_dir : str, optional
        Built llama.cpp checkout, for ``llamacpp`` variants.
    """
    if entry.serve is None:
        raise ValueError(f"checkpoint {entry.name!r} has no serve section")
    args = [
        "--engine",
        entry.serve.engine,
        "--model",
        model_path,
        "--served-model-name",
        entry.name,
    ]
    if entry.serve.quantization:
        args += ["--quantization", entry.serve.quantization]
    if entry.serve.engine == "llamacpp":
        if not llama_cpp_dir:
            raise ValueError("LLAMA_CPP_DIR is required to serve a GGUF variant")
        args += ["--llama-cpp-dir", llama_cpp_dir]
    return args


def main(argv: list[str] | None = None) -> int:
    """Print launch options for a named variant: ``launch-args NAME``."""
    parser = argparse.ArgumentParser(description="Variant serving helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("launch-args")
    cmd.add_argument("name")
    args = parser.parse_args(argv)
    entry = next(item for item in load_e1().checkpoints if item.name == args.name)
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    ckpt_root = os.environ.get("CKPT_ROOT", str(work / "checkpoints"))
    path = resolve_model(load_models(), entry.model, ckpt_root)
    print(shlex.join(launch_args(entry, path, os.environ.get("LLAMA_CPP_DIR"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
