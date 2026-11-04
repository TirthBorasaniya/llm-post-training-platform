"""Hugging Face generate baseline with static batching at batch sizes 1, 8, and 32."""

import sys

from llmplatform.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["hf", *sys.argv[1:]]))
