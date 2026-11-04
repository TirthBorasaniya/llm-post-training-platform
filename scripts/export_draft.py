"""Pad a draft model's embeddings to the target's padded vocabulary size and verify logits."""

import sys

from llmplatform.speculative.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["export-draft", *sys.argv[1:]]))
