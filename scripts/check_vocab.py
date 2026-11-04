"""Check that teacher and student vocabularies are compatible and write results/vocab_check.json."""

import sys

from llmplatform.distill.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-vocab", *sys.argv[1:]]))
