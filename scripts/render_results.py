"""Rewrite the results blocks in README.md and docs from results/*.json."""

import argparse
import sys

from llmplatform.common.paths import repo_root
from llmplatform.common.render import process


def main() -> int:
    """Render the blocks, or with --check report blocks that differ from the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when a block is out of date")
    args = parser.parse_args()
    root = repo_root()
    problems = process(root, root / "results", args.check)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
