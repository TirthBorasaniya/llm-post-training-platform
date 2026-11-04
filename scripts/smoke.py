"""Run the CPU smoke pipeline on tiny models and print one line per step."""

import argparse
import sys
from pathlib import Path

from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.paths import repo_root
from llmplatform.smoke.pipeline import run_smoke


def main() -> int:
    """Run the pipeline and return a nonzero status when any step failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(repo_root() / ".tmp" / "smoke"))
    parser.add_argument("--keep", action="store_true", help="reuse the work directory")
    args = parser.parse_args()
    configure_logging("smoke")
    results = run_smoke(Path(args.work_dir), o_clean=not args.keep)
    for item in results:
        detail = f" {item.detail}" if item.detail else ""
        print(f"{item.status:8} {item.seconds:7.1f}s  {item.name}{detail}")
    return 1 if any(item.status == "failed" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
