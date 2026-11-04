"""Verify that every flag the platform passes exists in the installed engine versions."""

import sys

from llmplatform.serve.flags import HELP_COMMANDS, check_tool


def main() -> int:
    """Check each tool whose command is available and report missing flags."""
    status = 0
    for tool in HELP_COMMANDS:
        try:
            missing = check_tool(tool)
        except (FileNotFoundError, OSError):
            print(f"skipped {tool}: command not available in this environment")
            continue
        if missing:
            status = 1
            print(f"FAIL {tool}: missing flags {missing}")
        else:
            print(f"ok {tool}")
    return status


if __name__ == "__main__":
    sys.exit(main())
