"""Repository path discovery."""

from pathlib import Path

# ============= Constants =============

ROOT_MARKER = "pyproject.toml"
RESULTS_DIRNAME = "results"
SYNC_SHA_FILENAME = ".sync_sha"


# ============= Paths =============


def repo_root(start: str | Path | None = None) -> Path:
    """
    Find the repository root by walking upward to the directory holding ``pyproject.toml``.

    Parameters
    ----------
    start : str or Path, optional
        Directory to start from. Defaults to this file's location.

    Returns
    -------
    root : Path
        Absolute repository root.
    """
    here = Path(start).resolve() if start else Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ROOT_MARKER).is_file() and (candidate / "src" / "llmplatform").is_dir():
            return candidate
    raise FileNotFoundError(f"repository root not found above {here}")


def results_dir(root: str | Path | None = None) -> Path:
    """Return the committed results directory, creating it when absent."""
    path = (Path(root) if root else repo_root()) / RESULTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path
