"""Repository-wide checks on tracked and to-be-tracked files: privacy, neutrality, and style."""

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# ============= Constants =============

MAX_FILE_BYTES = 5 * 1024 * 1024
PRIVATE_PATHS = (".planning/", "CLAUDE.md", ".claude/", ".env")
EM_DASH = chr(0x2014)
EMOJI_RANGES = [(0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x1F000, 0x1F2FF)]
EMOJI_PATTERN = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in EMOJI_RANGES) + "]")

# patterns are assembled from fragments so this file does not match itself
PROVIDER_PATTERNS = [
    r"north" + r"eastern",
    r"\bexplo" + r"rer\b",
    r"gpu-" + r"short",
    r"multi" + r"gpu",
    r"\ba" + r"ws\b",
    r"ama" + r"zon",
    r"\bg" + r"cp\b",
    r"google " + r"cloud",
    r"\baz" + r"ure\b",
    r"lambda " + r"labs",
    r"run" + r"pod",
    r"vast" + r"\.ai",
    r"core" + r"weave",
    r"paper" + r"space",
    r"digital" + r"ocean",
    r"\b(?:e" + r"ks|g" + r"ke|a" + r"ks)\b",
    r"het" + r"zner",
    r"neb" + r"ius",
]
REFERENCE_PATTERNS = [
    r"\bSP" + r"EC\b",
    r"SP" + r"EC\.md",
    r"CONVEN" + r"TIONS",
    r"\bP(?:[0-9]|1[01])b?\b",
    r"\bPha" + r"se \d",
]
PERSONAL_PATTERNS = [
    r"(?i)tir" + r"thborasaniya",
    r"(?i)borasaniya\." + r"t",
    r"/Us" + r"ers/",
    r"/ho" + r"me/[a-z]",
    r"114608276",
]
NAME_ALLOWED_IN = {"LICENSE", "pyproject.toml"}
SKIP_CONTENT_SUFFIXES = {".png", ".jpg", ".gif", ".ico", ".lock", ".gguf", ".safetensors"}
SKIP_CONTENT_NAMES = {"uv.lock", ".secrets.baseline"}


# ============= Helpers =============


def candidate_files(root: Path) -> list[Path]:
    """Tracked files plus untracked files that are not ignored."""
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [root / name for name in output.split("\0") if name and (root / name).is_file()]


def text_files(root: Path) -> list[Path]:
    """Files whose content is scanned."""
    selected = []
    for path in candidate_files(root):
        if path.suffix in SKIP_CONTENT_SUFFIXES or path.name in SKIP_CONTENT_NAMES:
            continue
        selected.append(path)
    return selected


def read_text(path: Path) -> str | None:
    """Return file text or None for binary files."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def scan(root: Path, patterns: list[str], o_ignore_case: bool = True) -> list[str]:
    """Return ``file:line`` hits for any pattern."""
    flags = re.IGNORECASE if o_ignore_case else 0
    compiled = [re.compile(pattern, flags) for pattern in patterns]
    hit_list = []
    for path in text_files(root):
        text = read_text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in compiled):
                hit_list.append(f"{path.relative_to(root)}:{number}")
    return hit_list


# ============= Tests =============


def test_no_private_paths_tracked(root: Path) -> None:
    names = [path.relative_to(root).as_posix() for path in candidate_files(root)]
    offenders = [
        name
        for name in names
        if any(name == p.rstrip("/") or name.startswith(p) for p in PRIVATE_PATHS)
        and name != ".env.example"
    ]
    assert offenders == []


def test_no_large_files(root: Path) -> None:
    offenders = [
        path.relative_to(root).as_posix()
        for path in candidate_files(root)
        if path.stat().st_size > MAX_FILE_BYTES
    ]
    assert offenders == []


def test_no_em_dashes_or_emojis(root: Path) -> None:
    hit_list = []
    for path in text_files(root):
        text = read_text(path)
        if text is None:
            continue
        if EM_DASH in text or EMOJI_PATTERN.search(text):
            hit_list.append(path.relative_to(root).as_posix())
    assert hit_list == []


def test_no_provider_or_cluster_names(root: Path) -> None:
    assert scan(root, PROVIDER_PATTERNS) == []


def test_no_spec_or_phase_references(root: Path) -> None:
    assert scan(root, REFERENCE_PATTERNS, o_ignore_case=False) == []


def test_no_personal_data_or_local_paths(root: Path) -> None:
    hit_list = []
    compiled = [re.compile(pattern) for pattern in PERSONAL_PATTERNS]
    for path in text_files(root):
        if path.name in NAME_ALLOWED_IN or path.name == "test_repo_hygiene.py":
            continue
        text = read_text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in compiled):
                hit_list.append(f"{path.relative_to(root)}:{number}")
    assert hit_list == []


def test_env_example_holds_placeholders_only(root: Path) -> None:
    text = (root / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        assert value == "" or value.startswith("replace-with-"), key


def test_license_is_mit_with_owner_name(root: Path) -> None:
    text = (root / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Tirth Borasaniya" in text


def test_gitignore_covers_local_artifacts(root: Path) -> None:
    text = (root / ".gitignore").read_text(encoding="utf-8")
    for entry in [".env", ".venv/", "/data/", "/models/", "/logs/", "/mlruns/", ".cache/"]:
        assert entry in text, entry


def test_no_hostnames_addresses_or_emails(root: Path) -> None:
    ip_pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    email_pattern = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
    allowed_ips = {"127.0.0.1", "0.0.0.0"}  # noqa: S104 - loopback and bind-all literals only
    hit_list = []
    for path in text_files(root):
        if path.parent.name == "envs" and path.suffix == ".txt":
            continue  # pinned requirement files hold four-part version numbers
        text = read_text(path)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            ips = set(ip_pattern.findall(line)) - allowed_ips
            if ips or email_pattern.search(line):
                hit_list.append(f"{path.relative_to(root)}:{number}")
    assert hit_list == []


def test_private_paths_are_not_in_gitignore(root: Path) -> None:
    lines = [line.strip() for line in (root / ".gitignore").read_text().splitlines()]
    for entry in (".planning/", ".planning", "CLAUDE.md", ".claude/", ".claude"):
        assert entry not in lines, entry


def test_private_paths_present_in_a_checkout_are_excluded_locally(root: Path) -> None:
    exclude = root / ".git" / "info" / "exclude"
    present = [name for name in (".planning/", "CLAUDE.md", ".claude/") if (root / name).exists()]
    if not present:
        return
    assert exclude.is_file(), "private paths exist but there is no local exclude file"
    lines = exclude.read_text().splitlines()
    for entry in present:
        assert entry in lines, entry


def test_every_workflow_job_skips_cleanly_without_secrets(root: Path) -> None:
    for path in sorted((root / ".github/workflows").glob("*.yml")):
        assert "secrets." not in path.read_text(), f"{path.name} reads a secret unguarded"
