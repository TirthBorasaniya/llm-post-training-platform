"""Create Hub repositories and upload model folders, GGUF files, and cards."""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from llmplatform.cards.render import REQUIRED_LICENSE
from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

logger = logging.getLogger(__name__)

# ============= Models =============


class RepoSpec(BaseModel):
    """One Hub repository to publish."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    source: str | None = None
    files: list[str] | None = None
    base: str


class PublishConfig(ConfigModel):
    """Contents of ``configs/publish.yaml``."""

    private: bool
    repos: dict[str, RepoSpec]
    ignore_patterns: list[str]


def load_publish(path: str | Path | None = None) -> PublishConfig:
    """Load the publishing config."""
    return load_config(path or repo_root() / "configs/publish.yaml", PublishConfig)


# ============= License check =============


def base_model_license(api: Any, hub_id: str, revision: str) -> str | None:
    """Return the license declared on a base model's Hub card."""
    info = api.model_info(hub_id, revision=revision)
    card = getattr(info, "card_data", None)
    return getattr(card, "license", None) if card is not None else None


def confirm_apache(api: Any, hub_id: str, revision: str) -> str:
    """Return the base model's license, raising unless it is Apache 2.0."""
    license_id = base_model_license(api, hub_id, revision)
    if license_id != REQUIRED_LICENSE:
        raise RuntimeError(
            f"{hub_id} declares license {license_id!r}, expected {REQUIRED_LICENSE!r}"
        )
    return license_id


# ============= Upload =============


def repo_id(namespace: str, name: str) -> str:
    """Join a namespace and repository name."""
    return f"{namespace}/{name}"


def list_upload_files(folder: Path, ignore_patterns: list[str]) -> list[str]:
    """List files that an upload of ``folder`` would send, relative to it."""
    from fnmatch import fnmatch

    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(folder).as_posix()
        if any(
            fnmatch(relative, pattern) or fnmatch(path.name, pattern) for pattern in ignore_patterns
        ):
            continue
        files.append(relative)
    return files


def publish_folder(
    api: Any,
    folder: Path,
    target: str,
    card: str,
    private: bool,
    ignore_patterns: list[str],
    o_dry_run: bool = False,
) -> list[str]:
    """
    Upload a model folder with its card.

    Returns
    -------
    files : list[str]
        Files sent (or that would be sent in a dry run), including ``README.md``.
    """
    files = [name for name in list_upload_files(folder, ignore_patterns) if name != "README.md"]
    if o_dry_run:
        return [*files, "README.md"]
    api.create_repo(target, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=target,
        repo_type="model",
        ignore_patterns=[*ignore_patterns, "README.md"],
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target,
        repo_type="model",
    )
    return [*files, "README.md"]


def publish_files(
    api: Any,
    paths: list[Path],
    target: str,
    card: str,
    private: bool,
    o_dry_run: bool = False,
) -> list[str]:
    """Upload individual files (GGUF) with a card."""
    names = [path.name for path in paths]
    if o_dry_run:
        return [*names, "README.md"]
    api.create_repo(target, repo_type="model", private=private, exist_ok=True)
    for path in paths:
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path.name, repo_id=target, repo_type="model"
        )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target,
        repo_type="model",
    )
    return [*names, "README.md"]
