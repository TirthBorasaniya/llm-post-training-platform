"""Typed YAML configuration loading with a version field and a SHA-256 fingerprint."""

from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from llmplatform.common.hashing import sha256_file
from llmplatform.common.paths import repo_root

# ============= Constants =============

SUPPORTED_VERSION = 1

ConfigT = TypeVar("ConfigT", bound="ConfigModel")


# ============= Models =============


class ConfigModel(BaseModel):
    """Base class for every YAML-backed configuration: strict keys, immutable, versioned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int

    def check_version(self) -> None:
        """Raise when the file declares a version this code does not understand."""
        if self.version != SUPPORTED_VERSION:
            raise ValueError(
                f"unsupported config version {self.version}, expected {SUPPORTED_VERSION}"
            )


class ConfigRef(BaseModel):
    """Location and content hash of the config file a run used."""

    path: str
    sha256: str


# ============= Loading =============


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML mapping from ``path``."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def apply_overrides(data: dict[str, Any], override_list: list[str]) -> dict[str, Any]:
    """
    Return a copy of ``data`` with dotted ``key.path=value`` overrides applied.

    Values are parsed as YAML scalars, so ``optim.max_steps=60`` yields an integer and
    ``quantization.enabled=false`` yields a boolean. Every key path must already exist.

    Parameters
    ----------
    data : dict
        Parsed configuration mapping.
    override_list : list[str]
        Strings of the form ``section.key=value``.
    """
    updated = deepcopy(data)
    for item in override_list:
        key_path, separator, raw = item.partition("=")
        if not separator or not key_path:
            raise ValueError(f"override {item!r} must look like section.key=value")
        keys = key_path.split(".")
        node = updated
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                raise KeyError(f"override path {key_path!r} does not exist")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"override path {key_path!r} does not exist")
        node[keys[-1]] = yaml.safe_load(raw)
    return updated


def load_config(
    path: str | Path, model: type[ConfigT], override_list: list[str] | None = None
) -> ConfigT:
    """
    Load and validate a YAML file against a pydantic config model.

    Parameters
    ----------
    path : str or Path
        YAML file to read.
    model : type[ConfigModel]
        Config class describing the expected keys.
    override_list : list[str], optional
        Dotted overrides applied before validation.

    Returns
    -------
    config : ConfigModel
        Validated, immutable configuration.
    """
    data = apply_overrides(load_yaml(path), override_list or [])
    try:
        config = model.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid config {path}: {exc}") from exc
    config.check_version()
    return config


def config_ref(path: str | Path, root: str | Path | None = None) -> ConfigRef:
    """Return the repository-relative path and SHA-256 of a config file."""
    base = Path(root) if root else repo_root()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(base.resolve())
    except ValueError:
        relative = Path(path)
    return ConfigRef(path=relative.as_posix(), sha256=sha256_file(resolved))
