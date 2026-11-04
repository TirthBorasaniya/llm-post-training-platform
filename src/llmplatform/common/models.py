"""Model registry: pinned Hub models and checkpoint paths produced by earlier stages."""

import argparse
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from llmplatform.common.config import ConfigModel, load_config
from llmplatform.common.paths import repo_root

# ============= Constants =============

MODELS_CONFIG_RELPATH = "configs/models.yaml"


# ============= Models =============


class ModelSpec(BaseModel):
    """One registry entry, either a pinned Hub model or a local checkpoint path."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["hub", "local"]
    hub_id: str | None = None
    revision: str | None = None
    license: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ModelSpec":
        if self.kind == "hub" and not (self.hub_id and self.revision):
            raise ValueError("hub models need hub_id and a pinned revision")
        if self.kind == "local" and not self.path:
            raise ValueError("local models need a path relative to the checkpoint root")
        return self


class ModelsConfig(ConfigModel):
    """Contents of ``configs/models.yaml``."""

    models: dict[str, ModelSpec]
    roles: dict[str, str]

    @model_validator(mode="after")
    def _roles_exist(self) -> "ModelsConfig":
        missing = [name for name in self.roles.values() if name not in self.models]
        if missing:
            raise ValueError(f"roles refer to unknown models: {missing}")
        return self


# ============= Resolution =============


def load_models(path: str | Path | None = None) -> ModelsConfig:
    """Load the model registry."""
    target = Path(path) if path else repo_root() / MODELS_CONFIG_RELPATH
    return load_config(target, ModelsConfig)


def resolve_model(config: ModelsConfig, name: str, ckpt_root: str | Path) -> str:
    """
    Return the identifier to pass to ``from_pretrained``.

    Parameters
    ----------
    config : ModelsConfig
        Loaded registry.
    name : str
        Registry key or role name.
    ckpt_root : str or Path
        Directory that holds local checkpoints.

    Returns
    -------
    location : str
        Hub id for hub models, absolute path for local models. Use ``revision_of`` for the pin.
    """
    key = config.roles.get(name, name)
    if key not in config.models:
        raise KeyError(f"unknown model {name!r}")
    spec = config.models[key]
    if spec.kind == "hub":
        assert spec.hub_id is not None
        return spec.hub_id
    assert spec.path is not None
    return str(Path(ckpt_root) / spec.path)


def revision_of(config: ModelsConfig, name: str) -> str | None:
    """Return the pinned revision of a hub model, or ``None`` for local checkpoints."""
    return config.models[config.roles.get(name, name)].revision


# ============= CLI =============


def main(argv: list[str] | None = None) -> int:
    """Print the location of a registry model for shell scripts: ``resolve NAME``."""
    parser = argparse.ArgumentParser(description="Query the model registry.")
    sub = parser.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve", help="print the path or Hub id of a model")
    resolve.add_argument("name")
    args = parser.parse_args(argv)
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    ckpt_root = os.environ.get("CKPT_ROOT", str(work / "checkpoints"))
    print(resolve_model(load_models(), args.name, ckpt_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
