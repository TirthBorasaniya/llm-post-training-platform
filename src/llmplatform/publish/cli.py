"""Command-line entry point that uploads models, GGUF files, and cards to the Hub."""

import argparse
import logging
import os
from pathlib import Path

from llmplatform.cards.cli import build_context
from llmplatform.cards.render import render_model_card
from llmplatform.common.logging_utils import configure_logging
from llmplatform.common.models import load_models, resolve_model, revision_of
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.settings import get_settings
from llmplatform.publish.hub import (
    confirm_apache,
    load_publish,
    publish_files,
    publish_folder,
    repo_id,
)

logger = logging.getLogger(__name__)

# ============= Command =============


def main(argv: list[str] | None = None) -> int:
    """Publish one configured repository."""
    configure_logging("publish")
    parser = argparse.ArgumentParser(description="Publish models and cards to the Hub.")
    parser.add_argument("--repo", choices=["teacher", "student", "gguf"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="list files without uploading")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    settings = get_settings()
    publish = load_publish()
    models = load_models()
    spec = publish.repos[args.repo]
    namespace = settings.require_hf_namespace()
    work = Path(os.environ.get("PLATFORM_ROOT", "~/llmplatform")).expanduser()
    ckpt_root = os.environ.get("CKPT_ROOT", str(work / "checkpoints"))
    api = HfApi(token=None if args.dry_run else settings.require_hf_token())

    base = models.models[spec.base]
    assert base.hub_id is not None
    confirm_apache(api, base.hub_id, revision_of(models, spec.base) or "main")

    ctx = build_context(args.repo, publish, namespace, results_dir(repo_root()))
    card = render_model_card(ctx)
    target = repo_id(namespace, spec.name)
    if spec.kind == "gguf":
        paths = [Path(resolve_model(models, name, ckpt_root)) for name in spec.files or []]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"GGUF files not found: {missing}")
        sent = publish_files(api, paths, target, card, publish.private, args.dry_run)
    else:
        assert spec.source is not None
        folder = Path(resolve_model(models, spec.source, ckpt_root))
        if not folder.is_dir():
            raise FileNotFoundError(f"model folder not found: {folder}")
        sent = publish_folder(
            api, folder, target, card, publish.private, publish.ignore_patterns, args.dry_run
        )
    action = "would upload" if args.dry_run else "uploaded"
    print(f"{action} {len(sent)} files to {target} (private={publish.private})")
    for name in sent:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
