"""Command-line entry point that renders a model card from the results directory."""

import argparse
import os
from pathlib import Path

from llmplatform.cards.render import CardContext, render_model_card
from llmplatform.common.models import load_models, resolve_model
from llmplatform.common.paths import repo_root, results_dir
from llmplatform.common.settings import get_settings
from llmplatform.data.config import load_data_config
from llmplatform.publish.hub import PublishConfig, load_publish, repo_id

# ============= Rendering =============


def build_context(
    key: str, publish: PublishConfig, namespace: str, results_root: Path
) -> CardContext:
    """Create the card context of one configured repository."""
    spec = publish.repos[key]
    models = load_models()
    base = models.models[spec.base]
    if base.hub_id is None or base.license is None:
        raise ValueError(f"base model {spec.base!r} needs a hub id and license in the registry")
    kind = {"teacher": "teacher", "student": "student", "gguf": "gguf"}[key]
    return CardContext(
        kind=kind,
        repo_id=repo_id(namespace, spec.name),
        base_model_id=base.hub_id,
        base_model_license=base.license,
        models=models,
        data=load_data_config(),
        results_root=results_root,
        gguf_files=[Path(resolve_model(models, name, "")).name for name in spec.files or []]
        or None,
    )


def main(argv: list[str] | None = None) -> int:
    """Render a card to a file or standard output."""
    parser = argparse.ArgumentParser(description="Render a Hub model card from results.")
    parser.add_argument("--repo", choices=["teacher", "student", "gguf"], required=True)
    parser.add_argument("--out", default=None, help="output file; standard output when omitted")
    args = parser.parse_args(argv)
    namespace = get_settings().hf_namespace or os.environ.get("HF_NAMESPACE") or "<namespace>"
    ctx = build_context(args.repo, load_publish(), namespace, results_dir(repo_root()))
    card = render_model_card(ctx)
    if args.out:
        Path(args.out).write_text(card, encoding="utf-8")
    else:
        print(card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
