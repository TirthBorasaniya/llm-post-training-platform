"""Pad a draft model's vocabulary to the target's padded size without changing real-token logits."""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llmplatform.distill.vocab_check import (
    collect_logits,
    logit_tolerance,
    padded_vocab_size,
    tokenizers_match,
)

logger = logging.getLogger(__name__)

# ============= Models =============


@dataclass
class ExportReport:
    """Verification of a padded draft."""

    tokenizers_identical: bool
    tokenizer_length: int
    draft_size_before: int
    draft_size_after: int
    target_size: int
    padded_rows_zeroed: bool
    prompts_checked: int
    real_logits_max_abs_diff: float
    logits_atol: float
    padded_argmax_count: int
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


# ============= Padding =============


def zero_padded_rows(model: Any, n_real: int) -> None:
    """Set embedding rows beyond the real vocabulary to zero, including an untied output head."""
    import torch

    with torch.no_grad():
        model.get_input_embeddings().weight[n_real:].zero_()
        head = model.get_output_embeddings()
        if (
            head is not None
            and head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
        ):
            head.weight[n_real:].zero_()


def pad_draft(
    draft: Any,
    draft_tokenizer: Any,
    target_tokenizer: Any,
    target_size: int,
    prompt_texts: list[str],
    max_length: int = 256,
    batch_size: int = 10,
    atol_float32: float = 1e-4,
    atol_half: float = 0.125,
) -> ExportReport:
    """
    Resize the draft's embeddings to ``target_size`` and verify nothing real changed.

    The new rows are zero, so their logits are zero. The report confirms that logits over the
    real tokens are unchanged (within the dtype tolerance) at every attended position and
    that the argmax never lands in a padded row on the test prompts, so greedy drafting
    never proposes one.

    Parameters
    ----------
    draft : PreTrainedModel
        Loaded draft model, modified in place.
    draft_tokenizer, target_tokenizer : PreTrainedTokenizerBase
        Must be identical.
    target_size : int
        Padded embedding size of the target model.
    prompt_texts : list[str]
        Prompts for the logit comparison.
    """
    import torch

    vocab_same, special_same, _ = tokenizers_match(draft_tokenizer, target_tokenizer)
    n_real = len(draft_tokenizer)
    size_before = padded_vocab_size(draft)
    if not (vocab_same and special_same) or n_real >= target_size or size_before > target_size:
        return ExportReport(
            vocab_same and special_same,
            n_real,
            size_before,
            size_before,
            target_size,
            False,
            0,
            0.0,
            0.0,
            0,
            False,
        )
    device = next(draft.parameters()).device
    atol = logit_tolerance(draft, atol_float32, atol_half)
    draft.eval()
    draft_tokenizer.padding_side = "left"
    batches = [
        draft_tokenizer(
            prompt_texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        for start in range(0, len(prompt_texts), batch_size)
    ]
    before = [collect_logits(draft, batch, n_real)[0] for batch in batches]
    draft.resize_token_embeddings(target_size)
    zero_padded_rows(draft, n_real)
    draft.eval()
    max_diff, padded_argmax = 0.0, 0
    for batch, reference in zip(batches, before, strict=True):
        with torch.no_grad():
            full = draft(**batch).logits.float().cpu()
        mask = batch["attention_mask"].cpu().bool()
        max_diff = max(max_diff, float((full[..., :n_real] - reference).abs()[mask].max()))
        padded_argmax += int((full.argmax(dim=-1)[mask] >= n_real).sum())
    return ExportReport(
        tokenizers_identical=True,
        tokenizer_length=n_real,
        draft_size_before=size_before,
        draft_size_after=padded_vocab_size(draft),
        target_size=target_size,
        padded_rows_zeroed=True,
        prompts_checked=len(prompt_texts),
        real_logits_max_abs_diff=max_diff,
        logits_atol=atol,
        padded_argmax_count=padded_argmax,
        passed=max_diff <= atol and padded_argmax == 0 and padded_vocab_size(draft) == target_size,
    )


def export_draft(
    draft_path: str,
    draft_revision: str | None,
    target_path: str,
    target_revision: str | None,
    output_dir: Path,
    prompt_texts: list[str],
    o_cpu: bool = False,
) -> ExportReport:
    """Load a draft, pad it to the target size, verify it, and save the padded model."""
    import os

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from llmplatform.common.io import write_json

    kwargs: dict[str, Any] = {"dtype": torch.float32 if o_cpu else torch.bfloat16}
    if draft_revision and not os.path.isdir(draft_path):
        kwargs["revision"] = draft_revision
    draft: Any = AutoModelForCausalLM.from_pretrained(draft_path, **kwargs)
    if not o_cpu and torch.cuda.is_available():
        draft.to("cuda")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, revision=kwargs.get("revision"))
    target_tokenizer = AutoTokenizer.from_pretrained(target_path, revision=target_revision)
    target_config = AutoConfig.from_pretrained(target_path, revision=target_revision)
    report = pad_draft(
        draft,
        draft_tokenizer,
        target_tokenizer,
        int(target_config.get_text_config().vocab_size),
        prompt_texts,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "export_report.json", report.as_dict())
    if not report.passed:
        raise RuntimeError(f"draft export verification failed: {report.as_dict()}")
    draft.to("cpu").save_pretrained(str(output_dir), safe_serialization=True)
    draft_tokenizer.save_pretrained(str(output_dir))
    return report
