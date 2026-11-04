"""Vocabulary compatibility check and teacher alignment before distillation."""

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ============= Models =============


@dataclass
class VocabReport:
    """Outcome of the vocabulary pre-check, written to ``results/vocab_check.json``."""

    tokenizers_identical: bool
    special_tokens_identical: bool
    chat_templates_identical: bool
    tokenizer_length: int
    teacher_padded_size: int
    student_padded_size: int
    tokenizer_below_both_padded_sizes: bool
    resized_teacher: bool
    teacher_padded_size_after: int
    prompts_checked: int
    logits_max_abs_diff: float
    logits_atol: float
    argmax_identical: bool
    dropped_mass_max: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


# ============= Tokenizer checks =============


def tokenizers_match(teacher_tokenizer: Any, student_tokenizer: Any) -> tuple[bool, bool, bool]:
    """
    Compare two tokenizers.

    Returns
    -------
    vocab_identical, special_identical, template_identical : tuple[bool, bool, bool]
        Whether vocabularies (including added tokens), special tokens and ids, and chat
        templates are identical.
    """
    vocab_identical = teacher_tokenizer.get_vocab() == student_tokenizer.get_vocab()
    special_identical = (
        teacher_tokenizer.special_tokens_map == student_tokenizer.special_tokens_map
        and teacher_tokenizer.all_special_ids == student_tokenizer.all_special_ids
    )
    template_identical = getattr(teacher_tokenizer, "chat_template", None) == getattr(
        student_tokenizer, "chat_template", None
    )
    return vocab_identical, special_identical, template_identical


def padded_vocab_size(model: Any) -> int:
    """Return the embedding row count a model's config declares."""
    return int(model.config.get_text_config().vocab_size)


# ============= Logit comparison =============


def collect_logits(model: Any, batch: Any, n_real: int) -> tuple[Any, Any]:
    """
    Run a batch and return real-token logits plus the probability mass beyond them.

    Returns
    -------
    logits : torch.Tensor
        Shape (batch, seq, n_real), float32, on the CPU.
    outside_mass : torch.Tensor
        Shape (batch, seq), softmax mass on the padded rows beyond ``n_real``.
    """
    import torch

    with torch.no_grad():
        full = model(**batch).logits.float()
    outside = torch.softmax(full, dim=-1)[..., n_real:].sum(dim=-1)
    return full[..., :n_real].cpu(), outside.cpu()


def logit_tolerance(model: Any, atol_float32: float, atol_half: float) -> float:
    """Return the comparison tolerance for the model's compute dtype."""
    import torch

    dtype = next(model.parameters()).dtype
    return atol_float32 if dtype == torch.float32 else atol_half


def check_and_align(
    teacher: Any,
    student: Any,
    teacher_tokenizer: Any,
    student_tokenizer: Any,
    prompt_texts: list[str],
    max_length: int = 256,
    atol_float32: float = 1e-4,
    atol_half: float = 0.125,
    batch_size: int = 10,
) -> VocabReport:
    """
    Verify tokenizer equality and align the teacher's embedding size to the student's.

    The two tokenizers must be identical and shorter than both padded embedding sizes. When
    the padded sizes differ, ``teacher.resize_token_embeddings`` shrinks the teacher to the
    student's size in place, and the teacher's logits on every real token at every attended
    position are compared before and after.

    Parameters
    ----------
    teacher, student : PreTrainedModel
        Loaded models; the teacher is modified in place when resized.
    teacher_tokenizer, student_tokenizer : PreTrainedTokenizerBase
        Their tokenizers.
    prompt_texts : list[str]
        Prompts used for the logit comparison (50 in the real check).
    max_length : int
        Truncation length of the comparison batches.
    atol_float32, atol_half : float
        Maximum allowed absolute logit difference for float32 and half-precision models.
    batch_size : int
        Prompts per forward pass.

    Returns
    -------
    report : VocabReport
        Every check and the overall verdict.
    """
    import torch

    vocab_same, special_same, template_same = tokenizers_match(teacher_tokenizer, student_tokenizer)
    n_real = len(student_tokenizer)
    teacher_size, student_size = padded_vocab_size(teacher), padded_vocab_size(student)
    below_both = n_real < teacher_size and n_real < student_size

    max_diff = 0.0
    argmax_same = True
    dropped_mass = 0.0
    resized = teacher_size != student_size
    atol = logit_tolerance(teacher, atol_float32, atol_half)
    if below_both and vocab_same and prompt_texts:
        device = next(teacher.parameters()).device
        teacher.eval()
        tokenizer = teacher_tokenizer
        before_list = []
        batches = []
        for start in range(0, len(prompt_texts), batch_size):
            texts = prompt_texts[start : start + batch_size]
            batch = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            batches.append(batch)
            logits, outside = collect_logits(teacher, batch, n_real)
            before_list.append(logits)
            dropped_mass = max(dropped_mass, float(outside.max()))
        if resized:
            teacher.resize_token_embeddings(student_size)
            teacher.eval()
        for batch, before in zip(batches, before_list, strict=True):
            after, _ = collect_logits(teacher, batch, n_real)
            mask = batch["attention_mask"].cpu().bool()
            diff = (after - before).abs()[mask]
            max_diff = max(max_diff, float(diff.max()))
            argmax_same = argmax_same and bool(
                torch.equal(after.argmax(dim=-1)[mask], before.argmax(dim=-1)[mask])
            )
    elif resized and below_both and vocab_same:
        teacher.resize_token_embeddings(student_size)

    passed = (
        vocab_same
        and special_same
        and below_both
        and padded_vocab_size(teacher) == student_size
        and max_diff <= atol
        and argmax_same
    )
    return VocabReport(
        tokenizers_identical=vocab_same,
        special_tokens_identical=special_same,
        chat_templates_identical=template_same,
        tokenizer_length=n_real,
        teacher_padded_size=teacher_size,
        student_padded_size=student_size,
        tokenizer_below_both_padded_sizes=below_both,
        resized_teacher=resized,
        teacher_padded_size_after=padded_vocab_size(teacher),
        prompts_checked=len(prompt_texts),
        logits_max_abs_diff=max_diff,
        logits_atol=atol,
        argmax_identical=argmax_same,
        dropped_mass_max=dropped_mass,
        passed=bool(passed),
    )
