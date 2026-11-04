"""Exact comparison of greedy outputs with and without speculative decoding."""

from dataclasses import dataclass, field
from typing import Any

# ============= Constants =============

PREVIEW_CHARS = 80


# ============= Comparison =============


@dataclass
class EqualityReport:
    """Result of comparing two output lists."""

    n: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every output matched exactly."""
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "n": self.n,
            "passed": self.passed,
            "mismatch_count": len(self.mismatches),
            "mismatches": self.mismatches,
        }


def first_difference(first: str, second: str) -> int:
    """Return the index of the first differing character, or the shorter length."""
    for index, (left, right) in enumerate(zip(first, second, strict=False)):
        if left != right:
            return index
    return min(len(first), len(second))


def compare_outputs(baseline: list[str], candidate: list[str]) -> EqualityReport:
    """
    Compare candidate outputs with the non-speculative baseline, character for character.

    Parameters
    ----------
    baseline, candidate : list[str]
        Greedy completions for the same prompts in the same order.

    Returns
    -------
    report : EqualityReport
        Passes only when the lists have equal length and every pair is identical.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"output counts differ: {len(baseline)} versus {len(candidate)}")
    report = EqualityReport(n=len(baseline))
    for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        if left != right:
            position = first_difference(left, right)
            report.mismatches.append(
                {
                    "index": index,
                    "first_difference_char": position,
                    "baseline_excerpt": left[position : position + PREVIEW_CHARS],
                    "candidate_excerpt": right[position : position + PREVIEW_CHARS],
                }
            )
    return report
