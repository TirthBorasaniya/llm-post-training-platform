"""Win-rate aggregation and bootstrap confidence intervals."""

from dataclasses import dataclass

import numpy as np

# ============= Constants =============

WIN, LOSS, TIE = "win", "loss", "tie"
SCORE_BY_OUTCOME = {WIN: 1.0, TIE: 0.5, LOSS: 0.0}


# ============= Outcomes =============


def combine_orders(first_order: str, second_order: str) -> str:
    """
    Combine the two position-swapped verdicts for one prompt.

    Parameters
    ----------
    first_order : str
        Verdict when the candidate was shown first: ``"candidate"``, ``"reference"``, or ``"tie"``.
    second_order : str
        Verdict when the reference was shown first, already mapped to the same labels.

    Returns
    -------
    outcome : str
        ``win`` if both orders prefer the candidate, ``loss`` if both prefer the reference,
        otherwise ``tie``.
    """
    if first_order == "candidate" and second_order == "candidate":
        return WIN
    if first_order == "reference" and second_order == "reference":
        return LOSS
    return TIE


@dataclass(frozen=True)
class WinRate:
    """Win rate with a bootstrap interval."""

    n: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    ci_low: float
    ci_high: float


def win_rate(outcome_list: list[str], resamples: int, seed: int, confidence: float) -> WinRate:
    """
    Compute ``(wins + 0.5 * ties) / n`` with a percentile bootstrap interval.

    Parameters
    ----------
    outcome_list : list[str]
        One of ``win``, ``loss``, ``tie`` per prompt.
    resamples : int
        Number of bootstrap resamples of the prompts.
    seed : int
        Seed of the resampling generator.
    confidence : float
        Interval coverage, for example 0.95.
    """
    if not outcome_list:
        raise ValueError("no outcomes to aggregate")
    scores = np.array([SCORE_BY_OUTCOME[item] for item in outcome_list], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scores), size=(resamples, len(scores)))
    means = scores[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return WinRate(
        n=len(scores),
        wins=outcome_list.count(WIN),
        losses=outcome_list.count(LOSS),
        ties=outcome_list.count(TIE),
        win_rate=float(scores.mean()),
        ci_low=float(low),
        ci_high=float(high),
    )
