"""Exact and MinHash near-duplicate matching used to decontaminate training prompts."""

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

# ============= Constants =============

MERSENNE_PRIME = (1 << 61) - 1
PERMUTATION_LIMIT = 1 << 31
HASH_MASK = (1 << 32) - 1
PUNCTUATION = re.compile(r"[^\w\s]")
WHITESPACE = re.compile(r"\s+")

RowT = TypeVar("RowT")


# ============= Normalization =============


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    return WHITESPACE.sub(" ", PUNCTUATION.sub("", text.lower())).strip()


def exact_key(text: str) -> str:
    """Return a digest of the normalized text for exact matching."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def shingle_hashes(text: str, shingle_words: int) -> np.ndarray:
    """
    Hash word n-grams of the normalized text to 32-bit integers.

    Texts shorter than ``shingle_words`` words yield a single shingle covering the whole text.

    Returns
    -------
    hashes : np.ndarray
        Unique hashes, shape (n_shingles,), dtype uint64.
    """
    words = normalize_text(text).split(" ")
    if len(words) <= shingle_words:
        grams = [" ".join(words)]
    else:
        grams = [
            " ".join(words[i : i + shingle_words]) for i in range(len(words) - shingle_words + 1)
        ]
    values = {
        int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(), "big")
        for gram in grams
    }
    return np.fromiter(values, dtype=np.uint64, count=len(values))


# ============= MinHash =============


class MinHasher:
    """Computes MinHash signatures from shingle hashes with seeded universal hash functions."""

    def __init__(self, num_perm: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, PERMUTATION_LIMIT, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, PERMUTATION_LIMIT, size=num_perm, dtype=np.uint64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        """Return the signature of a shingle set, shape (num_perm,)."""
        mixed = (np.outer(hashes, self.a) + self.b) % np.uint64(MERSENNE_PRIME)
        return (mixed & np.uint64(HASH_MASK)).min(axis=0)


def estimated_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Estimate Jaccard similarity as the fraction of equal signature positions."""
    return float(np.mean(first == second))


# ============= Index =============


@dataclass
class NearDuplicateConfig:
    """Parameters of the near-duplicate matcher."""

    num_perm: int = 128
    rows_per_band: int = 4
    shingle_words: int = 5
    threshold: float = 0.8
    seed: int = 20260101


@dataclass
class ReferenceIndex:
    """Index over reference texts answering whether a candidate duplicates one of them."""

    config: NearDuplicateConfig = field(default_factory=NearDuplicateConfig)

    def __post_init__(self) -> None:
        if self.config.num_perm % self.config.rows_per_band:
            raise ValueError("num_perm must be divisible by rows_per_band")
        self._hasher = MinHasher(self.config.num_perm, self.config.seed)
        self._exact: set[str] = set()
        self._signatures: list[np.ndarray] = []
        self._buckets: dict[tuple[int, bytes], list[int]] = {}

    def __len__(self) -> int:
        return len(self._signatures)

    def _band_keys(self, signature: np.ndarray) -> list[tuple[int, bytes]]:
        rows = self.config.rows_per_band
        return [
            (band, signature[band * rows : (band + 1) * rows].tobytes())
            for band in range(self.config.num_perm // rows)
        ]

    def _signature(self, text: str) -> np.ndarray:
        return self._hasher.signature(shingle_hashes(text, self.config.shingle_words))

    def add(self, text: str) -> None:
        """Add a reference text to the index."""
        self._exact.add(exact_key(text))
        signature = self._signature(text)
        position = len(self._signatures)
        self._signatures.append(signature)
        for key in self._band_keys(signature):
            self._buckets.setdefault(key, []).append(position)

    def add_many(self, texts: Iterable[str]) -> None:
        """Add every text of an iterable."""
        for text in texts:
            self.add(text)

    def match(self, text: str) -> str | None:
        """
        Classify a candidate against the index.

        Returns
        -------
        kind : str or None
            ``"exact"`` for a normalized-text match, ``"near"`` when an estimated Jaccard
            similarity reaches the threshold, otherwise ``None``.
        """
        if exact_key(text) in self._exact:
            return "exact"
        if not self._signatures:
            return None
        signature = self._signature(text)
        seen: set[int] = set()
        for key in self._band_keys(signature):
            for position in self._buckets.get(key, ()):
                if position in seen:
                    continue
                seen.add(position)
                if (
                    estimated_jaccard(signature, self._signatures[position])
                    >= self.config.threshold
                ):
                    return "near"
        return None


# ============= Filtering =============


@dataclass
class FilterStats:
    """Counts of rows kept and dropped by one filter."""

    name: str
    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        """Record one dropped row."""
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {"name": self.name, "seen": self.seen, "kept": self.kept, "dropped": self.dropped}


def decontaminate(
    rows: Iterable[RowT], text_of: Callable[[RowT], str], index: ReferenceIndex
) -> tuple[list[RowT], FilterStats]:
    """
    Remove rows whose text duplicates or nearly duplicates a reference text.

    Parameters
    ----------
    rows : iterable
        Candidate rows.
    text_of : callable
        Maps a row to the text compared against the index.
    index : ReferenceIndex
        Reference texts such as judge prompts and benchmark questions.

    Returns
    -------
    kept : list
        Rows with no match.
    stats : FilterStats
        Counts by match kind.
    """
    stats = FilterStats(name="decontaminate")
    kept = []
    for row in rows:
        stats.seen += 1
        kind = index.match(text_of(row))
        if kind is None:
            kept.append(row)
            stats.kept += 1
        else:
            stats.drop(kind)
    return kept, stats


def unique_by_text(
    rows: Iterable[RowT], text_of: Callable[[RowT], str]
) -> tuple[list[RowT], FilterStats]:
    """Keep the first row for each normalized text."""
    stats = FilterStats(name="unique")
    seen_keys: set[str] = set()
    kept = []
    for row in rows:
        stats.seen += 1
        key = exact_key(text_of(row))
        if key in seen_keys:
            stats.drop("exact")
            continue
        seen_keys.add(key)
        kept.append(row)
        stats.kept += 1
    return kept, stats
