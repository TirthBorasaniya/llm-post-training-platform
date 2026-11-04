"""Synthetic chat, preference, and prompt rows for smoke tests; no external data is used."""

from typing import Any

# ============= Constants =============

TOPICS = [
    "binary search",
    "the water cycle",
    "compound interest",
    "photosynthesis",
    "the printing press",
    "plate tectonics",
    "prime numbers",
    "supply and demand",
]


# ============= Rows =============


def _question(index: int) -> str:
    return f"Explain {TOPICS[index % len(TOPICS)]} in two sentences. Variant {index}."


def _answer(index: int) -> str:
    return f"{TOPICS[index % len(TOPICS)].capitalize()} is a useful idea. Here is variant {index}."


def sft_rows(count: int) -> list[dict[str, Any]]:
    """Return ``count`` single-turn SFT records."""
    return [
        {
            "id": f"sft-{i}",
            "messages": [
                {"role": "user", "content": _question(i)},
                {"role": "assistant", "content": _answer(i)},
            ],
        }
        for i in range(count)
    ]


def preference_rows(count: int) -> list[dict[str, Any]]:
    """Return ``count`` conversational preference records."""
    return [
        {
            "id": f"pref-{i}",
            "prompt": [{"role": "user", "content": _question(i)}],
            "chosen": [{"role": "assistant", "content": _answer(i)}],
            "rejected": [{"role": "assistant", "content": f"I do not know about variant {i}."}],
        }
        for i in range(count)
    ]


def prompt_rows(count: int) -> list[dict[str, Any]]:
    """Return ``count`` prompt-only records."""
    return [
        {"id": f"prompt-{i}", "prompt": [{"role": "user", "content": _question(i)}]}
        for i in range(count)
    ]


def judge_prompt_rows(count: int) -> list[dict[str, Any]]:
    """Return ``count`` judge prompt records with plain-text prompts."""
    return [
        {"id": f"judge-{i:03d}", "prompt_id": str(i), "prompt": _question(i)} for i in range(count)
    ]
