"""Conversion of public dataset rows into the conversational formats the trainers read."""

from typing import Any

# ============= Constants =============

VALID_ROLES = {"system", "user", "assistant"}


# ============= Helpers =============


def clean_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep only ``role`` and ``content`` of each message."""
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


def is_valid_dialogue(messages: list[dict[str, str]]) -> bool:
    """
    Check that a dialogue alternates user and assistant turns and ends with the assistant.

    A leading system message is allowed. Every content string must be non-blank.
    """
    turns = messages[1:] if messages and messages[0]["role"] == "system" else messages
    if len(turns) < 2 or len(turns) % 2:
        return False
    for index, item in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if item["role"] != expected or not item["content"].strip():
            return False
    return True


def last_user_text(prompt: Any) -> str:
    """Extract the final user text from a string or a list of messages."""
    if isinstance(prompt, str):
        return prompt
    for item in reversed(prompt):
        if item.get("role") == "user":
            return str(item["content"])
    return ""


# ============= Row converters =============


def sft_record(row: dict[str, Any], row_id: str) -> dict[str, Any] | None:
    """Convert a row with ``messages`` into an SFT record, or ``None`` if invalid."""
    messages = clean_messages(row.get("messages") or [])
    if not is_valid_dialogue(messages):
        return None
    return {"id": row_id, "messages": messages}


def preference_record(row: dict[str, Any], row_id: str) -> dict[str, Any] | None:
    """
    Convert a binarized UltraFeedback row into a conversational preference record.

    The shared prefix of ``chosen`` and ``rejected`` becomes the prompt; the final assistant
    messages become the two completions. Rows with identical completions or mismatched prompts
    are rejected.
    """
    chosen = clean_messages(row.get("chosen") or [])
    rejected = clean_messages(row.get("rejected") or [])
    if not is_valid_dialogue(chosen) or not is_valid_dialogue(rejected):
        return None
    if chosen[:-1] != rejected[:-1]:
        return None
    if chosen[-1]["content"].strip() == rejected[-1]["content"].strip():
        return None
    return {
        "id": row_id,
        "prompt": chosen[:-1],
        "chosen": [chosen[-1]],
        "rejected": [rejected[-1]],
    }


def prompt_record(row: dict[str, Any], row_id: str) -> dict[str, Any] | None:
    """Convert a prompt-only row into ``{"id", "prompt": [user message]}``."""
    text = last_user_text(row.get("prompt", row.get("pompt", ""))).strip()
    if not text:
        return None
    return {"id": row_id, "prompt": [{"role": "user", "content": text}]}


def record_text(record: dict[str, Any]) -> str:
    """Return the text used for decontamination: the final user turn of the prompt or dialogue."""
    if "prompt" in record:
        return last_user_text(record["prompt"])
    users = [item["content"] for item in record["messages"] if item["role"] == "user"]
    return users[0] if users else ""
