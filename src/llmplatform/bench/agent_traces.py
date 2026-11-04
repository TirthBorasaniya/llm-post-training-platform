"""Generate multi-turn agent conversations with a shared system prompt and tool schemas."""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmplatform.bench.config import AgentConfig

# ============= Constants =============

SYSTEM_SUFFIX = ".system.txt"
TOOLS_SUFFIX = ".tools.json"
VOCABULARY = """
service deploy latency queue alert incident rollback metric cluster replica restart
timeout error budget trace region dependency cache shard index schema migration
ticket owner runbook severity pager dashboard threshold baseline regression release
canary traffic request response status failure retry backoff circuit breaker pool
memory disk network packet loss cpu load saturation capacity forecast cost invoice
usage tenant workspace report summary anomaly drift sample window percentile
""".split()  # noqa: SIM905 - a fixed word list kept readable as prose


# ============= Sources =============


@dataclass(frozen=True)
class AgentSource:
    """System prompt and tool schemas of one agent."""

    name: str
    system_prompt: str
    tools: list[dict[str, Any]]


def load_sources(source_dir: str | Path) -> list[AgentSource]:
    """
    Load every ``<name>.system.txt`` with its ``<name>.tools.json``.

    Raises
    ------
    FileNotFoundError
        If the directory holds no complete agent, so no trace is ever built from invented
        prompts.
    """
    root = Path(source_dir)
    source_list = []
    for system_file in sorted(root.glob(f"*{SYSTEM_SUFFIX}")):
        name = system_file.name[: -len(SYSTEM_SUFFIX)]
        tools_file = root / f"{name}{TOOLS_SUFFIX}"
        if not tools_file.is_file():
            raise FileNotFoundError(f"missing {tools_file.name} for agent {name!r}")
        tools = json.loads(tools_file.read_text(encoding="utf-8"))
        if not isinstance(tools, list):
            raise ValueError(f"{tools_file.name} must hold a JSON list of tool schemas")
        source_list.append(
            AgentSource(name, system_file.read_text(encoding="utf-8").strip(), tools)
        )
    if not source_list:
        raise FileNotFoundError(
            f"no agent sources in {root}; add <name>.system.txt and <name>.tools.json files"
        )
    return source_list


def system_message(source: AgentSource) -> str:
    """Return the system message: the prompt followed by the tool schemas as JSON text."""
    return (
        f"{source.system_prompt}\n\n# Tools\n{json.dumps(source.tools, indent=2, sort_keys=True)}"
    )


# ============= Generation =============


def _words(rng: random.Random, bounds: list[int]) -> str:
    count = rng.randint(bounds[0], bounds[1])
    return " ".join(rng.choice(VOCABULARY) for _ in range(count))


def build_conversation(
    source: AgentSource, conversation_id: int, config: AgentConfig, seed: int
) -> list[dict[str, Any]]:
    """
    Build the request of every turn of one conversation.

    Turn ``k`` repeats all messages of turn ``k - 1`` and appends the previous assistant reply
    and a new user message with a synthetic tool result, so each request extends the previous
    prompt by a suffix. Text is drawn from a fixed vocabulary with a seeded generator.
    """
    rng = random.Random(f"{seed}-{conversation_id}")
    messages: list[dict[str, str]] = [{"role": "system", "content": system_message(source)}]
    rows = []
    for turn in range(config.turns_per_conversation):
        if turn > 0:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Calling tool. {_words(rng, config.assistant_words)}",
                }
            )
        request_text = _words(rng, config.user_words)
        tool_result = _words(rng, config.tool_result_words)
        user = f"{request_text}\nTool result: {tool_result}"
        messages.append({"role": "user", "content": user})
        rows.append(
            {
                "conversation_id": conversation_id,
                "turn": turn,
                "agent": source.name,
                "messages": [dict(message) for message in messages],
                "max_tokens": config.max_tokens,
            }
        )
    return rows


def build_traces(
    sources: list[AgentSource], config: AgentConfig, seed: int | None = None
) -> list[dict[str, Any]]:
    """Build every conversation, alternating agents, in conversation then turn order."""
    base_seed = config.seed if seed is None else seed
    rows: list[dict[str, Any]] = []
    for conversation_id in range(config.conversations):
        source = sources[conversation_id % len(sources)]
        rows.extend(build_conversation(source, conversation_id, config, base_seed))
    return rows


def shared_prefix_chars(rows: list[dict[str, Any]]) -> int:
    """Return the length of the system message shared by all conversations of the first agent."""
    return len(rows[0]["messages"][0]["content"]) if rows else 0
