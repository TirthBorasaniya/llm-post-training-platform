"""Pairwise judge prompt."""

# ============= Constants =============

SYSTEM_PROMPT = (
    "You are an impartial judge comparing two assistant responses to the same user request. "
    "Judge helpfulness, correctness, instruction following, and clarity. Do not prefer a "
    "response because it is longer or because of the order in which it appears. "
    'Reply with only a JSON object of the form {"winner": "A"} or {"winner": "B"} or '
    '{"winner": "tie"}, optionally with a short "reason" field.'
)
USER_TEMPLATE = (
    "[User request]\n{prompt}\n\n"
    "[Response A]\n{response_a}\n\n"
    "[Response B]\n{response_b}\n\n"
    "Which response is better? Answer with the JSON object only."
)
RETRY_SUFFIX = (
    "\n\nYour previous reply could not be parsed. "
    'Reply with exactly {"winner": "A"}, {"winner": "B"}, or {"winner": "tie"}.'
)


# ============= Builders =============


def build_messages(prompt: str, response_a: str, response_b: str) -> list[dict[str, str]]:
    """Return the chat messages asking the judge to compare two responses."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                prompt=prompt, response_a=response_a, response_b=response_b
            ),
        },
    ]
