# Agent trace sources

Place the real system prompt and tool schemas of each agent here before generating traces.
Only code text is allowed: no secrets, no workspace data.

For each agent `<name>`, provide two files:

- `<name>.system.txt`: the system prompt, as plain text.
- `<name>.tools.json`: the tool schemas, as a JSON list of objects.

Then run `make agent-traces`, which writes `eval/agent_traces/traces.jsonl`. Conversations
alternate between the agents found in this directory.
