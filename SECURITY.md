# Security policy

## Secrets

The platform reads `HF_TOKEN` and `GROQ_API_KEY` from the environment and never writes them to a file, a log, or a results file. They are held as `SecretStr` values. Keep `.env` out of version control and out of every sync (`.syncignore` excludes it), and keep `~/.llmplatform.env` on job machines at mode 600. `detect-secrets` scans tracked files in pre-commit against `.secrets.baseline`.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the repository owner through the contact method on the owner's profile, not in a public issue. Include the affected file, a description, and steps to reproduce.

## Scope

Model weights and datasets come from their upstream licenses and are pinned by revision. Uploads create private Hub repositories. The Helm chart verifies the SHA-256 of a downloaded GGUF and refuses to render without a checksum for the llama.cpp backend unless `model.allowUnverified` is set.
