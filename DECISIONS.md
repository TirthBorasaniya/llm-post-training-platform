# Design decisions

## D-001: One set of job bodies for two compute backends
Status: accepted (2026-11-04)
Context: The platform must run on any SLURM cluster and on a single rented GPU machine without naming either.
Decision: Job bodies live once in `jobs/`. `slurm/submit.sh` builds `sbatch` options from `configs/compute.yaml`, and `local/run.sh` runs the same body directly with restart after a timeout or crash. Partition, GPU type, time limit, and the GPU ceiling per job are configuration.
Alternatives: Separate script sets per backend (rejected: they drift), a workflow engine (rejected: a second system to install on every machine), Kubernetes batch jobs (rejected: no scheduler on the single-machine case).
Consequences: Every long stage must checkpoint and resume, because both backends can stop a job. Results record the provider label and price from the same file.

## D-002: Data-parallel ZeRO-2, not ZeRO-3 and not pipeline parallelism
Status: accepted (2026-11-04)
Context: A 4-bit 7B base with LoRA adapters fits on one modern GPU, so extra GPUs add throughput rather than capacity.
Decision: DeepSpeed ZeRO-2 shards optimizer state and gradients across data-parallel ranks with low communication overhead. ZeRO-3 parameter sharding is unnecessary because the quantized base already fits on each GPU, and it does not combine with bitsandbytes quantized weights.
Alternatives: ZeRO-3 (rejected: sharding parameters that already fit adds communication), pipeline or tensor parallelism (rejected: the model does not need to be split), plain DDP (rejected: optimizer state is replicated for no benefit).
Consequences: Scaling is measured as data-parallel efficiency on one node and is limited by that node's interconnect.

## D-003: Preference objectives run under identical settings
Status: accepted (2026-11-04)
Context: DPO, ORPO, and SimPO are compared, so any difference in optimizer, schedule, lengths, adapters, data, or seed would confound the comparison.
Decision: The three configs share every field except the objective-specific values (`beta`, `simpo_gamma`), and a test enforces the equality. All start from the merged SFT checkpoint.
Alternatives: Per-objective tuned learning rates (rejected: it turns the study into a tuning contest), one shared file with overrides (rejected: separate files keep each run self-describing).
Consequences: Objective-specific tuning could change the ranking; the result is a comparison at one fixed setting.

## D-004: Position-swapped pairwise judging with a disk cache
Status: accepted (2026-11-04)
Context: Language-model judges favor the first position and longer answers.
Decision: Each pair is judged in both orders. A win needs both orders to prefer the candidate, a loss needs both to prefer the reference, and anything else is a tie. The judge differs from every candidate, mean response length is reported with every win rate, and judgments are cached by the SHA-256 of model, template version, prompt, and both responses so reruns cost nothing.
Alternatives: Single-order judging (rejected: position bias), averaging scores (rejected: the judge returns a preference, not a calibrated score).
Consequences: Disagreement between orders counts as a tie, which lowers the contrast between close systems.

## D-005: Decontamination before any training set is written
Status: accepted (2026-11-04)
Context: Training prompts that overlap the judge prompts or benchmark questions inflate every result.
Decision: Rows are removed by normalized exact match and by MinHash near-duplicate match (128 permutations, 5-word shingles, estimated Jaccard similarity of at least 0.8) against the judge prompts and the question fields of the six benchmarks. Counts per filter are written to the data manifest.
Alternatives: Exact matching only (rejected: misses reformatted copies), embedding similarity (rejected: it needs a model and a threshold with no ground truth).
Consequences: Benchmark sources are read at their default branch head when the data is prepared; their sizes are recorded.

## D-006: The teacher is resized to the student's embedding size
Status: accepted (2026-11-04)
Context: TRL's distillation trainer requires equal `vocab_size`, and the 7B and 1.5B models pad their embeddings differently.
Decision: After verifying identical tokenizers and that the tokenizer is shorter than both padded sizes, `resize_token_embeddings` shrinks the teacher to the student's size. Logits over real tokens are compared before and after on 50 prompts with a tolerance chosen by dtype, and the probability mass on the dropped rows is recorded. Distillation refuses to start when the check fails.
Alternatives: Padding the student up to the teacher (rejected: it changes the student that is published), a custom loss over the shared vocabulary (rejected: it reimplements the trainer).
Consequences: The dropped rows are untrained padding, and the recorded mass on them shows how little the teacher distribution changes.

## D-007: Float32 student weights with bf16 autocast
Status: accepted (2026-11-04)
Context: The trainer loads string-specified models in float32 unless told otherwise, which doubles memory.
Decision: `model_init_kwargs` sets the dtype explicitly. The student's master weights are float32 and compute runs in bf16 autocast, so updates at a learning rate of 1e-6 are not rounded away. The SFT baseline uses the same weight dtype, and both are configurable (`student_training.dtype`, `student_dtype`).
Alternatives: Pure bf16 weights (rejected: small updates at 1e-6 can be lost to rounding), leaving the trainer default implicit (rejected: the dtype should be stated, not inherited).
Consequences: Weights, gradients, and optimizer state for the 1.5B student take about 24 GB, so full fine-tuning still fits an 80 GB GPU alongside the 7B teacher and colocated generation; `student_training.dtype: bfloat16` is the lower-memory option.

## D-008: The SFT baseline stops at the distillation GPU-hours
Status: accepted (2026-11-04)
Context: Distillation is compared with SFT at equal compute, not equal steps.
Decision: Teacher completions for the same prompts are generated with SGLang, then the student trains with a constant schedule until the ledger reaches the distillation run's GPU-hours. Teacher generation hours are recorded separately.
Alternatives: Equal epochs (rejected: unequal cost), a cosine schedule with a precomputed step count (rejected: step time is unknown before the run).
Consequences: A constant schedule does not anneal, which slightly disadvantages the baseline.

## D-009: Time-based checkpoints and a GPU-hour ledger
Status: accepted (2026-11-04)
Context: Jobs are killed at the time limit and step duration varies with rollout length.
Decision: A checkpoint is requested whenever 40 minutes have passed since the last one, and a ledger records the start and last heartbeat of every segment so that a killed segment counts only what it ran.
Alternatives: A fixed `save_steps` (rejected: lost work then depends on step time).
Consequences: A kill loses at most 40 minutes of work plus one step.

## D-010: FP8 online on Ada and Hopper, AWQ elsewhere
Status: accepted (2026-11-04)
Context: The right GPU quantization depends on the GPU generation.
Decision: The method is chosen from the compute capability. The server quantizes to FP8 at load on capability 8.9 or higher, and older GPUs use AWQ 4-bit weights produced by llm-compressor. Both variants are optional entries in the quality experiment.
Alternatives: AWQ everywhere (rejected: slower to produce and unnecessary on FP8 GPUs), FP8 checkpoints written to disk (rejected: the server's online path needs no files).
Consequences: Results state which method ran.

## D-011: llama.cpp tools resolved by name or subcommand
Status: accepted (2026-11-04)
Context: Recent llama.cpp releases ship one `llama` binary with subcommands, while older ones ship `llama-quantize`, `llama-cli`, and `llama-server`.
Decision: The tool resolver prefers the unified binary and falls back to the legacy names. The release tag is pinned in `configs/quantize.yaml`.
Alternatives: Pinning only the legacy names (rejected: they fail on current releases).
Consequences: Verification prompts use the single-turn flag of the command-line tool.

## D-012: Engine flags verified against the pinned sources
Status: accepted (2026-11-04)
Context: Serving flags change between releases. SGLang's benchmark client moved to `sglang.benchmark.serving`, and vLLM pairs `--enable-prefix-caching` with `--disable-prefix-caching`.
Decision: The flags used by the command builders were checked against the pinned releases, and `scripts/check_engine_flags.py` rechecks them against installed help output on the serving machine.
Alternatives: Trusting documentation (rejected: it lags the code).
Consequences: A flag removed in a later pin is caught by the script before a run.

## D-013: Prefill and decode disaggregation uses NIXL
Status: accepted (2026-11-04)
Context: Mooncake needs a kernel module that cannot be installed on a shared cluster.
Decision: SGLang's NIXL transfer backend moves the cache between a prefill server on one GPU and a decode server on the other, behind the router in prefill-decode mode. The baseline is two colocated replicas behind the same router.
Alternatives: Mooncake (rejected: installation), no router (rejected: the comparison needs identical admission).
Consequences: If NIXL cannot run on the available machine, the same job runs on a rented two-GPU node and the results record the hardware.

## D-014: The disaggregation SLO is defined on the chat class
Status: accepted (2026-11-04)
Context: Long prefills delay short requests on colocated replicas, which is the interference disaggregation addresses.
Decision: A run meets the SLO when chat-class p95 time to first token and overall p95 inter-token latency are within the limits and no request failed. Long-prompt time to first token is reported separately.
Alternatives: A single overall time to first token (rejected: long prompts dominate it for both setups).
Consequences: A null or negative result is reported as measured.

## D-015: Agent traces keep the prefix real and the history synthetic
Status: accepted (2026-11-04)
Context: Prefix caching depends on a long shared prefix that grows across turns.
Decision: System prompts and tool schemas come from the real agents. User messages, tool results, and assistant turns are synthetic and seeded, so every run replays identical prefixes. The prefix cache is flushed between runs.
Alternatives: Recording live agent sessions (rejected: not reproducible and may contain workspace data).
Consequences: Output length behavior is not that of a real agent.

## D-016: Results are numbers from committed scripts only
Status: accepted (2026-11-04)
Context: A result that cannot be traced to code and configuration is not evidence.
Decision: Every results file records the code version, the SHA-256 of its config, the GPU model and count, the provider label, and the hourly price. README tables are rendered from those files, and a check fails when they differ.
Alternatives: Hand-written tables (rejected: they drift), a database (rejected: files diff in review).
Consequences: A GPU run cannot be recorded until the price is configured.

## D-017: Hub repositories start private
Status: accepted (2026-11-04)
Context: Publishing is outward-facing and hard to undo.
Decision: Uploads create private repositories, verify the base model license against the Hub before rendering a card, and read the namespace and token from the environment. Making a repository public is a deliberate configuration change.
Alternatives: Public by default (rejected: a card or file error becomes public immediately).
Consequences: The demo needs public repositories, so the switch must be made before it is linked.
