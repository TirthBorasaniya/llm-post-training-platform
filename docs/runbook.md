# Runbook

An ordered sequence from a clean checkout to a full set of results. Commands that start with `make` run on the machine named in `configs/compute.yaml`; for the slurm backend run them on the cluster login node after `make sync`, and for the local backend run them on the GPU machine. `make pull-results` copies `results/` and `logs/` back.

## 0. Preparation on the development machine

1. `make setup`, `make check`, and `make smoke` confirm that the code works on tiny models.
2. Create `.env` from `.env.example` with `REMOTE_HOST` and `REMOTE_DIR` (slurm backend), and `HF_TOKEN`, `GROQ_API_KEY`, and `HF_NAMESPACE` for upload and judge stages.
3. Edit `configs/compute.yaml`: `backend`, `provider_label`, `gpu_hourly_usd` with `price_source` and `price_date`, and for slurm the partition, GPU type, time limit, and `max_gpus_per_job`.
4. `make judge-prompts` writes `eval/judge_prompts.jsonl`; commit it.
5. Put the real system prompts and tool schemas of the agents in `eval/agent_traces/source/` (see its README), then `make agent-traces` and commit `eval/agent_traces/traces.jsonl`.

## 1. Prepare the machine

1. Slurm: `make sync`, then on the cluster create `~/.llmplatform.env` (mode 600) with the three secrets and optionally `PLATFORM_ROOT`. Local: clone the repository on the machine, create the same file, and optionally set `local.use_docker`.
2. `make setup-envs` (add `--apptainer-fallback` through `scripts/run_job.sh setup_envs --apptainer-fallback` if serving packages fail to install).
3. `make env-info` prints the node, GPUs, driver, and CUDA version.
4. `make data` downloads and filters the datasets and writes `results/data_manifest.json`.

## 2. Alignment

1. `make train STAGE=sft CHAIN=3` (increase `CHAIN` until the run finishes; finished runs exit at once).
2. `make train STAGE=dpo CHAIN=3`, then the same for `orpo` and `simpo`. The three can run in parallel once SFT has finished.
3. `make scaling` runs 1, 2, 4, and 8 GPUs on one node. Counts above `max_gpus_per_job` are skipped, so the report is marked incomplete until a node that allows them is used.

## 3. Evaluation

1. For each of `base_7b`, `sft`, `dpo`, `orpo`, `simpo`: `make eval CHECKPOINT=NAME`, `make generate CHECKPOINT=NAME`, then `make judge CANDIDATE=NAME`.
2. `make e1-summary` after the files exist.

## 4. Distillation

1. `make set-teacher STAGE=dpo` points `teacher/merged` under the checkpoint root at the merged checkpoint of the stage you chose as the teacher.
2. `make teacher-gen`, then `make distill CHAIN=4`, then `make sft-baseline CHAIN=4`. For the alternative learning rate run `scripts/run_job.sh distill -- --alternative-lr`.
3. Evaluate `teacher`, `base_1_5b`, `sft_baseline_1_5b`, and `distilled_1_5b` as in step 3, plus `make judge CANDIDATE=distilled_1_5b REFERENCE=base_1_5b` for the direct comparison.
4. `make e3`.

## 5. Quantization and publishing

1. `make quantize` (AWQ on GPUs without FP8) and `make gguf` (builds the pinned llama.cpp on first use).
2. `make eval-variant CHECKPOINT=NAME` for each variant in `configs/eval/e1.yaml` that applies, then `make judge CANDIDATE=NAME`, then `make e1-summary`.
3. `make cards REPO=teacher` (also `student`, `gguf`) to review the generated cards, then `make publish REPO=teacher` and the other two. Repositories are private until `private: false` is set in `configs/publish.yaml`.

## 6. Serving experiments

1. `make serve-bench`, `make agent-bench`, `make speculative`.
2. `scripts/run_job.sh disagg --gpus 2` on a node with two GPUs (`make disagg`).
3. `scripts/check_engine_flags.py` inside each serving environment confirms the flags against the installed versions.

## 7. Kubernetes

1. CPU on kind: set `MODEL_REPO`, `MODEL_FILE`, `MODEL_SHA256` (from `results/quantization.json`), then `make k8s-up`, `make k8s-load`, `make k8s-report KIND=cpu`, `make k8s-down`. Nothing else may be running in Docker.
2. GPU: follow `docs/gpu_validation.md`, then `make k8s-report KIND=gpu` with the provider, GPU model, and price options.

## 8. Finish

`make pull-results`, `make results`, `make results-check`, then commit the results together with the code and configuration that produced them. `make demo` and the Space instructions in `space/README.md` start the public demo once the GGUF repository is public.
