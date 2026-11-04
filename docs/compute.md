# Compute backends

All heavy stages run through `scripts/run_job.sh JOB [--gpus N] [--chain K] [--time HH:MM:SS] [--smoke] [-- job arguments]`. The script reads `configs/compute.yaml` and hands the job to the configured backend. Job bodies are in `jobs/` and are the same for both backends.

## Configuration

| Key | Meaning |
|---|---|
| `backend` | `slurm` or `local` |
| `provider_label` | Free-text description stored in every results file |
| `gpu_hourly_usd`, `price_source`, `price_date` | Price per GPU hour used for cost, with its source and date; a results file for a GPU run cannot be written while the price is unset |
| `gpu_model` | Optional override when `nvidia-smi` is not visible to the job |
| `work_root` | Storage for environments, caches, datasets, and checkpoints; `PLATFORM_ROOT` overrides it |
| `slurm.partition`, `cpu_partition`, `smoke_partition` | Partitions for GPU, CPU-only, and smoke jobs |
| `slurm.gpu_type`, `max_gpus_per_job`, `time_limit`, `smoke_time_limit` | Generic resource request and per-job GPU ceiling |
| `slurm.cpus_per_gpu`, `mem_per_gpu_gb`, `account`, `qos`, `extra_args` | Further `sbatch` options |
| `slurm.max_chain` | Longest `--chain` accepted |
| `local.max_gpus`, `time_limit_hours`, `max_restarts` | Single-machine limits and restart policy |
| `local.use_docker`, `docker_image`, `docker_extra_args` | Run job bodies inside the container |
| `jobs.<name>.gpus` | Default GPU count of each job body |

A job that asks for more GPUs than the backend allows fails before submission with a message naming the limit. Multi-GPU jobs need a partition or machine that provides them; the scaling experiment skips GPU counts above `max_gpus_per_job`.

## Slurm

`slurm/submit.sh` turns the configuration into `sbatch` options and submits `slurm/run_job.sbatch`, which runs the matching body in `jobs/`. With `--chain K` it submits K jobs with `--dependency=afterany`. Finished runs write a completion marker, so later jobs in a chain exit at once, and unfinished runs resume from the newest complete checkpoint. `--dry-run` prints the commands without submitting.

## Local

`local/run.sh` runs the job body directly. It restarts after a time-limit stop or a failure, up to `local.max_restarts` times, and each restart resumes from the newest checkpoint. Run it under `tmux` or `nohup` for long jobs. With `local.use_docker: true` it delegates to `local/docker_run.sh`, which mounts the repository and the storage root into the image built from `docker/Dockerfile` and forwards `HF_TOKEN`, `GROQ_API_KEY`, and `HF_NAMESPACE` by name.

## Secrets and storage

Secrets for job bodies live in `~/.llmplatform.env` on the machine that runs them (mode 600, never synced), with `HF_TOKEN`, `GROQ_API_KEY`, `HF_NAMESPACE`, and optionally `PLATFORM_ROOT` and `LLAMA_CPP_DIR`. `jobs/common.sh` sources it. `.env` in the repository holds `REMOTE_HOST` and `REMOTE_DIR` for `make sync` and `make pull-results` and is excluded from every sync.

## Environments

`make setup-envs` runs `jobs/setup_envs.sh`, which creates one virtual environment per file in `envs/` under `$PLATFORM_ROOT/envs`. If installing SGLang or vLLM fails against the machine's CUDA, `jobs/setup_envs.sh --apptainer-fallback` pulls the official container named in `envs/images.txt` and job bodies run inside it. Results record whether a virtual environment or a container ran.
