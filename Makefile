# Project targets. Heavy jobs run through scripts/run_job.sh, which uses the backend in
# configs/compute.yaml (slurm or local). Variables are passed as VAR=value.
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.cache/uv-python
export HF_HOME := $(CURDIR)/.cache/hf

RUN := scripts/run_job.sh
GPUS ?= 1
CHAIN ?= 1
STAGE ?= sft
SCALING_GPUS ?= 1 2 4 8
CHECKPOINT ?=
CANDIDATE ?=
REFERENCE ?=
REPO ?= teacher
KIND ?= cpu

.PHONY: setup lint test check test-slow test-integration clean \
	sync pull-results setup-envs env-info compile-envs \
	data judge-prompts train scaling merge set-teacher eval generate eval-variant judge judge-local e1-summary \
	distill teacher-gen sft-baseline e3 quantize gguf cards publish \
	serve-bench benchmark agent-traces agent-bench speculative disagg \
	chart-check up down logs loadtest k8s-up k8s-load k8s-report k8s-down \
	demo smoke results results-check

# ============= Development =============

setup:
	uv sync --python 3.11 --frozen --extra ml

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

test-slow:
	uv run pytest -m slow tests/e2e

test-integration:
	uv run pytest -m "integration or slow" tests

check: lint test

smoke:
	uv run python scripts/smoke.py

results:
	uv run python scripts/render_results.py

results-check:
	uv run python scripts/render_results.py --check

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist .tmp/smoke

# ============= Machines and environments =============

sync:
	scripts/sync.sh

pull-results:
	scripts/pull_results.sh

setup-envs:
	$(RUN) setup_envs

env-info:
	$(RUN) env_info

compile-envs:
	scripts/compile_envs.sh

# ============= Data =============

data:
	$(RUN) data

judge-prompts:
	uv run python scripts/prepare_data.py --judge-only

# ============= Alignment =============

# make train STAGE=sft|dpo|orpo|simpo [GPUS=N CHAIN=K]
train:
	$(RUN) train --gpus $(GPUS) --chain $(CHAIN) -- --stage $(STAGE)

# one point of the scaling experiment per GPU count; counts above the backend limit are skipped
scaling:
	@for n in $(SCALING_GPUS); do \
		echo "scaling with $$n GPUs"; \
		$(RUN) scaling --gpus $$n || echo "skipped $$n GPUs"; \
	done

merge:
	$(RUN) merge -- --stage $(STAGE)

# make set-teacher STAGE=dpo points the teacher at that stage's merged checkpoint
set-teacher:
	scripts/set_teacher.sh $(STAGE)

# ============= Evaluation =============

# make eval CHECKPOINT=name ; make generate CHECKPOINT=name ; make judge CANDIDATE=name
eval:
	$(RUN) eval --gpus $(GPUS) -- --checkpoint $(CHECKPOINT)

generate:
	$(RUN) generate --gpus $(GPUS) -- --checkpoint $(CHECKPOINT)

eval-variant:
	$(RUN) eval_variant --gpus $(GPUS) -- --checkpoint $(CHECKPOINT)

# judge runs where the responses are stored; judge-local runs in the development environment
judge:
	$(RUN) judge -- --candidate $(CANDIDATE) $(if $(REFERENCE),--reference $(REFERENCE))

judge-local:
	uv run python scripts/eval.py judge --candidate $(CANDIDATE) $(if $(REFERENCE),--reference $(REFERENCE))

e1-summary:
	uv run python scripts/eval.py summary

# ============= Distillation =============

distill:
	$(RUN) distill --gpus $(GPUS) --chain $(CHAIN)

teacher-gen:
	$(RUN) teacher_gen --gpus $(GPUS)

sft-baseline:
	$(RUN) sft_baseline --gpus $(GPUS) --chain $(CHAIN)

e3:
	$(RUN) e3

# ============= Quantization and publishing =============

quantize:
	$(RUN) quantize --gpus $(GPUS)

gguf:
	$(RUN) gguf

cards:
	uv run python scripts/render_model_card.py --repo $(REPO)

publish:
	$(RUN) publish -- --repo $(REPO)

# ============= Serving experiments =============

serve-bench:
	$(RUN) serve_bench --gpus 1

benchmark: serve-bench

agent-traces:
	uv run python scripts/bench.py agent-traces

agent-bench:
	$(RUN) agent_bench --gpus 1

speculative:
	$(RUN) speculative --gpus 1

disagg:
	$(RUN) disagg --gpus 2

# ============= Kubernetes =============

chart-check:
	scripts/validate_chart.sh

up: k8s-up

down: k8s-down

logs:
	KUBECONFIG=$(CURDIR)/.kube/config kubectl logs -l app.kubernetes.io/name=llm-serve --tail=200 --all-containers

loadtest: k8s-load

k8s-up:
	scripts/k8s_up.sh

k8s-load:
	scripts/k8s_load.sh

k8s-report:
	uv run python scripts/k8s_report.py --kind $(KIND)

k8s-down:
	scripts/k8s_down.sh

# ============= Demo =============

demo:
	scripts/demo.sh
