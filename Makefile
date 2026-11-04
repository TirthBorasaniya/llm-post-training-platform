# Targets are added as each component lands.
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.cache/uv-python
export HF_HOME := $(CURDIR)/.cache/hf

.PHONY: setup lint test check test-slow clean sync pull-results setup-envs env-info compile-envs data judge-prompts

JOB_RUN := scripts/run_job.sh

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

check: lint test

sync:
	scripts/sync.sh

pull-results:
	scripts/pull_results.sh

setup-envs:
	$(JOB_RUN) setup_envs

env-info:
	$(JOB_RUN) env_info

compile-envs:
	scripts/compile_envs.sh

data:
	$(JOB_RUN) data

judge-prompts:
	uv run python scripts/prepare_data.py --judge-only

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
