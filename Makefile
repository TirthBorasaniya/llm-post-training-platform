# Targets are added as each component lands.
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.cache/uv-python
export HF_HOME := $(CURDIR)/.cache/hf

.PHONY: setup lint test clean

setup:
	uv sync --python 3.11 --frozen --extra ml

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
