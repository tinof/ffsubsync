# Makefile for easy development workflows.
# See DEVELOPMENT.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

# Use only the checked-in project configuration. Otherwise uv merges user- and
# system-level settings into uv.lock, which can make it fail on another machine.
UV_CONFIG_FILE := $(CURDIR)/uv.toml
export UV_CONFIG_FILE

# Safe default for every dependency resolution invoked through this Makefile.
UV_EXCLUDE_NEWER ?= 14 days
export UV_EXCLUDE_NEWER

SYNC_ARGS := --all-groups

.PHONY: default install lint lint-check typecheck test test-integration upgrade build clean

default: install lint test

install:
	uv sync $(SYNC_ARGS)

lint:
	uv run python devtools/lint.py

# Check-only lint, matching CI (does not modify files).
lint-check:
	uv run python devtools/lint.py --check

typecheck:
	uv run basedpyright ffsubsync

test:
	uv run pytest -m 'not integration'

test-integration:
	INTEGRATION=1 uv run pytest -m integration

upgrade:
	uv sync --upgrade $(SYNC_ARGS)

build: install
	uv build --no-build-isolation

clean:
	-rm -rf dist/
	-rm -rf build/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .ruff_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +
