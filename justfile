# FoldGemma Project Justfile
# Run `just` to see all available commands

set shell := ["bash", "-uc"]

# Show available commands
default:
    @just --list

# Clean Python virtual environments
clean:
    rm -rf site
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

install: clean
    uv sync

# Run the test suite
test: install
    uv run pytest tests/

# Format Python code
fmt:
    uv run black .

# Check Python formatting
fmt-check:
    uv run black --check .

# Lint Python code
lint:
    uv run ruff check .
    uv run ty check

# Run the full CI pipeline locally (format check, lint, test)
ci: fmt-check lint test

# Build the Python package
build:
    uv build

# Publish the Python package to PyPI
publish:
    uv publish

# Prepare Steinegger Lab AFDB data into TFRecords for training
build-dataset tsv fcz out_dir:
    uv run foldgemma prep --tsv {{tsv}} --fcz {{fcz}} --out-dir {{out_dir}}
