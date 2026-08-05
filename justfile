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
    find . -type d -name ".karva_cache" -exec rm -rf {} +

install: clean
    uv sync

# Run the test suite
test: install
    uv run karva test tests/

# Format Python code
fmt:
    uvx ruff format .

# Check Python formatting
fmt-check:
    uvx ruff format --check .

# Lint Python code
lint:
    -uvx ruff check .
    -uvx ty check .

# Run the full CI pipeline locally (format check, lint, test)
ci: fmt-check lint test

# Build the Python package
build:
    uv build

# Publish the Python package to PyPI
publish:
    uv publish

# --- FoldGemma CLI Commands ---

# Train the FoldGemma model
train tfrecord_pattern model_size="small" batch_size="32" epochs="1" steps="1000":
    uv run --extra train foldgemma train {{tfrecord_pattern}} --model-size {{model_size}} --batch-size {{batch_size}} --epochs {{epochs}} --steps-per-epoch {{steps}}

# Run inference to predict 3di structures
infer input output model_size="small":
    uv run foldgemma infer --input {{input}} --output {{output}} --model-size {{model_size}}

# Prepare Steinegger Lab AFDB data into TFRecords for training
prep db_path out_dir num_workers="4":
    uv run --extra train foldgemma prep {{db_path}} {{out_dir}} --num-workers {{num_workers}}

# Deploy a trained model to the Hugging Face Hub
deploy repo_id model_path:
    uv run foldgemma deploy --repo-id {{repo_id}} --model-path {{model_path}}

# --- Utility Scripts ---

# Run the 3di comparison script to validate model against Foldcomp structure
compare-3di struct_fasta model_fasta:
    uv run --extra dev python scripts/compare_3di.py {{struct_fasta}} {{model_fasta}}

# Test mini3di package explicitly
test-mini3di fasta_file:
    uv run --extra dev python scripts/test_mini3di.py {{fasta_file}}
