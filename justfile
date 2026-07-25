lint:
	uvx run ruff check .

typecheck:
	uvx run ty check

test:
	uv run python -m pytest tests/

data:
	uv run python -m foldgemma.data.generate_synthetic

train:
	uv run python -m foldgemma.train.train

convert:
	uv run python scripts/convert_jax_to_pytorch.py

deploy repo_id:
	uv run python scripts/deploy_to_hf.py --repo_id {{repo_id}}
