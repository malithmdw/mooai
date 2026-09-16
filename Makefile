.PHONY: help install dev-install run api streamlit ingest index index-init test lint typecheck check fmt clean

help:
	@echo "Available targets:"
	@echo "  install      Install runtime dependencies"
	@echo "  dev-install  Install runtime + dev dependencies"
	@echo "  api          Run the FastAPI app (uvicorn, reload)"
	@echo "  streamlit    Run the Streamlit UI"
	@echo "  ingest       Run the document ingestion pipeline (data/knowledge → stdout JSONL)"
	@echo "  index        Ingest, embed, and upsert documents to Pinecone"
	@echo "  index-init   Create the Pinecone index (run once before first index build)"
	@echo "  test         Run pytest"
	@echo "  lint         Run ruff check"
	@echo "  fmt          Run ruff format"
	@echo "  typecheck    Run mypy"
	@echo "  check        Run lint + typecheck + test (CI gate)"
	@echo "  clean        Remove caches and build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	streamlit run scripts/streamlit_app.py

ingest:
	python -m scripts.ingest

index:
	python -m scripts.index

index-init:
	python -m scripts.index --init

test:
	pytest

lint:
	ruff check .

fmt:
	ruff format .

typecheck:
	mypy src

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
