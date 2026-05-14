-include .env
export

.PHONY: install test test-fast test-slow ingest qdrant-up qdrant-down qdrant-rm

install:
	uv sync

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

test-slow:
	uv run pytest -m slow

ingest:
	uv run python -m src.pipeline $(or $(path),.)

qdrant-up:
	docker run -d --name qdrant -p 127.0.0.1:6333:6333 -v qdrant_data:/qdrant/storage -e QDRANT__SERVICE__API_KEY=$(QDRANT_API_KEY) qdrant/qdrant

qdrant-down:
	docker stop qdrant

qdrant-rm:
	docker rm qdrant
