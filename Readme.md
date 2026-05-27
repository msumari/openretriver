# OpenRetriver

Index any codebase and ask questions about it. OpenRetriver loads your project files, chunks them intelligently (docs by paragraph, code by function/class using tree-sitter), embeds everything with dense + sparse vectors, stores it in Qdrant, and answers questions using hybrid search with LLM generation.

## What it does

1. **Ingests** your project: walks the directory, skips junk (node_modules, .venv, etc.), splits files into meaningful chunks
2. **Embeds** each chunk with both dense vectors (semantic) and sparse BM25 (keyword matching)
3. **Searches** using hybrid retrieval: combines semantic similarity with exact term matching via Reciprocal Rank Fusion
4. **Answers** your questions grounded in the actual code, with source citations

## Supported languages

| Type | Extensions |
|------|-----------|
| Documentation | `.md`, `.txt` |
| Python | `.py` |
| JavaScript | `.js` |
| TypeScript | `.ts` |
| Rust | `.rs` |

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker (for Qdrant)
- AWS credentials configured (for Bedrock LLM calls)

## Setup

You need Qdrant running. Pick your API key and start it with Docker:

```bash
docker run -d --name qdrant -p 127.0.0.1:6333:6333 \
  -v qdrant_data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY='your-key-here' \
  qdrant/qdrant
```

Then install openretriver:

```bash
git clone <repo-url>
cd openretriver
uv tool install -e .
```

## Usage

Export your Qdrant API key before running any command:

```bash
export QDRANT_API_KEY='your-key-here'

openretriver ingest ~/Projects/my-app
openretriver ask "how does authentication work"
openretriver chat
```

### Multi project support

Each project gets its own collection. Specify a name to keep them separate:

```bash
openretriver ingest ~/Projects/backend --name backend
openretriver ingest ~/Projects/frontend --name frontend

openretriver ask "how does auth work" --name backend
openretriver chat --name frontend
```

When ingesting without `--name`, the collection is named after the directory (e.g. `~/Projects/backend` becomes `backend`). For `ask` and `chat` you always need `--name` to tell it which collection to search.

## How it works

**Ingestion is incremental.** On re-ingest, only new and changed files are processed. Unchanged files are skipped. Deleted files have their stale chunks removed. This is tracked via SHA-256 content hashes stored alongside the vectors.

**Search is hybrid.** Two independent signals (dense semantic + sparse BM25) are fused with Reciprocal Rank Fusion server-side. This means both "find function parse_config" (exact match) and "how does config loading work" (semantic) give good results.

**Answers are grounded.** The LLM only answers from retrieved context. If the indexed docs don't cover your question, it says so instead of making things up.

## Configuration

Set these in `.env` or export them:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_API_KEY` | (none) | Qdrant API key |
| `MODEL_PROVIDER` | `bedrock` | LLM provider |
| `BEDROCK_REGION` | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | `global.anthropic.claude-sonnet-4-6` | Model to use |

## Development

```bash
make test          # run all tests
make test-fast     # skip slow tests (embedding/storage)
make test-slow     # only slow tests

# Run a single test
uv run pytest tests/test_loader.py -v
```

## File structure

```
src/
├── cli.py              # CLI entry point (ingest, ask, chat subcommands)
├── loader.py           # Walks project directory, filters files
├── chunker_docs.py     # Splits docs by paragraph with token budget
├── chunker_code.py     # Splits code by AST symbols via tree-sitter
├── embedder.py         # Dense + sparse BM25 embeddings via FastEmbed
├── change_detection.py # SHA-256 manifest for incremental re-ingestion
├── storage.py          # Qdrant client, collection setup, upsert/delete
├── search.py           # Hybrid search with RRF fusion
├── rag.py              # Context formatting, LLM call, cited answers
├── model_provider.py   # Pluggable LLM provider (default: Bedrock)
├── session.py          # Interactive REPL with conversation memory
└── models.py           # Shared dataclasses and constants
```

## License

MIT
