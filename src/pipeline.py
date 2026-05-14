import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.models import LoadedFile, Chunk, DOC_EXTENSIONS
from src.loader import load_files
from src.chunker_docs import chunk_doc
from src.chunker_code import chunk_code
from src.embedder import embed_chunks
from src.storage import get_client, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)


def _chunk_file(loaded_file: LoadedFile) -> list[Chunk]:
    if loaded_file.extension in DOC_EXTENSIONS:
        return chunk_doc(loaded_file)
    return chunk_code(loaded_file)


def ingest(
    project_path: str | Path,
    client: QdrantClient | None = None,
) -> int:
    if client is None:
        client = get_client()

    logger.info("Loading files from %s", project_path)
    files = load_files(project_path)
    if not files:
        logger.warning("No supported files found in %s", project_path)
        return 0
    logger.info("Loaded %d files", len(files))

    all_chunks = []
    for f in files:
        chunks = _chunk_file(f)
        logger.info("  %s → %d chunks", f.path, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced from %d files", len(files))
        return 0
    logger.info("Chunked into %d total chunks", len(all_chunks))

    logger.info("Embedding %d chunks...", len(all_chunks))
    embedded = embed_chunks(all_chunks)
    logger.info("Embedding complete")

    logger.info("Storing in Qdrant...")
    ensure_collection(client)
    stored = upsert_chunks(client, embedded)
    logger.info("Stored %d chunks", stored)

    return stored


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    path = sys.argv[1] if len(sys.argv) > 1 else "."
    count = ingest(path)
    print(f"Ingested {count} chunks from {path}")
