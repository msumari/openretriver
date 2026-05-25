import logging
from pathlib import Path

from qdrant_client import QdrantClient

from src.models import LoadedFile, Chunk, DOC_EXTENSIONS
from src.loader import load_files
from src.chunker_docs import chunk_doc
from src.chunker_code import chunk_code
from src.embedder import embed_chunks
from src.storage import (
    get_client,
    ensure_collection,
    upsert_chunks,
    delete_source,
    store_manifest,
    fetch_manifest,
)
from src.change_detection import detect_changes

logger = logging.getLogger(__name__)


def _chunk_file(loaded_file: LoadedFile) -> list[Chunk]:
    if loaded_file.extension in DOC_EXTENSIONS:
        return chunk_doc(loaded_file)
    return chunk_code(loaded_file)


def _derive_collection_name(project_path: str | Path) -> str:
    return Path(project_path).resolve().name


def ingest(
    project_path: str | Path,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
) -> int:
    if client is None:
        client = get_client()

    if collection_name is None:
        collection_name = _derive_collection_name(project_path)

    logger.info("Loading files from %s (collection: %s)", project_path, collection_name)
    files = load_files(project_path)
    if not files:
        logger.warning("No supported files found in %s", project_path)
        return 0
    logger.info("Loaded %d files", len(files))

    ensure_collection(client, collection_name)

    manifest = fetch_manifest(client, collection_name)
    changes = detect_changes(files, manifest)
    logger.info(
        "Changes: %d new, %d changed, %d unchanged, %d deleted",
        len(changes.new), len(changes.changed), len(changes.unchanged), len(changes.deleted),
    )

    to_process = changes.new + changes.changed
    if not to_process and not changes.deleted:
        logger.info("Nothing changed, skipping ingestion")
        return 0

    for source in changes.deleted:
        logger.info("  Deleting stale chunks for %s", source)
        delete_source(client, source, collection_name)

    for f in changes.changed:
        logger.info("  Deleting old chunks for changed file %s", f.path)
        delete_source(client, f.path, collection_name)

    if not to_process:
        logger.info("Only deletions, no new chunks to embed")
        return 0

    all_chunks = []
    new_manifest_entries = {}
    for f in to_process:
        chunks = _chunk_file(f)
        logger.info("  %s → %d chunks", f.path, len(chunks))
        all_chunks.extend(chunks)
        new_manifest_entries[f.path] = {
            "content_hash": changes.hashes[f.path],
            "chunk_count": len(chunks),
        }

    logger.info("Embedding %d chunks...", len(all_chunks))
    embedded = embed_chunks(all_chunks)
    logger.info("Embedding complete")

    logger.info("Storing in Qdrant...")
    stored = upsert_chunks(client, embedded, collection_name)
    logger.info("Stored %d chunks", stored)

    store_manifest(client, new_manifest_entries, collection_name)

    return stored


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    count = ingest(args.path, collection_name=args.name)
    print(f"Ingested {count} chunks from {args.path}")
