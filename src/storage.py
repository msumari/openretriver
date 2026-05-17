import dataclasses
import hashlib
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    IntegerIndexParams,
    IntegerIndexType,
    PayloadSchemaType,
)

from src.models import Chunk, EmbeddedChunk, FILE_TYPE_MANIFEST

COLLECTION_NAME = "openretriver"
VECTOR_SIZE = 384


def get_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def _create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=TextIndexParams(
            type=TextIndexType.TEXT,
            tokenizer=TokenizerType.WORD,
            lowercase=True,
        ),
    )
    for field in ("source", "file_type", "language", "symbol_name", "symbol_type", "section_heading"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_index",
        field_schema=IntegerIndexParams(
            type=IntegerIndexType.INTEGER,
            lookup=True,
            range=True,
        ),
    )


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        _create_payload_indexes(client, collection_name)


def upsert_chunks(
    client: QdrantClient,
    embedded_chunks: list[EmbeddedChunk],
    collection_name: str = COLLECTION_NAME,
) -> int:
    if not embedded_chunks:
        return 0

    points = [
        PointStruct(
            id=_make_point_id(ec.chunk.source, ec.chunk.chunk_index),
            vector=ec.vector,
            payload=_chunk_to_payload(ec.chunk),
        )
        for ec in embedded_chunks
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def _make_point_id(source: str, chunk_index: int) -> int:
    key = f"{source}:{chunk_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _chunk_to_payload(chunk: Chunk) -> dict:
    return dataclasses.asdict(chunk)


def delete_source(
    client: QdrantClient,
    source: str,
    collection_name: str = COLLECTION_NAME,
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]
        ),
    )


def _make_manifest_point_id(source: str) -> int:
    key = f"manifest:{source}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def store_manifest(
    client: QdrantClient,
    manifest: dict[str, dict],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not manifest:
        return

    dummy_vector = [0.0] * VECTOR_SIZE
    dummy_vector[0] = 1.0

    points = [
        PointStruct(
            id=_make_manifest_point_id(source),
            vector=dummy_vector,
            payload={
                "source": source,
                "file_type": FILE_TYPE_MANIFEST,
                "content_hash": meta["content_hash"],
                "chunk_count": meta["chunk_count"],
            },
        )
        for source, meta in manifest.items()
    ]
    client.upsert(collection_name=collection_name, points=points)


def fetch_manifest(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> dict[str, dict]:
    manifest = {}
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_MANIFEST))]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = record.payload
            manifest[payload["source"]] = {
                "content_hash": payload["content_hash"],
                "chunk_count": payload["chunk_count"],
            }
        if next_offset is None:
            break
        offset = next_offset

    return manifest
