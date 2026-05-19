import dataclasses
import hashlib
import os
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    IntegerIndexParams,
    IntegerIndexType,
    PayloadSchemaType,
    SparseVectorParams,
    SparseVector as QdrantSparseVector,
    Modifier,
)

from src.models import Chunk, EmbeddedChunk, SessionMessage, FILE_TYPE_MANIFEST, FILE_TYPE_SESSION

COLLECTION_NAME = "openretriver"
VECTOR_SIZE = 384
VECTOR_NAME_DENSE = "dense"
VECTOR_NAME_SPARSE = "bm25"


def _hash_to_point_id(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _dummy_vector() -> dict:
    v = [0.0] * VECTOR_SIZE
    v[0] = 1.0
    return {VECTOR_NAME_DENSE: v}


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
            vectors_config={VECTOR_NAME_DENSE: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)},
            sparse_vectors_config={VECTOR_NAME_SPARSE: SparseVectorParams(modifier=Modifier.IDF)},
        )
        _create_payload_indexes(client, collection_name)


def _build_vector(ec: EmbeddedChunk) -> dict:
    vectors = {VECTOR_NAME_DENSE: ec.vector}
    if ec.sparse_vector:
        vectors[VECTOR_NAME_SPARSE] = QdrantSparseVector(
            indices=ec.sparse_vector.indices,
            values=ec.sparse_vector.values,
        )
    return vectors


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
            vector=_build_vector(ec),
            payload=_chunk_to_payload(ec.chunk),
        )
        for ec in embedded_chunks
    ]
    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def _make_point_id(source: str, chunk_index: int) -> int:
    return _hash_to_point_id(f"{source}:{chunk_index}")


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
    return _hash_to_point_id(f"manifest:{source}")


def store_manifest(
    client: QdrantClient,
    manifest: dict[str, dict],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if not manifest:
        return

    points = [
        PointStruct(
            id=_make_manifest_point_id(source),
            vector=_dummy_vector(),
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


def _make_session_point_id(session_id: str, timestamp: int, role: str) -> int:
    return _hash_to_point_id(f"session:{session_id}:{timestamp}:{role}")


def ensure_session_indexes(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    client.create_payload_index(
        collection_name=collection_name,
        field_name="session_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="timestamp",
        field_schema=IntegerIndexParams(
            type=IntegerIndexType.INTEGER,
            lookup=True,
            range=True,
        ),
    )


def store_session_message(
    client: QdrantClient,
    messages: SessionMessage | list[SessionMessage],
    collection_name: str = COLLECTION_NAME,
) -> None:
    if isinstance(messages, SessionMessage):
        messages = [messages]
    if not messages:
        return
    points = [
        PointStruct(
            id=_make_session_point_id(msg.session_id, msg.timestamp, msg.role),
            vector=_dummy_vector(),
            payload={"file_type": FILE_TYPE_SESSION, **dataclasses.asdict(msg)},
        )
        for msg in messages
    ]
    client.upsert(collection_name=collection_name, points=points)


def fetch_session_history(
    client: QdrantClient,
    session_id: str,
    limit: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> list[SessionMessage]:
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_SESSION)),
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
            ]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    messages = [
        SessionMessage(
            session_id=r.payload["session_id"],
            timestamp=r.payload["timestamp"],
            role=r.payload["role"],
            content=r.payload["content"],
        )
        for r in records
    ]
    messages.sort(key=lambda m: m.timestamp)
    return messages[-limit:]


def cleanup_expired_sessions(
    client: QdrantClient,
    ttl_hours: int = 24,
    collection_name: str = COLLECTION_NAME,
) -> None:
    cutoff_ms = int((time.time() - ttl_hours * 3600) * 1000)
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_SESSION)),
                FieldCondition(key="timestamp", range=Range(lt=cutoff_ms)),
            ]
        ),
    )


def delete_session(
    client: QdrantClient,
    session_id: str,
    collection_name: str = COLLECTION_NAME,
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_SESSION)),
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
            ]
        ),
    )
