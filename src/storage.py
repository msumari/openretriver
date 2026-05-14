import hashlib
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.models import EmbeddedChunk

COLLECTION_NAME = "openretriver"
VECTOR_SIZE = 384


def get_client(
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = api_key or os.environ.get("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


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


def _chunk_to_payload(chunk) -> dict:
    return {
        "source": chunk.source,
        "file_type": chunk.file_type,
        "language": chunk.language,
        "section_heading": chunk.section_heading,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "symbol_name": chunk.symbol_name,
        "symbol_type": chunk.symbol_type,
    }
