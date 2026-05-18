import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    Fusion,
    MatchText,
    MatchValue,
    Prefetch,
)

from src.embedder import embed_query
from src.models import FILE_TYPE_MANIFEST
from src.storage import get_client, COLLECTION_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    score: float
    file_type: str
    language: str | None
    symbol_name: str | None
    section_heading: str | None
    chunk_index: int


def _embed_query(query: str) -> list[float]:
    return embed_query(query)


def _manifest_filter() -> Filter:
    return Filter(
        must_not=[FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_MANIFEST))]
    )


def search(
    query: str,
    client: QdrantClient | None = None,
    top_k: int = 5,
    score_threshold: float = 0.6,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchResult]:
    if client is None:
        client = get_client()

    query_vector = _embed_query(query)
    exclude_manifest = _manifest_filter()

    vector_prefetch = Prefetch(
        query=query_vector,
        filter=exclude_manifest,
        limit=top_k * 2,
    )

    text_prefetch = Prefetch(
        query=query_vector,
        filter=Filter(
            must=[FieldCondition(key="text", match=MatchText(text=query))],
            must_not=[FieldCondition(key="file_type", match=MatchValue(value=FILE_TYPE_MANIFEST))],
        ),
        limit=top_k * 2,
    )

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[vector_prefetch, text_prefetch],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        if point.score < score_threshold:
            continue
        payload = point.payload
        results.append(SearchResult(
            text=payload["text"],
            source=payload["source"],
            score=point.score,
            file_type=payload["file_type"],
            language=payload.get("language"),
            symbol_name=payload.get("symbol_name"),
            section_heading=payload.get("section_heading"),
            chunk_index=payload.get("chunk_index", 0),
        ))

    return results
