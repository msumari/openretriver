import pytest
from qdrant_client import QdrantClient

from src.models import Chunk, EmbeddedChunk
from src.storage import ensure_collection, upsert_chunks, _make_point_id, COLLECTION_NAME


@pytest.fixture
def memory_client():
    return QdrantClient(location=":memory:")


def _make_embedded_chunk(text: str, index: int, source: str = "test.md") -> EmbeddedChunk:
    chunk = Chunk(
        text=text,
        source=source,
        chunk_index=index,
        file_type="doc",
        language=None,
        section_heading=None,
        symbol_name=None,
        symbol_type=None,
    )
    return EmbeddedChunk(chunk=chunk, vector=[0.1] * 384)


def test_ensure_collection_creates(memory_client):
    ensure_collection(memory_client)
    info = memory_client.get_collection(COLLECTION_NAME)
    assert info is not None


def test_ensure_collection_idempotent(memory_client):
    ensure_collection(memory_client)
    ensure_collection(memory_client)
    info = memory_client.get_collection(COLLECTION_NAME)
    assert info is not None


def test_upsert_and_count(memory_client):
    ensure_collection(memory_client)
    embedded = [_make_embedded_chunk(f"chunk {i}", i) for i in range(5)]
    upsert_chunks(memory_client, embedded)
    count = memory_client.count(COLLECTION_NAME).count
    assert count == 5


def test_upsert_idempotent(memory_client):
    ensure_collection(memory_client)
    embedded = [_make_embedded_chunk(f"chunk {i}", i) for i in range(5)]
    upsert_chunks(memory_client, embedded)
    upsert_chunks(memory_client, embedded)
    count = memory_client.count(COLLECTION_NAME).count
    assert count == 5


def test_payload_fields_present(memory_client):
    ensure_collection(memory_client)
    ec = _make_embedded_chunk("hello world", 0)
    upsert_chunks(memory_client, [ec])

    points = memory_client.scroll(COLLECTION_NAME, limit=1)[0]
    payload = points[0].payload
    assert payload["source"] == "test.md"
    assert payload["file_type"] == "doc"
    assert payload["language"] is None
    assert payload["chunk_index"] == 0
    assert payload["text"] == "hello world"
    assert payload["section_heading"] is None
    assert payload["symbol_name"] is None
    assert payload["symbol_type"] is None


def test_point_id_deterministic():
    id1 = _make_point_id("test.md", 0)
    id2 = _make_point_id("test.md", 0)
    assert id1 == id2


def test_point_id_different_for_different_inputs():
    id1 = _make_point_id("a.md", 0)
    id2 = _make_point_id("b.md", 0)
    id3 = _make_point_id("a.md", 1)
    assert id1 != id2
    assert id1 != id3


def test_search_returns_results(memory_client):
    ensure_collection(memory_client)
    embedded = [_make_embedded_chunk(f"chunk {i}", i) for i in range(3)]
    upsert_chunks(memory_client, embedded)

    results = memory_client.query_points(
        collection_name=COLLECTION_NAME,
        query=[0.1] * 384,
        limit=2,
    )
    assert len(results.points) == 2
