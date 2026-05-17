import pytest
from qdrant_client import QdrantClient

from src.models import Chunk, EmbeddedChunk, FILE_TYPE_DOC, FILE_TYPE_CODE, FILE_TYPE_MANIFEST
from src.models import compute_file_hash
from src.storage import (
    ensure_collection,
    upsert_chunks,
    delete_source,
    store_manifest,
    fetch_manifest,
    _make_point_id,
    COLLECTION_NAME,
)


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


# --- Payload index tests ---


def test_ensure_collection_creates_payload_indexes(memory_client):
    ensure_collection(memory_client)
    ec = _make_embedded_chunk("hello world", 0)
    upsert_chunks(memory_client, [ec])
    points = memory_client.scroll(COLLECTION_NAME, limit=1)[0]
    payload = points[0].payload
    assert "text" in payload
    assert "source" in payload
    assert "file_type" in payload
    assert "language" in payload
    assert "symbol_name" in payload
    assert "symbol_type" in payload
    assert "section_heading" in payload
    assert "chunk_index" in payload


# --- Delete source tests ---


def test_delete_source_removes_all_chunks_for_file(memory_client):
    ensure_collection(memory_client)
    chunks_a = [_make_embedded_chunk(f"a chunk {i}", i, source="a.py") for i in range(3)]
    chunks_b = [_make_embedded_chunk(f"b chunk {i}", i, source="b.py") for i in range(2)]
    upsert_chunks(memory_client, chunks_a + chunks_b)
    assert memory_client.count(COLLECTION_NAME).count == 5

    delete_source(memory_client, "a.py")
    assert memory_client.count(COLLECTION_NAME).count == 2


def test_delete_source_noop_for_nonexistent(memory_client):
    ensure_collection(memory_client)
    chunks = [_make_embedded_chunk(f"chunk {i}", i, source="a.py") for i in range(2)]
    upsert_chunks(memory_client, chunks)

    delete_source(memory_client, "nonexistent.py")
    assert memory_client.count(COLLECTION_NAME).count == 2


def test_delete_source_then_upsert_replaces_cleanly(memory_client):
    ensure_collection(memory_client)
    old_chunks = [_make_embedded_chunk(f"old {i}", i, source="file.py") for i in range(5)]
    upsert_chunks(memory_client, old_chunks)
    assert memory_client.count(COLLECTION_NAME).count == 5

    delete_source(memory_client, "file.py")
    new_chunks = [_make_embedded_chunk(f"new {i}", i, source="file.py") for i in range(3)]
    upsert_chunks(memory_client, new_chunks)
    assert memory_client.count(COLLECTION_NAME).count == 3

    points = memory_client.scroll(COLLECTION_NAME, limit=10)[0]
    texts = {p.payload["text"] for p in points}
    assert texts == {"new 0", "new 1", "new 2"}


# --- Manifest tests ---


def test_compute_file_hash_deterministic():
    h1 = compute_file_hash("hello world")
    h2 = compute_file_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64


def test_compute_file_hash_differs_for_different_content():
    h1 = compute_file_hash("hello")
    h2 = compute_file_hash("world")
    assert h1 != h2


def test_store_and_fetch_manifest(memory_client):
    ensure_collection(memory_client)
    manifest = {
        "src/loader.py": {"content_hash": "abc123", "chunk_count": 5},
        "README.md": {"content_hash": "def456", "chunk_count": 2},
    }
    store_manifest(memory_client, manifest)

    fetched = fetch_manifest(memory_client)
    assert fetched == manifest


def test_store_manifest_overwrites_on_rerun(memory_client):
    ensure_collection(memory_client)
    manifest_v1 = {"file.py": {"content_hash": "aaa", "chunk_count": 3}}
    store_manifest(memory_client, manifest_v1)

    manifest_v2 = {"file.py": {"content_hash": "bbb", "chunk_count": 5}}
    store_manifest(memory_client, manifest_v2)

    fetched = fetch_manifest(memory_client)
    assert fetched["file.py"]["content_hash"] == "bbb"
    assert fetched["file.py"]["chunk_count"] == 5


def test_fetch_manifest_empty_collection(memory_client):
    ensure_collection(memory_client)
    fetched = fetch_manifest(memory_client)
    assert fetched == {}


def test_manifest_points_coexist_with_chunk_points(memory_client):
    ensure_collection(memory_client)
    chunks = [_make_embedded_chunk(f"chunk {i}", i, source="file.py") for i in range(3)]
    upsert_chunks(memory_client, chunks)

    manifest = {"file.py": {"content_hash": "abc", "chunk_count": 3}}
    store_manifest(memory_client, manifest)

    total = memory_client.count(COLLECTION_NAME).count
    assert total == 4

    fetched = fetch_manifest(memory_client)
    assert fetched == manifest


def test_delete_source_also_removes_manifest_point(memory_client):
    ensure_collection(memory_client)
    chunks = [_make_embedded_chunk(f"chunk {i}", i, source="file.py") for i in range(3)]
    upsert_chunks(memory_client, chunks)
    manifest = {"file.py": {"content_hash": "abc", "chunk_count": 3}}
    store_manifest(memory_client, manifest)
    assert memory_client.count(COLLECTION_NAME).count == 4

    delete_source(memory_client, "file.py")
    assert memory_client.count(COLLECTION_NAME).count == 0
    assert fetch_manifest(memory_client) == {}
