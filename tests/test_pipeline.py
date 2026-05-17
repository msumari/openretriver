from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.pipeline import ingest
from src.storage import COLLECTION_NAME, fetch_manifest


@pytest.fixture
def mini_project(tmp_path: Path) -> Path:
    (tmp_path / "readme.md").write_text("# Hello\n\nThis is a test project.")
    (tmp_path / "main.py").write_text("def greet():\n    return 'hi'\n")
    return tmp_path


@pytest.mark.slow
def test_ingest_end_to_end(mini_project):
    client = QdrantClient(location=":memory:")
    count = ingest(mini_project, client=client)
    assert count > 0


@pytest.mark.slow
def test_ingest_returns_chunk_count(mini_project):
    client = QdrantClient(location=":memory:")
    count = ingest(mini_project, client=client)
    assert isinstance(count, int)
    assert count >= 2


@pytest.mark.slow
def test_ingest_empty_project(tmp_path):
    client = QdrantClient(location=":memory:")
    count = ingest(tmp_path, client=client)
    assert count == 0


@pytest.mark.slow
def test_re_ingest_idempotent(mini_project):
    client = QdrantClient(location=":memory:")
    count1 = ingest(mini_project, client=client)
    assert count1 > 0
    count2 = ingest(mini_project, client=client)
    assert count2 == 0


@pytest.mark.slow
def test_ingest_stores_manifest(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client)
    manifest = fetch_manifest(client)
    assert "readme.md" in manifest
    assert "main.py" in manifest
    assert "content_hash" in manifest["readme.md"]
    assert "chunk_count" in manifest["readme.md"]


@pytest.mark.slow
def test_reingest_unchanged_skips_embedding(mini_project):
    client = QdrantClient(location=":memory:")
    count1 = ingest(mini_project, client=client)
    count2 = ingest(mini_project, client=client)
    total = client.count(COLLECTION_NAME).count
    manifest_size = len(fetch_manifest(client))
    assert total == count1 + manifest_size
    assert count2 == 0


@pytest.mark.slow
def test_reingest_after_file_change_updates_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client)
    count_before = client.count(COLLECTION_NAME).count

    (mini_project / "main.py").write_text("def greet():\n    return 'hello'\n\ndef farewell():\n    return 'bye'\n")
    count2 = ingest(mini_project, client=client)
    assert count2 > 0

    manifest = fetch_manifest(client)
    assert manifest["main.py"]["chunk_count"] > 0


@pytest.mark.slow
def test_reingest_after_file_delete_removes_stale_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client)

    (mini_project / "main.py").unlink()
    ingest(mini_project, client=client)

    manifest = fetch_manifest(client)
    assert "main.py" not in manifest

    points = client.scroll(COLLECTION_NAME, limit=100)[0]
    sources = {p.payload["source"] for p in points if p.payload.get("file_type") != "manifest"}
    assert "main.py" not in sources


@pytest.mark.slow
def test_reingest_after_rename_no_ghost_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client)

    (mini_project / "main.py").rename(mini_project / "app.py")
    ingest(mini_project, client=client)

    points = client.scroll(COLLECTION_NAME, limit=100)[0]
    sources = {p.payload["source"] for p in points if p.payload.get("file_type") != "manifest"}
    assert "main.py" not in sources
    assert "app.py" in sources
