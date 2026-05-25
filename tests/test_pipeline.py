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
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)
    manifest = fetch_manifest(client)
    assert "readme.md" in manifest
    assert "main.py" in manifest
    assert "content_hash" in manifest["readme.md"]
    assert "chunk_count" in manifest["readme.md"]


@pytest.mark.slow
def test_reingest_unchanged_skips_embedding(mini_project):
    client = QdrantClient(location=":memory:")
    count1 = ingest(mini_project, client=client, collection_name=COLLECTION_NAME)
    count2 = ingest(mini_project, client=client, collection_name=COLLECTION_NAME)
    total = client.count(COLLECTION_NAME).count
    manifest_size = len(fetch_manifest(client))
    assert total == count1 + manifest_size
    assert count2 == 0


@pytest.mark.slow
def test_reingest_after_file_change_updates_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)

    (mini_project / "main.py").write_text("def greet():\n    return 'hello'\n\ndef farewell():\n    return 'bye'\n")
    count2 = ingest(mini_project, client=client, collection_name=COLLECTION_NAME)
    assert count2 > 0

    manifest = fetch_manifest(client)
    assert manifest["main.py"]["chunk_count"] > 0


@pytest.mark.slow
def test_reingest_after_file_delete_removes_stale_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)

    (mini_project / "main.py").unlink()
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)

    manifest = fetch_manifest(client)
    assert "main.py" not in manifest

    points = client.scroll(COLLECTION_NAME, limit=100)[0]
    sources = {p.payload["source"] for p in points if p.payload.get("file_type") != "manifest"}
    assert "main.py" not in sources


@pytest.mark.slow
def test_reingest_after_rename_no_ghost_chunks(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)

    (mini_project / "main.py").rename(mini_project / "app.py")
    ingest(mini_project, client=client, collection_name=COLLECTION_NAME)

    points = client.scroll(COLLECTION_NAME, limit=100)[0]
    sources = {p.payload["source"] for p in points if p.payload.get("file_type") != "manifest"}
    assert "main.py" not in sources
    assert "app.py" in sources


@pytest.mark.slow
def test_ingest_uses_custom_collection_name(mini_project):
    client = QdrantClient(location=":memory:")
    count = ingest(mini_project, client=client, collection_name="my-repo")
    assert count > 0
    assert client.collection_exists("my-repo")


@pytest.mark.slow
def test_ingest_derives_name_from_path(mini_project):
    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client)
    expected_name = mini_project.resolve().name
    assert client.collection_exists(expected_name)


@pytest.mark.slow
def test_search_isolated_between_collections(mini_project, tmp_path):
    project_b = tmp_path / "project-b"
    project_b.mkdir()
    (project_b / "other.md").write_text("# Unrelated\n\nThis is about quantum physics.")

    client = QdrantClient(location=":memory:")
    ingest(mini_project, client=client, collection_name="repo-a")
    ingest(project_b, client=client, collection_name="repo-b")

    manifest_a = fetch_manifest(client, "repo-a")
    manifest_b = fetch_manifest(client, "repo-b")

    assert "readme.md" in manifest_a
    assert "other.md" not in manifest_a
    assert "other.md" in manifest_b
    assert "readme.md" not in manifest_b
