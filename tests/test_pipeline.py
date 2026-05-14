from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.pipeline import ingest
from src.storage import COLLECTION_NAME


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
    count2 = ingest(mini_project, client=client)
    assert count1 == count2
