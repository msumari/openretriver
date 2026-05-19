import pytest
from unittest.mock import patch
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector as QdrantSparseVector

from src.models import Chunk, EmbeddedChunk, SparseVector, FILE_TYPE_DOC, FILE_TYPE_CODE
from src.storage import ensure_collection, upsert_chunks, COLLECTION_NAME


DUMMY_DENSE = [0.1] * 384


def _make_embedded(text, index, source, sparse_indices, sparse_values,
                   file_type=FILE_TYPE_DOC, language=None, symbol_name=None):
    chunk = Chunk(
        text=text,
        source=source,
        chunk_index=index,
        file_type=file_type,
        language=language,
        section_heading=None,
        symbol_name=symbol_name,
        symbol_type=None,
    )
    sparse = SparseVector(indices=sparse_indices, values=sparse_values)
    return EmbeddedChunk(chunk=chunk, vector=DUMMY_DENSE, sparse_vector=sparse)


@pytest.fixture
def memory_client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def populated_client(memory_client):
    ensure_collection(memory_client)
    embedded = [
        _make_embedded(
            "The loader reads files recursively from a directory",
            0, "src/loader.py",
            sparse_indices=[10, 20, 30], sparse_values=[1.0, 0.8, 0.5],
            file_type=FILE_TYPE_CODE, language="python", symbol_name="load_files",
        ),
        _make_embedded(
            "SUPPORTED_EXTENSIONS contains .md .txt .py .js .ts .rs",
            1, "src/models.py",
            sparse_indices=[40, 50, 60], sparse_values=[1.0, 0.7, 0.6],
            file_type=FILE_TYPE_CODE, language="python", symbol_name="SUPPORTED_EXTENSIONS",
        ),
        _make_embedded(
            "The chunker splits markdown by paragraphs",
            2, "src/chunker_docs.py",
            sparse_indices=[70, 80, 90], sparse_values=[1.0, 0.9, 0.4],
            file_type=FILE_TYPE_CODE, language="python", symbol_name="chunk_doc",
        ),
    ]
    upsert_chunks(memory_client, embedded)
    return memory_client


def test_collection_has_named_vectors(memory_client):
    ensure_collection(memory_client)
    info = memory_client.get_collection(COLLECTION_NAME)
    assert "dense" in info.config.params.vectors


def test_collection_has_sparse_vectors(memory_client):
    ensure_collection(memory_client)
    info = memory_client.get_collection(COLLECTION_NAME)
    assert info.config.params.sparse_vectors is not None
    assert "bm25" in info.config.params.sparse_vectors


def test_upsert_with_sparse_vectors(populated_client):
    count = populated_client.count(COLLECTION_NAME).count
    assert count == 3


@patch("src.search.embed_query", return_value=DUMMY_DENSE)
@patch("src.search._embed_query_sparse", return_value=QdrantSparseVector(indices=[10, 20], values=[1.0, 0.8]))
def test_search_uses_both_vectors(mock_sparse, mock_dense, populated_client):
    from src.search import search

    results = search("loader files", client=populated_client, score_threshold=0.0)
    assert len(results) > 0


@patch("src.search.embed_query", return_value=DUMMY_DENSE)
@patch("src.search._embed_query_sparse", return_value=QdrantSparseVector(indices=[10, 20], values=[1.0, 0.8]))
def test_sparse_match_boosts_score(mock_sparse, mock_dense, populated_client):
    from src.search import search

    results = search("loader", client=populated_client, score_threshold=0.0)
    if len(results) > 1:
        assert results[0].source == "src/loader.py"
