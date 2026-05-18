import pytest
from unittest.mock import patch
from qdrant_client import QdrantClient

from src.models import Chunk, EmbeddedChunk, FILE_TYPE_DOC, FILE_TYPE_CODE, FILE_TYPE_MANIFEST
from src.storage import ensure_collection, upsert_chunks, store_manifest, COLLECTION_NAME


DUMMY_VECTOR = [0.1] * 384


def _make_embedded(text, index, source, file_type=FILE_TYPE_DOC,
                   language=None, symbol_name=None, section_heading=None):
    chunk = Chunk(
        text=text,
        source=source,
        chunk_index=index,
        file_type=file_type,
        language=language,
        section_heading=section_heading,
        symbol_name=symbol_name,
        symbol_type=None,
    )
    return EmbeddedChunk(chunk=chunk, vector=DUMMY_VECTOR)


@pytest.fixture
def memory_client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def populated_client(memory_client):
    ensure_collection(memory_client)
    embedded = [
        _make_embedded("The loader reads files recursively from a directory", 0, "src/loader.py",
                       FILE_TYPE_CODE, "python", "load_files"),
        _make_embedded("SUPPORTED_EXTENSIONS contains .md .txt .py .js .ts .rs", 1, "src/models.py",
                       FILE_TYPE_CODE, "python", "SUPPORTED_EXTENSIONS"),
        _make_embedded("The chunker splits markdown by paragraphs", 2, "src/chunker_docs.py",
                       FILE_TYPE_CODE, "python", "chunk_doc"),
        _make_embedded("## Installation\n\nRun make install to set up the project", 3, "docs/guide.md",
                       FILE_TYPE_DOC, None, None, "## Installation"),
        _make_embedded("Tree-sitter parses source code into AST nodes", 4, "src/chunker_code.py",
                       FILE_TYPE_CODE, "python", "chunk_code"),
    ]
    upsert_chunks(memory_client, embedded)
    return memory_client


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_returns_results(mock_embed, populated_client):
    from src.search import search, SearchResult

    results = search("how does file loading work", client=populated_client, score_threshold=0.0)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_result_fields(mock_embed, populated_client):
    from src.search import search

    results = search("loader", client=populated_client, score_threshold=0.0)
    assert len(results) > 0
    r = results[0]
    assert r.text
    assert r.source
    assert r.score > 0
    assert r.file_type in (FILE_TYPE_DOC, FILE_TYPE_CODE)
    assert isinstance(r.chunk_index, int)


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_respects_top_k(mock_embed, populated_client):
    from src.search import search

    results = search("python", client=populated_client, top_k=2, score_threshold=0.0)
    assert len(results) <= 2


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_excludes_manifest_points(mock_embed, populated_client):
    from src.search import search

    store_manifest(populated_client, {"src/loader.py": {"content_hash": "abc", "chunk_count": 1}})
    results = search("loader", client=populated_client, top_k=10, score_threshold=0.0)
    for r in results:
        assert r.file_type != FILE_TYPE_MANIFEST


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_score_threshold_filters_low_scores(mock_embed, populated_client):
    from src.search import search

    results = search("anything", client=populated_client, score_threshold=999.0)
    assert results == []


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_results_ordered_by_score_descending(mock_embed, populated_client):
    from src.search import search

    results = search("file loading", client=populated_client, top_k=5, score_threshold=0.0)
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


@patch("src.search._embed_query", return_value=DUMMY_VECTOR)
def test_search_empty_collection(mock_embed, memory_client):
    from src.search import search

    ensure_collection(memory_client)
    results = search("anything", client=memory_client)
    assert results == []
