import pytest

from src.models import Chunk
from src.embedder import embed_chunks


def _make_chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        source="test.md",
        chunk_index=index,
        file_type="doc",
        language=None,
        section_heading=None,
        symbol_name=None,
        symbol_type=None,
    )


@pytest.mark.slow
def test_embed_returns_correct_count():
    chunks = [_make_chunk("Hello world"), _make_chunk("Goodbye world", 1), _make_chunk("Test", 2)]
    results = embed_chunks(chunks)
    assert len(results) == 3


@pytest.mark.slow
def test_embedding_dimension():
    chunks = [_make_chunk("Some text to embed")]
    results = embed_chunks(chunks)
    assert len(results[0].vector) == 384


@pytest.mark.slow
def test_embedding_is_list_of_floats():
    chunks = [_make_chunk("Testing float types")]
    results = embed_chunks(chunks)
    assert all(isinstance(v, float) for v in results[0].vector)


@pytest.mark.slow
def test_chunk_preserved():
    chunk = _make_chunk("Preserve me")
    results = embed_chunks([chunk])
    assert results[0].chunk is chunk


@pytest.mark.slow
def test_empty_input():
    results = embed_chunks([])
    assert results == []


@pytest.mark.slow
def test_ordering_preserved():
    chunks = [_make_chunk(f"Chunk number {i}", i) for i in range(5)]
    results = embed_chunks(chunks)
    for i, ec in enumerate(results):
        assert ec.chunk.chunk_index == i
