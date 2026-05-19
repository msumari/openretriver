import logging
from functools import lru_cache

from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25

from src.models import Chunk, EmbeddedChunk, SparseVector

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    logger.info("Loading embedding model (first call downloads ~90MB)...")
    model = TextEmbedding()
    logger.info("Embedding model ready")
    return model


@lru_cache(maxsize=1)
def _get_sparse_model() -> Bm25:
    logger.info("Loading BM25 sparse model...")
    model = Bm25("Qdrant/bm25")
    logger.info("BM25 model ready")
    return model


def embed_query(text: str) -> list[float]:
    model = _get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def embed_query_sparse(text: str) -> SparseVector:
    model = _get_sparse_model()
    embeddings = list(model.embed([text]))
    e = embeddings[0]
    return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    dense_model = _get_model()
    sparse_model = _get_sparse_model()

    texts = [c.text for c in chunks]
    dense_embeddings = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))

    return [
        EmbeddedChunk(
            chunk=chunk,
            vector=dense.tolist(),
            sparse_vector=SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
        )
        for chunk, dense, sparse in zip(chunks, dense_embeddings, sparse_embeddings)
    ]
