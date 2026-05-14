import logging
from functools import lru_cache

from fastembed import TextEmbedding

from src.models import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_model() -> TextEmbedding:
    logger.info("Loading embedding model (first call downloads ~90MB)...")
    model = TextEmbedding()
    logger.info("Embedding model ready")
    return model


def embed_chunks(chunks: list[Chunk], model: TextEmbedding | None = None) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    model = model or _get_model()
    texts = [c.text for c in chunks]
    embeddings = list(model.embed(texts))

    return [
        EmbeddedChunk(chunk=chunk, vector=[float(x) for x in embedding])
        for chunk, embedding in zip(chunks, embeddings)
    ]
