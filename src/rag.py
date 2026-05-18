import logging
import re
import sys
from dataclasses import dataclass

from qdrant_client import QdrantClient

from src.search import search, SearchResult
from src.model_provider import get_provider, GenerateFn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior software engineer that answers questions using ONLY the provided context.

Rules:
1. Answer based solely on the context below. Do not use prior knowledge.
2. Cite sources using [source: filename] format after each claim.
3. If the context does not contain enough information to answer the question, say "I don't have enough information in the indexed documents to answer this question."
4. If the answer spans multiple sources, cite each one where relevant.
5. For code-related answers, include relevant code snippets from the context."""


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[str]
    chunks_used: int


def format_context(results: list[SearchResult]) -> str:
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results, 1):
        header_parts = [f"Source: {r.source}", f"Type: {r.file_type}"]
        if r.language:
            header_parts.append(f"Language: {r.language}")
        if r.symbol_name:
            header_parts.append(f"Symbol: {r.symbol_name}")
        if r.section_heading:
            header_parts.append(f"Section: {r.section_heading}")

        header = f"[{i}] " + " | ".join(header_parts)
        blocks.append(f"{header}\n{r.text}")

    return "\n\n".join(blocks)


def build_user_message(question: str, results: list[SearchResult]) -> str:
    context = format_context(results)
    if not context:
        return f"--- Context ---\n\nNo relevant documents found.\n\n--- Question ---\n{question}"
    return f"--- Context ---\n\n{context}\n\n--- Question ---\n{question}"


def _extract_sources(answer: str) -> list[str]:
    pattern = r"\[source:\s*([^\]]+)\]"
    return list(dict.fromkeys(re.findall(pattern, answer)))


def ask(
    question: str,
    client: QdrantClient | None = None,
    top_k: int = 5,
    score_threshold: float = 0.6,
    generate: GenerateFn | None = None,
) -> RAGResponse:
    results = search(
        question, client=client, top_k=top_k, score_threshold=score_threshold
    )
    logger.info("Retrieved %d chunks for question", len(results))
    for r in results:
        logger.info("  %.4f  %s", r.score, r.source)

    if not results:
        return RAGResponse(
            answer="I don't have enough information in the indexed documents to answer this question.",
            sources=[],
            chunks_used=0,
        )

    if generate is None:
        generate = get_provider()

    user_message = build_user_message(question, results)
    answer = generate(SYSTEM_PROMPT, user_message)

    sources = _extract_sources(answer)
    return RAGResponse(answer=answer, sources=sources, chunks_used=len(results))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag 'your question here'")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    response = ask(question)

    print(f"\nAnswer:\n{response.answer}")
    if response.sources:
        print(f"\nSources used:")
        for source in response.sources:
            print(f"  - {source}")
