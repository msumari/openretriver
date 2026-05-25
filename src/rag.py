import logging
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient

from src.search import search, SearchResult
from src.models import SessionMessage
from src.model_provider import get_provider, GenerateFn
from src.storage import COLLECTION_NAME

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


def format_history(messages: list[SessionMessage]) -> str:
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role_label}: {msg.content}")
    return "\n".join(lines)


def build_user_message(
    question: str,
    results: list[SearchResult],
    history: list[SessionMessage] | None = None,
) -> str:
    context = format_context(results)
    history_text = format_history(history or [])

    parts = []
    parts.append("--- Context ---")
    parts.append(context if context else "No relevant documents found.")
    if history_text:
        parts.append("--- Conversation History ---")
        parts.append(history_text)
    parts.append("--- Question ---")
    parts.append(question)

    return "\n\n".join(parts)


def _extract_sources(answer: str) -> list[str]:
    pattern = r"\[source:\s*([^\]]+)\]"
    return list(dict.fromkeys(re.findall(pattern, answer)))


def ask(
    question: str,
    client: QdrantClient | None = None,
    top_k: int = 5,
    score_threshold: float = 0.5,
    generate: GenerateFn | None = None,
    history: list[SessionMessage] | None = None,
    collection_name: str | None = None,
) -> RAGResponse:
    collection_name = collection_name or COLLECTION_NAME

    try:
        results = search(
            question, client=client, top_k=top_k,
            score_threshold=score_threshold, collection_name=collection_name,
        )
    except Exception as e:
        logger.error("Search failed: %s", e)
        return RAGResponse(
            answer="Search is currently unavailable. Please check that Qdrant is running and accessible.",
            sources=[],
            chunks_used=0,
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

    user_message = build_user_message(question, results, history=history)

    try:
        answer = generate(SYSTEM_PROMPT, user_message)
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        sources = [r.source for r in results]
        return RAGResponse(
            answer="LLM is currently unavailable. Retrieved sources: " + ", ".join(sources),
            sources=sources,
            chunks_used=len(results),
        )

    sources = _extract_sources(answer)
    return RAGResponse(answer=answer, sources=sources, chunks_used=len(results))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    question = " ".join(args.question)
    response = ask(question, collection_name=args.name)

    print(f"\nAnswer:\n{response.answer}")
    if response.sources:
        print(f"\nSources used:")
        for source in response.sources:
            print(f"  - {source}")
