import logging
import time
import uuid

from qdrant_client import QdrantClient

from src.models import SessionMessage
from src.rag import ask
from src.storage import (
    COLLECTION_NAME,
    get_client,
    ensure_collection,
    store_session_message,
    fetch_session_history,
    cleanup_expired_sessions,
    ensure_session_indexes,
)
from src.model_provider import get_provider

logger = logging.getLogger(__name__)

SESSION_TTL_HOURS = 24
MAX_HISTORY_MESSAGES = 10


def generate_session_id() -> str:
    return uuid.uuid4().hex


def record_exchange(
    client: QdrantClient, session_id: str, question: str, answer: str,
    collection_name: str = COLLECTION_NAME,
) -> None:
    now_ms = int(time.time() * 1000)
    user_msg = SessionMessage(session_id=session_id, timestamp=now_ms, role="user", content=question)
    assistant_msg = SessionMessage(session_id=session_id, timestamp=now_ms + 1, role="assistant", content=answer)
    store_session_message(client, [user_msg, assistant_msg], collection_name=collection_name)


def get_history(
    client: QdrantClient, session_id: str, limit: int = MAX_HISTORY_MESSAGES,
    collection_name: str = COLLECTION_NAME,
) -> list[SessionMessage]:
    return fetch_session_history(client, session_id, limit=limit, collection_name=collection_name)


def repl(client=None, collection_name: str | None = None) -> None:
    if client is None:
        try:
            client = get_client()
        except Exception as e:
            logger.error("Failed to connect to Qdrant: %s", e)
            print("Error: Cannot connect to Qdrant. Please check that it is running.")
            return

    collection_name = collection_name or COLLECTION_NAME

    try:
        ensure_collection(client, collection_name)
        ensure_session_indexes(client, collection_name)
        cleanup_expired_sessions(client, ttl_hours=SESSION_TTL_HOURS, collection_name=collection_name)
    except Exception as e:
        logger.warning("Session setup failed: %s. Continuing without session persistence.", e)

    session_id = generate_session_id()

    try:
        generate = get_provider()
    except Exception as e:
        logger.error("Failed to initialize LLM provider: %s", e)
        print("Error: Cannot initialize LLM provider. Check your credentials and configuration.")
        return

    history: list[SessionMessage] = []

    print(f"Session started: {session_id[:8]}...")
    print("Type 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        recent_history = history[-MAX_HISTORY_MESSAGES:]
        response = ask(
            question,
            client=client,
            generate=generate,
            history=recent_history,
            collection_name=collection_name,
        )

        print(f"\n{response.answer}")
        if response.sources:
            print(f"\nSources: {', '.join(response.sources)}")
        print()

        now_ms = int(time.time() * 1000)
        history.append(SessionMessage(session_id=session_id, timestamp=now_ms, role="user", content=question))
        history.append(SessionMessage(session_id=session_id, timestamp=now_ms + 1, role="assistant", content=response.answer))

        try:
            record_exchange(client, session_id, question, response.answer, collection_name=collection_name)
        except Exception as e:
            logger.warning("Failed to persist session message: %s", e)

    print("Session ended.")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    repl(collection_name=args.name)
