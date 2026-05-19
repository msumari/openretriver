import logging
import time
import uuid

from src.models import SessionMessage
from src.rag import ask
from src.storage import (
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


def record_exchange(client, session_id: str, question: str, answer: str) -> None:
    now_ms = int(time.time() * 1000)
    user_msg = SessionMessage(session_id=session_id, timestamp=now_ms, role="user", content=question)
    assistant_msg = SessionMessage(session_id=session_id, timestamp=now_ms + 1, role="assistant", content=answer)
    store_session_message(client, [user_msg, assistant_msg])


def get_history(client, session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> list[SessionMessage]:
    return fetch_session_history(client, session_id, limit=limit)


def repl(client=None) -> None:
    if client is None:
        client = get_client()

    ensure_collection(client)
    ensure_session_indexes(client)
    cleanup_expired_sessions(client, ttl_hours=SESSION_TTL_HOURS)

    session_id = generate_session_id()
    generate = get_provider()
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
        )

        print(f"\n{response.answer}")
        if response.sources:
            print(f"\nSources: {', '.join(response.sources)}")
        print()

        now_ms = int(time.time() * 1000)
        history.append(SessionMessage(session_id=session_id, timestamp=now_ms, role="user", content=question))
        history.append(SessionMessage(session_id=session_id, timestamp=now_ms + 1, role="assistant", content=response.answer))
        record_exchange(client, session_id, question, response.answer)

    print("Session ended.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repl()
