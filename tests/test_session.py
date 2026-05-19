import time
import pytest
from qdrant_client import QdrantClient

from src.models import FILE_TYPE_SESSION, FILE_TYPE_DOC, SessionMessage
from src.storage import (
    ensure_collection,
    upsert_chunks,
    store_session_message,
    fetch_session_history,
    cleanup_expired_sessions,
    delete_session,
    ensure_session_indexes,
    COLLECTION_NAME,
)
from src.models import Chunk, EmbeddedChunk


@pytest.fixture
def memory_client():
    return QdrantClient(location=":memory:")


@pytest.fixture
def ready_client(memory_client):
    ensure_collection(memory_client)
    ensure_session_indexes(memory_client)
    return memory_client


def _msg(session_id, role, content, timestamp_ms=None):
    return SessionMessage(
        session_id=session_id,
        timestamp=timestamp_ms or int(time.time() * 1000),
        role=role,
        content=content,
    )


def test_store_and_fetch_session_message(ready_client):
    msg = _msg("sess-1", "user", "How does loading work?")
    store_session_message(ready_client, msg)

    history = fetch_session_history(ready_client, "sess-1")
    assert len(history) == 1
    assert history[0].content == "How does loading work?"
    assert history[0].role == "user"
    assert history[0].session_id == "sess-1"


def test_fetch_returns_chronological_order(ready_client):
    now = int(time.time() * 1000)
    store_session_message(ready_client, _msg("sess-1", "user", "first", now))
    store_session_message(ready_client, _msg("sess-1", "assistant", "second", now + 100))
    store_session_message(ready_client, _msg("sess-1", "user", "third", now + 200))

    history = fetch_session_history(ready_client, "sess-1")
    assert len(history) == 3
    assert history[0].content == "first"
    assert history[1].content == "second"
    assert history[2].content == "third"


def test_fetch_respects_limit(ready_client):
    now = int(time.time() * 1000)
    for i in range(20):
        store_session_message(ready_client, _msg("sess-1", "user", f"msg-{i}", now + i))

    history = fetch_session_history(ready_client, "sess-1", limit=10)
    assert len(history) == 10
    assert history[-1].content == "msg-19"


def test_session_isolation(ready_client):
    now = int(time.time() * 1000)
    store_session_message(ready_client, _msg("sess-A", "user", "question A", now))
    store_session_message(ready_client, _msg("sess-B", "user", "question B", now + 1))

    history_a = fetch_session_history(ready_client, "sess-A")
    history_b = fetch_session_history(ready_client, "sess-B")

    assert len(history_a) == 1
    assert history_a[0].content == "question A"
    assert len(history_b) == 1
    assert history_b[0].content == "question B"


def test_cleanup_expired_sessions(ready_client):
    now = int(time.time() * 1000)
    old_ts = now - (25 * 3600 * 1000)
    recent_ts = now - (1 * 3600 * 1000)

    store_session_message(ready_client, _msg("old-sess", "user", "old message", old_ts))
    store_session_message(ready_client, _msg("new-sess", "user", "recent message", recent_ts))

    cleanup_expired_sessions(ready_client, ttl_hours=24)

    old_history = fetch_session_history(ready_client, "old-sess")
    new_history = fetch_session_history(ready_client, "new-sess")

    assert len(old_history) == 0
    assert len(new_history) == 1


def test_cleanup_does_not_affect_chunks(ready_client):
    chunk = Chunk(
        text="some code", source="file.py", chunk_index=0,
        file_type=FILE_TYPE_DOC, language="python",
        section_heading=None, symbol_name=None, symbol_type=None,
    )
    embedded = EmbeddedChunk(chunk=chunk, vector=[0.1] * 384)
    upsert_chunks(ready_client, [embedded])

    now = int(time.time() * 1000)
    old_ts = now - (25 * 3600 * 1000)
    store_session_message(ready_client, _msg("old-sess", "user", "old", old_ts))

    cleanup_expired_sessions(ready_client, ttl_hours=24)

    count = ready_client.count(COLLECTION_NAME).count
    assert count == 1


def test_delete_session(ready_client):
    now = int(time.time() * 1000)
    store_session_message(ready_client, _msg("sess-1", "user", "q1", now))
    store_session_message(ready_client, _msg("sess-1", "assistant", "a1", now + 1))
    store_session_message(ready_client, _msg("sess-2", "user", "q2", now + 2))

    delete_session(ready_client, "sess-1")

    assert len(fetch_session_history(ready_client, "sess-1")) == 0
    assert len(fetch_session_history(ready_client, "sess-2")) == 1
