import pytest
from unittest.mock import patch

from src.models import FILE_TYPE_DOC, FILE_TYPE_CODE
from src.search import SearchResult


def _make_result(text, source, score=0.8, file_type=FILE_TYPE_CODE,
                 language="python", symbol_name=None, section_heading=None):
    return SearchResult(
        text=text,
        source=source,
        score=score,
        file_type=file_type,
        language=language,
        symbol_name=symbol_name,
        section_heading=section_heading,
        chunk_index=0,
    )


# --- Context formatting tests ---


def test_format_context_includes_source_and_text():
    from src.rag import format_context

    results = [_make_result("def load_files(): pass", "src/loader.py", symbol_name="load_files")]
    context = format_context(results)
    assert "src/loader.py" in context
    assert "def load_files" in context


def test_format_context_numbers_chunks():
    from src.rag import format_context

    results = [
        _make_result("chunk one", "a.py", symbol_name="foo"),
        _make_result("chunk two", "b.py", symbol_name="bar"),
    ]
    context = format_context(results)
    assert "[1]" in context
    assert "[2]" in context


def test_format_context_includes_metadata():
    from src.rag import format_context

    results = [
        _make_result("install instructions", "docs/guide.md", file_type=FILE_TYPE_DOC,
                     language=None, section_heading="## Installation"),
    ]
    context = format_context(results)
    assert "## Installation" in context
    assert "doc" in context


def test_format_context_empty_results():
    from src.rag import format_context

    assert format_context([]) == ""


# --- User message building ---


def test_build_user_message_contains_context_and_question():
    from src.rag import build_user_message

    results = [_make_result("some text", "file.py", symbol_name="func")]
    message = build_user_message("What does func do?", results)
    assert "some text" in message
    assert "What does func do?" in message


def test_build_user_message_with_no_results():
    from src.rag import build_user_message

    message = build_user_message("question?", [])
    assert "question?" in message


# --- System prompt ---


def test_system_prompt_enforces_grounding():
    from src.rag import SYSTEM_PROMPT

    lower = SYSTEM_PROMPT.lower()
    assert "only" in lower
    assert "source" in lower
    assert "don't have enough information" in lower or "do not" in lower


# --- ask() integration with mocked provider ---


@patch("src.rag.search")
@patch("src.rag.get_provider")
def test_ask_returns_rag_response(mock_get_provider, mock_search):
    from src.rag import ask, RAGResponse

    mock_search.return_value = [
        _make_result("The loader walks directories", "src/loader.py", symbol_name="load_files"),
    ]
    mock_get_provider.return_value = lambda system, msg: "The loader walks directories. [source: src/loader.py]"

    response = ask("How does loading work?")
    assert isinstance(response, RAGResponse)
    assert "loader" in response.answer
    assert "src/loader.py" in response.sources
    assert response.chunks_used == 1


@patch("src.rag.search")
@patch("src.rag.get_provider")
def test_ask_with_no_results(mock_get_provider, mock_search):
    from src.rag import ask

    mock_search.return_value = []
    mock_get_provider.return_value = lambda system, msg: "I don't have enough information"

    response = ask("What is quantum computing?")
    assert response.chunks_used == 0
    assert "don't have enough information" in response.answer


@patch("src.rag.search")
@patch("src.rag.get_provider")
def test_ask_extracts_multiple_sources(mock_get_provider, mock_search):
    from src.rag import ask

    mock_search.return_value = [
        _make_result("change detection logic", "src/change_detection.py", symbol_name="detect_changes"),
        _make_result("pipeline orchestration", "src/pipeline.py", symbol_name="ingest"),
    ]
    mock_get_provider.return_value = lambda system, msg: (
        "Changes are detected [source: src/change_detection.py] "
        "then the pipeline runs [source: src/pipeline.py]"
    )

    response = ask("How does change detection work?")
    assert "src/change_detection.py" in response.sources
    assert "src/pipeline.py" in response.sources
    assert response.chunks_used == 2


# --- History formatting tests ---


def test_format_history_empty():
    from src.rag import format_history
    from src.models import SessionMessage

    assert format_history([]) == ""


def test_format_history_formats_turns():
    from src.rag import format_history
    from src.models import SessionMessage

    messages = [
        SessionMessage(session_id="s1", timestamp=1000, role="user", content="What is X?"),
        SessionMessage(session_id="s1", timestamp=2000, role="assistant", content="X is a thing."),
    ]
    result = format_history(messages)
    assert "User: What is X?" in result
    assert "Assistant: X is a thing." in result


def test_build_user_message_with_history():
    from src.rag import build_user_message
    from src.models import SessionMessage

    results = [_make_result("some context", "file.py", symbol_name="func")]
    history = [
        SessionMessage(session_id="s1", timestamp=1000, role="user", content="prev question"),
        SessionMessage(session_id="s1", timestamp=2000, role="assistant", content="prev answer"),
    ]
    message = build_user_message("follow-up?", results, history=history)
    assert "prev question" in message
    assert "prev answer" in message
    assert "follow-up?" in message
    assert "some context" in message
