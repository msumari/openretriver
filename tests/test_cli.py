from unittest.mock import patch, MagicMock
import pytest


def test_main_no_args_shows_help(capsys):
    from src.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code in (0, 2)


def test_ingest_subcommand_calls_pipeline():
    from src.cli import main

    with patch("src.cli.ingest") as mock_ingest:
        mock_ingest.return_value = 5
        main(["ingest", "/tmp/x"])
        mock_ingest.assert_called_once_with("/tmp/x", collection_name=None)


def test_ingest_default_path_is_cwd():
    from src.cli import main

    with patch("src.cli.ingest") as mock_ingest:
        mock_ingest.return_value = 0
        main(["ingest"])
        mock_ingest.assert_called_once_with(".", collection_name=None)


def test_ingest_with_name():
    from src.cli import main

    with patch("src.cli.ingest") as mock_ingest:
        mock_ingest.return_value = 3
        main(["ingest", ".", "--name", "foo"])
        mock_ingest.assert_called_once_with(".", collection_name="foo")


def test_ask_subcommand_calls_rag():
    from src.cli import main

    mock_response = MagicMock(answer="test answer", sources=[], chunks_used=1)
    with patch("src.cli.ask") as mock_ask:
        mock_ask.return_value = mock_response
        main(["ask", "what is this?"])
        mock_ask.assert_called_once_with("what is this?", collection_name=None)


def test_ask_multi_word_question():
    from src.cli import main

    mock_response = MagicMock(answer="test", sources=[], chunks_used=1)
    with patch("src.cli.ask") as mock_ask:
        mock_ask.return_value = mock_response
        main(["ask", "how", "does", "auth", "work"])
        mock_ask.assert_called_once_with("how does auth work", collection_name=None)


def test_ask_with_name():
    from src.cli import main

    mock_response = MagicMock(answer="test", sources=[], chunks_used=1)
    with patch("src.cli.ask") as mock_ask:
        mock_ask.return_value = mock_response
        main(["ask", "question", "--name", "myrepo"])
        mock_ask.assert_called_once_with("question", collection_name="myrepo")


def test_chat_subcommand_calls_repl():
    from src.cli import main

    with patch("src.cli.repl") as mock_repl:
        main(["chat"])
        mock_repl.assert_called_once_with(collection_name=None)


def test_chat_with_name():
    from src.cli import main

    with patch("src.cli.repl") as mock_repl:
        main(["chat", "--name", "myrepo"])
        mock_repl.assert_called_once_with(collection_name="myrepo")


def test_ingest_qdrant_connection_error(capsys):
    from src.cli import main
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("src.cli.ingest") as mock_ingest:
        mock_ingest.side_effect = ResponseHandlingException(ConnectionError("[Errno 61] Connection refused"))
        with pytest.raises(SystemExit) as exc_info:
            main(["ingest", "."])
        assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Could not connect to Qdrant" in captured.err


def test_ask_qdrant_connection_error(capsys):
    from src.cli import main
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("src.cli.ask") as mock_ask:
        mock_ask.side_effect = ResponseHandlingException(ConnectionError("[Errno 61] Connection refused"))
        with pytest.raises(SystemExit) as exc_info:
            main(["ask", "hello"])
        assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Could not connect to Qdrant" in captured.err


def test_chat_qdrant_connection_error(capsys):
    from src.cli import main
    from qdrant_client.http.exceptions import ResponseHandlingException

    with patch("src.cli.repl") as mock_repl:
        mock_repl.side_effect = ResponseHandlingException(ConnectionError("[Errno 61] Connection refused"))
        with pytest.raises(SystemExit) as exc_info:
            main(["chat"])
        assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Could not connect to Qdrant" in captured.err
