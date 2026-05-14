import dataclasses

import pytest

from src.models import (
    SUPPORTED_EXTENSIONS,
    DOC_EXTENSIONS,
    CODE_EXTENSIONS,
    IGNORE_DIRS,
    EXTENSION_TO_LANGUAGE,
    LoadedFile,
    Chunk,
    EmbeddedChunk,
)


class TestLoadedFile:
    def test_creation(self):
        f = LoadedFile(path="src/main.py", extension=".py", content="print('hi')")
        assert f.path == "src/main.py"
        assert f.extension == ".py"
        assert f.content == "print('hi')"

    def test_frozen(self):
        f = LoadedFile(path="src/main.py", extension=".py", content="")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.path = "other.py"


class TestChunk:
    def test_creation_doc(self):
        c = Chunk(
            text="Some documentation text.",
            source="docs/setup.md",
            chunk_index=0,
            file_type="doc",
            language=None,
            section_heading="## Setup",
            symbol_name=None,
            symbol_type=None,
        )
        assert c.file_type == "doc"
        assert c.language is None
        assert c.section_heading == "## Setup"
        assert c.symbol_name is None
        assert c.symbol_type is None

    def test_creation_code(self):
        c = Chunk(
            text="def parse(): ...",
            source="src/parser.py",
            chunk_index=2,
            file_type="code",
            language="python",
            section_heading=None,
            symbol_name="parse",
            symbol_type="function",
        )
        assert c.file_type == "code"
        assert c.language == "python"
        assert c.symbol_name == "parse"
        assert c.symbol_type == "function"

    def test_frozen(self):
        c = Chunk(
            text="x",
            source="a.py",
            chunk_index=0,
            file_type="code",
            language="python",
            section_heading=None,
            symbol_name=None,
            symbol_type=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.text = "y"


class TestEmbeddedChunk:
    def test_creation(self):
        c = Chunk(
            text="hello",
            source="a.md",
            chunk_index=0,
            file_type="doc",
            language=None,
            section_heading=None,
            symbol_name=None,
            symbol_type=None,
        )
        vec = [0.1] * 384
        ec = EmbeddedChunk(chunk=c, vector=vec)
        assert ec.chunk is c
        assert len(ec.vector) == 384

    def test_frozen(self):
        c = Chunk(
            text="hello",
            source="a.md",
            chunk_index=0,
            file_type="doc",
            language=None,
            section_heading=None,
            symbol_name=None,
            symbol_type=None,
        )
        ec = EmbeddedChunk(chunk=c, vector=[0.0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            ec.vector = [1.0]


class TestConstants:
    def test_supported_extensions(self):
        assert SUPPORTED_EXTENSIONS == {".md", ".txt", ".py", ".js", ".ts", ".rs"}

    def test_doc_extensions(self):
        assert DOC_EXTENSIONS == {".md", ".txt"}

    def test_code_extensions(self):
        assert CODE_EXTENSIONS == {".py", ".js", ".ts", ".rs"}

    def test_ignore_dirs(self):
        for d in [".git", "node_modules", "__pycache__", "target", "venv", "dist"]:
            assert d in IGNORE_DIRS

    def test_extension_to_language(self):
        assert EXTENSION_TO_LANGUAGE[".py"] == "python"
        assert EXTENSION_TO_LANGUAGE[".js"] == "javascript"
        assert EXTENSION_TO_LANGUAGE[".ts"] == "typescript"
        assert EXTENSION_TO_LANGUAGE[".rs"] == "rust"

    def test_doc_and_code_union_equals_supported(self):
        assert DOC_EXTENSIONS | CODE_EXTENSIONS == SUPPORTED_EXTENSIONS

    def test_doc_and_code_no_overlap(self):
        assert DOC_EXTENSIONS & CODE_EXTENSIONS == set()
