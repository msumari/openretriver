from src.models import LoadedFile
from src.chunker_docs import chunk_doc


def _make_doc(content: str, path: str = "docs/test.md", ext: str = ".md") -> LoadedFile:
    return LoadedFile(path=path, extension=ext, content=content)


def test_single_short_paragraph():
    doc = _make_doc("A short paragraph.")
    chunks = chunk_doc(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "A short paragraph."


def test_multiple_paragraphs_merged():
    paragraphs = ["Short paragraph one."] * 10
    content = "\n\n".join(paragraphs)
    doc = _make_doc(content)
    chunks = chunk_doc(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.text) > 0


def test_large_paragraph_sentence_split():
    long_para = ". ".join(["This is sentence number " + str(i) for i in range(200)]) + "."
    doc = _make_doc(long_para)
    chunks = chunk_doc(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert not c.text.startswith(". ")


def test_overlap_present():
    paragraphs = ["Paragraph " + str(i) + ". " + ("word " * 80) for i in range(10)]
    content = "\n\n".join(paragraphs)
    doc = _make_doc(content)
    chunks = chunk_doc(doc)
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-50:]
            assert tail in chunks[i + 1].text


def test_markdown_heading_tracked():
    content = "## Setup\n\nInstall the package.\n\nRun the tests."
    doc = _make_doc(content)
    chunks = chunk_doc(doc)
    assert all(c.section_heading == "## Setup" for c in chunks)


def test_heading_changes_mid_file():
    section_one = "## First\n\n" + "\n\n".join(["Word " * 80 + "." for _ in range(5)])
    section_two = "## Second\n\n" + "\n\n".join(["Other " * 80 + "." for _ in range(5)])
    content = section_one + "\n\n" + section_two
    doc = _make_doc(content)
    chunks = chunk_doc(doc)
    headings = [c.section_heading for c in chunks]
    assert "## First" in headings
    assert "## Second" in headings


def test_txt_file_heading_is_none():
    doc = _make_doc("Some plain text content.", path="notes.txt", ext=".txt")
    chunks = chunk_doc(doc)
    assert all(c.section_heading is None for c in chunks)


def test_chunk_index_sequential():
    paragraphs = ["Paragraph " + str(i) + ". " + ("word " * 80) for i in range(10)]
    content = "\n\n".join(paragraphs)
    doc = _make_doc(content)
    chunks = chunk_doc(doc)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_file_type_is_doc():
    doc = _make_doc("Hello world.")
    chunks = chunk_doc(doc)
    assert all(c.file_type == "doc" for c in chunks)


def test_language_is_none():
    doc = _make_doc("Hello world.")
    chunks = chunk_doc(doc)
    assert all(c.language is None for c in chunks)


def test_source_matches_file():
    doc = _make_doc("Content here.", path="docs/readme.md")
    chunks = chunk_doc(doc)
    assert all(c.source == "docs/readme.md" for c in chunks)


def test_empty_file():
    doc = _make_doc("")
    chunks = chunk_doc(doc)
    assert chunks == []
