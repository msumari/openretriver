from src.models import LoadedFile
from src.change_detection import detect_changes
from src.models import compute_file_hash


def _make_file(path: str, content: str) -> LoadedFile:
    ext = "." + path.rsplit(".", 1)[-1]
    return LoadedFile(path=path, extension=ext, content=content)


def test_all_new_when_manifest_empty():
    files = [_make_file("a.py", "print('hello')"), _make_file("b.md", "# Docs")]
    result = detect_changes(files, {})
    assert len(result.new) == 2
    assert len(result.changed) == 0
    assert len(result.unchanged) == 0
    assert len(result.deleted) == 0


def test_all_unchanged_when_hashes_match():
    files = [_make_file("a.py", "print('hello')")]
    manifest = {
        "a.py": {"content_hash": compute_file_hash("print('hello')"), "chunk_count": 1}
    }
    result = detect_changes(files, manifest)
    assert len(result.new) == 0
    assert len(result.changed) == 0
    assert len(result.unchanged) == 1
    assert len(result.deleted) == 0


def test_changed_when_hash_differs():
    files = [_make_file("a.py", "print('world')")]
    manifest = {
        "a.py": {"content_hash": compute_file_hash("print('hello')"), "chunk_count": 1}
    }
    result = detect_changes(files, manifest)
    assert len(result.new) == 0
    assert len(result.changed) == 1
    assert result.changed[0].path == "a.py"
    assert len(result.unchanged) == 0
    assert len(result.deleted) == 0


def test_deleted_when_in_manifest_but_not_on_disk():
    files = [_make_file("a.py", "print('hello')")]
    manifest = {
        "a.py": {"content_hash": compute_file_hash("print('hello')"), "chunk_count": 1},
        "removed.py": {"content_hash": "old_hash", "chunk_count": 3},
    }
    result = detect_changes(files, manifest)
    assert len(result.deleted) == 1
    assert "removed.py" in result.deleted


def test_mixed_scenario():
    files = [
        _make_file("unchanged.py", "same"),
        _make_file("changed.py", "new content"),
        _make_file("brand_new.py", "fresh"),
    ]
    manifest = {
        "unchanged.py": {"content_hash": compute_file_hash("same"), "chunk_count": 1},
        "changed.py": {"content_hash": compute_file_hash("old content"), "chunk_count": 2},
        "deleted.py": {"content_hash": "whatever", "chunk_count": 5},
    }
    result = detect_changes(files, manifest)
    assert len(result.new) == 1
    assert result.new[0].path == "brand_new.py"
    assert len(result.changed) == 1
    assert result.changed[0].path == "changed.py"
    assert len(result.unchanged) == 1
    assert result.unchanged[0].path == "unchanged.py"
    assert len(result.deleted) == 1
    assert "deleted.py" in result.deleted


def test_empty_files_and_manifest():
    result = detect_changes([], {})
    assert len(result.new) == 0
    assert len(result.changed) == 0
    assert len(result.unchanged) == 0
    assert len(result.deleted) == 0
