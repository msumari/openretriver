from pathlib import Path

from src.loader import load_files


def test_finds_all_supported_types(sample_project_dir):
    results = load_files(sample_project_dir)
    extensions = {f.extension for f in results}
    assert extensions == {".md", ".txt", ".py", ".js", ".ts", ".rs"}


def test_ignores_unsupported_extensions(sample_project_dir):
    results = load_files(sample_project_dir)
    extensions = {f.extension for f in results}
    assert ".jpg" not in extensions
    assert ".csv" not in extensions


def test_skips_ignored_dirs(sample_project_dir):
    results = load_files(sample_project_dir)
    paths = {f.path for f in results}
    for p in paths:
        assert ".git" not in p
        assert "node_modules" not in p
        assert "__pycache__" not in p


def test_recursive_discovery(sample_project_dir):
    results = load_files(sample_project_dir)
    paths = {f.path for f in results}
    assert any("docs/" in p for p in paths)
    assert any("src/" in p for p in paths)


def test_paths_are_relative(sample_project_dir):
    results = load_files(sample_project_dir)
    for f in results:
        assert not Path(f.path).is_absolute()


def test_content_matches(sample_project_dir):
    results = load_files(sample_project_dir)
    by_path = {f.path: f for f in results}
    guide = by_path["docs/guide.md"]
    assert guide.content == "# Guide\n\nSome docs here."


def test_empty_directory(empty_project_dir):
    results = load_files(empty_project_dir)
    assert results == []


def test_sorted_by_path(sample_project_dir):
    results = load_files(sample_project_dir)
    paths = [f.path for f in results]
    assert paths == sorted(paths)


def test_handles_binary_file_gracefully(tmp_path):
    (tmp_path / "binary.py").write_bytes(b"\x80\x81\x82\x83")
    results = load_files(tmp_path)
    assert results == []
