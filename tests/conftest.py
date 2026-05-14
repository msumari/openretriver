import pytest
from pathlib import Path


@pytest.fixture
def sample_project_dir(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n\nSome docs here.")
    (tmp_path / "docs" / "notes.txt").write_text("Plain text notes.")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main():\n    pass\n")
    (tmp_path / "src" / "app.js").write_text("function app() {}\n")
    (tmp_path / "src" / "types.ts").write_text("interface Config {}\n")
    (tmp_path / "src" / "lib.rs").write_text("fn main() {}\n")

    # unsupported files — should be ignored
    (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n")

    # ignored directories — files inside should be skipped
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("git internal")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("module")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"\x00")

    return tmp_path


@pytest.fixture
def empty_project_dir(tmp_path: Path) -> Path:
    (tmp_path / "empty").mkdir()
    return tmp_path / "empty"
