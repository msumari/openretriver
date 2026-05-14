import logging
from pathlib import Path

from src.models import LoadedFile, SUPPORTED_EXTENSIONS, IGNORE_DIRS

logger = logging.getLogger(__name__)


def _has_ignored_parent(file_path: Path, root: Path) -> bool:
    relative = file_path.relative_to(root)
    return any(part in IGNORE_DIRS for part in relative.parts)


def _read_text_safe(file_path: Path) -> str | None:
    try:
        return file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        logger.warning("Could not read file %s, skipping", file_path)
        return None


def load_files(root: str | Path) -> list[LoadedFile]:
    root = Path(root)
    results = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if _has_ignored_parent(file_path, root):
            continue
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        content = _read_text_safe(file_path)
        if content is None:
            continue
        results.append(
            LoadedFile(
                path=str(file_path.relative_to(root)),
                extension=file_path.suffix,
                content=content,
            )
        )
    return results
