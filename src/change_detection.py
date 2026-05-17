from dataclasses import dataclass

from src.models import LoadedFile
from src.models import compute_file_hash


@dataclass(frozen=True)
class ChangeResult:
    new: list[LoadedFile]
    changed: list[LoadedFile]
    unchanged: list[LoadedFile]
    deleted: set[str]
    hashes: dict[str, str]


def detect_changes(
    files: list[LoadedFile],
    manifest: dict[str, dict],
) -> ChangeResult:
    new = []
    changed = []
    unchanged = []
    hashes = {}

    seen_sources = set()

    for f in files:
        seen_sources.add(f.path)
        current_hash = compute_file_hash(f.content)
        hashes[f.path] = current_hash
        previous = manifest.get(f.path)

        if previous is None:
            new.append(f)
        elif previous["content_hash"] != current_hash:
            changed.append(f)
        else:
            unchanged.append(f)

    deleted = set(manifest.keys()) - seen_sources

    return ChangeResult(new=new, changed=changed, unchanged=unchanged, deleted=deleted, hashes=hashes)
