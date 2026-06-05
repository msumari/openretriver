import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedFile:
    path: str
    extension: str
    content: str


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    chunk_index: int
    file_type: str
    language: str | None
    section_heading: str | None
    symbol_name: str | None
    symbol_type: str | None


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]
    sparse_vector: SparseVector | None = None


FILE_TYPE_DOC = "doc"
FILE_TYPE_CODE = "code"
FILE_TYPE_MANIFEST = "manifest"
FILE_TYPE_SESSION = "session"
SYMBOL_TYPE_PREAMBLE = "preamble"

DOC_EXTENSIONS: set[str] = {".md", ".txt"}
CODE_EXTENSIONS: set[str] = {".py", ".js", ".ts", ".rs", ".json", ".yaml", ".yml", ".sh", ".bash", ".mk"}
SUPPORTED_EXTENSIONS: set[str] = DOC_EXTENSIONS | CODE_EXTENSIONS
IGNORE_FILES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "Cargo.lock",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
}
SUPPORTED_FILENAMES: dict[str, str] = {
    "Makefile": ".mk",
    "makefile": ".mk",
    "GNUmakefile": ".mk",
}
IGNORE_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".ruff_cache",
    "target",
    "dist",
    "build",
    ".eggs",
    "site-packages",
    ".next",
    ".nuxt",
    ".aws-sam",
}
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
    ".bash": "bash",
    ".mk": "make",
}
NODE_TYPE_TO_SYMBOL_TYPE = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",
    "class_definition": "class",
    "class_declaration": "class",
    "lexical_declaration": "function",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "impl_item": "impl",
    "struct_item": "struct",
    "enum_item": "enum",
    "pair": "pair",
    "block_mapping_pair": "pair",
    "variable_assignment": "variable",
    "rule": "rule",
}


@dataclass(frozen=True)
class SessionMessage:
    session_id: str
    timestamp: int
    role: str
    content: str


def compute_file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
