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
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]


FILE_TYPE_DOC = "doc"
FILE_TYPE_CODE = "code"
FILE_TYPE_MANIFEST = "manifest"
SYMBOL_TYPE_PREAMBLE = "preamble"

DOC_EXTENSIONS: set[str] = {".md", ".txt"}
CODE_EXTENSIONS: set[str] = {".py", ".js", ".ts", ".rs"}
SUPPORTED_EXTENSIONS: set[str] = DOC_EXTENSIONS | CODE_EXTENSIONS
IGNORE_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    "target",
    "venv",
    "dist",
}
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".rs": "rust",
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
}


def compute_file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
