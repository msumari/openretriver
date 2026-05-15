import logging
import re

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript
import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser, Node

from src.models import (
    LoadedFile,
    Chunk,
    EXTENSION_TO_LANGUAGE,
    NODE_TYPE_TO_SYMBOL_TYPE,
    FILE_TYPE_CODE,
    SYMBOL_TYPE_PREAMBLE,
)

logger = logging.getLogger(__name__)

_language_cache: dict[str, dict] = {}


def _make_code_chunk(
    loaded_file: LoadedFile,
    language: str,
    text: str,
    index: int,
    symbol_name: str | None = None,
    symbol_type: str = SYMBOL_TYPE_PREAMBLE,
) -> Chunk:
    return Chunk(
        text=text,
        source=loaded_file.path,
        chunk_index=index,
        file_type=FILE_TYPE_CODE,
        language=language,
        section_heading=None,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
    )


def _get_language_config(ext: str) -> dict:
    if ext not in _language_cache:
        configs = {
            ".py": lambda: {
                "language": Language(tspython.language()),
                "top_level": ["function_definition", "class_definition"],
            },
            ".js": lambda: {
                "language": Language(tsjavascript.language()),
                "top_level": [
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                ],
            },
            ".ts": lambda: {
                "language": Language(tstypescript.language_typescript()),
                "top_level": [
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                    "interface_declaration",
                    "type_alias_declaration",
                ],
            },
            ".rs": lambda: {
                "language": Language(tsrust.language()),
                "top_level": [
                    "function_item",
                    "impl_item",
                    "struct_item",
                    "enum_item",
                ],
            },
        }
        config = configs[ext]()
        config["parser"] = Parser(config["language"])
        _language_cache[ext] = config
    return _language_cache[ext]


def _extract_node_name(node: Node, extension: str) -> str:
    if extension == ".rs" and node.type == "impl_item":
        type_node = node.child_by_field_name("type")
        if type_node:
            return type_node.text.decode("utf-8")
        return "impl"

    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8")

    if extension in (".js", ".ts") and node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                vname = child.child_by_field_name("name")
                if vname:
                    return vname.text.decode("utf-8")

    logger.warning(
        "Could not extract name from %s node in %s file", node.type, extension
    )
    return "<anonymous>"


def _chunk_with_treesitter(loaded_file: LoadedFile) -> list[Chunk]:
    config = _get_language_config(loaded_file.extension)
    source_bytes = loaded_file.content.encode("utf-8")
    tree = config["parser"].parse(source_bytes)

    language = EXTENSION_TO_LANGUAGE[loaded_file.extension]
    chunks = []
    preamble_end = 0
    top_level_types = set(config["top_level"])

    top_nodes = []
    for node in tree.root_node.children:
        if node.type in top_level_types:
            top_nodes.append(node)

    if top_nodes:
        preamble_end = top_nodes[0].start_byte
        preamble_text = source_bytes[:preamble_end].decode("utf-8").strip()
        if preamble_text:
            chunks.append(
                _make_code_chunk(loaded_file, language, preamble_text, 0)
            )

    for node in top_nodes:
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
        name = _extract_node_name(node, loaded_file.extension)
        symbol_type = NODE_TYPE_TO_SYMBOL_TYPE.get(node.type, "function")

        chunks.append(
            _make_code_chunk(loaded_file, language, text, len(chunks), name, symbol_type)
        )

    if not chunks:
        chunks.append(
            _make_code_chunk(loaded_file, language, loaded_file.content.strip(), 0)
        )

    return chunks


def chunk_code(loaded_file: LoadedFile) -> list[Chunk]:
    if not loaded_file.content.strip():
        return []

    try:
        return _chunk_with_treesitter(loaded_file)
    except (KeyError, ValueError, UnicodeDecodeError):
        logger.warning(
            "Tree-sitter parse failed for %s, falling back to regex", loaded_file.path
        )
        return _chunk_with_regex(loaded_file)


# --- Regex fallback ---

_PYTHON_PATTERN = re.compile(r"^(class|def)\s+(\w+)", re.MULTILINE)
_JS_PATTERN = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)|^(?:export\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_TS_PATTERN = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)|^(?:export\s+)?class\s+(\w+)|^(?:export\s+)?interface\s+(\w+)",
    re.MULTILINE,
)
_RUST_PATTERN = re.compile(
    r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)|^(?:pub\s+)?struct\s+(\w+)|^(?:pub\s+)?enum\s+(\w+)|^impl\s+(\w+)",
    re.MULTILINE,
)


def _guess_symbol_type(match_text: str) -> str:
    lower = match_text.lower()
    if "class" in lower:
        return "class"
    if "interface" in lower:
        return "interface"
    if "struct" in lower:
        return "struct"
    if "enum" in lower:
        return "enum"
    if "impl" in lower:
        return "impl"
    return "function"


def _chunk_with_regex(loaded_file: LoadedFile) -> list[Chunk]:
    language = EXTENSION_TO_LANGUAGE[loaded_file.extension]

    patterns = {
        ".py": _PYTHON_PATTERN,
        ".js": _JS_PATTERN,
        ".ts": _TS_PATTERN,
        ".rs": _RUST_PATTERN,
    }

    pattern = patterns.get(loaded_file.extension)
    if not pattern:
        return [_make_code_chunk(loaded_file, language, loaded_file.content, 0)]

    matches = list(pattern.finditer(loaded_file.content))
    if not matches:
        return [_make_code_chunk(loaded_file, language, loaded_file.content.strip(), 0)]

    chunks = []
    first_match_pos = matches[0].start()
    preamble = loaded_file.content[:first_match_pos].strip()
    if preamble:
        chunks.append(_make_code_chunk(loaded_file, language, preamble, 0))

    for i, match in enumerate(matches):
        start = match.start()
        end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(loaded_file.content)
        )
        text = loaded_file.content[start:end].rstrip()
        name = next((g for g in match.groups() if g), "<anonymous>")
        symbol_type = _guess_symbol_type(match.group(0))

        chunks.append(
            _make_code_chunk(loaded_file, language, text, len(chunks), name, symbol_type)
        )

    return chunks
