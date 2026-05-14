import re
from typing import NamedTuple

from src.models import LoadedFile, Chunk


class _Block(NamedTuple):
    text: str
    heading: str | None


TARGET_TOKENS = 500
OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4.0


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s for s in parts if s.strip()]


def _estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


def _merge_into_blocks(paragraphs: list[str], track_headings: bool) -> list[_Block]:
    target_chars = int(TARGET_TOKENS * CHARS_PER_TOKEN)
    blocks: list[_Block] = []
    current_text = ""
    current_heading = None

    for para in paragraphs:
        heading = _extract_heading(para) if track_headings else None
        if heading:
            current_heading = heading
            continue

        if _estimate_tokens(para) > TARGET_TOKENS:
            if current_text:
                blocks.append(_Block(current_text.strip(), current_heading))
                current_text = ""
            for sentence_block in _split_large_paragraph(para, target_chars):
                blocks.append(_Block(sentence_block, current_heading))
            continue

        if len(current_text) + len(para) + 2 > target_chars and current_text:
            blocks.append(_Block(current_text.strip(), current_heading))
            current_text = ""

        current_text += "\n\n" + para if current_text else para

    if current_text.strip():
        blocks.append(_Block(current_text.strip(), current_heading))

    return blocks


def _split_large_paragraph(para: str, target_chars: int) -> list[str]:
    sentences = _split_sentences(para)
    blocks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > target_chars and current:
            blocks.append(current.strip())
            current = ""
        current += " " + sentence if current else sentence

    if current.strip():
        blocks.append(current.strip())

    return blocks


def _apply_overlap(blocks: list[_Block]) -> list[_Block]:
    if len(blocks) <= 1:
        return blocks

    overlap_chars = int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
    result = [blocks[0]]

    for i in range(1, len(blocks)):
        prev_text = blocks[i - 1].text
        overlap = (
            prev_text[-overlap_chars:] if len(prev_text) >= overlap_chars else prev_text
        )
        result.append(_Block(overlap + " " + blocks[i].text, blocks[i].heading))

    return result


def _extract_heading(paragraph: str) -> str | None:
    if re.match(r"^#{1,6}\s", paragraph):
        return paragraph.strip()
    return None


def chunk_doc(loaded_file: LoadedFile) -> list[Chunk]:
    if not loaded_file.content.strip():
        return []

    is_markdown = loaded_file.extension == ".md"
    paragraphs = _split_paragraphs(loaded_file.content)
    blocks = _merge_into_blocks(paragraphs, is_markdown)
    blocks_with_overlap = _apply_overlap(blocks)

    return [
        Chunk(
            text=block.text,
            source=loaded_file.path,
            chunk_index=i,
            file_type="doc",
            language=None,
            section_heading=block.heading if is_markdown else None,
            symbol_name=None,
            symbol_type=None,
        )
        for i, block in enumerate(blocks_with_overlap)
    ]
