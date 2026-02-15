"""
VectorSearcher - Text Splitter

Splits text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
Falls back to a built-in recursive splitter if LangChain is unavailable.

Supports both English and Chinese text:
- Chinese punctuation (。！？；，) is included in the separator hierarchy.
- `keep_separator="end"` ensures punctuation stays attached to the preceding text.
"""

from dataclasses import dataclass

from app import config
from app.logger import get_logger

logger = get_logger(__name__)

# Separator hierarchy optimized for Chinese + English:
# paragraph > line > sentence end (ZH/EN) > clause (ZH) > space > char
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "，", " ", ""]


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int


# ---------------------------------------------------------------------------
# Primary: LangChain RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

def _split_with_langchain(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Split text using LangChain's RecursiveCharacterTextSplitter.

    Uses keep_separator="end" so Chinese punctuation stays at the end
    of the preceding chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator="end",
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Fallback: Built-in recursive splitter
# ---------------------------------------------------------------------------

def _split_keeping_separator(text: str, separator: str) -> list[str]:
    """Split text by separator, keeping the separator attached to the preceding part."""
    if not separator:
        return list(text)

    raw_parts = text.split(separator)
    parts: list[str] = []
    for i, part in enumerate(raw_parts):
        if i < len(raw_parts) - 1:
            parts.append(part + separator)
        elif part:
            parts.append(part)
    return [p for p in parts if p.strip()]


def _split_text_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using a hierarchy of separators (built-in fallback)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = ""
    remaining_separators: list[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    parts = _split_keeping_separator(text, separator)

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in parts:
        part_len = len(part)

        if current_length + part_len > chunk_size and current_chunk:
            merged = "".join(current_chunk)
            if merged.strip():
                chunks.append(merged.strip())

            effective_overlap = min(chunk_overlap, current_length // 2)
            overlap_parts: list[str] = []
            overlap_len = 0
            for prev_part in reversed(current_chunk):
                if overlap_len + len(prev_part) > effective_overlap:
                    break
                overlap_parts.insert(0, prev_part)
                overlap_len += len(prev_part)

            current_chunk = overlap_parts
            current_length = overlap_len

        current_chunk.append(part)
        current_length += part_len

    if current_chunk:
        merged = "".join(current_chunk)
        if merged.strip():
            if len(merged) > chunk_size and remaining_separators:
                sub_chunks = _split_text_recursive(
                    merged, remaining_separators, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(merged.strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size and remaining_separators:
            sub_chunks = _split_text_recursive(
                chunk, remaining_separators, chunk_size, chunk_overlap
            )
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    return final_chunks


# ---------------------------------------------------------------------------
# Post-processing: merge undersized chunks
# ---------------------------------------------------------------------------

def merge_small_chunks(
    chunks: list[str], min_size: int, max_size: int
) -> list[str]:
    """Merge chunks below min_size with their neighbors.

    Strategy:
    - Try to merge a small chunk into the previous chunk (append).
    - If that would exceed max_size, keep it as-is.
    - Finally, check if the first chunk is still too small and merge forward.
    """
    if not chunks or min_size <= 0:
        return chunks

    result: list[str] = []
    for chunk in chunks:
        if result and len(chunk) < min_size:
            combined = result[-1] + "\n" + chunk
            if len(combined) <= max_size:
                result[-1] = combined
                continue
        result.append(chunk)

    # Handle the case where the first chunk is still too small
    if len(result) > 1 and len(result[0]) < min_size:
        combined = result[0] + "\n" + result[1]
        if len(combined) <= max_size:
            result = [combined] + result[2:]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    min_chunk_size: int | None = None,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter by default.
    Falls back to a built-in implementation if LangChain is not available.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap characters between chunks (default from config).
        min_chunk_size: Minimum characters per chunk; smaller chunks are
                        merged with neighbors (default from config).
        separators: Custom separator hierarchy (default: paragraph > line > sentence > word).

    Returns:
        A list of Chunk objects with text and metadata.
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    min_size = min_chunk_size if min_chunk_size is not None else config.MIN_CHUNK_SIZE
    seps = separators if separators is not None else DEFAULT_SEPARATORS

    try:
        raw_chunks = _split_with_langchain(text, size, overlap, seps)
    except Exception:
        logger.warning("LangChain splitter unavailable, using built-in fallback.")
        raw_chunks = _split_text_recursive(text, seps, size, overlap)

    raw_chunks = merge_small_chunks(raw_chunks, min_size, size)

    # Build Chunk objects with position metadata
    chunks: list[Chunk] = []
    search_start = 0
    for i, chunk_text in enumerate(raw_chunks):
        start = text.find(chunk_text, search_start)
        if start == -1:
            start = search_start  # fallback
        end = start + len(chunk_text)

        chunks.append(
            Chunk(
                text=chunk_text,
                chunk_index=i,
                start_char=start,
                end_char=end,
            )
        )
        search_start = max(search_start, start + 1)

    logger.info(
        "Split text (%d chars) into %d chunks (size=%d, overlap=%d).",
        len(text),
        len(chunks),
        size,
        overlap,
    )

    preview_count = 5
    omitted = len(chunks) > preview_count * 2
    show_chunks = chunks[:preview_count] + chunks[-preview_count:] if omitted else chunks
    for chunk in show_chunks:
        if omitted and chunk.chunk_index == chunks[-preview_count].chunk_index:
            logger.debug("  ... (%d chunks omitted) ...", len(chunks) - preview_count * 2)
        if len(chunk.text) <= 200:
            display = chunk.text.replace("\n", "\\n")
            logger.info(
                "  Chunk #%d [chars %d-%d, %d chars]:\n%s",
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                len(chunk.text),
                display,
            )
        else:
            head = chunk.text[:100].replace("\n", "\\n")
            tail = chunk.text[-100:].replace("\n", "\\n")
            logger.info(
                "  Chunk #%d [chars %d-%d, %d chars]:\n%s...%s",
                chunk.chunk_index,
                chunk.start_char,
                chunk.end_char,
                len(chunk.text),
                head,
                tail,
            )

    return chunks
