"""
VectorVault - Recursive Character Text Splitter

Splits text into overlapping chunks using a hierarchy of separators.
This ensures chunks respect natural text boundaries (paragraphs, sentences,
words) before falling back to character-level splitting.
"""

from dataclasses import dataclass

from app import config
from app.logger import get_logger

logger = get_logger(__name__)

# Default separator hierarchy: double newline > single newline > sentence end > space > char
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_index: int
    start_char: int
    end_char: int


def _split_text_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using a hierarchy of separators.

    Args:
        text: The text to split.
        separators: Ordered list of separators to try (most preferred first).
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that exists in the text
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    # Split with the chosen separator
    if separator:
        parts = text.split(separator)
    else:
        parts = list(text)

    # Merge parts into chunks that respect chunk_size
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for part in parts:
        part_with_sep = part + separator if separator else part
        part_len = len(part_with_sep)

        if current_length + part_len > chunk_size and current_chunk:
            # Finalize the current chunk
            merged = separator.join(current_chunk) if separator else "".join(current_chunk)
            if merged.strip():
                chunks.append(merged.strip())

            # Keep overlap: walk backwards to find overlap content
            overlap_parts: list[str] = []
            overlap_len = 0
            for prev_part in reversed(current_chunk):
                prev_len = len(prev_part) + len(separator)
                if overlap_len + prev_len > chunk_overlap:
                    break
                overlap_parts.insert(0, prev_part)
                overlap_len += prev_len

            current_chunk = overlap_parts
            current_length = overlap_len

        current_chunk.append(part)
        current_length += part_len

    # Finalize any remaining content
    if current_chunk:
        merged = separator.join(current_chunk) if separator else "".join(current_chunk)
        if merged.strip():
            # If this final piece is still too large, recurse with next separator
            if len(merged) > chunk_size and remaining_separators:
                sub_chunks = _split_text_recursive(
                    merged, remaining_separators, chunk_size, chunk_overlap
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(merged.strip())

    # Post-process: if any chunk is still too large, recurse with remaining separators
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


def split_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks with metadata.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap characters between chunks (default from config).
        separators: Custom separator hierarchy (default: paragraph > line > sentence > word).

    Returns:
        A list of Chunk objects with text and metadata.
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    seps = separators if separators is not None else DEFAULT_SEPARATORS

    raw_chunks = _split_text_recursive(text, seps, size, overlap)

    chunks: list[Chunk] = []
    search_start = 0
    for i, chunk_text in enumerate(raw_chunks):
        # Find the position of this chunk in the original text
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
        # Allow overlap: don't advance search_start past the overlap region
        search_start = max(search_start, start + 1)

    logger.info(
        "Split text (%d chars) into %d chunks (size=%d, overlap=%d).",
        len(text),
        len(chunks),
        size,
        overlap,
    )
    logger.debug(str(chunks))
    return chunks
