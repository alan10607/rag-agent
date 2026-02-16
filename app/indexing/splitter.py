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
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

logger = get_logger(__name__)

# Separator hierarchy optimized for Chinese + English:
DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    ". ",
    "！",
    "! ",
    "？",
    "? ",
    "；",
    ";",
    "，",
    ", ",
    " ",
    ""
]
DEFAULT_HEADERS_SEPARATORS = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

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
    ext: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
    headers_separators: list[str],
) -> list[str]:
    """Split text using LangChain's RecursiveCharacterTextSplitter.

    Uses keep_separator="end" so Chinese punctuation stays at the end
    of the preceding chunk.
    """

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator="end",
    )

    if ext == ".md":
        # Markdown: header + recursive
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_separators
        )
        header_chunks = md_splitter.split_text(text)

        raw_chunks: list[str] = []
        for doc in header_chunks:
            raw_chunks.extend(
                splitter.split_text(doc.page_content)
            )

        return raw_chunks

    else:
        return splitter.split_text(text)


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
    ext: str = "txt",
    chunk_size: int = config.CHUNK_SIZE ,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    chunk_min_size: int = config.CHUNK_MIN_SIZE,
    separators: list[str] = DEFAULT_SEPARATORS,
    headers_separators: list[str] = DEFAULT_HEADERS_SEPARATORS,
) -> list[Chunk]:
    """Split text into overlapping chunks with metadata.

    Uses LangChain's RecursiveCharacterTextSplitter by default.
    Falls back to a built-in implementation if LangChain is not available.

    Args:
        text: The full text to split.
        ext: The file extension of the text.
        chunk_size: Maximum characters per chunk (default from config).
        chunk_overlap: Overlap characters between chunks (default from config).
        chunk_min_size: Minimum characters per chunk; smaller chunks are
                        merged with neighbors (default from config).
        separators: Custom separator hierarchy (default: paragraph > line > sentence > word).

    Returns:
        A list of Chunk objects with text and metadata.
    """
    logger.info(
        "Start splitting %s extension file (%d chars) into chunks (size=%d, overlap=%d, min_size=%d).",
        ext, len(text), chunk_size, chunk_overlap, chunk_min_size
    )

    raw_chunks = _split_with_langchain(text, ext, chunk_size, chunk_overlap, separators, headers_separators)
    raw_chunks = merge_small_chunks(raw_chunks, chunk_min_size, chunk_size)

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
        "Split %s extension file (%d chars) into %d chunks (size=%d, overlap=%d, min_size=%d).",
        ext, len(text), len(chunks), chunk_size, chunk_overlap, chunk_min_size
    )
    return chunks
