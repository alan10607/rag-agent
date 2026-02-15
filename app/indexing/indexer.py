"""
VectorSearcher - Ingestion Pipeline

Scans a local data directory, processes supported document types (.pdf, .txt, .md),
splits them into chunks, generates embeddings, and upserts into Qdrant.

Features:
- Idempotent: uses deterministic UUIDs based on filename + chunk_index
- Batched upsert for performance
- Comprehensive logging
"""

import os
import re
import uuid
from datetime import datetime, timezone

from pypdf import PdfReader
from qdrant_client.http.models import PointStruct

from app import config
from app.core import embedding
from app.indexing import splitter
from app.storage import vector_store
from app.logger import get_logger, setup_logging

logger = get_logger(__name__)

# UUID namespace for deterministic ID generation
_NAMESPACE = uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890")


def _generate_point_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic UUID based on source filename and chunk index.

    This ensures idempotent ingestion — re-running on the same file
    will overwrite rather than duplicate entries.
    """
    key = f"{source}::{chunk_index}"
    return str(uuid.uuid5(_NAMESPACE, key))


def _extract_text_from_txt(filepath: str) -> str:
    """Extract text content from a .txt file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def _strip_markdown_syntax(text: str) -> str:
    """Strip Markdown formatting syntax, keeping the semantic text content.

    Removes or simplifies:
    - Image references ![alt](url) → alt
    - Links [text](url) → text
    - Header markers (# ## ### etc.)
    - Emphasis markers (**, *, __, _)
    - Inline code backticks
    - Fenced code block markers (``` and ~~~)
    - Horizontal rules (---, ***, ___)
    - Blockquote markers (>)
    - HTML tags
    - Normalizes excessive blank lines
    """
    # Remove fenced code block markers (``` or ~~~), keep content inside
    text = re.sub(r'^```\w*\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^~~~\w*\s*$', '', text, flags=re.MULTILINE)

    # Remove images: ![alt](url) → alt
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)

    # Convert links: [text](url) → text
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)

    # Remove reference-style links: [text][ref] → text
    text = re.sub(r'\[([^\]]*)\]\[[^\]]*\]', r'\1', text)

    # Remove link definitions: [ref]: url
    text = re.sub(r'^\[([^\]]*)\]:\s*\S+.*$', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove horizontal rules (lines with only ---, ***, or ___)
    text = re.sub(r'^[\s]*([-*_]){3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove header markers (keep the text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove blockquote markers (keep the text)
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # Remove bold/italic markers: **text** → text, *text* → text
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_\s][^_]*)_', r'\1', text)

    # Remove strikethrough: ~~text~~ → text
    text = re.sub(r'~~([^~]+)~~', r'\1', text)

    # Remove inline code backticks: `code` → code
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove unordered list markers (-, *, +) at line start
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)

    # Remove ordered list markers (1., 2., etc.) at line start
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Normalize multiple blank lines into at most two newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _extract_text_from_markdown(filepath: str) -> str:
    """Extract text content from a .md file with Markdown syntax stripped.

    Reads the file as UTF-8, strips Markdown formatting syntax,
    and returns clean text suitable for embedding.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return _strip_markdown_syntax(raw)


def _clean_chinese_text(text: str) -> str:
    """Clean up common PDF extraction artifacts in Chinese text.

    - Remove spaces between CJK characters
    - Normalize multiple blank lines into one
    - Strip trailing whitespace per line
    """
    # Normalize multiple blank lines first (before space removal)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove spaces (NOT newlines) between CJK characters
    # CJK Unified Ideographs: \u4e00-\u9fff, CJK punctuation: \u3000-\u303f
    text = re.sub(
        r'([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])[ \t]+([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])',
        r'\1\2',
        text,
    )
    # Apply twice to handle consecutive cases (A B C → AB C → ABC)
    text = re.sub(
        r'([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])[ \t]+([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])',
        r'\1\2',
        text,
    )
    return text.strip()


def _extract_text_from_pdf(filepath: str) -> tuple[str, list[dict]]:
    """Extract text from a PDF, joining all pages into a single document.

    Returns:
        A tuple of:
        - full_text: The entire document text (cleaned).
        - page_map: A list of dicts with 'page' and 'start_char' keys,
                     used to map chunk positions back to page numbers.
    """
    reader = PdfReader(filepath)
    full_text = ""
    page_map: list[dict] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        text = _clean_chinese_text(text)
        page_map.append({"page": i + 1, "start_char": len(full_text)})
        full_text += text + "\n"

    return full_text.strip(), page_map


def _find_page_number(start_char: int, page_map: list[dict]) -> int | None:
    """Find which page a character position belongs to.

    Args:
        start_char: The character offset in the full document text.
        page_map: Sorted list of dicts with 'page' and 'start_char'.

    Returns:
        The page number, or None if not found.
    """
    page_num = None
    for entry in page_map:
        if entry["start_char"] <= start_char:
            page_num = entry["page"]
        else:
            break
    return page_num


def _scan_directory(data_dir: str) -> list[str]:
    """Scan directory for supported files."""
    supported = []
    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in config.SUPPORTED_EXTENSIONS:
            supported.append(os.path.join(data_dir, filename))
    logger.info("Found %d supported files in '%s'.", len(supported), data_dir)
    logger.info("Supported files: %s", ", ".join(os.path.basename(fp) for fp in supported))
    return supported


def _process_file(filepath: str) -> list[PointStruct]:
    """Process a single file: extract text, split into chunks, generate embeddings.

    Returns:
        A list of PointStruct objects ready for upsert.
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    logger.info("Processing file: %s", filename)

    # --- Step 1: Extract text (format-specific) ---
    page_map: list[dict] | None = None

    if ext == ".txt":
        text = _extract_text_from_txt(filepath)
    elif ext == ".pdf":
        text, page_map = _extract_text_from_pdf(filepath)
    elif ext == ".md":
        text = _extract_text_from_markdown(filepath)
    else:
        logger.warning("Unsupported file type: '%s'.", ext)
        return []

    if not text:
        logger.warning("No text extracted from '%s'.", filename)
        return []

    # --- Step 2: Split into chunks ---
    chunks = splitter.split_text(text)
    texts = [c.text for c in chunks]

    if not texts:
        logger.warning("No chunks extracted from '%s'.", filename)
        return []

    # --- Step 3: Generate embeddings ---
    embeddings = embedding.encode_batch(texts)

    # --- Step 4: Build points ---
    now = datetime.now(timezone.utc).isoformat()
    points: list[PointStruct] = []

    for chunk, emb in zip(chunks, embeddings):
        page = _find_page_number(chunk.start_char, page_map) if page_map else None
        point_id = _generate_point_id(filename, chunk.chunk_index)
        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "text": chunk.text,
                    "source": filename,
                    "chunk_index": chunk.chunk_index,
                    "page": page,
                    "created_at": now,
                },
            )
        )

    logger.info("Generated %d points from '%s'.", len(points), filename)
    return points


def ingest(data_dir: str | None = None) -> int:
    """Run the full ingestion pipeline.

    Args:
        data_dir: Path to the data directory (default from config).

    Returns:
        Total number of points ingested.
    """
    directory = data_dir or config.DATA_DIR
    logger.info("Starting ingestion from '%s'...", directory)

    if not os.path.isdir(directory):
        logger.error("Data directory '%s' does not exist.", directory)
        raise FileNotFoundError(f"Data directory not found: {directory}")

    # Initialize Qdrant
    client = vector_store.get_client()
    vector_store.ensure_collection(client)

    # Scan and process files
    filepaths = _scan_directory(directory)
    all_points: list[PointStruct] = []

    for filepath in filepaths:
        points = _process_file(filepath)
        all_points.extend(points)

    if not all_points:
        logger.warning("No points generated. Nothing to ingest.")
        return 0

    # Upsert into Qdrant
    vector_store.upsert_points(client, all_points)
    logger.info("Ingestion complete. Total points: %d", len(all_points))

    return len(all_points)


def main() -> None:
    """CLI entry point for ingestion."""
    setup_logging(module="ingest")
    total = ingest()
    print(f"\nIngestion complete. Total chunks ingested: {total}")


if __name__ == "__main__":
    main()
