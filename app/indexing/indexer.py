"""
VectorSearcher - Ingestion Pipeline

Scans a local data directory, processes supported document types (.pdf, .txt, .md),
splits them into chunks, generates embeddings, and upserts into Qdrant.

Features:
- Idempotent: uses deterministic UUIDs based on filename + chunk_index
- Batched upsert for performance
- Comprehensive logging
"""

import argparse
import os
import re
import uuid
from datetime import datetime, timezone

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

def _extract_text_from_docx(filepath: str) -> str:
    """Extract text content from a .docx file."""
    from docx import Document

    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

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
    from pypdf import PdfReader

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

    if ext == ".pdf":
        text, page_map = _extract_text_from_pdf(filepath)
    elif ext == ".docx":
        text = _extract_text_from_docx(filepath)
    else:
        text = _extract_text_from_txt(filepath)

    if not text:    
        logger.warning("No text extracted from '%s'.", filename)
        return []

    # --- Step 2: Split into chunks ---
    chunks = splitter.split_text(text, ext=ext)
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


def ingest(data_dir: str | None = None) -> tuple[int, int, int]:
    """Run the full ingestion pipeline.

    Args:
        data_dir: Path to the data directory (default from config).

    Returns:
        A tuple of:
        - total number of points ingested.
        - number of successful files processed.
        - number of failed files processed.
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
    success_count = 0
    failed_count = 0

    for filepath in filepaths:
        try:
            points = _process_file(filepath)
            all_points.extend(points)
            success_count += 1
        except Exception as e:
            logger.exception(
                "Failed to process file '%s'. Skipping. Error: %s",
                filepath,
                str(e),
            )
            failed_count += 1
            continue


    if all_points:
        # Upsert into Qdrant
        vector_store.upsert_points(client, all_points)
        logger.info("Ingestion complete. Total points: %d", len(all_points))
    else:
        logger.warning("No points generated. Nothing to ingest.")

    return len(all_points), success_count, failed_count


def main() -> None:
    """CLI entry point for ingestion."""
    parser = argparse.ArgumentParser(
        description="VectorSearcher Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.DATA_DIR,
        help="The path to the data directory. Default from config if not provided.",
    )

    args = parser.parse_args()

    setup_logging(module="ingest")  

    total, success_count, failed_count = ingest(args.data_dir)
    print(f"\nIngestion complete. Total points: {total}, success: {success_count}, failed: {failed_count}")


if __name__ == "__main__":
    main()
