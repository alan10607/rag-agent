"""
VectorVault - Ingestion Pipeline

Scans a local data directory, processes supported document types (.pdf, .txt),
splits them into chunks, generates embeddings, and upserts into Qdrant.

Features:
- Idempotent: uses deterministic UUIDs based on filename + chunk_index
- Batched upsert for performance
- Comprehensive logging
"""

import os
import uuid
from datetime import datetime, timezone

from pypdf import PdfReader
from qdrant_client.http.models import PointStruct

from app import config, embedding, qdrant_client
from app.ingest import splitter
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


def _extract_text_from_pdf(filepath: str) -> list[dict]:
    """Extract text content from a PDF file, page by page.

    Returns:
        A list of dicts with 'text' and 'page' keys.
    """
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages


def _scan_directory(data_dir: str) -> list[str]:
    """Scan directory for supported files."""
    supported = []
    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in config.SUPPORTED_EXTENSIONS:
            supported.append(os.path.join(data_dir, filename))
    logger.info("Found %d supported files in '%s'.", len(supported), data_dir)
    return supported


def _process_file(filepath: str) -> list[PointStruct]:
    """Process a single file: extract text, split into chunks, generate embeddings.

    Returns:
        A list of PointStruct objects ready for upsert.
    """
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    logger.info("Processing file: %s", filename)

    points: list[PointStruct] = []
    now = datetime.now(timezone.utc).isoformat()

    if ext == ".txt":
        text = _extract_text_from_txt(filepath)
        chunks = splitter.split_text(text)
        texts = [c.text for c in chunks]

        if not texts:
            logger.warning("No chunks extracted from '%s'.", filename)
            return []

        embeddings = embedding.encode_batch(texts)

        for chunk, emb in zip(chunks, embeddings):
            point_id = _generate_point_id(filename, chunk.chunk_index)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload={
                        "text": chunk.text,
                        "source": filename,
                        "chunk_index": chunk.chunk_index,
                        "page": None,
                        "created_at": now,
                    },
                )
            )

    elif ext == ".pdf":
        pages = _extract_text_from_pdf(filepath)
        global_chunk_index = 0
        for page_info in pages:
            chunks = splitter.split_text(page_info["text"])
            texts = [c.text for c in chunks]

            if not texts:
                continue

            embeddings = embedding.encode_batch(texts)

            for chunk, emb in zip(chunks, embeddings):
                point_id = _generate_point_id(filename, global_chunk_index)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "text": chunk.text,
                            "source": filename,
                            "chunk_index": global_chunk_index,
                            "page": page_info["page"],
                            "created_at": now,
                        },
                    )
                )
                global_chunk_index += 1

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
    client = qdrant_client.get_client()
    qdrant_client.ensure_collection(client)

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
    qdrant_client.upsert_points(client, all_points)
    logger.info("Ingestion complete. Total points: %d", len(all_points))

    return len(all_points)


def main() -> None:
    """CLI entry point for ingestion."""
    setup_logging(module="ingest")
    total = ingest()
    print(f"\nIngestion complete. Total chunks ingested: {total}")


if __name__ == "__main__":
    main()
