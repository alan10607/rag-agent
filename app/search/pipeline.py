"""
VectorVault - Query Pipeline (Semantic Search)

Provides a CLI interface for semantic search against the Qdrant vector store.

Usage:
    python -m app.search "your query here" --top_k 5
"""

import argparse

from app import config, embedding, qdrant_client
from app.logger import get_logger, setup_logging

logger = get_logger(__name__)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Perform a semantic search against the vector store.

    Args:
        query: The user's text query.
        top_k: Number of results to return (default from config).

    Returns:
        A list of result dicts with score, source, and text snippet.
    """
    k = top_k or config.DEFAULT_TOP_K
    logger.info("Searching for: '%s' (top_k=%d)", query, k)

    # Convert query to embedding
    query_vector = embedding.encode(query)

    # Search in Qdrant
    client = qdrant_client.get_client()
    results = qdrant_client.search_vectors(client, query_vector, top_k=k)

    return results


def _format_results(results: list[dict]) -> str:
    """Format search results for CLI display."""
    if not results:
        return "No results found."

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Found {len(results)} result(s)")
    lines.append(f"{'='*60}\n")

    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        score = result.get("score", 0.0)
        source = payload.get("source", "unknown")
        text = payload.get("text", "")
        page = payload.get("page")
        chunk_idx = payload.get("chunk_index", "?")

        # Truncate long text for display
        preview = text[:200] + "..." if len(text) > 200 else text

        lines.append(f"  [{i}] Score: {score:.4f}")
        lines.append(f"      Source: {source}")
        if page is not None:
            lines.append(f"      Page: {page}")
        lines.append(f"      Chunk: #{chunk_idx}")
        lines.append(f"      Preview:")
        lines.append(f"        {preview}")
        lines.append(f"  {'-'*56}")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for semantic search."""
    parser = argparse.ArgumentParser(
        description="VectorVault Semantic Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        help="The text query to search for.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of results to return (default: {config.DEFAULT_TOP_K}).",
    )

    args = parser.parse_args()

    setup_logging(module="search")

    results = search(args.query, top_k=args.top_k)
    print(_format_results(results))


if __name__ == "__main__":
    main()
