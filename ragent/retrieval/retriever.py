"""
Ragent - Query Pipeline (Semantic Search)

Searches the vector store for the most relevant documents based on the query.
"""

import argparse

from ragent import config
from ragent.core import embedding
from ragent.storage import vector_store
from ragent.logger import get_logger, setup_logging

MAX_TOP_K = 20

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

    if k > MAX_TOP_K:
        k = MAX_TOP_K
        logger.warning("top_k is greater than %d, setting to %d, k=%d", MAX_TOP_K, MAX_TOP_K, k)
        
    # Convert query to embedding
    query_vector = embedding.encode(query)

    # Search in Qdrant
    client = vector_store.get_client()
    results = vector_store.search_vectors(client, query_vector, top_k=k)

    return results


def format_results(results: list[dict]) -> str:
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
        preview = payload.get("text", "")
        page = payload.get("page")
        chunk_idx = payload.get("chunk_index", "?")

        lines.append(f"  [{i}] Score: {score:.4f}")
        lines.append(f"      Source: {source}")
        if page is not None:
            lines.append(f"      Page: {page}")
        lines.append(f"      Chunk: #{chunk_idx}")
        lines.append(f"      Preview:")
        lines.append(f"        {preview}")
        lines.append(f"  {'-'*56}")

    return "\n".join(lines)

def format_results_json(results: list[dict]) -> list[dict]:
    """
    Format search results as structured JSON-friendly list of dicts.

    Each result dict contains:
    - index
    - score
    - source
    - page
    - chunk_index
    - text
    """
    formatted_results = []

    for i, result in enumerate(results, 1):
        payload = result.get("payload", {})
        formatted_result = {
            "index": i,
            "score": result.get("score", 0.0),
            "source": payload.get("source", "unknown"),
            "page": payload.get("page"),
            "chunk_index": payload.get("chunk_index", "?"),
            "text": payload.get("text", "")
        }
        formatted_results.append(formatted_result)

    return formatted_results



def main() -> None:
    """CLI entry point for semantic search."""
    parser = argparse.ArgumentParser(
        description="Ragent Semantic Search",
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
    print(format_results(results))


if __name__ == "__main__":
    main()
