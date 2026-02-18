"""
Retrieval tools for MCP Server
Provides semantic search integration.
"""

from typing import Dict, Any, List
from mcp.types import Tool

from ragent.retrieval.retriever import search, format_results_json
from ragent.logger import get_logger

logger = get_logger(__name__)


def _semantic_search(query: str, top_k: int | None = None) -> str:
    """Perform semantic search in Qdrant vector database."""
    results = search(query, top_k=top_k)
    return format_results_json(results)


async def handle_tool(
    name: str, args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Handle retrieval tool calls."""
    try:
        if name == "semantic_search":
            result = _semantic_search(args["query"], args.get("top_k"))
            return [{"type": "text", "text": result}]
        else:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

    except Exception as e:
        logger.error(
            "Retrieval tool error - tool=%s error=%s",
            name,
            str(e),
        )
        return [{"type": "text", "text": f"Error in {name}: {str(e)}"}]

def get_tools() -> List[Tool]:
    """Return list of available retrieval tools."""
    return [
        Tool(
            name="semantic_search",
            description="Perform semantic search in Qdrant vector database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text query to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
    ]
