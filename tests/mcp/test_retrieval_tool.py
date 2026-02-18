"""Unit test for MCP tool handlers"""

import pytest
import asyncio

from ragent.mcp import retrieval_tool


class TestRetrievalTool:
    """Unit tests for app.mcp.retrieval_tool."""

    @pytest.mark.asyncio
    async def test_semantic_search_returns_text(self):
        """semantic_search should return a text dict with results."""
        args = {"query": "test query", "top_k": 2}
        result = await retrieval_tool.handle_tool("semantic_search", args)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert isinstance(result[0]["text"], list)
        assert len(result[0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_text(self):
        """Calling an unknown tool should return an error message."""
        result = await retrieval_tool.handle_tool("nonexistent_tool", {})
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "Unknown tool" in result[0]["text"]

    @pytest.mark.asyncio
    async def test_missing_query_raises_keyerror(self):
        """Missing 'query' in args should be caught and returned as error text."""
        result = await retrieval_tool.handle_tool("semantic_search", {})
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "Error in semantic_search" in result[0]["text"]

    def test_get_tools_returns_semantic_search_tool(self):
        """get_tools() should return a list with semantic_search Tool object."""
        tools = retrieval_tool.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name == "semantic_search"
        assert "Perform semantic search" in tool.description
        # Check inputSchema
        schema = tool.inputSchema
        assert "query" in schema["properties"]
        assert "top_k" in schema["properties"]
        assert "query" in schema["required"]
