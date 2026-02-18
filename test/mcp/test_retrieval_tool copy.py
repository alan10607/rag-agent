"""Unit test for MCP tool handlers"""

import asyncio
from app.mcp import retrieval_tool


async def test_semantic_search():
    """測試 semantic_search tool handler"""
    query = "Python MCP server"
    args = {"query": query, "top_k": 3}

    result = await retrieval_tool.handle_tool("semantic_search", args)

    print("=== Test Result ===")
    for item in result:
        print(f"[{item['type']}] {item.get('text', '')}")


async def test_unknown_tool():
    """Test unknown tool handler"""
    args = {}
    result = await retrieval_tool.handle_tool("unknown_tool", args)
    print("=== Unknown Tool Test ===")
    for item in result:
        print(f"[{item['type']}] {item.get('text', '')}")


if __name__ == "__main__":
    asyncio.run(test_semantic_search())
    asyncio.run(test_unknown_tool())
