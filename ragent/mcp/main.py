"""
Ragent MCP Server

Production-ready MCP server
Supports future HTTP-based tools
"""

import asyncio
from typing import Any, Dict, Callable, Awaitable, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool

from ragent.mcp import retrieval_tool
from ragent.logger import get_logger, setup_logging


setup_logging(module="mcp")
logger = get_logger(__name__)

server = Server("Ragent MCP Server")
TOOL_MODULES = [retrieval_tool]
TOOL_DEFINITIONS: List[Tool] = []
TOOL_HANDLERS: Dict[str, Callable[[str, dict], Awaitable[List[dict]]]] = {}


# ==========================================================
# Tool Registration
# ==========================================================
def register_tools():
    """Register all tools into registry."""
    for module in TOOL_MODULES:
        logger.info("Registering tools from module: %s", module.__name__)
        for tool in module.get_tools():
            logger.info("Registering tool: %s", tool.name)
            TOOL_DEFINITIONS.append(tool)
            TOOL_HANDLERS[tool.name] = module.handle_tool


# ==========================================================
# MCP List Tools
# ==========================================================
@server.list_tools()
async def list_tools():
    return TOOL_DEFINITIONS


# ==========================================================
# MCP Call Tool
# ==========================================================
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.info("Tool called: %s", name)

    if name not in TOOL_HANDLERS:
        return {
            "content": [
                {"type": "text", "text": f"Unknown tool: {name}"}
            ]
        }

    try:
        handler = TOOL_HANDLERS[name]

        if asyncio.iscoroutinefunction(handler):
            result = await handler(name, arguments)
        else:
            result = handler(name, arguments)

        return {"content": result}

    except Exception as e:
        logger.exception("Tool execution failed: %s", name)
        return {
            "content": [
                {"type": "text", "text": f"Error executing tool {name}: {str(e)}"}
            ]
        }



# ==========================================================
# MCP Server Startup
# ==========================================================
async def main():
    logger.info("Starting Ragent MCP Server...")

    # Register tools before server startup
    register_tools()
    logger.info("Tools registered: %s", TOOL_DEFINITIONS)
    logger.info("Tool handlers: %s", TOOL_HANDLERS)

    # async with stdio_server() as (reader, writer):
    #     # MCP v0.5+ must provide initialization_options
    #     await server.run(reader, writer, initialization_options={})


    async with stdio_server() as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )

if __name__ == "__main__":
    try:
        logger.info("Starting MCP Server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
