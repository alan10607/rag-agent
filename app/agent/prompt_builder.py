"""
VectorSearcher - RAG Prompt Builder with MCP Tools

Constructs prompts that combine retrieved context chunks with the user's question,
following a standard RAG (Retrieval-Augmented Generation) pattern.
"""

import re
import json
from mcp.types import Tool
from app.logger import get_logger
from app.mcp import retrieval_tool
from typing import List

logger = get_logger(__name__)

_TOOL_MODULES = [retrieval_tool]

def _get_available_mcp_tools() -> List[Tool]:
    available_tools = []
    for module in _TOOL_MODULES:
        available_tools.extend(module.get_tools())
    return available_tools


def _detect_language(text: str) -> str:
    """Detect the dominant language of the text based on character distribution.

    Returns a human-readable language name (e.g. "Traditional Chinese",
    "English", "Japanese").
    """
    if not text:
        return "English"

    cjk = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))
    jp_only = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))  # hiragana + katakana
    kr = len(re.findall(r'[\uac00-\ud7af\u1100-\u11ff]', text))

    total = len(text)
    if total == 0:
        return "English"

    if jp_only > 0 and (jp_only + cjk) / total > 0.1:
        return "Japanese"
    if kr / total > 0.1:
        return "Korean"
    if cjk / total > 0.1:
        return "Traditional Chinese"

    return "English"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a knowledgeable assistant. Answer the user's question based on the \
provided reference materials below.

Rules:
1. Prioritize information from the reference materials and cite the source.
2. If the reference materials are insufficient, clearly state which parts \
come from the references and which are your own supplementation.
3. **CRITICAL language rule**: You MUST reply in {lang_name} because the \
user's question is written in {lang_name}. Do NOT switch to any other \
language regardless of the language used in the reference materials.
4. Keep the answer concise and well-structured.
"""

_MCP_TOOLS_TEMPLATE = """\
You must use the following MCP tools to get the information you need:
{mcp_tools}
"""

def build_prompt(question: str) -> str:
    """Build a RAG prompt from the user question and retrieved context chunks.

    Ensures all available MCP tools are included in the prompt.

    Args:
        question: The user's natural-language question.

    Returns:
        A fully assembled prompt string ready to be sent to the LLM.
    """
    lang_name = _detect_language(question)
    system_block = _SYSTEM_TEMPLATE.format(lang_name=lang_name)

    # Build MCP tools block dynamically
    tools = _get_available_mcp_tools()
    tools_str = "\n".join(
        f"- {t.name}: {t.description}\n  Schema: {json.dumps(t.inputSchema, indent=2)}" 
        for t in tools
    )
    mcp_tools_block = _MCP_TOOLS_TEMPLATE.format(mcp_tools=tools_str)

    prompt = f"{system_block}\n{mcp_tools_block}\nUser question: {question}\n"

    logger.info(
        "Built RAG prompt: question=%r, lang=%s, prompt_length=%d",
        question,
        lang_name,
        len(prompt),
    )
    return prompt