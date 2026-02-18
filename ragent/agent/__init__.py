"""
Ragent - LLM Agent Module

Provides RAG-based question answering using Cursor Agent CLI as the LLM backend.
Retrieves relevant context from Qdrant vector store and feeds it to the LLM.
"""

from ragent.agent.llm_agent import ask, main

__all__ = ["ask", "main"]
