import pytest
import json
from ragent.agent.cli_runner import (
    MCPToolResult, 
    SemanticSearchResult,
)

# ---------------------------------------------------------------------------
# Tests: MCPToolResult
# ---------------------------------------------------------------------------

class TestMCPToolResult:
    """Tests for MCPToolResult dataclass and its factory method."""

    def test_returns_dataclass_instance(self):
        result = MCPToolResult.from_event({})
        assert isinstance(result, MCPToolResult)

    def test_extracts_nested_content_from_list(self):
        event = {
            "type": "tool_call",
            "subtype": "completed",
            "tool_call": {
                "mcpToolCall": {
                    "args": {
                        "name": "ragent-semantic_search",
                        "args": {
                            "query": "my-query",
                            "top_k": 5
                        }
                    },
                    "result": {
                        "success": {
                            "content": [
                                {
                                    "text": {
                                        "text": "hello world"
                                    }
                                }
                            ],
                            "isError": False
                        }
                    }
                }
            },
            "timestamp_ms": 123456789
        }

        result = MCPToolResult.from_event(event)

        assert result.tool_name == "ragent-semantic_search"
        assert result.tool_args == {"query": "my-query", "top_k": 5}
        assert result.success is True  # success is not isError
        assert result.timestamp_ms == 123456789
        assert result.raw_text == "hello world"
        assert result.is_semantic_search is True

    def test_defaults_when_missing_fields(self):
        result = MCPToolResult.from_event({})

        assert result.tool_name == "unknown"
        assert result.tool_args == {}
        assert result.success is True
        assert result.timestamp_ms == 0
        assert result.raw_text == ""
        assert result.is_semantic_search is False

    def test_handles_empty_content_list(self):
        event = {
            "tool_call": {
                "mcpToolCall": {
                    "result": {
                        "success": {
                            "content": []
                        }
                    }
                }
            }
        }
        result = MCPToolResult.from_event(event)
        assert result.raw_text == ""


# ---------------------------------------------------------------------------
# Tests: SemanticSearchResult
# ---------------------------------------------------------------------------

class TestSemanticSearchResult:
    """Tests for SemanticSearchResult parsing logic."""

    def test_parses_real_mcp_format(self):
        raw_text = json.dumps({
            "content": [
                {
                    "type": "text",
                    "text": [
                        {
                            "index": 1,
                            "score": 0.8682998,
                            "source": "test.txt",
                            "page": None,
                            "chunk_index": 1,
                            "text": "hello world"
                        }
                    ]
                }
            ]
        })

        # Encapsulate into MCPToolResult to test conversion
        mcp_result = MCPToolResult(
            tool_name="ragent-semantic_search",
            tool_args={},
            success=True,
            timestamp_ms=123,
            raw_text=raw_text
        )

        results = mcp_result.to_semantic_results()

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, SemanticSearchResult)
        assert item.index == 1
        assert item.score == 0.8682998
        assert item.source == "test.txt"
        assert item.page is None
        assert item.chunk_index == 1
        assert item.text == "hello world"

    def test_invalid_json_returns_empty_list(self):
        mcp_result = MCPToolResult(
            tool_name="ragent-semantic_search",
            tool_args={},
            success=True,
            timestamp_ms=123,
            raw_text="not json"
        )
        assert mcp_result.to_semantic_results() == []

    def test_missing_content_returns_empty_list(self):
        mcp_result = MCPToolResult(
            tool_name="ragent-semantic_search",
            tool_args={},
            success=True,
            timestamp_ms=123,
            raw_text=json.dumps({})
        )
        assert mcp_result.to_semantic_results() == []