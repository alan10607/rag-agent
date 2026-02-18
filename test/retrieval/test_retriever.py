"""Tests for the search pipeline (app.search.search)."""

import pytest

from app.retrieval.retriever import format_results


class TestFormatResults:
    """Tests for format_results function."""

    def test_no_results_message(self):
        """Empty results should return a 'no results' message."""
        output = format_results([])
        assert "No results found" in output

    def test_single_result_formatted(self):
        """A single result should be formatted with score and source."""
        results = [
            {
                "id": "abc-123",
                "score": 0.9512,
                "payload": {
                    "text": "This is a test chunk.",
                    "source": "test.txt",
                    "chunk_index": 0,
                    "page": None,
                },
            }
        ]
        output = format_results(results)
        assert "0.9512" in output
        assert "test.txt" in output
        assert "This is a test chunk." in output

    def test_multiple_results_all_shown(self):
        """Multiple results should all appear in the output."""
        results = [
            {
                "id": f"id-{i}",
                "score": 0.9 - i * 0.1,
                "payload": {
                    "text": f"Chunk {i} text.",
                    "source": f"file{i}.txt",
                    "chunk_index": i,
                    "page": None,
                },
            }
            for i in range(3)
        ]
        output = format_results(results)
        assert "3 result(s)" in output
        for i in range(3):
            assert f"file{i}.txt" in output

    def test_page_number_shown_when_present(self):
        """Page number should appear when provided in payload."""
        results = [
            {
                "id": "id-1",
                "score": 0.85,
                "payload": {
                    "text": "PDF chunk.",
                    "source": "doc.pdf",
                    "chunk_index": 5,
                    "page": 3,
                },
            }
        ]
        output = format_results(results)
        assert "Page: 3" in output

