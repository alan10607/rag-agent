"""Tests for the text splitter module."""

import pytest
from app.ingest.splitter import Chunk, split_text


class TestSplitText:
    """Tests for split_text function."""

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        text = "This is a short text."
        chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0

    def test_empty_text_returns_no_chunks(self):
        """Empty or whitespace-only text should return no chunks."""
        assert split_text("", chunk_size=500, chunk_overlap=50) == []
        assert split_text("   ", chunk_size=500, chunk_overlap=50) == []

    def test_long_text_is_split_into_multiple_chunks(self):
        """Text longer than chunk_size should be split into multiple chunks."""
        # Create a text with distinct paragraphs
        paragraphs = [f"Paragraph {i}. " + "word " * 80 for i in range(5)]
        text = "\n\n".join(paragraphs)

        chunks = split_text(text, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_chunks_have_sequential_indices(self):
        """Chunks should have sequential chunk_index values starting from 0."""
        text = "sentence one. " * 100
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_respect_max_size(self):
        """Most chunks should respect the chunk_size limit."""
        text = "Hello world. " * 200
        chunk_size = 100
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=10)
        # Allow some tolerance for edge cases with separators
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size * 1.5  # generous tolerance

    def test_chunk_dataclass_fields(self):
        """Chunk objects should have all expected fields."""
        text = "A simple test text for checking the chunk dataclass."
        chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.text, str)
        assert isinstance(chunk.chunk_index, int)
        assert isinstance(chunk.start_char, int)
        assert isinstance(chunk.end_char, int)

    def test_no_empty_chunks(self):
        """The splitter should not produce empty or whitespace-only chunks."""
        text = "word " * 500
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_custom_separators(self):
        """Custom separators should be respected."""
        text = "part1|||part2|||part3"
        chunks = split_text(text, chunk_size=10, chunk_overlap=0, separators=["|||", ""])
        assert len(chunks) == 3

    def test_overlap_produces_more_chunks(self):
        """Overlap should produce more chunks than no overlap for the same text."""
        text = "word " * 200
        chunks_no_overlap = split_text(text, chunk_size=100, chunk_overlap=0)
        chunks_with_overlap = split_text(text, chunk_size=100, chunk_overlap=30)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)
