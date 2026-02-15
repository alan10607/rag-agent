"""Tests for the text splitter module (app.indexing.splitter)."""

import pytest
from app.indexing.splitter import Chunk, merge_small_chunks, split_text


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

    def test_chinese_separators_kept(self):
        """Chinese punctuation should be preserved in chunk text."""
        text = "第一句話。第二句話。第三句話。" * 20
        chunks = split_text(text, chunk_size=50, chunk_overlap=10, min_chunk_size=0)
        for chunk in chunks:
            # Chunks should not start with a separator
            assert not chunk.text.startswith("。")

    def test_no_near_duplicate_chunks(self):
        """Overlap should not produce near-identical chunks."""
        # Use unique sentences so prefix comparison is meaningful
        text = "Short Title\n\n" + " ".join(
            f"Sentence number {i} with unique content here." for i in range(30)
        )
        chunks = split_text(text, chunk_size=200, chunk_overlap=80, min_chunk_size=0)
        for i in range(len(chunks) - 1):
            # No two consecutive chunks should share > 80% prefix
            shorter = min(len(chunks[i].text), len(chunks[i + 1].text))
            overlap_len = 0
            for c1, c2 in zip(chunks[i].text, chunks[i + 1].text):
                if c1 == c2:
                    overlap_len += 1
                else:
                    break
            if shorter > 0:
                assert overlap_len / shorter < 0.8, (
                    f"Chunks #{i} and #{i+1} are near-duplicates "
                    f"({overlap_len}/{shorter} = {overlap_len/shorter:.0%} prefix overlap)"
                )

    def test_small_chunks_merged_with_neighbor(self):
        """Chunks below min_chunk_size should be merged with neighbors."""
        # Title (short) + long paragraph + title (short) + long paragraph
        text = (
            "Title One\n\n"
            + "Long paragraph with enough content to fill a chunk. " * 5
            + "\n\nTitle Two\n\n"
            + "Another long paragraph with enough content. " * 5
        )
        chunks = split_text(text, chunk_size=300, chunk_overlap=40, min_chunk_size=50)
        for chunk in chunks:
            assert len(chunk.text) >= 50, (
                f"Chunk too small ({len(chunk.text)} chars): '{chunk.text[:60]}...'"
            )

    def test_min_chunk_size_zero_disables_merge(self):
        """Setting min_chunk_size=0 should skip merging."""
        text = "Hi\n\n" + "word " * 100
        chunks_merged = split_text(text, chunk_size=200, chunk_overlap=20, min_chunk_size=50)
        chunks_raw = split_text(text, chunk_size=200, chunk_overlap=20, min_chunk_size=0)
        # With merging, small 'Hi' should be absorbed → fewer chunks
        assert len(chunks_merged) <= len(chunks_raw)

    def test_heading_plus_body_markdown_style(self):
        """Simulate Markdown: short headings should merge into body text."""
        text = (
            "Introduction\n\n"
            "Vector search represents data as high-dimensional vectors. "
            "It enables semantic matching beyond simple keyword lookup. "
            "This approach has transformed information retrieval.\n\n"
            "How It Works\n\n"
            "First, text is converted into embeddings using a neural model. "
            "Then, similarity is measured using distance metrics like cosine. "
            "The closest vectors are returned as search results."
        )
        chunks = split_text(text, chunk_size=300, chunk_overlap=40, min_chunk_size=50)
        # No chunk should be just a heading
        for chunk in chunks:
            assert len(chunk.text) >= 50


# ---------------------------------------------------------------------------
# Tests: merge_small_chunks
# ---------------------------------------------------------------------------

class TestMergeSmallChunks:
    """Tests for merge_small_chunks function."""

    def test_no_change_when_all_large(self):
        chunks = ["a" * 100, "b" * 100]
        assert merge_small_chunks(chunks, min_size=50, max_size=300) == chunks

    def test_small_chunk_merged_into_previous(self):
        result = merge_small_chunks(["Large chunk here" * 5, "Hi"], min_size=50, max_size=300)
        assert len(result) == 1
        assert "Hi" in result[0]

    def test_first_small_chunk_merged_into_next(self):
        result = merge_small_chunks(["Hi", "Large chunk here" * 5], min_size=50, max_size=300)
        assert len(result) == 1
        assert "Hi" in result[0]

    def test_respects_max_size(self):
        """Should not merge if combined would exceed max_size."""
        result = merge_small_chunks(["a" * 280, "Hi"], min_size=50, max_size=300)
        # 280 + 1 (newline) + 2 = 283 <= 300, should merge
        assert len(result) == 1
        result2 = merge_small_chunks(["a" * 299, "Hi"], min_size=50, max_size=300)
        # 299 + 1 + 2 = 302 > 300, should not merge
        assert len(result2) == 2

    def test_empty_input(self):
        assert merge_small_chunks([], min_size=50, max_size=300) == []

    def test_min_size_zero_returns_unchanged(self):
        chunks = ["Hi", "There"]
        assert merge_small_chunks(chunks, min_size=0, max_size=300) == chunks

    def test_single_small_chunk_unchanged(self):
        result = merge_small_chunks(["Hi"], min_size=50, max_size=300)
        assert result == ["Hi"]

    def test_consecutive_small_chunks_merged(self):
        result = merge_small_chunks(
            ["Title", "Subtitle", "Long body text " * 10],
            min_size=50, max_size=300,
        )
        # "Title" merges into [], then stays as result[0]
        # "Subtitle" merges into result[0] ("Title")
        # "Long body" tries merge into "Title\nSubtitle" (27 chars < 50) → merge
        # Actually "Title\nSubtitle" is 14 chars < 50, next chunk is 150 chars
        # 14+1+150 = 165 < 300, so it merges
        assert len(result) == 1
        assert "Title" in result[0]
        assert "Subtitle" in result[0]
