"""Tests for the embedding module.

Note: These tests require the sentence-transformers model to be downloaded.
They are integration tests that validate the embedding pipeline works correctly.
"""

import pytest
from app.core.embedding import encode, encode_batch


class TestEncode:
    """Tests for the encode function."""

    def test_encode_returns_list_of_floats(self):
        """encode() should return a list of floats."""
        result = encode("Hello, world!")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_encode_returns_correct_dimension(self):
        """Embedding vector should have 384 dimensions (all-MiniLM-L6-v2)."""
        result = encode("Test sentence for dimension check.")
        assert len(result) == 384

    def test_similar_texts_have_high_similarity(self):
        """Semantically similar texts should produce similar embeddings."""
        vec1 = encode("The cat sat on the mat.")
        vec2 = encode("A feline rested on the rug.")
        vec3 = encode("Quantum physics and black holes.")

        # Cosine similarity helper
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x**2 for x in a) ** 0.5
            norm_b = sum(x**2 for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_similar = cosine_sim(vec1, vec2)
        sim_different = cosine_sim(vec1, vec3)

        # Similar sentences should have higher similarity
        assert sim_similar > sim_different

    def test_encode_empty_string(self):
        """Encoding an empty string should still return a valid vector."""
        result = encode("")
        assert isinstance(result, list)
        assert len(result) == 384


class TestEncodeBatch:
    """Tests for the encode_batch function."""

    def test_batch_encode_returns_correct_count(self):
        """Batch encoding should return one embedding per input text."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        results = encode_batch(texts)
        assert len(results) == 3

    def test_batch_encode_dimensions(self):
        """Each embedding in the batch should have 384 dimensions."""
        texts = ["Hello", "World"]
        results = encode_batch(texts)
        for vec in results:
            assert len(vec) == 384

    def test_batch_encode_matches_single_encode(self):
        """Batch encoding should produce the same results as individual encoding."""
        texts = ["The quick brown fox.", "Jumped over the lazy dog."]
        batch_results = encode_batch(texts)
        single_results = [encode(t) for t in texts]

        for batch_vec, single_vec in zip(batch_results, single_results):
            # Allow small floating point differences
            for b, s in zip(batch_vec, single_vec):
                assert abs(b - s) < 1e-5
