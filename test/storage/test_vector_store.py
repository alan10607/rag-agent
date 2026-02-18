"""Tests for the vector store wrapper (app.storage.vector_store).

Note: Tests that require a running Qdrant instance are marked with
pytest.mark.integration and skipped by default.
"""

import pytest

from ragent.storage import vector_store


class TestGetClient:
    """Tests for get_client function."""

    @pytest.mark.integration
    def test_get_client_returns_client(self):
        """get_client should return a QdrantClient instance."""
        from qdrant_client import QdrantClient

        client = vector_store.get_client()
        assert isinstance(client, QdrantClient)


class TestEnsureCollection:
    """Tests for ensure_collection function."""

    @pytest.mark.integration
    def test_ensure_collection_creates_or_exists(self):
        """ensure_collection should not raise when called."""
        client = vector_store.get_client()
        # Should not raise
        vector_store.ensure_collection(client, collection_name="test_collection")


class TestSearchVectors:
    """Tests for search_vectors function."""

    @pytest.mark.integration
    def test_search_returns_list(self):
        """search_vectors should return a list."""
        client = vector_store.get_client()
        vector_store.ensure_collection(client, collection_name="test_collection")
        results = vector_store.search_vectors(
            client,
            query_vector=[0.0] * 384,
            top_k=3,
            collection_name="test_collection",
        )
        assert isinstance(results, list)
