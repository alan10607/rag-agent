"""Tests for the configuration module (app.config)."""

import os

import pytest

from ragent import config


class TestConfigValues:
    """Tests for configuration value loading."""

    def test_qdrant_host_is_string(self):
        assert isinstance(config.QDRANT_HOST, str)
        assert len(config.QDRANT_HOST) > 0

    def test_qdrant_port_is_int(self):
        assert isinstance(config.QDRANT_PORT, int)
        assert config.QDRANT_PORT > 0

    def test_collection_name_is_string(self):
        assert isinstance(config.COLLECTION_NAME, str)
        assert len(config.COLLECTION_NAME) > 0

    def test_embedding_model_name_is_string(self):
        assert isinstance(config.EMBEDDING_MODEL_NAME, str)
        assert len(config.EMBEDDING_MODEL_NAME) > 0

    def test_vector_size_is_positive_int(self):
        assert isinstance(config.VECTOR_SIZE, int)
        assert config.VECTOR_SIZE > 0

    def test_default_top_k_is_positive_int(self):
        assert isinstance(config.DEFAULT_TOP_K, int)
        assert config.DEFAULT_TOP_K > 0

    def test_chunk_size_is_positive_int(self):
        assert isinstance(config.CHUNK_SIZE, int)
        assert config.CHUNK_SIZE > 0

    def test_chunk_overlap_is_non_negative_int(self):
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert config.CHUNK_OVERLAP >= 0

    def test_chunk_overlap_less_than_chunk_size(self):
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

    def test_data_dir_is_string(self):
        assert isinstance(config.DATA_DIR, str)

    def test_supported_extensions_contains_pdf_and_txt(self):
        assert ".pdf" in config.SUPPORTED_EXTENSIONS
        assert ".txt" in config.SUPPORTED_EXTENSIONS

    def test_upsert_batch_size_is_positive(self):
        assert isinstance(config.UPSERT_BATCH_SIZE, int)
        assert config.UPSERT_BATCH_SIZE > 0

    def test_log_level_is_valid(self):
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert config.LOG_LEVEL in valid_levels

    def test_log_level_int_is_int(self):
        assert isinstance(config.LOG_LEVEL_INT, int)
