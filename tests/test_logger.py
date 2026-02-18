"""Tests for the logger module (app.logger)."""

import logging
import os

import pytest

from ragent.logger import get_logger, setup_logging


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_instance(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        logger = get_logger("my.custom.name")
        assert logger.name == "my.custom.name"

    def test_different_names_return_different_loggers(self):
        logger_a = get_logger("module.a")
        logger_b = get_logger("module.b")
        assert logger_a is not logger_b


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_does_not_raise(self):
        """setup_logging should not raise any exceptions."""
        setup_logging(module=None, log_to_file=False, log_to_console=False)

    def test_setup_with_custom_level(self):
        """setup_logging with a custom level should set the root logger level."""
        setup_logging(level=logging.WARNING, log_to_file=False, log_to_console=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING

