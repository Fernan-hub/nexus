"""Tests for ExportationStrategy base class."""

import unittest
from unittest.mock import MagicMock

import pytest

from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig


class _ConcreteExporter(ExportationStrategy):
    pass


class TestExportationStrategy(unittest.TestCase):
    """Unit tests for ExportationStrategy base class default method behaviour."""

    def setUp(self) -> None:
        self.exporter = _ConcreteExporter(ExportationStrategyConfig())

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_scalar_result_raises_not_implemented(self) -> None:
        """Test that export_scalar_result raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.exporter.export_scalar_result(MagicMock())

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_vector_result_raises_not_implemented(self) -> None:
        """Test that export_vector_result raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.exporter.export_vector_result(MagicMock())

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_matrix_result_raises_not_implemented(self) -> None:
        """Test that export_matrix_result raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.exporter.export_matrix_result(MagicMock())
