"""Tests for MatrixResult."""

import unittest
from unittest.mock import MagicMock

import pytest

from nexus.analysis.results.matrix_result import MatrixElement, MatrixResult


class TestMatrixResult(unittest.TestCase):
    """Unit tests for MatrixResult."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_accept(self) -> None:
        """Test that accept calls export_matrix_result on the exporter with self."""
        result = MatrixResult(
            algorithm="transfer_entropy",
            matrix=[MatrixElement(row="A", column="B", value=0.5)],
        )
        mock_exporter = MagicMock()

        result.accept(mock_exporter)

        mock_exporter.export_matrix_result.assert_called_once_with(result)
