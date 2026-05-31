"""Tests for ScalarResult."""

import unittest
from unittest.mock import MagicMock

import pytest

from nexus.analysis.results.scalar_result import ScalarResult


class TestScalarResult(unittest.TestCase):
    """Unit tests for ScalarResult."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_accept(self) -> None:
        """Test that accept calls export_scalar_result on the exporter with self."""
        result = ScalarResult(algorithm="entropy", value=1.5, label="H")
        mock_exporter = MagicMock()

        result.accept(mock_exporter)

        mock_exporter.export_scalar_result.assert_called_once_with(result)
