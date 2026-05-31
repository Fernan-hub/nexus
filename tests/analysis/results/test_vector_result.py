"""Tests for VectorResult."""

import unittest
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nexus.analysis.results.vector_result import VectorResult


class TestVectorResult(unittest.TestCase):
    """Unit tests for VectorResult."""

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_accept(self) -> None:
        """Test that accept calls export_vector_result on the exporter with self."""
        result = VectorResult(
            algorithm="mutual_info",
            vector=pd.DataFrame({"mi": [0.1, 0.2]}),
        )
        mock_exporter = MagicMock()

        result.accept(mock_exporter)

        mock_exporter.export_vector_result.assert_called_once_with(result)
