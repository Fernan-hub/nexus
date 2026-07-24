"""Tests for CSVExporter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nexus.analysis.results.matrix_result import MatrixElement, MatrixResult
from nexus.analysis.results.scalar_result import ScalarResult
from nexus.analysis.results.vector_result import VectorResult
from nexus.exportation.strategies.csv_exporter import CSVExporter, CSVExporterConfig


def _make_scalar_result() -> ScalarResult:
    return ScalarResult(algorithm="entropy", value=1.5, label="entropy")


def _make_vector_result() -> VectorResult:
    return VectorResult(
        algorithm="mutual_info",
        vector=pd.DataFrame({"mutual_info": [0.1, 0.2, 0.3]}),
    )


def _make_matrix_result() -> MatrixResult:
    return MatrixResult(
        algorithm="transfer_entropy",
        matrix=[
            MatrixElement(row="A", column="B", value=0.5),
            MatrixElement(row="A", column="A", value=0.0),
            MatrixElement(row="B", column="A", value=0.8),
            MatrixElement(row="B", column="B", value=0.0),
        ],
        row_label="from",
        col_label="to",
    )


class TestCSVExporter(unittest.TestCase):
    """Unit tests for CSVExporter."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch.object(CSVExporter, "_dataframe_to_csv")
    def test_export_scalar_result(self, mock_to_csv: MagicMock) -> None:
        """Test export_scalar_result wraps scalar in a single-row DataFrame."""
        exporter = CSVExporter(CSVExporterConfig())

        exporter.export_scalar_result(_make_scalar_result())

        mock_to_csv.assert_called_once()
        df_arg = mock_to_csv.call_args[0][0]
        self.assertIsInstance(df_arg, pd.DataFrame)
        self.assertEqual(list(df_arg.columns), ["entropy"])
        self.assertAlmostEqual(df_arg["entropy"].iloc[0], 1.5)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch.object(CSVExporter, "_dataframe_to_csv")
    def test_export_vector_result(self, mock_to_csv: MagicMock) -> None:
        """Test export_vector_result passes DataFrame directly to _dataframe_to_csv."""
        vector_result = _make_vector_result()
        exporter = CSVExporter(CSVExporterConfig())

        exporter.export_vector_result(vector_result)

        mock_to_csv.assert_called_once()
        df_arg = mock_to_csv.call_args[0][0]
        self.assertIs(df_arg, vector_result.vector)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch.object(CSVExporter, "_dataframe_to_csv")
    def test_export_matrix_result(self, mock_to_csv: MagicMock) -> None:
        """Test export_matrix_result passes a pivoted DataFrame with row/col names."""
        exporter = CSVExporter(CSVExporterConfig())

        exporter.export_matrix_result(_make_matrix_result())

        mock_to_csv.assert_called_once()
        df_arg = mock_to_csv.call_args[0][0]
        self.assertEqual(df_arg.index.name, "from/to")
        self.assertIsNone(df_arg.columns.name)
        self.assertAlmostEqual(df_arg.loc["A", "B"], 0.5)
        self.assertAlmostEqual(df_arg.loc["B", "A"], 0.8)


class TestIntegrationCSVExporter(unittest.TestCase):
    """Integration tests for CSVExporter with real file I/O."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_scalar_result(self) -> None:
        """Test export_scalar_result writes a single-row CSV with the correct value."""
        output_path = str(self.temp_path / "scalar.csv")
        exporter = CSVExporter(CSVExporterConfig(output_file_path=output_path))

        exporter.export_scalar_result(_make_scalar_result())

        df = pd.read_csv(output_path)
        self.assertEqual(list(df.columns), ["entropy"])
        self.assertAlmostEqual(df["entropy"].iloc[0], 1.5)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_vector_result(self) -> None:
        """Test that export_vector_result writes the vector DataFrame as CSV."""
        output_path = str(self.temp_path / "vector.csv")
        exporter = CSVExporter(CSVExporterConfig(output_file_path=output_path))

        exporter.export_vector_result(_make_vector_result())

        df = pd.read_csv(output_path)
        self.assertEqual(list(df.columns), ["mutual_info"])
        self.assertAlmostEqual(df["mutual_info"].iloc[0], 0.1)
        self.assertEqual(len(df), 3)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_matrix_result(self) -> None:
        """Test that export_matrix_result writes a pivoted CSV with row index."""
        output_path = str(self.temp_path / "matrix.csv")
        exporter = CSVExporter(CSVExporterConfig(output_file_path=output_path))

        exporter.export_matrix_result(_make_matrix_result())

        df = pd.read_csv(output_path, index_col=0)
        self.assertAlmostEqual(df.loc["A", "B"], 0.5)
        self.assertAlmostEqual(df.loc["B", "A"], 0.8)

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_uses_custom_delimiter(self) -> None:
        """Test that the configured delimiter is used in the output file."""
        output_path = str(self.temp_path / "semicolon.csv")
        exporter = CSVExporter(
            CSVExporterConfig(output_file_path=output_path, delimiter=";")
        )

        exporter.export_matrix_result(_make_matrix_result())

        content = Path(output_path).read_text(encoding="utf-8")
        self.assertIn(";", content)
        self.assertNotIn(",", content)
