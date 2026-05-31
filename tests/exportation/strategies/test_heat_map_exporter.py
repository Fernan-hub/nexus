"""Tests for HeatMapExporter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from nexus.analysis.results.matrix_result import MatrixElement, MatrixResult
from nexus.analysis.results.scalar_result import ScalarResult
from nexus.analysis.results.vector_result import VectorResult
from nexus.exportation.strategies.heat_map_exporter import (
    HeatMapExporter,
    HeatMapExporterConfig,
)
from tests.utils import Expected, Given, Scenario


def _make_matrix_result(row_label: str = "from", col_label: str = "to") -> MatrixResult:
    return MatrixResult(
        algorithm="transfer_entropy",
        matrix=[
            MatrixElement(row="A", column="B", value=0.5),
            MatrixElement(row="B", column="A", value=0.3),
        ],
        row_label=row_label,
        col_label=col_label,
    )


class TestHeatMapExporter(unittest.TestCase):
    """Unit tests for HeatMapExporter."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.heat_map_exporter.plt")
    @patch("nexus.exportation.strategies.heat_map_exporter.sns")
    def test_export_matrix_result_calls_heatmap_with_pivot(
        self, mock_sns: MagicMock, _mock_plt: MagicMock
    ) -> None:
        """Test that sns.heatmap receives the pivot of the matrix DataFrame."""
        matrix_result = _make_matrix_result()
        exporter = HeatMapExporter(HeatMapExporterConfig(output_file_path="out.png"))

        exporter.export_matrix_result(matrix_result)

        mock_sns.heatmap.assert_called_once()
        df_arg = mock_sns.heatmap.call_args[0][0]
        self.assertAlmostEqual(df_arg.loc["A", "B"], 0.5)
        self.assertAlmostEqual(df_arg.loc["B", "A"], 0.3)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.heat_map_exporter.plt")
    @patch("nexus.exportation.strategies.heat_map_exporter.sns")
    def test_export_matrix_result_passes_cmap_and_cbar_to_heatmap(
        self, mock_sns: MagicMock, _mock_plt: MagicMock
    ) -> None:
        """Test that cmap and cbar from config are forwarded to sns.heatmap."""
        scenarios = [
            Scenario(
                name="default viridis with cbar",
                given=Given(data={"cmap": "viridis", "cbar": True}),
                expected=Expected(data={"cmap": "viridis", "cbar": True}),
            ),
            Scenario(
                name="custom coolwarm without cbar",
                given=Given(data={"cmap": "coolwarm", "cbar": False}),
                expected=Expected(data={"cmap": "coolwarm", "cbar": False}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_sns.reset_mock()
                exporter = HeatMapExporter(
                    HeatMapExporterConfig(
                        output_file_path="out.png",
                        cmap=scenario.given.data["cmap"],
                        cbar=scenario.given.data["cbar"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                _, heatmap_kwargs = mock_sns.heatmap.call_args
                self.assertEqual(heatmap_kwargs["cmap"], scenario.expected.data["cmap"])
                self.assertEqual(heatmap_kwargs["cbar"], scenario.expected.data["cbar"])

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.heat_map_exporter.plt")
    @patch("nexus.exportation.strategies.heat_map_exporter.sns")
    def test_export_matrix_result_axis_labels(
        self, _mock_sns: MagicMock, mock_plt: MagicMock
    ) -> None:
        """Test that axis labels come from config when set, or fall back to result's row/col labels."""
        scenarios = [
            Scenario(
                name="labels from config",
                given=Given(data={"x_label": "Target", "y_label": "Source"}),
                expected=Expected(data={"x_label": "Target", "y_label": "Source"}),
            ),
            Scenario(
                name="labels fall back to result",
                given=Given(data={"x_label": None, "y_label": None}),
                expected=Expected(data={"x_label": "to", "y_label": "from"}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_plt.reset_mock()
                exporter = HeatMapExporter(
                    HeatMapExporterConfig(
                        output_file_path="out.png",
                        x_label=scenario.given.data["x_label"],
                        y_label=scenario.given.data["y_label"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                mock_plt.xlabel.assert_called_once_with(
                    scenario.expected.data["x_label"]
                )
                mock_plt.ylabel.assert_called_once_with(
                    scenario.expected.data["y_label"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.heat_map_exporter.plt")
    @patch("nexus.exportation.strategies.heat_map_exporter.sns")
    def test_export_matrix_result_uses_plt_configuration(
        self, _mock_sns: MagicMock, mock_plt: MagicMock
    ) -> None:
        """Test that plt.title and plt.tight_layout are called only when their config options are set."""
        scenarios = [
            Scenario(
                name="title and tight_layout enabled",
                given=Given(data={"fig_title": "My Chart", "tight_layout": True}),
                expected=Expected(
                    calls={
                        "title": [call("My Chart")],
                        "tight_layout": [call()],
                    }
                ),
            ),
            Scenario(
                name="title and tight_layout disabled",
                given=Given(data={"fig_title": None, "tight_layout": False}),
                expected=Expected(
                    calls={
                        "title": [],
                        "tight_layout": [],
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_plt.reset_mock()
                exporter = HeatMapExporter(
                    HeatMapExporterConfig(
                        output_file_path="out.png",
                        fig_title=scenario.given.data["fig_title"],
                        tight_layout=scenario.given.data["tight_layout"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                mock_plt.title.assert_has_calls(scenario.expected.calls["title"])
                mock_plt.tight_layout.assert_has_calls(
                    scenario.expected.calls["tight_layout"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_scalar_result_raises_not_implemented(self) -> None:
        """Test that export_scalar_result raises NotImplementedError."""
        exporter = HeatMapExporter(HeatMapExporterConfig(output_file_path="out.png"))
        scalar_result = ScalarResult(algorithm="entropy", value=1.0, label="H")

        with self.assertRaises(NotImplementedError):
            exporter.export_scalar_result(scalar_result)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_vector_result_raises_not_implemented(self) -> None:
        """Test that export_vector_result raises NotImplementedError."""
        exporter = HeatMapExporter(HeatMapExporterConfig(output_file_path="out.png"))
        vector_result = VectorResult(
            algorithm="mutual_info", vector=pd.DataFrame({"mi": [0.1]})
        )

        with self.assertRaises(NotImplementedError):
            exporter.export_vector_result(vector_result)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.heat_map_exporter.plt")
    @patch("nexus.exportation.strategies.heat_map_exporter.sns")
    def test_export_matrix_result_saves_and_clears_figure(
        self, _mock_sns: MagicMock, mock_plt: MagicMock
    ) -> None:
        """Test that export_matrix_result saves to the configured path and clears the figure."""
        exporter = HeatMapExporter(HeatMapExporterConfig(output_file_path="out.png"))

        exporter.export_matrix_result(_make_matrix_result())

        mock_plt.savefig.assert_called_once_with("out.png")
        mock_plt.clf.assert_called_once()


class TestIntegrationHeatMapExporter(unittest.TestCase):
    """Integration tests for HeatMapExporter with real file I/O and libraries."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_matrix_result(self) -> None:
        """Test that export_matrix_result writes a non-empty file for minimal and fully-configured exporters."""
        scenarios = [
            Scenario(
                name="minimal config",
                given=Given(data={"filename": "heatmap.png", "extra_config": {}}),
                expected=Expected(data={"file_exists": True}),
            ),
            Scenario(
                name="all config options",
                given=Given(
                    data={
                        "filename": "heatmap_full.png",
                        "extra_config": {
                            "fig_title": "Test Chart",
                            "x_label": "Target",
                            "y_label": "Source",
                            "tight_layout": True,
                            "cmap": "coolwarm",
                            "cbar": False,
                        },
                    }
                ),
                expected=Expected(data={"file_exists": True}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                output_path = str(self.temp_path / scenario.given.data["filename"])
                exporter = HeatMapExporter(
                    HeatMapExporterConfig(
                        output_file_path=output_path,
                        **scenario.given.data["extra_config"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                self.assertEqual(
                    Path(output_path).exists(), scenario.expected.data["file_exists"]
                )
                self.assertGreater(Path(output_path).stat().st_size, 0)
