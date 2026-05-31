"""Tests for ConnGraphExporter."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from nexus.analysis.results.matrix_result import MatrixElement, MatrixResult
from nexus.analysis.results.scalar_result import ScalarResult
from nexus.analysis.results.vector_result import VectorResult
from nexus.exportation.strategies.conn_graph_exporter import (
    ConnGraphExporter,
    ConnGraphExporterConfig,
)
from tests.utils import Expected, Given, Scenario


def _make_matrix_result() -> MatrixResult:
    return MatrixResult(
        algorithm="transfer_entropy",
        matrix=[
            MatrixElement(row="A", column="B", value=0.5),
            MatrixElement(row="B", column="A", value=0.3),
        ],
    )


class TestConnGraphExporter(unittest.TestCase):
    """Unit tests for ConnGraphExporter."""

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.conn_graph_exporter.plt")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.draw_networkx")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.circular_layout")
    def test_export_matrix_result_adds_all_nodes(
        self, _mock_layout: MagicMock, mock_draw: MagicMock, _mock_plt: MagicMock
    ) -> None:
        """Test that all distinct row and column values from the matrix are added as nodes."""
        matrix_result = MatrixResult(
            algorithm="te",
            matrix=[
                MatrixElement(row="A", column="B", value=0.5),
                MatrixElement(row="B", column="C", value=0.3),
            ],
        )
        exporter = ConnGraphExporter(
            ConnGraphExporterConfig(output_file_path="out.png")
        )

        exporter.export_matrix_result(matrix_result)

        graph = mock_draw.call_args[0][0]
        self.assertSetEqual(set(graph.nodes), {"A", "B", "C"})

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.conn_graph_exporter.plt")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.draw_networkx")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.circular_layout")
    def test_export_matrix_result_only_adds_edges_above_threshold(
        self, _mock_layout: MagicMock, mock_draw: MagicMock, _mock_plt: MagicMock
    ) -> None:
        """Test that only edges with value strictly greater than threshold are included."""
        scenarios = [
            Scenario(
                name="threshold below value: edge included",
                given=Given(data={"threshold": 0.0}),
                expected=Expected(data={"edges": {("A", "B")}}),
            ),
            Scenario(
                name="threshold above value: edge excluded",
                given=Given(data={"threshold": 0.5}),
                expected=Expected(data={"edges": set()}),
            ),
            Scenario(
                name="threshold equal to value: edge excluded",
                given=Given(data={"threshold": 0.3}),
                expected=Expected(data={"edges": set()}),
            ),
        ]

        matrix_result = MatrixResult(
            algorithm="te",
            matrix=[MatrixElement(row="A", column="B", value=0.3)],
        )

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_draw.reset_mock()
                exporter = ConnGraphExporter(
                    ConnGraphExporterConfig(
                        output_file_path="out.png",
                        threshold=scenario.given.data["threshold"],
                    )
                )

                exporter.export_matrix_result(matrix_result)

                graph = mock_draw.call_args[0][0]
                self.assertEqual(set(graph.edges), scenario.expected.data["edges"])

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.conn_graph_exporter.plt")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.draw_networkx")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.circular_layout")
    def test_export_matrix_result_edge_widths_scaled_by_multiplier(
        self, _mock_layout: MagicMock, mock_draw: MagicMock, _mock_plt: MagicMock
    ) -> None:
        """Test that edge widths equal element value times width_multiplier."""
        matrix_result = MatrixResult(
            algorithm="te",
            matrix=[
                MatrixElement(row="A", column="B", value=0.4),
                MatrixElement(row="B", column="A", value=0.6),
            ],
        )
        exporter = ConnGraphExporter(
            ConnGraphExporterConfig(output_file_path="out.png", width_multiplier=2.0)
        )

        exporter.export_matrix_result(matrix_result)

        _, draw_kwargs = mock_draw.call_args
        self.assertAlmostEqual(draw_kwargs["width"][0], 0.4 * 2.0)
        self.assertAlmostEqual(draw_kwargs["width"][1], 0.6 * 2.0)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.conn_graph_exporter.plt")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.draw_networkx")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.circular_layout")
    def test_export_matrix_result_uses_plt_configuration(
        self, _mock_layout: MagicMock, _mock_draw: MagicMock, mock_plt: MagicMock
    ) -> None:
        """Test that plt.figure, plt.title, and plt.tight_layout are called only when their config options are set."""
        scenarios = [
            Scenario(
                name="fig_size, title, and tight_layout enabled",
                given=Given(
                    data={
                        "fig_size": (10.0, 8.0),
                        "fig_title": "Graph Title",
                        "tight_layout": True,
                    }
                ),
                expected=Expected(
                    calls={
                        "figure": [call(figsize=(10.0, 8.0))],
                        "title": [call("Graph Title")],
                        "tight_layout": [call()],
                    }
                ),
            ),
            Scenario(
                name="fig_size, title, and tight_layout disabled",
                given=Given(
                    data={"fig_size": None, "fig_title": None, "tight_layout": False}
                ),
                expected=Expected(
                    calls={
                        "figure": [],
                        "title": [],
                        "tight_layout": [],
                    }
                ),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                mock_plt.reset_mock()
                exporter = ConnGraphExporter(
                    ConnGraphExporterConfig(
                        output_file_path="out.png",
                        fig_size=scenario.given.data["fig_size"],
                        fig_title=scenario.given.data["fig_title"],
                        tight_layout=scenario.given.data["tight_layout"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                mock_plt.figure.assert_has_calls(scenario.expected.calls["figure"])
                mock_plt.title.assert_has_calls(scenario.expected.calls["title"])
                mock_plt.tight_layout.assert_has_calls(
                    scenario.expected.calls["tight_layout"]
                )

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_scalar_result_raises_not_implemented(self) -> None:
        """Test that export_scalar_result raises NotImplementedError."""
        exporter = ConnGraphExporter(
            ConnGraphExporterConfig(output_file_path="out.png")
        )
        scalar_result = ScalarResult(algorithm="entropy", value=1.0, label="H")

        with self.assertRaises(NotImplementedError):
            exporter.export_scalar_result(scalar_result)

    @pytest.mark.unit
    @pytest.mark.strategy
    def test_export_vector_result_raises_not_implemented(self) -> None:
        """Test that export_vector_result raises NotImplementedError."""
        exporter = ConnGraphExporter(
            ConnGraphExporterConfig(output_file_path="out.png")
        )
        vector_result = VectorResult(
            algorithm="mutual_info", vector=pd.DataFrame({"mi": [0.1]})
        )

        with self.assertRaises(NotImplementedError):
            exporter.export_vector_result(vector_result)

    @pytest.mark.unit
    @pytest.mark.strategy
    @patch("nexus.exportation.strategies.conn_graph_exporter.plt")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.draw_networkx")
    @patch("nexus.exportation.strategies.conn_graph_exporter.nx.circular_layout")
    def test_export_matrix_result_saves_and_clears_figure(
        self, _mock_layout: MagicMock, _mock_draw: MagicMock, mock_plt: MagicMock
    ) -> None:
        """Test that export_matrix_result saves to the configured path and clears the figure."""
        exporter = ConnGraphExporter(
            ConnGraphExporterConfig(output_file_path="graph.png")
        )

        exporter.export_matrix_result(_make_matrix_result())

        mock_plt.savefig.assert_called_once_with("graph.png")
        mock_plt.clf.assert_called_once()


class TestIntegrationConnGraphExporter(unittest.TestCase):
    """Integration tests for ConnGraphExporter with real file I/O and libraries."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @pytest.mark.integration
    @pytest.mark.strategy
    def test_export_matrix_result(self) -> None:
        """Test that export_matrix_result writes a non-empty file with and without edges above threshold."""
        scenarios = [
            Scenario(
                name="with edges above threshold",
                given=Given(data={"filename": "graph.png", "threshold": 0.0}),
                expected=Expected(data={"file_exists": True}),
            ),
            Scenario(
                name="with no edges above threshold",
                given=Given(data={"filename": "graph_empty.png", "threshold": 1.0}),
                expected=Expected(data={"file_exists": True}),
            ),
        ]

        for scenario in scenarios:
            with self.subTest(msg=scenario.name):
                output_path = str(self.temp_path / scenario.given.data["filename"])
                exporter = ConnGraphExporter(
                    ConnGraphExporterConfig(
                        output_file_path=output_path,
                        threshold=scenario.given.data["threshold"],
                    )
                )

                exporter.export_matrix_result(_make_matrix_result())

                self.assertEqual(
                    Path(output_path).exists(), scenario.expected.data["file_exists"]
                )
                self.assertGreater(Path(output_path).stat().st_size, 0)
