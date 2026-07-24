"""Exportation strategy that renders a MatrixResult as a directed connectivity graph."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx

from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig
from nexus.exportation.mixins import PlotMixin

if TYPE_CHECKING:
    from nexus.analysis.results import MatrixResult


@dataclass
class ConnGraphExporterConfig(ExportationStrategyConfig, PlotMixin):
    """Configuration for ConnGraphExporter.

    Parameters
    ----------
    output_file_path : str or None, optional
        Destination path for the saved figure. Inherited from ExportationStrategyConfig.
    fig_title : str or None, optional
        Title rendered above the graph. Inherited from PlotMixin.
    fig_size : tuple of (float, float) or None, optional
        Figure dimensions in inches. Inherited from PlotMixin.
    x_label : str or None, optional
        x-axis label. Inherited from PlotMixin (rarely used for graph plots).
    y_label : str or None, optional
        y-axis label. Inherited from PlotMixin (rarely used for graph plots).
    tight_layout : bool, optional
        Whether to call plt.tight_layout() before saving. Defaults to True.
    threshold : float, optional
        Minimum matrix value required for an edge to be drawn. Defaults to 0.0.
    width_multiplier : float, optional
        Scales each edge's width by its matrix value. Defaults to 1.0.
    edge_color : str, optional
        Colour applied to all drawn edges. Defaults to "black".
    node_color : str, optional
        Colour applied to all nodes. Defaults to "lightblue".
    """

    threshold: float = 0.0
    width_multiplier: float = 1.0
    edge_color: str = "black"
    node_color: str = "lightblue"


class ConnGraphExporter(ExportationStrategy):
    """Renders a MatrixResult as a directed graph and saves the figure to disk.

    Edges are drawn only for matrix values that exceed the configured threshold.
    Edge width is proportional to the matrix value scaled by width_multiplier.

    Parameters
    ----------
    config : ConnGraphExporterConfig
        Configuration specifying graph appearance, threshold, and the output path.
    """

    def __init__(self, config: ConnGraphExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        """Render the matrix as a directed connectivity graph and save the figure.

        Parameters
        ----------
        matrix_result : MatrixResult
            The pairwise analysis result whose elements define the graph edges.
        """
        graph = nx.DiGraph()

        nodes = sorted(
            {e.row for e in matrix_result.matrix}
            | {e.column for e in matrix_result.matrix}
        )
        graph.add_nodes_from(nodes)

        edge_widths: list[float] = []
        for element in matrix_result.matrix:
            if element.value > self._config.threshold:
                graph.add_edge(element.row, element.column)
                edge_widths.append(element.value * self._config.width_multiplier)

        if self._config.fig_size is not None:
            plt.figure(figsize=self._config.fig_size)

        pos = nx.circular_layout(graph)
        nx.draw_networkx(
            graph,
            pos=pos,
            arrows=True,
            width=edge_widths,
            edge_color=self._config.edge_color,
            node_color=self._config.node_color,
            connectionstyle="arc3,rad=0.2",
        )

        if self._config.fig_title is not None:
            plt.title(self._config.fig_title)

        if self._config.tight_layout:
            plt.tight_layout()

        plt.savefig(self._get_output_file_path(matrix_result))
        plt.clf()
