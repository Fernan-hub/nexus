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
    threshold: float = 0.0
    width_multiplier: float = 1.0
    edge_color: str = "black"
    node_color: str = "lightblue"


class ConnGraphExporter(ExportationStrategy):
    def __init__(self, config: ConnGraphExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
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
