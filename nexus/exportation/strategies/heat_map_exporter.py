from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig
from nexus.exportation.mixins import PlotMixin

if TYPE_CHECKING:
    from nexus.analysis.results import MatrixResult


@dataclass
class HeatMapExporterConfig(ExportationStrategyConfig, PlotMixin):
    cbar: bool = True
    cmap: str = "viridis"


class HeatMapExporter(ExportationStrategy):
    def __init__(self, config: HeatMapExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        matrix_df_pivot = matrix_result.matrix_df.pivot(
            index="row",
            columns="column",
            values="value",
        )
        sns.heatmap(
            matrix_df_pivot,
            annot=True,
            cmap=self._config.cmap,
            cbar=self._config.cbar,
        )

        if self._config.fig_title is not None:
            plt.title(self._config.fig_title)

        plt.xlabel(
            self._config.x_label
            if self._config.x_label is not None
            else matrix_result.col_label
        )
        plt.ylabel(
            self._config.y_label
            if self._config.y_label is not None
            else matrix_result.row_label
        )

        if self._config.tight_layout:
            plt.tight_layout()

        plt.savefig(self._get_output_file_path(matrix_result))
        plt.clf()
