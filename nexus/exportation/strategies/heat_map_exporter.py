from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig
from nexus.exportation.mixins import PlotMixin

if TYPE_CHECKING:
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


@dataclass
class HeatMapExporterConfig(ExportationStrategyConfig, PlotMixin):
    cbar: bool = True
    cmap: str = "viridis"


class HeatMapExporter(ExportationStrategy):
    def __init__(self, config: HeatMapExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        matrix_df = pd.DataFrame(
            [element.to_dict() for element in matrix_result.matrix]
        )
        matrix_df_pivot = matrix_df.pivot(
            index="row",
            columns="col",
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

        if self._config.tight_layout:
            plt.tight_layout()

        plt.savefig(self._config.output_file_path)
        plt.clf()

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        raise NotImplementedError(
            "HeatMapExporter does not support exporting vector results."
        )

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        raise NotImplementedError(
            "HeatMapExporter does not support exporting scalar results."
        )
