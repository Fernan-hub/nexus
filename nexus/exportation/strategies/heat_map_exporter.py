"""Exportation strategy that renders a MatrixResult as a seaborn heat map."""

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
    """Configuration for HeatMapExporter.

    Parameters
    ----------
    output_file_path : str or None, optional
        Destination path for the saved figure. Inherited from ExportationStrategyConfig.
    fig_title : str or None, optional
        Title rendered above the heat map. Inherited from PlotMixin.
    fig_size : tuple of (float, float) or None, optional
        Figure dimensions in inches. Inherited from PlotMixin.
    x_label : str or None, optional
        x-axis label; falls back to the result's col_label when None.
    y_label : str or None, optional
        y-axis label; falls back to the result's row_label when None.
    tight_layout : bool, optional
        Whether to call plt.tight_layout() before saving. Defaults to True.
    cbar : bool, optional
        Whether to display the colour bar. Defaults to True.
    cmap : str, optional
        Matplotlib/seaborn colour map name. Defaults to "viridis".
    """

    cbar: bool = True
    cmap: str = "viridis"


class HeatMapExporter(ExportationStrategy):
    """Renders a MatrixResult as an annotated seaborn heat map and saves it.

    Parameters
    ----------
    config : HeatMapExporterConfig
        Configuration specifying figure appearance and the output path.
    """

    def __init__(self, config: HeatMapExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        """Render the matrix as a heat map and save the figure to disk.

        Parameters
        ----------
        matrix_result : MatrixResult
            The pairwise analysis result to visualise.
        """
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
