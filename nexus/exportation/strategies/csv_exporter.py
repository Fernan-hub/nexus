"""Exportation strategy that writes analysis results to CSV files."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from nexus.exportation.interfaces import ExportationStrategy
from nexus.exportation.models import ExportationStrategyConfig

if TYPE_CHECKING:
    from nexus.analysis.interfaces import AnalysisResult
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


@dataclass
class CSVExporterConfig(ExportationStrategyConfig):
    """Configuration for CSVExporter.

    Parameters
    ----------
    output_file_path : str or None, optional
        Destination path for the CSV file. Inherited from ExportationStrategyConfig.
    delimiter : str, optional
        Column separator used in the output file. Defaults to ",".
    """

    delimiter: str = ","


class CSVExporter(ExportationStrategy):
    """Exports analysis results to CSV files via pandas.

    Parameters
    ----------
    config : CSVExporterConfig
        Configuration specifying the output path and delimiter.
    """

    def __init__(self, config: CSVExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        """Write a MatrixResult to CSV as a pivot table (rows x columns).

        Parameters
        ----------
        matrix_result : MatrixResult
            The pairwise analysis result to serialise.
        """
        pivot_df = (
            matrix_result.matrix_df.pivot(index="row", columns="column", values="value")
            .rename_axis(f"{matrix_result.row_label}/{matrix_result.col_label}", axis=0)
            .rename_axis(None, axis=1)
        )
        self._dataframe_to_csv(pivot_df, matrix_result, index=True)

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        """Write a VectorResult DataFrame to CSV.

        Parameters
        ----------
        vector_result : VectorResult
            The per-signal analysis result to serialise.
        """
        self._dataframe_to_csv(vector_result.vector, vector_result)

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        """Write a ScalarResult to a single-row CSV.

        Parameters
        ----------
        scalar_result : ScalarResult
            The single-value analysis result to serialise.
        """
        self._dataframe_to_csv(
            pd.DataFrame([scalar_result.value], columns=[scalar_result.label]),
            scalar_result,
        )

    def _dataframe_to_csv(
        self, df: pd.DataFrame, analysis_result: "AnalysisResult", index: bool = False
    ) -> None:
        output_file_path = self._get_output_file_path(analysis_result)
        df.to_csv(output_file_path, index=index, sep=self._config.delimiter)
