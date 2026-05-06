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
    delimiter: str = ","


class CSVExporter(ExportationStrategy):
    def __init__(self, config: CSVExporterConfig) -> None:
        super().__init__(config)

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        self._dataframe_to_csv(matrix_result.matrix_df, matrix_result)

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        self._dataframe_to_csv(vector_result.vector, vector_result)

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        self._dataframe_to_csv(
            pd.DataFrame([scalar_result.value], columns=["value"]), scalar_result
        )

    def _dataframe_to_csv(
        self, df: pd.DataFrame, analysis_result: "AnalysisResult"
    ) -> None:
        output_file_path = self._get_output_file_path(analysis_result)
        df.to_csv(output_file_path, index=False, sep=self._config.delimiter)
