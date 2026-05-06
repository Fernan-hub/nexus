from abc import ABC

from typing import TYPE_CHECKING

from nexus.exportation.models import ExportationStrategyConfig

if TYPE_CHECKING:
    from nexus.analysis.interfaces import AnalysisResult
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


class ExportationStrategy(ABC):
    def __init__(self, config: ExportationStrategyConfig) -> None:
        self._config = config

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting matrix results."
        )

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting vector results."
        )

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exporting scalar results."
        )

    def _get_output_file_path(self, analysis_result: "AnalysisResult") -> str:
        return self._config.output_file_path or analysis_result.algorithm
