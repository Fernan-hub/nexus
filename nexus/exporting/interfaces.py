from abc import ABC

from typing import TYPE_CHECKING

from nexus.exporting.models import ExportationStrategyConfig

if TYPE_CHECKING:
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


class ExportationStrategy(ABC):
    def __init__(self, config: ExportationStrategyConfig) -> None:
        self._config = config

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        raise NotImplementedError(
            "ExportationStrategy subclasses must implement export_matrix_result method."
        )

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        raise NotImplementedError(
            "ExportationStrategy subclasses must implement export_vector_result method."
        )

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        raise NotImplementedError(
            "ExportationStrategy subclasses must implement export_scalar_result method."
        )
