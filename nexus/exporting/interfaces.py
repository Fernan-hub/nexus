from abc import ABC

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.analysis.results import MatrixResult, ScalarResult, VectorResult


class ExporterStrategy(ABC):
    def __init__(self, output_path: str | None) -> None:
        self.output_path = output_path

    def export_matrix_result(self, matrix_result: "MatrixResult") -> None:
        raise NotImplementedError(
            "ExporterStrategy subclasses must implement export_matrix_result method."
        )

    def export_vector_result(self, vector_result: "VectorResult") -> None:
        raise NotImplementedError(
            "ExporterStrategy subclasses must implement export_vector_result method."
        )

    def export_scalar_result(self, scalar_result: "ScalarResult") -> None:
        raise NotImplementedError(
            "ExporterStrategy subclasses must implement export_scalar_result method."
        )
