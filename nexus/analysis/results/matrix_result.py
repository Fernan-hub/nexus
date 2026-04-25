from pandas import DataFrame

from nexus.analysis.interfaces import AnalysisResult
from nexus.exporting.interfaces import ExporterStrategy


class MatrixResult(AnalysisResult):
    def __init__(self, algorithm: str, matrix: DataFrame) -> None:
        super().__init__(algorithm)
        self.matrix = matrix

    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        exporter_strategy.export_matrix_result(self)
