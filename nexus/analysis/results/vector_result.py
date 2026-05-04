from pandas import DataFrame

from nexus.analysis.interfaces import AnalysisResult
from nexus.exporting.interfaces import ExportationStrategy


class VectorResult(AnalysisResult):
    def __init__(self, algorithm: str, vector: DataFrame) -> None:
        super().__init__(algorithm)
        self.vector = vector

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_vector_result(self)
