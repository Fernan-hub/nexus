from dataclasses import dataclass, asdict

from nexus.analysis.interfaces import AnalysisResult
from nexus.exporting.interfaces import ExportationStrategy


@dataclass
class MatrixElement:
    row: str
    column: str
    value: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


class MatrixResult(AnalysisResult):
    def __init__(self, algorithm: str, matrix: list[MatrixElement]) -> None:
        super().__init__(algorithm)
        self.matrix = matrix

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_matrix_result(self)
