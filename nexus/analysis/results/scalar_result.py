from nexus.analysis.interfaces import AnalysisResult
from nexus.exporting.interfaces import ExportationStrategy


class ScalarResult(AnalysisResult):
    def __init__(self, algorithm: str, scalar: float) -> None:
        super().__init__(algorithm)
        self.scalar = scalar

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_scalar_result(self)
