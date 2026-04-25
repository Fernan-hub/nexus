from nexus.analysis.interfaces import AnalysisResult
from nexus.exporting.interfaces import ExporterStrategy


class ScalarResult(AnalysisResult):
    def __init__(self, algorithm: str, scalar: float) -> None:
        super().__init__(algorithm)
        self.scalar = scalar

    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        exporter_strategy.export_scalar_result(self)
