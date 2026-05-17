from nexus.analysis.interfaces import AnalysisResult
from nexus.exportation.interfaces import ExportationStrategy


class ScalarResult(AnalysisResult):
    def __init__(self, algorithm: str, value: float, label: str) -> None:
        super().__init__(algorithm)
        self.value = value
        self.label = label

    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        exporter_strategy.export_scalar_result(self)
