from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.exporting.interfaces import ExporterStrategy


class AnalysisResult(ABC):
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @abstractmethod
    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        pass


class AnalysisStrategy(ABC):
    @abstractmethod
    def run_analysis(
        self, *args: list[DataObject], **kwargs: list[DataObject]
    ) -> AnalysisResult:
        pass
