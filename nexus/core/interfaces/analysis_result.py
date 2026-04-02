from abc import ABC, abstractmethod


from nexus.core.interfaces.exporter_strategy import ExporterStrategy


class AnalysisResult(ABC):
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @abstractmethod
    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        pass
