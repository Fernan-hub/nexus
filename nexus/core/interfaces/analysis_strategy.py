from abc import ABC, abstractmethod
from neo.core.dataobject import DataObject

from nexus.core.interfaces.analysis_result import AnalysisResult


class AnalysisStrategy(ABC):
    @abstractmethod
    def run_analysis(self, signals: list[DataObject]) -> AnalysisResult:
        pass
