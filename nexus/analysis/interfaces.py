from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from nexus.common.interfaces import FilterCriteria
from nexus.exporting.interfaces import ExporterStrategy


class AnalysisResult(ABC):
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @abstractmethod
    def accept(self, exporter_strategy: ExporterStrategy) -> None:
        pass


DataInputT = TypeVar("DataInputT")


class AnalysisStrategy(ABC, Generic[DataInputT]):
    @property
    @abstractmethod
    def filter_criteria_type(self) -> type[FilterCriteria]:
        pass

    @property
    @abstractmethod
    def data_input_type(self) -> type[DataInputT]:
        pass

    @abstractmethod
    def run_analysis(self, inputs: DataInputT) -> AnalysisResult:
        pass
