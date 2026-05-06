from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, TypeVar, Any

from infomeasure.utils.types import EstimatorType, LogBaseType

from nexus.common.interfaces import FilterCriteria
from nexus.exportation.interfaces import ExportationStrategy


class AnalysisResult(ABC):
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @abstractmethod
    def accept(self, exporter_strategy: ExportationStrategy) -> None:
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
    def run_analysis(self, data_input: DataInputT) -> AnalysisResult:
        pass


class EstimatorConfigBase(ABC):
    base: LogBaseType = 2

    @abstractmethod
    def get_estimator_class(self) -> EstimatorType:
        pass

    def get_conditional_estimator_class(self) -> EstimatorType:
        raise NotImplementedError(
            "This estimator does not support conditional estimation."
        )

    def get_estimator_kwargs(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


class EstimatorBasedAnalysisStrategy(AnalysisStrategy[DataInputT], ABC):
    def __init__(self, config: EstimatorConfigBase) -> None:
        self._config = config
        self._estimator_class = config.get_estimator_class()
        self._estimator_kwargs = config.get_estimator_kwargs()
