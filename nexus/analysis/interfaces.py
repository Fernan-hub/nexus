"""Abstract base classes for analysis strategies, results, and estimator configs."""

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, TypeVar, Any

from infomeasure.utils.types import EstimatorType, LogBaseType

from nexus.common.interfaces import FilterCriteria
from nexus.exportation.interfaces import ExportationStrategy


class AnalysisResult(ABC):
    """Base class for all analysis results.

    Parameters
    ----------
    algorithm : str
        Name of the analysis algorithm that produced this result.
    """

    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @abstractmethod
    def accept(self, exporter_strategy: ExportationStrategy) -> None:
        """Dispatch this result to the appropriate exporter method.

        Parameters
        ----------
        exporter_strategy : ExportationStrategy
            The exporter that will consume this result.
        """


DataInputT = TypeVar("DataInputT")


class AnalysisStrategy(ABC, Generic[DataInputT]):
    """Abstract base class for analysis strategies."""

    @property
    @abstractmethod
    def filter_criteria_type(self) -> type[FilterCriteria]:
        """Return the FilterCriteria subclass used to query proxies for this strategy.

        Returns
        -------
        type[FilterCriteria]
            The FilterCriteria subclass specific to this strategy.
        """

    @property
    @abstractmethod
    def data_input_type(self) -> type[DataInputT]:
        """Return the data input dataclass type expected by this strategy.

        Returns
        -------
        type[DataInputT]
            The data input type specific to this strategy.
        """

    @abstractmethod
    def run_analysis(self, data_input: DataInputT) -> AnalysisResult:
        """Run the analysis on the provided data and return a result.

        Parameters
        ----------
        data_input : DataInputT
            The loaded data, structured as the strategy-specific input type.

        Returns
        -------
        AnalysisResult
            The computed analysis result.
        """


class EstimatorConfigBase(ABC):
    """Abstract base class for infomeasure estimator configurations.

    Subclasses are dataclasses whose fields are passed as keyword arguments
    to the estimator constructor via ``get_estimator_kwargs``.
    """

    base: LogBaseType = 2

    @abstractmethod
    def get_approach_name(self) -> str:
        """Return the estimator approach identifier (e.g. ``'discrete'``, ``'kernel'``).

        Returns
        -------
        str
            Short string identifying the estimation approach.
        """

    @abstractmethod
    def get_estimator_class(self) -> EstimatorType:
        """Return the estimator class to use for unconditional estimation.

        Returns
        -------
        EstimatorType
            The infomeasure estimator class.
        """

    def get_conditional_estimator_class(self) -> EstimatorType:
        """Return the estimator class to use for conditional estimation.

        Returns
        -------
        EstimatorType
            The infomeasure conditional estimator class.

        Raises
        ------
        NotImplementedError
            If the estimator does not support conditional estimation.
        """
        raise NotImplementedError(
            "This estimator does not support conditional estimation."
        )

    def get_estimator_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments to pass to the estimator constructor.

        Returns
        -------
        dict[str, Any]
            Mapping of field names to values, excluding private fields.
        """
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


class EstimatorBasedAnalysisStrategy(AnalysisStrategy[DataInputT], ABC):
    """Base class for analysis strategies that delegate to an Infomeasure estimator.

    Parameters
    ----------
    config : EstimatorConfigBase
        Estimator configuration used to instantiate the estimator.
    """

    def __init__(self, config: EstimatorConfigBase) -> None:
        self._config = config
        self._estimator_class = config.get_estimator_class()
        self._estimator_kwargs = config.get_estimator_kwargs()
