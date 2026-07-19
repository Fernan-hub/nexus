"""Analysis strategy for computing the joint entropy of a set of signals."""

from dataclasses import dataclass

from neo.core.dataobject import DataObject

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria
from nexus.analysis.results.scalar_result import ScalarResult


@dataclass
class JointEntropyStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in joint entropy computation.

    Parameters
    ----------
    data : Criteria or None, optional
        Criteria for selecting all signals whose joint entropy is computed.
    """

    data: Criteria | None = None


@dataclass
class JointEntropyStrategyDataInput:
    """Data input container for joint entropy computation.

    Parameters
    ----------
    data : list[DataObject]
        Signals over which joint entropy H(X1, X2, ..., Xn) is computed.
    """

    data: list[DataObject]


class JointEntropyStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes the joint entropy of all input signals.

    Parameters
    ----------
    config : EntropyConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "joint_entropy"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[JointEntropyStrategyFilterCriteria]:
        """Return the filter criteria type for joint entropy analysis.

        Returns
        -------
        type[JointEntropyStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return JointEntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[JointEntropyStrategyDataInput]:
        """Return the data input type for joint entropy analysis.

        Returns
        -------
        type[JointEntropyStrategyDataInput]
            The data input class for this strategy.
        """
        return JointEntropyStrategyDataInput

    def _get_estimator_result(self, signals: list[DataObject]) -> float:
        joint_data = tuple(signals)
        estimator = self._estimator_class(joint_data, **self._estimator_kwargs)
        return estimator.result()

    def _get_column_name(self, signals: list[DataObject] | None) -> str:
        signals_names = [
            signal.name or f"signal_{i}" for i, signal in enumerate(signals)
        ]
        return "Joint entropy of " + ", ".join(signals_names)

    def run_analysis(self, data_input: JointEntropyStrategyDataInput) -> ScalarResult:
        """Compute the joint entropy of all signals in ``data_input.data``.

        Parameters
        ----------
        data_input : JointEntropyStrategyDataInput
            Loaded signals to compute joint entropy over.

        Returns
        -------
        ScalarResult
            A ScalarResult containing the single joint entropy value.
        """
        joint_entropy_result = self._get_estimator_result(data_input.data)
        return ScalarResult(
            algorithm=self._ALGORITHM_NAME,
            value=joint_entropy_result,
            label=self._get_column_name(data_input.data),
        )
