"""Analysis strategy for computing (conditional) entropy over a set of signals."""

from dataclasses import dataclass

from neo.core.dataobject import DataObject
import pandas as pd

from nexus.analysis.estimators.entropy import EntropyConfig
from nexus.analysis.interfaces import EstimatorBasedAnalysisStrategy
from nexus.common.interfaces import FilterCriteria
from nexus.types import Criteria
from nexus.analysis.results.vector_result import VectorResult


@dataclass
class EntropyStrategyFilterCriteria(FilterCriteria):
    """Filter criteria for selecting signals used in entropy computation.

    Parameters
    ----------
    data : Criteria or None, optional
        Criteria for selecting the signals whose entropy is computed.
    cond : Criteria or None, optional
        Criteria for selecting the conditioning signal. When provided,
        conditional entropy H(data | cond) is computed.
    """

    data: Criteria | None = None
    cond: Criteria | None = None


@dataclass
class EntropyStrategyDataInput:
    """Data input container for entropy computation.

    Parameters
    ----------
    data : list[DataObject]
        Signals whose entropy is computed.
    cond : list[DataObject] or None, optional
        Conditioning signal. When provided, conditional entropy is computed.
    """

    data: list[DataObject]
    cond: list[DataObject] | None = None


class EntropyStrategy(EstimatorBasedAnalysisStrategy):
    """Analysis strategy that computes (conditional) entropy for each input signal.

    Parameters
    ----------
    config : EntropyConfig
        Estimator configuration specifying the estimation approach.
    """

    _ALGORITHM_NAME = "entropy"

    def __init__(self, config: EntropyConfig) -> None:
        super().__init__(config)

    @property
    def filter_criteria_type(self) -> type[EntropyStrategyFilterCriteria]:
        """Return the filter criteria type for entropy analysis.

        Returns
        -------
        type[EntropyStrategyFilterCriteria]
            The criteria class used to select input signals.
        """
        return EntropyStrategyFilterCriteria

    @property
    def data_input_type(self) -> type[EntropyStrategyDataInput]:
        """Return the data input type for entropy analysis.

        Returns
        -------
        type[EntropyStrategyDataInput]
            The data input class for this strategy.
        """
        return EntropyStrategyDataInput

    def _compute_joint_entropy(
        self, signal: DataObject, cond: list[DataObject]
    ) -> float:
        joint_data = (signal, cond[0])
        estimator = self._estimator_class(joint_data, **self._estimator_kwargs)
        return estimator.result()

    def _compute_simple_entropy(self, signal: DataObject) -> float:
        estimator = self._estimator_class(signal, **self._estimator_kwargs)
        return estimator.result()

    def _get_estimator_result(
        self, signal: DataObject, cond: list[DataObject] | None
    ) -> float:
        if cond is None or len(cond) == 0:
            return self._compute_simple_entropy(signal)

        joint_entropy = self._compute_joint_entropy(signal, cond)
        cond_entropy = self._compute_simple_entropy(cond[0])
        return joint_entropy - cond_entropy

    def _get_column_name(self, cond: list[DataObject] | None) -> str:
        if cond is None or len(cond) == 0:
            return "entropy"
        if cond[0].name is not None:
            return f"cond_entropy_given_{cond[0].name}"
        return "cond_entropy"

    def run_analysis(self, data_input: EntropyStrategyDataInput) -> VectorResult:
        """Compute entropy (or conditional entropy) for each signal in ``data``.

        Parameters
        ----------
        data_input : EntropyStrategyDataInput
            Loaded signals and optional conditioning signal.

        Returns
        -------
        VectorResult
            A VectorResult whose DataFrame contains one entropy value per signal.
        """
        results_list = [
            self._get_estimator_result(data, data_input.cond)
            for data in data_input.data
        ]
        columns = [self._get_column_name(data_input.cond)]
        results_df = pd.DataFrame(results_list, columns=columns)
        return VectorResult(algorithm=self._ALGORITHM_NAME, vector=results_df)
